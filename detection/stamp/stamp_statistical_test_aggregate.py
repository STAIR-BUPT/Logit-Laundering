"""
STAMP statistical test — supports normal model and Adaptive Fusion.
Detects whether a model was trained on specific data (data contamination detection).

Example (normal model):

python detection/stamp/stamp_statistical_test_aggregate.py \
    --dataset_file /absolute/path/to/stamp_dataset.json \
    --public_key origin_watermarked \
    --private_keys watermarked_text_1 watermarked_text_2 \
    --model_type normal \
    --normal_model_path /absolute/path/to/trained-base-model \
    --group_by_field book_name

Example (adaptive fusion):

python detection/stamp/stamp_statistical_test_aggregate.py \
    --dataset_file /absolute/path/to/stamp_dataset.json \
    --public_key origin_watermarked \
    --private_keys watermarked_text_1 watermarked_text_2 \
    --model_type adaptive_fusion \
    --base_model_path /absolute/path/to/trained-base-model \
    --assist_model_path /absolute/path/to/trained-assistant-model \
    --alpha_min 0.17 --alpha_max 0.56 --alpha_s 1.0 \
    --group_by_field book_name
"""
import json
import numpy as np
import argparse
import math
import sys
import os
import zlib
import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# Add project root to path
RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "ward_core"))

# ======================
# Model configuration
# ======================
GPU_ID = None


def resolve_device(gpu_id=None):
    if not torch.cuda.is_available():
        return "cpu"
    if gpu_id is None:
        return "cuda"
    return f"cuda:{gpu_id}"


DEVICE = resolve_device(GPU_ID)

# Model type: "adaptive_fusion" or "normal"
MODEL_TYPE = "adaptive_fusion"  # options: "adaptive_fusion" / "normal"

# Normal model path (used when MODEL_TYPE="normal")
NORMAL_MODEL_PATH = ""  # set via --normal_model_path (e.g. /...)

# Adaptive Fusion model paths (used when MODEL_TYPE="adaptive_fusion")
BASE_MODEL_PATH = ""  # set via --base_model_path (e.g. /...)
ASSIST_MODEL_PATH = ""  # set via --assist_model_path (e.g. /...)

ALPHA_MIN = 0.17
ALPHA_MAX = 0.38
ALPHA_S = 1.0
GROUP_BY_FIELD = None
TEXT_KEY = "watermarked_text"
MINK_RATIO = 0.2
METRIC_NAMES = ("ppl", "mink", "minkplus", "lowercase", "zlib")
DISPLAY_METRIC_NAMES = {
    "ppl": "ppl",
    "mink": "mink",
    "minkplus": "mink++",
    "lowercase": "lowercase",
    "zlib": "zlib",
}

# ======================
# Load model (supports both normal model and Adaptive Fusion)
# ======================
def load_normal_model():
    """Load the normal (non-fusion) model."""
    print(f"🚀 Loading normal model...")
    print(f"   Model path: {NORMAL_MODEL_PATH}")
    
    model = AutoModelForCausalLM.from_pretrained(
        NORMAL_MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(NORMAL_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model.tokenizer = tokenizer
    print("✅ Normal model loaded\n")
    return model, tokenizer


def load_adaptive_fusion_model():
    """Load the base and assistant models required for Adaptive Fusion."""
    print("🚀 Loading Adaptive Fusion model...")
    print(f"   Base model: {BASE_MODEL_PATH}")
    print(f"   Assistant model: {ASSIST_MODEL_PATH}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    assist_model = AutoModelForCausalLM.from_pretrained(
        ASSIST_MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model_bundle = {
        "base_model": base_model,
        "assist_model": assist_model,
    }
    print("✅ Adaptive Fusion model loaded\n")
    return model_bundle, tokenizer


def compute_adaptive_alpha(base_logits_last):
    """Compute alpha_t following the formula in Top-1_adaptive_fusion_aggregate.py."""
    top2_val = torch.topk(base_logits_last.float(), k=2).values
    margin_t = top2_val[0].item() - top2_val[1].item()
    alpha_t = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * margin_t / (margin_t + ALPHA_S)
    return alpha_t, margin_t


def get_adaptive_fused_logits(model_bundle, input_ids, attention_mask=None):
    """Return the last-step logits after adaptive fusion."""
    base_model = model_bundle["base_model"]
    assist_model = model_bundle["assist_model"]

    model_kwargs = {"input_ids": input_ids, "return_dict": True}
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask

    base_last = base_model(**model_kwargs).logits[0, -1, :]
    alpha_t, margin_t = compute_adaptive_alpha(base_last)
    assist_last = assist_model(**model_kwargs).logits[0, -1, :]
    fused_logits = base_last - alpha_t * assist_last
    return fused_logits, alpha_t, margin_t


def load_model(model_type=MODEL_TYPE):
    """
    Load model according to configuration.
    
    Parameters:
        model_type: "adaptive_fusion" or "normal"
    
    Returns:
        model, tokenizer
    """
    if model_type == "normal":
        return load_normal_model()
    elif model_type == "adaptive_fusion":
        return load_adaptive_fusion_model()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'normal' or 'adaptive_fusion'")

def compute_text_token_statistics(model, tokenizer, text):
    """Compute per-token statistics for text."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1024
    ).to(DEVICE)

    if MODEL_TYPE == "adaptive_fusion":
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs.get("attention_mask")
        if len(input_ids) < 2:
            empty = np.array([], dtype=float)
            return {
                "token_log_probs": empty,
                "token_mean_log_probs": empty,
                "token_std_log_probs": empty,
            }

        token_log_probs = []
        token_mean_log_probs = []
        token_std_log_probs = []
        for i in range(1, len(input_ids)):
            prefix_ids = input_ids[:i].unsqueeze(0)
            prefix_attn = None
            if attention_mask is not None:
                prefix_attn = attention_mask[0, :i].unsqueeze(0)

            fused_logits, _, _ = get_adaptive_fused_logits(
                model,
                prefix_ids,
                attention_mask=prefix_attn,
            )
            log_probs = torch.log_softmax(fused_logits.float(), dim=-1)
            probs = torch.softmax(fused_logits.float(), dim=-1)
            target = input_ids[i].item()
            token_log_probs.append(log_probs[target].item())
            mean_log_prob = torch.sum(probs * log_probs).item()
            second_moment = torch.sum(probs * torch.square(log_probs)).item()
            variance = max(0.0, second_moment - mean_log_prob ** 2)
            token_mean_log_probs.append(mean_log_prob)
            token_std_log_probs.append(math.sqrt(variance))
        return {
            "token_log_probs": np.array(token_log_probs, dtype=float),
            "token_mean_log_probs": np.array(token_mean_log_probs, dtype=float),
            "token_std_log_probs": np.array(token_std_log_probs, dtype=float),
        }

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = torch.softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        token_mean_log_probs = (probs * log_probs).sum(dim=-1)
        token_var_log_probs = (probs * torch.square(log_probs)).sum(dim=-1) - torch.square(token_mean_log_probs)
        token_std_log_probs = torch.sqrt(torch.clamp(token_var_log_probs, min=0.0))
    return {
        "token_log_probs": token_log_probs[0].detach().cpu().numpy().astype(float),
        "token_mean_log_probs": token_mean_log_probs[0].detach().cpu().numpy().astype(float),
        "token_std_log_probs": token_std_log_probs[0].detach().cpu().numpy().astype(float),
    }


def compute_text_log_probs(model, tokenizer, text):
    """Compatibility wrapper: return only the per-token log-prob sequence."""
    return compute_text_token_statistics(model, tokenizer, text)["token_log_probs"]


def compute_ppl_from_log_probs(log_probs):
    """Compute PPL from per-token log-prob sequence."""
    if len(log_probs) == 0:
        return float("inf")
    return float(math.exp(-float(np.mean(log_probs))))


def compute_mink_from_log_probs(log_probs, mink_ratio=MINK_RATIO):
    """Compute min-k metric as defined in perp.py."""
    if len(log_probs) == 0:
        return float("inf")
    k_length = max(1, int(len(log_probs) * mink_ratio))
    mink_slice = np.sort(log_probs)[:k_length]
    return float(-np.mean(mink_slice))


def compute_minkplus_from_token_statistics(
    log_probs,
    token_mean_log_probs,
    token_std_log_probs,
    mink_ratio=MINK_RATIO,
):
    """Compute min-k++ metric as defined in perp.py."""
    if len(log_probs) == 0:
        return float("inf")
    safe_std = np.maximum(np.asarray(token_std_log_probs, dtype=float), 1e-12)
    minkplus_scores = (
        np.asarray(log_probs, dtype=float) - np.asarray(token_mean_log_probs, dtype=float)
    ) / safe_std
    k_length = max(1, int(len(minkplus_scores) * mink_ratio))
    minkplus_slice = np.sort(minkplus_scores)[:k_length]
    return float(-np.mean(minkplus_slice))


def compute_metrics_from_log_probs(
    text,
    log_probs,
    lowercase_log_probs,
    token_mean_log_probs=None,
    token_std_log_probs=None,
    mink_ratio=MINK_RATIO,
):
    """Derive metrics from log-prob sequences of the original and lowercased text."""
    ppl = compute_ppl_from_log_probs(log_probs)
    mink = compute_mink_from_log_probs(log_probs, mink_ratio=mink_ratio)
    if token_mean_log_probs is None or token_std_log_probs is None:
        minkplus = float("inf")
    else:
        minkplus = compute_minkplus_from_token_statistics(
            log_probs,
            token_mean_log_probs,
            token_std_log_probs,
            mink_ratio=mink_ratio,
        )

    nll = -float(np.mean(log_probs)) if len(log_probs) > 0 else float("inf")
    lowercase_nll = (
        -float(np.mean(lowercase_log_probs))
        if len(lowercase_log_probs) > 0
        else float("inf")
    )
    if not np.isfinite(nll) or nll == 0.0 or not np.isfinite(lowercase_nll):
        lowercase_metric = float("inf")
    else:
        lowercase_metric = float(-(lowercase_nll / nll))

    compressed_len = len(zlib.compress(text.encode("utf-8")))
    if not np.isfinite(nll) or compressed_len == 0:
        zlib_metric = float("inf")
    else:
        zlib_metric = float(nll / compressed_len)

    return {
        "ppl": ppl,
        "mink": mink,
        "minkplus": minkplus,
        "lowercase": lowercase_metric,
        "zlib": zlib_metric,
    }


def compute_text_metrics(model, tokenizer, text):
    """Compute all metrics for a single text sample."""
    try:
        token_stats = compute_text_token_statistics(model, tokenizer, text)
        lowercase_token_stats = compute_text_token_statistics(model, tokenizer, text.lower())
        return compute_metrics_from_log_probs(
            text,
            token_stats["token_log_probs"],
            lowercase_token_stats["token_log_probs"],
            token_mean_log_probs=token_stats["token_mean_log_probs"],
            token_std_log_probs=token_stats["token_std_log_probs"],
            mink_ratio=MINK_RATIO,
        )
    except Exception as exc:
        print(f"Metric computation failed: {exc}")
        return {metric_name: float("inf") for metric_name in METRIC_NAMES}

def remove_outliers(metrics, remove_frac=0.05, outliers="clip"):
    """
    Remove outliers from a metric array.
    
    Args:
        metrics: metric array
        remove_frac: fraction to remove (default 5%, i.e. 2.5% from each tail)
        outliers: handling method ("clip" | "zero" | "mean" | "keep")
    """
    # Sort to find outlier indices
    sorted_ids = np.argsort(metrics)
    
    total_elements = len(metrics)
    elements_to_remove_each_side = int(total_elements * remove_frac / 2)
    
    if elements_to_remove_each_side * 2 > total_elements:
        raise ValueError("remove_frac too large; no elements remain")
    if elements_to_remove_each_side == 0 or total_elements == 0:
        return np.copy(metrics)
    
    # Find the lowest and highest outlier indices
    lowest_ids = sorted_ids[:elements_to_remove_each_side]
    highest_ids = sorted_ids[-elements_to_remove_each_side:]
    all_ids = np.concatenate((lowest_ids, highest_ids))
    
    trimmed_metrics = np.copy(metrics)
    
    if outliers == "zero":
        trimmed_metrics[all_ids] = 0
    elif outliers == "mean":
        trimmed_metrics[all_ids] = np.mean(trimmed_metrics)
    elif outliers == "clip":
        # Clip to boundary values (recommended)
        highest_val_permissible = trimmed_metrics[highest_ids[0]]
        lowest_val_permissible = trimmed_metrics[lowest_ids[-1]]
        trimmed_metrics[highest_ids] = highest_val_permissible
        trimmed_metrics[lowest_ids] = lowest_val_permissible
    elif outliers == "keep":
        pass  # keep original values
    else:
        raise ValueError(f"Unknown outliers parameter: {outliers}")
    
    return trimmed_metrics

def calculate_metrics_for_texts(model, tokenizer, texts):
    """Compute metrics for a batch of texts."""
    metrics_by_name = {metric_name: [] for metric_name in METRIC_NAMES}

    for text in texts:
        metric_bundle = compute_text_metrics(model, tokenizer, text)
        for metric_name in METRIC_NAMES:
            metrics_by_name[metric_name].append(metric_bundle[metric_name])
    return {
        metric_name: np.array(values, dtype=float)
        for metric_name, values in metrics_by_name.items()
    }


def load_json_or_jsonl(path):
    """Read a JSON or JSONL file."""
    if path.endswith(".json") and not path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"JSON file format error, expected list: {path}")
        return data

    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def collect_group_texts(records, group_by_field, text_key):
    """Collect text from records by field name."""
    grouped_texts = defaultdict(list)
    for record in records:
        text = str(record.get(text_key, "")).strip()
        if not text:
            continue
        group_name = str(record.get(group_by_field) or f"<missing-{group_by_field}>")
        grouped_texts[group_name].append(text)
    return dict(sorted(grouped_texts.items()))


def collect_group_samples(records, group_by_field, text_key, sample_id_field="id"):
    """Collect text with sample_id from records by field name."""
    grouped_samples = defaultdict(list)
    for index, record in enumerate(records):
        text = str(record.get(text_key, "")).strip()
        if not text:
            continue
        group_name = str(record.get(group_by_field) or f"<missing-{group_by_field}>")
        sample_id = record.get(sample_id_field, index)
        grouped_samples[group_name].append(
            {
                "sample_id": str(sample_id),
                "text": text,
            }
        )
    return dict(sorted(grouped_samples.items()))


def format_metric_bundle(metric_bundle):
    return (
        f"ppl={metric_bundle['ppl']:.4f}, "
        f"mink={metric_bundle['mink']:.4f}, "
        f"mink++={metric_bundle['minkplus']:.4f}, "
        f"lowercase={metric_bundle['lowercase']:.4f}, "
        f"zlib={metric_bundle['zlib']:.4f}"
    )


def calculate_group_mean_metrics(model, tokenizer, grouped_texts, dataset_label):
    """Compute mean of each metric per group."""
    group_metrics = {metric_name: {} for metric_name in METRIC_NAMES}
    total_groups = len(grouped_texts)

    for index, (group_name, texts) in enumerate(grouped_texts.items(), start=1):
        print(
            f"\n[{dataset_label}] Processing group {index}/{total_groups}: "
            f"{group_name} (samples={len(texts)})"
        )
        metrics_by_name = calculate_metrics_for_texts(model, tokenizer, texts)
        group_mean_bundle = {}
        for metric_name in METRIC_NAMES:
            mean_value = float(np.mean(metrics_by_name[metric_name]))
            group_metrics[metric_name][group_name] = mean_value
            group_mean_bundle[metric_name] = mean_value
        print(f"   [{dataset_label}] {group_name}: {format_metric_bundle(group_mean_bundle)}")

    return group_metrics


def calculate_group_metric_details(model, tokenizer, grouped_samples, dataset_label):
    """Compute mean of each metric per group and retain per-sample metrics."""
    group_mean_metrics = {metric_name: {} for metric_name in METRIC_NAMES}
    group_sample_metrics = {metric_name: {} for metric_name in METRIC_NAMES}
    total_groups = len(grouped_samples)

    for index, (group_name, samples) in enumerate(grouped_samples.items(), start=1):
        texts = [sample["text"] for sample in samples]
        print(
            f"\n[{dataset_label}] Processing group {index}/{total_groups}: "
            f"{group_name} (samples={len(texts)})"
        )
        metrics_by_name = calculate_metrics_for_texts(model, tokenizer, texts)
        group_mean_bundle = {}
        for metric_name in METRIC_NAMES:
            values = metrics_by_name[metric_name]
            group_mean_metrics[metric_name][group_name] = float(np.mean(values))
            group_sample_metrics[metric_name][group_name] = {
                sample["sample_id"]: float(values[i])
                for i, sample in enumerate(samples)
            }
            group_mean_bundle[metric_name] = group_mean_metrics[metric_name][group_name]
        print(f"   [{dataset_label}] {group_name}: {format_metric_bundle(group_mean_bundle)}")

    return {
        "mean_metrics": group_mean_metrics,
        "sample_metrics": group_sample_metrics,
    }


def aggregate_metrics_by_group(dataset, public_metrics, private_metrics_by_key, group_by_field):
    """Aggregate metrics by field, treating each group as one statistical unit."""
    groups = defaultdict(list)
    for index, item in enumerate(dataset):
        group_name = str(item.get(group_by_field) or f"<missing-{group_by_field}>")
        groups[group_name].append(index)

    group_names = []
    public_group_metrics = []
    private_group_metrics = {key: [] for key in private_metrics_by_key}
    group_summaries = []

    for group_name, indices in sorted(groups.items()):
        group_names.append(group_name)
        public_mean = float(np.mean(public_metrics[indices]))
        public_group_metrics.append(public_mean)

        private_means_by_key = {}
        for key, metrics in private_metrics_by_key.items():
            private_mean = float(np.mean(metrics[indices]))
            private_group_metrics[key].append(private_mean)
            private_means_by_key[key] = private_mean

        mean_private_metric = (
            float(np.mean(list(private_means_by_key.values())))
            if private_means_by_key
            else float("nan")
        )
        group_summaries.append(
            {
                "group_name": group_name,
                "sample_count": len(indices),
                "public_mean": public_mean,
                "private_means_by_key": private_means_by_key,
                "mean_private_metric": mean_private_metric,
            }
        )

    private_group_metrics = {
        key: np.array(values, dtype=float) for key, values in private_group_metrics.items()
    }
    if private_group_metrics:
        mean_private_metrics = np.mean(
            np.array(list(private_group_metrics.values()), dtype=float), axis=0
        )
    else:
        mean_private_metrics = np.array([], dtype=float)

    return {
        "group_names": group_names,
        "group_indices": dict(sorted(groups.items())),
        "public_metrics": np.array(public_group_metrics, dtype=float),
        "private_metrics_by_key": private_group_metrics,
        "mean_private_metrics": mean_private_metrics,
        "group_summaries": group_summaries,
    }


def align_group_metrics(public_group_metrics, private_group_metrics_by_key):
    """Align group-aggregated metrics across versions, retaining only groups common to all."""
    common_groups = set(public_group_metrics)
    for metrics in private_group_metrics_by_key.values():
        common_groups &= set(metrics)
    group_names = sorted(common_groups)

    aligned_public_metrics = np.array(
        [public_group_metrics[name] for name in group_names], dtype=float
    )
    aligned_private_metrics = {
        key: np.array([metrics[name] for name in group_names], dtype=float)
        for key, metrics in private_group_metrics_by_key.items()
    }
    if aligned_private_metrics:
        mean_private_metrics = np.mean(
            np.array(list(aligned_private_metrics.values()), dtype=float), axis=0
        )
    else:
        mean_private_metrics = np.array([], dtype=float)

    return {
        "group_names": group_names,
        "public_metrics": aligned_public_metrics,
        "private_metrics_by_key": aligned_private_metrics,
        "mean_private_metrics": mean_private_metrics,
    }


def compute_statistical_test_result(
    metric_name,
    public_metrics,
    private_metrics_by_key,
    use_paired_test=True,
    remove_outlier_frac=0.05,
    outlier_method="clip",
    group_by_field=None,
    group_summaries=None,
    raw_sample_count=None,
):
    """Run one statistical test and return result dict and cleaned metrics."""
    public_metrics = np.array(public_metrics, dtype=float)
    public_metrics_clean = remove_outliers(
        public_metrics,
        remove_frac=remove_outlier_frac,
        outliers=outlier_method,
    )

    cleaned_private_metrics_by_key = {}
    all_private_metrics = []
    for private_key, private_metrics in private_metrics_by_key.items():
        private_metrics = np.array(private_metrics, dtype=float)
        private_metrics_clean = remove_outliers(
            private_metrics,
            remove_frac=remove_outlier_frac,
            outliers=outlier_method,
        )
        cleaned_private_metrics_by_key[private_key] = private_metrics_clean
        all_private_metrics.append(private_metrics_clean)

    all_private_metrics = np.array(all_private_metrics, dtype=float)
    mean_private_metrics = np.mean(all_private_metrics, axis=0)

    if use_paired_test:
        test_result = stats.ttest_rel(
            public_metrics_clean,
            mean_private_metrics,
            alternative="less",
        )
    else:
        test_result = stats.ttest_ind(
            public_metrics_clean,
            mean_private_metrics,
            alternative="less",
        )

    p_value = test_result.pvalue
    t_statistic = test_result.statistic

    if p_value == 0.0 or p_value < 1e-300:
        n = len(public_metrics_clean)
        if use_paired_test:
            df = n - 1
            p_value_precise = stats.t.sf(-t_statistic, df)
        else:
            n2 = len(mean_private_metrics)
            df = n + n2 - 2
            p_value_precise = stats.t.sf(-t_statistic, df)

        if p_value_precise > 0:
            log10_p_value_display = np.log10(p_value_precise)
            if log10_p_value_display < -300:
                p_value_display = "< 1e-300"
                log10_p_value_display = -300.0
            else:
                p_value_display = f"{p_value_precise:.6e}"
        else:
            p_value_display = "< 1e-300"
            log10_p_value_display = -300.0
    else:
        p_value_precise = p_value
        p_value_display = f"{p_value:.6e}"
        log10_p_value_display = np.log10(p_value) if p_value > 0 else -np.inf

    group_average_summary = compute_group_average_summary(group_summaries)

    result = {
        "metric": metric_name,
        "analysis_unit": group_by_field if group_by_field else "sample",
        "group_by_field": group_by_field,
        "group_count": len(group_summaries) if group_summaries is not None else None,
        "group_summaries": group_summaries,
        "raw_sample_count": raw_sample_count,
        "test_sample_count": int(len(public_metrics_clean)),
        "t_statistic": float(t_statistic),
        "p_value": float(p_value_precise) if p_value_precise >= 1e-300 else 0.0,
        "p_value_display": p_value_display,
        "log10_p_value": float(log10_p_value_display),
        "contaminated": bool(p_value_precise < 0.05),
        "public_mean": float(np.mean(public_metrics_clean)),
        "private_mean": float(np.mean(mean_private_metrics)),
        "public_std": float(np.std(public_metrics_clean)),
        "private_std": float(np.std(mean_private_metrics)),
    }
    if group_average_summary is not None:
        result.update(group_average_summary)

    return result, public_metrics_clean, cleaned_private_metrics_by_key, mean_private_metrics


def format_p_value_for_display(p_value):
    """Format a p-value, handling very small values."""
    if p_value <= 0.0 or p_value < 1e-300:
        return 0.0, "< 1e-300", -300.0
    return float(p_value), f"{p_value:.6e}", float(np.log10(p_value))


def compute_group_average_summary(group_summaries):
    """Compute simple average of book_p_value / book_t_statistic across books."""
    if not group_summaries:
        return None

    mean_book_p_value = float(np.mean([summary["book_p_value"] for summary in group_summaries]))
    normalized_mean_book_p_value, mean_book_p_value_display, mean_book_log10_p_value = (
        format_p_value_for_display(mean_book_p_value)
    )
    detected_group_count = int(
        sum(1 for summary in group_summaries if summary.get("book_contaminated"))
    )
    group_count = int(len(group_summaries))

    return {
        "group_count": group_count,
        "mean_book_p_value": normalized_mean_book_p_value,
        "mean_book_p_value_display": mean_book_p_value_display,
        "mean_book_log10_p_value": mean_book_log10_p_value,
        "mean_book_t_statistic": float(
            np.mean([summary["book_t_statistic"] for summary in group_summaries])
        ),
        "mean_book_contaminated": bool(mean_book_p_value < 0.05),
        "detected_group_count": detected_group_count,
        "detected_group_fraction": float(detected_group_count / group_count),
    }


def build_group_test_summaries_from_index_groups(
    metric_name,
    group_indices,
    public_metrics,
    private_metrics_by_key,
    use_paired_test=True,
    remove_outlier_frac=0.05,
    outlier_method="clip",
):
    """Run a t-test on samples within each group."""
    summaries = []
    for group_name, indices in sorted(group_indices.items()):
        group_public_metrics = np.array(public_metrics[indices], dtype=float)
        group_private_metrics_by_key = {
            key: np.array(metrics[indices], dtype=float)
            for key, metrics in private_metrics_by_key.items()
        }
        private_means_by_key = {
            key: float(np.mean(values))
            for key, values in group_private_metrics_by_key.items()
        }
        test_result, _, _, _ = compute_statistical_test_result(
            metric_name=metric_name,
            public_metrics=group_public_metrics,
            private_metrics_by_key=group_private_metrics_by_key,
            use_paired_test=use_paired_test,
            remove_outlier_frac=remove_outlier_frac,
            outlier_method=outlier_method,
            raw_sample_count=len(indices),
        )
        summaries.append(
            {
                "group_name": group_name,
                "sample_count": int(len(indices)),
                "aligned_sample_count": int(len(indices)),
                "public_mean": float(np.mean(group_public_metrics)),
                "private_means_by_key": private_means_by_key,
                "mean_private_metric": float(np.mean(list(private_means_by_key.values()))),
                "book_t_statistic": test_result["t_statistic"],
                "book_p_value": test_result["p_value"],
                "book_p_value_display": test_result["p_value_display"],
                "book_log10_p_value": test_result["log10_p_value"],
                "book_contaminated": test_result["contaminated"],
                "test_sample_count": test_result["test_sample_count"],
            }
        )
    return summaries


def build_group_test_summaries_from_sample_metrics(
    metric_name,
    public_group_sample_metrics,
    private_group_sample_metrics_by_key,
    use_paired_test=True,
    remove_outlier_frac=0.05,
    outlier_method="clip",
):
    """Run a t-test on aligned samples within each group."""
    common_groups = set(public_group_sample_metrics)
    for metrics_by_group in private_group_sample_metrics_by_key.values():
        common_groups &= set(metrics_by_group)

    summaries = []
    for group_name in sorted(common_groups):
        private_samples_by_key = {
            key: metrics_by_group[group_name]
            for key, metrics_by_group in private_group_sample_metrics_by_key.items()
            if group_name in metrics_by_group
        }
        aligned = align_group_metrics(
            public_group_sample_metrics[group_name],
            private_samples_by_key,
        )
        if not aligned["group_names"]:
            continue

        private_means_by_key = {
            key: float(np.mean(list(sample_metrics.values())))
            for key, sample_metrics in private_samples_by_key.items()
        }
        test_result, _, _, _ = compute_statistical_test_result(
            metric_name=metric_name,
            public_metrics=aligned["public_metrics"],
            private_metrics_by_key=aligned["private_metrics_by_key"],
            use_paired_test=use_paired_test,
            remove_outlier_frac=remove_outlier_frac,
            outlier_method=outlier_method,
            raw_sample_count=len(aligned["group_names"]),
        )
        summaries.append(
            {
                "group_name": group_name,
                "sample_count": int(len(public_group_sample_metrics[group_name])),
                "aligned_sample_count": int(len(aligned["group_names"])),
                "public_mean": float(np.mean(aligned["public_metrics"])),
                "private_means_by_key": private_means_by_key,
                "mean_private_metric": float(np.mean(list(private_means_by_key.values()))),
                "book_t_statistic": test_result["t_statistic"],
                "book_p_value": test_result["p_value"],
                "book_p_value_display": test_result["p_value_display"],
                "book_log10_p_value": test_result["log10_p_value"],
                "book_contaminated": test_result["contaminated"],
                "test_sample_count": test_result["test_sample_count"],
            }
        )
    return summaries


def build_single_group_test_summary(
    metric_name,
    group_name,
    public_sample_metrics,
    private_sample_metrics_by_key,
    use_paired_test=True,
    remove_outlier_frac=0.05,
    outlier_method="clip",
):
    """Run a t-test on aligned samples for a single group."""
    aligned = align_group_metrics(public_sample_metrics, private_sample_metrics_by_key)
    if not aligned["group_names"]:
        return None

    private_means_by_key = {
        key: float(np.mean(list(sample_metrics.values())))
        for key, sample_metrics in private_sample_metrics_by_key.items()
    }
    test_result, _, _, _ = compute_statistical_test_result(
        metric_name=metric_name,
        public_metrics=aligned["public_metrics"],
        private_metrics_by_key=aligned["private_metrics_by_key"],
        use_paired_test=use_paired_test,
        remove_outlier_frac=remove_outlier_frac,
        outlier_method=outlier_method,
        raw_sample_count=len(aligned["group_names"]),
    )
    return {
        "group_name": group_name,
        "sample_count": int(len(public_sample_metrics)),
        "aligned_sample_count": int(len(aligned["group_names"])),
        "public_mean": float(np.mean(aligned["public_metrics"])),
        "private_means_by_key": private_means_by_key,
        "mean_private_metric": float(np.mean(list(private_means_by_key.values()))),
        "book_t_statistic": test_result["t_statistic"],
        "book_p_value": test_result["p_value"],
        "book_p_value_display": test_result["p_value_display"],
        "book_log10_p_value": test_result["log10_p_value"],
        "book_contaminated": test_result["contaminated"],
        "test_sample_count": test_result["test_sample_count"],
    }


def add_bool_argument(parser, name, default, help_text):
    """Boolean flag compatible with Python 3.8+."""
    disable_help = help_text[len("Enable "):] if help_text.startswith("Enable ") else help_text
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}",
        dest=name,
        action="store_false",
        help=f"Disable {disable_help}",
    )
    parser.set_defaults(**{name: default})


def build_arg_parser():
    parser = argparse.ArgumentParser(description="STAMP statistical test — supports normal model and Adaptive Fusion")

    parser.add_argument("--dataset_file", type=str, default=None, help="merged dataset file path (JSON format)")
    parser.add_argument("--public_key", type=str, default=None, help="column name for the public version")
    parser.add_argument("--private_keys", nargs="+", default=None, help="list of column names for private versions")
    parser.add_argument("--public_dataset_file", type=str, default=None, help="public version dataset file path (JSON/JSONL")
    parser.add_argument("--private_dataset_files", nargs="+", default=None, help="list of private version dataset file paths (JSON/JSONL")
    parser.add_argument("--text_key", type=str, default=TEXT_KEY, help="text column name (single-file mode)")

    parser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_TYPE,
        choices=["normal", "adaptive_fusion"],
        help="model type: normal or adaptive_fusion",
    )
    parser.add_argument("--normal_model_path", type=str, default=None, help="normal model path (used when model_type=normal)")
    parser.add_argument("--base_model_path", type=str, default=None, help="Adaptive Fusion base model path")
    parser.add_argument("--assist_model_path", type=str, default=None, help="Adaptive Fusion assistant model path")
    parser.add_argument("--alpha_min", type=float, default=ALPHA_MIN, help="Adaptive Fusion alpha lower bound")
    parser.add_argument("--alpha_max", type=float, default=ALPHA_MAX, help="Adaptive Fusion alpha upper bound")
    parser.add_argument("--alpha_s", type=float, default=ALPHA_S, help="Adaptive Fusion smoothing term S")
    parser.add_argument("--gpu_id", type=int, default=GPU_ID, help="explicit GPU ID; default follows CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--group_by_field",
        type=str,
        default=GROUP_BY_FIELD,
        help="aggregate by specified field before running statistical test, e.g. book_name",
    )
    add_bool_argument(parser, "use_paired_test", default=True, help_text="Enable paired t-test")
    parser.add_argument("--remove_outlier_frac", type=float, default=0.05, help="fraction of outliers to remove")
    parser.add_argument(
        "--outlier_method",
        type=str,
        default="clip",
        choices=["clip", "zero", "mean", "keep"],
        help="outlier handling method",
    )
    return parser


def run_statistical_test(
    metric_name,
    public_metrics,
    private_metrics_by_key,
    use_paired_test=True,
    remove_outlier_frac=0.05,
    outlier_method="clip",
    group_by_field=None,
    group_summaries=None,
    raw_sample_count=None,
):
    print(f"\n3. Number of statistical units: {len(public_metrics)}")
    print(f"   Raw mean: {np.mean(public_metrics):.4f}")
    print(f"   Raw std: {np.std(public_metrics):.4f}")
    result, public_metrics_clean, cleaned_private_metrics_by_key, mean_private_metrics = compute_statistical_test_result(
        metric_name=metric_name,
        public_metrics=public_metrics,
        private_metrics_by_key=private_metrics_by_key,
        use_paired_test=use_paired_test,
        remove_outlier_frac=remove_outlier_frac,
        outlier_method=outlier_method,
        group_by_field=group_by_field,
        group_summaries=group_summaries,
        raw_sample_count=raw_sample_count,
    )
    print(f"   Cleaned mean: {np.mean(public_metrics_clean):.4f}")
    print(f"   Cleaned std: {np.std(public_metrics_clean):.4f}")

    private_keys = list(private_metrics_by_key)
    for private_key in private_keys:
        private_unit_metrics = np.array(private_metrics_by_key[private_key], dtype=float)
        private_metrics_clean = cleaned_private_metrics_by_key[private_key]
        print(
            f"   Private version {private_key}: unit mean={np.mean(private_unit_metrics):.4f}, "
            f"cleaned mean={np.mean(private_metrics_clean):.4f}"
        )
    print(f"\n   Overall mean across all private versions: {np.mean(mean_private_metrics):.4f}")
    print(f"   Overall std across all private versions: {np.std(mean_private_metrics):.4f}")

    print(f"\n4. Running statistical test")
    test_type = "paired t-test" if use_paired_test else "independent t-test"
    print(f"   Test type: {test_type}")
    print(f"   H0: public version {metric_name} >= private version {metric_name} (not contaminated)")
    print(f"   H1: public version {metric_name} < private version {metric_name} (contaminated)")

    print(f"\n{'='*60}")
    print(f"Test result ({metric_name})")
    print(f"{'='*60}")
    print(f"t-statistic: {result['t_statistic']:.4f}")
    print(f"p-value: {result['p_value_display']}")
    print(f"log10(p-value): {result['log10_p_value']:.2f}")
    if result["log10_p_value"] < -100:
        print(f"  (this means p ≈ 10^{result['log10_p_value']:.0f})")
    print(f"\nDecision (significance level α=0.05):")

    if result["contaminated"]:
        print(f"✓ p-value < 0.05: reject H0")
        print(f"  Conclusion: significant evidence of contamination (model may have been trained on the public version)")
    else:
        print(f"✗ p-value >= 0.05: cannot reject H0")
        print(f"  Conclusion: insufficient evidence of contamination")

    if "mean_book_p_value_display" in result:
        print(f"\nPer-book average summary (descriptive):")
        print(f"Mean per-book p-value: {result['mean_book_p_value_display']}")
        print(f"Mean per-book t-statistic: {result['mean_book_t_statistic']:.4f}")

    plot_results(
        public_metrics_clean,
        mean_private_metrics,
        metric_name,
        result["p_value"],
        result["p_value_display"],
        result["contaminated"],
    )

    return result

def stamp_detection(dataset, model, tokenizer, public_key, private_keys, 
                   use_paired_test=True, 
                   remove_outlier_frac=0.05, outlier_method="clip",
                   group_by_field=None):
    """
    STAMP statistical test for data contamination detection.
    Computes multiple metrics using a normal model or Adaptive Fusion.
    
    Parameters:
        dataset: dataset containing texts
        model: model object
        tokenizer: tokenizer
        public_key: key name for the public version
        private_keys: list of key names for private versions
        use_paired_test: use paired t-test (default True)
        remove_outlier_frac: fraction of outliers to remove
        outlier_method: outlier handling method
    """
    print(f"\n{'='*60}")
    print("STAMP Statistical Test - Metrics: ppl / mink / mink++ / lowercase / zlib")
    print(f"{'='*60}")
    
    # 1. Compute public version metrics
    print(f"\n1. Processing public version: {public_key}")
    public_texts = [item[public_key] for item in dataset]
    public_metrics_by_name = calculate_metrics_for_texts(model, tokenizer, public_texts)
    public_mean_bundle = {
        metric_name: float(np.mean(values))
        for metric_name, values in public_metrics_by_name.items()
    }
    print(f"   Public version mean: {format_metric_bundle(public_mean_bundle)}")
    
    # 2. Compute all private version metrics
    print(f"\n2. Processing private versions (total: {len(private_keys)})")
    private_metrics_by_key = {}
    
    for i, private_key in enumerate(private_keys, 1):
        print(f"\n   Private version {i} ({private_key}):")
        private_texts = [item[private_key] for item in dataset]
        private_metric_bundle = calculate_metrics_for_texts(model, tokenizer, private_texts)
        private_metrics_by_key[private_key] = private_metric_bundle
        private_mean_bundle = {
            metric_name: float(np.mean(values))
            for metric_name, values in private_metric_bundle.items()
        }
        print(f"     Private version mean: {format_metric_bundle(private_mean_bundle)}")

    results = {}
    for metric_name in METRIC_NAMES:
        print(f"\n{'#' * 60}")
        print(f"Running statistics for metric: {metric_name}")
        print(f"{'#' * 60}")

        public_metric_values = public_metrics_by_name[metric_name]
        private_metric_values_by_key = {
            key: metric_bundle[metric_name]
            for key, metric_bundle in private_metrics_by_key.items()
        }
        analysis_public_metrics = public_metric_values
        analysis_private_metrics_by_key = private_metric_values_by_key
        group_summaries = None
        group_count = None

        if group_by_field:
            aggregated = aggregate_metrics_by_group(
                dataset=dataset,
                public_metrics=public_metric_values,
                private_metrics_by_key=private_metric_values_by_key,
                group_by_field=group_by_field,
            )
            analysis_public_metrics = aggregated["public_metrics"]
            analysis_private_metrics_by_key = aggregated["private_metrics_by_key"]
            group_summaries = build_group_test_summaries_from_index_groups(
                metric_name=metric_name,
                group_indices=aggregated["group_indices"],
                public_metrics=public_metric_values,
                private_metrics_by_key=private_metric_values_by_key,
                use_paired_test=use_paired_test,
                remove_outlier_frac=remove_outlier_frac,
                outlier_method=outlier_method,
            )
            group_count = len(aggregated["group_names"])

            print(f"\n2.5 After aggregation by {group_by_field}: {group_count} statistical units:")
            for summary in group_summaries:
                print(
                    f"   - {summary['group_name']}: n={summary['sample_count']}, "
                    f"public_mean={summary['public_mean']:.4f}, "
                    f"private_mean={summary['mean_private_metric']:.4f}, "
                    f"p-value={summary['book_p_value_display']}, "
                    f"t-statistic={summary['book_t_statistic']:.4f}"
                )

        results[metric_name] = run_statistical_test(
            metric_name=metric_name,
            public_metrics=analysis_public_metrics,
            private_metrics_by_key=analysis_private_metrics_by_key,
            use_paired_test=use_paired_test,
            remove_outlier_frac=remove_outlier_frac,
            outlier_method=outlier_method,
            group_by_field=group_by_field,
            group_summaries=group_summaries,
            raw_sample_count=len(dataset),
        )

    return results


def stamp_detection_from_dataset_files(
    public_dataset_file,
    private_dataset_files,
    model,
    tokenizer,
    text_key,
    group_by_field,
    use_paired_test=True,
    remove_outlier_frac=0.05,
    outlier_method="clip",
):
    print(f"\n{'='*60}")
    print("STAMP Statistical Test - Metrics: ppl / mink / mink++ / lowercase / zlib")
    print(f"Input mode: separate files")
    print(f"Public file: {public_dataset_file}")
    print(f"Number of private files: {len(private_dataset_files)}")
    print(f"Text column: {text_key}")
    print(f"Group-by field: {group_by_field}")
    print(f"{'='*60}")

    public_records = load_json_or_jsonl(public_dataset_file)
    public_group_samples = collect_group_samples(public_records, group_by_field, text_key)
    print(f"Valid groups in public file: {len(public_group_samples)}")
    private_group_samples_by_key = {}
    private_group_sizes_by_key = {}
    for dataset_file in private_dataset_files:
        dataset_label = os.path.basename(dataset_file)
        private_records = load_json_or_jsonl(dataset_file)
        private_group_samples = collect_group_samples(private_records, group_by_field, text_key)
        private_group_samples_by_key[dataset_label] = private_group_samples
        private_group_sizes_by_key[dataset_label] = {
            name: len(samples) for name, samples in private_group_samples.items()
        }
        print(f"Valid groups in private file {dataset_label}: {len(private_group_samples)}")

    common_groups = set(public_group_samples)
    for grouped_samples in private_group_samples_by_key.values():
        common_groups &= set(grouped_samples)
    common_groups = sorted(common_groups)
    if not common_groups:
        raise ValueError("No common groups between public and private files; statistical test cannot be run")

    print(f"\nCommon valid books: {len(common_groups)}")

    public_group_metrics = {metric_name: {} for metric_name in METRIC_NAMES}
    private_group_metrics_by_key = {
        dataset_label: {metric_name: {} for metric_name in METRIC_NAMES}
        for dataset_label in private_group_samples_by_key
    }
    per_metric_group_summaries = {metric_name: [] for metric_name in METRIC_NAMES}

    for group_index, group_name in enumerate(common_groups, start=1):
        print(f"\n{'=' * 80}")
        print(f"Processing book {group_index}/{len(common_groups)}: {group_name}")
        print(f"{'=' * 80}")

        public_samples = public_group_samples[group_name]
        public_texts = [sample["text"] for sample in public_samples]
        public_metric_bundle = calculate_metrics_for_texts(model, tokenizer, public_texts)
        public_sample_metrics_by_name = {
            metric_name: {
                sample["sample_id"]: float(public_metric_bundle[metric_name][i])
                for i, sample in enumerate(public_samples)
            }
            for metric_name in METRIC_NAMES
        }
        public_mean_bundle = {}
        for metric_name in METRIC_NAMES:
            mean_value = float(np.mean(public_metric_bundle[metric_name]))
            public_group_metrics[metric_name][group_name] = mean_value
            public_mean_bundle[metric_name] = mean_value

        private_group_sample_metrics_by_key = {}
        for dataset_label, grouped_samples in private_group_samples_by_key.items():
            private_samples = grouped_samples[group_name]
            private_texts = [sample["text"] for sample in private_samples]
            private_metric_bundle = calculate_metrics_for_texts(model, tokenizer, private_texts)
            private_group_sample_metrics_by_key[dataset_label] = {
                metric_name: {
                    sample["sample_id"]: float(private_metric_bundle[metric_name][i])
                    for i, sample in enumerate(private_samples)
                }
                for metric_name in METRIC_NAMES
            }
            private_mean_bundle = {}
            for metric_name in METRIC_NAMES:
                mean_value = float(np.mean(private_metric_bundle[metric_name]))
                private_group_metrics_by_key[dataset_label][metric_name][group_name] = mean_value
                private_mean_bundle[metric_name] = mean_value

        print(f"\n[{group_name}] Per-book statistics")
        for metric_name in METRIC_NAMES:
            summary = build_single_group_test_summary(
                metric_name=metric_name,
                group_name=group_name,
                public_sample_metrics=public_sample_metrics_by_name[metric_name],
                private_sample_metrics_by_key={
                    dataset_label: metric_bundle[metric_name]
                    for dataset_label, metric_bundle in private_group_sample_metrics_by_key.items()
                },
                use_paired_test=use_paired_test,
                remove_outlier_frac=remove_outlier_frac,
                outlier_method=outlier_method,
            )
            if summary is None:
                continue
            summary["private_sample_counts_by_key"] = {
                key: sizes.get(group_name, 0)
                for key, sizes in private_group_sizes_by_key.items()
            }
            per_metric_group_summaries[metric_name].append(summary)
            print(
                f"   {DISPLAY_METRIC_NAMES[metric_name]}: public_mean={summary['public_mean']:.4f}, "
                f"private_mean={summary['mean_private_metric']:.4f}, "
                f"p-value={summary['book_p_value_display']}, "
                f"t-statistic={summary['book_t_statistic']:.4f}"
            )

    results = {}
    for metric_name in METRIC_NAMES:
        print(f"\n{'#' * 60}")
        print(f"Running statistics for metric: {metric_name}")
        print(f"{'#' * 60}")

        public_metric_groups = public_group_metrics[metric_name]
        private_metric_groups_by_key = {
            key: metrics_by_name[metric_name]
            for key, metrics_by_name in private_group_metrics_by_key.items()
        }
        aligned = align_group_metrics(public_metric_groups, private_metric_groups_by_key)
        if not aligned["group_names"]:
            raise ValueError("No common groups between public and private files; statistical test cannot be run")

        group_summaries = per_metric_group_summaries[metric_name]
        group_summaries_by_name = {summary["group_name"]: summary for summary in group_summaries}

        print(f"\nCommon statistical units: {len(group_summaries)}")
        for group_name in aligned["group_names"]:
            summary = group_summaries_by_name[group_name]
            print(
                f"   - {summary['group_name']}: public_n={summary['sample_count']}, "
                f"public_mean={summary['public_mean']:.4f}, "
                f"private_mean={summary['mean_private_metric']:.4f}, "
                f"p-value={summary['book_p_value_display']}, "
                f"t-statistic={summary['book_t_statistic']:.4f}"
            )

        results[metric_name] = run_statistical_test(
            metric_name=metric_name,
            public_metrics=aligned["public_metrics"],
            private_metrics_by_key=aligned["private_metrics_by_key"],
            use_paired_test=use_paired_test,
            remove_outlier_frac=remove_outlier_frac,
            outlier_method=outlier_method,
            group_by_field=group_by_field,
            group_summaries=group_summaries,
            raw_sample_count=len(public_records),
        )

    return results

def plot_results(public_metrics, private_metrics, metric_name, p_value, p_value_display, contaminated):
    """Visualize results."""
    metric_display_name = DISPLAY_METRIC_NAMES.get(metric_name, metric_name)
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Histogram Comparison
    plt.subplot(1, 3, 1)
    plt.hist(public_metrics, bins=30, alpha=0.6, label='Public Version', color='blue', density=True)
    plt.hist(private_metrics, bins=30, alpha=0.6, label='Private Version', color='red', density=True)
    plt.xlabel(metric_display_name)
    plt.ylabel('Density')
    plt.title(f'{metric_display_name} Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Boxplot
    plt.subplot(1, 3, 2)
    data_to_plot = [public_metrics, private_metrics]
    box = plt.boxplot(data_to_plot, labels=['Public', 'Private'], patch_artist=True)
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel(metric_display_name)
    plt.title(f'{metric_display_name} Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Statistical Info
    plt.subplot(1, 3, 3)
    plt.axis('off')
    
    result_color = 'red' if contaminated else 'green'
    result_text = 'contaminated' if contaminated else 'not contaminated'
    
    info_text = f"""
Statistical test results

p-value: {p_value_display}
Result: {result_text}

Public version:
  Mean: {np.mean(public_metrics):.4f}
  Std: {np.std(public_metrics):.4f}

Private version:
  Mean: {np.mean(private_metrics):.4f}
  Std: {np.std(private_metrics):.4f}

Difference: {np.mean(public_metrics) - np.mean(private_metrics):.4f}
    """
    
    plt.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'stamp_result_{metric_name}.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: stamp_result_{metric_name}.png")


def print_metric_result_summary(results_by_metric):
    """Print summary conclusions for all metrics."""
    print(f"\n{'=' * 60}")
    print("Metric summary")
    print(f"{'=' * 60}")
    for metric_name in METRIC_NAMES:
        result = results_by_metric[metric_name]
        if "mean_book_p_value_display" in result:
            mean_book_contaminated = result.get(
                "mean_book_contaminated",
                result.get("mean_book_p_value", 1.0) < 0.05,
            )
            detected_text = "detected" if mean_book_contaminated else "not detected"
            print(
                f"{DISPLAY_METRIC_NAMES[metric_name]:<10} {detected_text:<4} "
                f"mean per-book p-value={result['mean_book_p_value_display']} "
                f"mean per-book t-statistic={result['mean_book_t_statistic']:.4f}"
            )
        else:
            detected_text = "detected" if result["contaminated"] else "not detected"
            print(
                f"{DISPLAY_METRIC_NAMES[metric_name]:<10} {detected_text:<4} "
                f"p-value={result['p_value_display']} "
                f"t-statistic={result['t_statistic']:.4f}"
            )
    print(f"{'=' * 60}")

def main(args):
    using_separate_files = bool(args.public_dataset_file or args.private_dataset_files)
    if using_separate_files:
        if not args.public_dataset_file or not args.private_dataset_files:
            raise ValueError("Separate-file mode requires both --public_dataset_file and --private_dataset_files")
        if not args.group_by_field:
            args.group_by_field = "book_name"
    else:
        if not args.dataset_file or not args.public_key or not args.private_keys:
            raise ValueError("Merged-dataset mode requires --dataset_file, --public_key, and --private_keys")

        print(f"📚 Loading dataset file: {args.dataset_file}")
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"✓ Dataset loaded successfully")
        print(f"Total samples: {len(dataset)}")
        
        if len(dataset) > 0:
            sample_keys = list(dataset[0].keys())
            print(f"Dataset columns: {sample_keys}")
            
            if args.public_key not in sample_keys:
                raise ValueError(f"Public version key '{args.public_key}' not found in dataset")
            
            for private_key in args.private_keys:
                if private_key not in sample_keys:
                    raise ValueError(f"Private version key '{private_key}' not found in dataset")
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Model configuration:")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Model type: {MODEL_TYPE}")
    if args.group_by_field:
        print(f"Statistical unit: aggregated by {args.group_by_field}")
    else:
        print("Statistical unit: individual samples")
    if using_separate_files:
        print(f"Input mode: separate files")
        print(f"Public file: {args.public_dataset_file}")
        print(f"Number of private files: {len(args.private_dataset_files)}")
        print(f"Text column: {args.text_key}")
    else:
        print(f"Input mode: merged dataset")
    
    if MODEL_TYPE == "normal":
        print(f"Normal model path: {NORMAL_MODEL_PATH}")
    elif MODEL_TYPE == "adaptive_fusion":
        print(f"Base model: {BASE_MODEL_PATH}")
        print(f"Assistant model: {ASSIST_MODEL_PATH}")
        print(f"alpha_min: {ALPHA_MIN}")
        print(f"alpha_max: {ALPHA_MAX}")
        print(f"alpha_s: {ALPHA_S}")
        print("Fusion formula: fused = base_logits - alpha_t * assist_logits")
    
    print(f"{'='*60}\n")
    
    model, tokenizer = load_model(MODEL_TYPE)

    if using_separate_files:
        result = stamp_detection_from_dataset_files(
            public_dataset_file=args.public_dataset_file,
            private_dataset_files=args.private_dataset_files,
            model=model,
            tokenizer=tokenizer,
            text_key=args.text_key,
            group_by_field=args.group_by_field,
            use_paired_test=args.use_paired_test,
            remove_outlier_frac=args.remove_outlier_frac,
            outlier_method=args.outlier_method,
        )
    else:
        if isinstance(args.private_keys, list):
            private_keys = args.private_keys
        else:
            private_keys = args.private_keys.split()
        
        result = stamp_detection(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            public_key=args.public_key,
            private_keys=private_keys,
            use_paired_test=args.use_paired_test,
            remove_outlier_frac=args.remove_outlier_frac,
            outlier_method=args.outlier_method,
            group_by_field=args.group_by_field,
        )
    
    # Save results
    output_file = "stamp_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "metrics": result,
                "input_mode": "separate_files" if using_separate_files else "merged_dataset",
                "public_dataset_file": args.public_dataset_file,
                "private_dataset_files": args.private_dataset_files,
                "dataset_file": args.dataset_file,
                "public_key": args.public_key,
                "private_keys": args.private_keys,
                "text_key": args.text_key,
                "model_type": MODEL_TYPE,
                "normal_model_path": NORMAL_MODEL_PATH if MODEL_TYPE == "normal" else None,
                "base_model_path": BASE_MODEL_PATH if MODEL_TYPE == "adaptive_fusion" else None,
                "assist_model_path": ASSIST_MODEL_PATH if MODEL_TYPE == "adaptive_fusion" else None,
                "alpha_min": ALPHA_MIN if MODEL_TYPE == "adaptive_fusion" else None,
                "alpha_max": ALPHA_MAX if MODEL_TYPE == "adaptive_fusion" else None,
                "alpha_s": ALPHA_S if MODEL_TYPE == "adaptive_fusion" else None,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print_metric_result_summary(result)
    print(f"\n{'='*60}")
    print(f"✓ All results saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Update global configuration from command-line arguments
    if args.model_type:
        globals()['MODEL_TYPE'] = args.model_type
    if args.normal_model_path:
        globals()['NORMAL_MODEL_PATH'] = args.normal_model_path
    if args.base_model_path:
        globals()['BASE_MODEL_PATH'] = args.base_model_path
    if args.assist_model_path:
        globals()['ASSIST_MODEL_PATH'] = args.assist_model_path
    globals()['ALPHA_MIN'] = args.alpha_min
    globals()['ALPHA_MAX'] = args.alpha_max
    globals()['ALPHA_S'] = args.alpha_s
    globals()['GPU_ID'] = args.gpu_id
    globals()['DEVICE'] = resolve_device(args.gpu_id)
    
    main(args)
