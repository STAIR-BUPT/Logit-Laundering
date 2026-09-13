#!/usr/bin/env python3
"""
Top-1 watermark detection — Adaptive Fusion aggregation.
Aggregates top-1 prediction green-word hits per prefix position grouped by book,
computes per-book z-score / p-value, and writes JSON results.

Fusion form:
    fused_logits = base_logits - alpha_t * assist_logits
    alpha_t = alpha_min + (alpha_max - alpha_min) * margin_t / (margin_t + S)
    margin_t = top1_logit - top2_logit  (logit space, not probability)
"""

import sys
import os
import json
import math
import torch
import argparse
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import norm

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "ward_core"))

from src.watermarks.watermark_factory import get_watermark
from src.config.watermark_config import MetaConfig, WatermarkScheme
from types import SimpleNamespace

# ======================
# Configuration
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_PATH = ""  # set via --base_model_path (e.g. /...)

# BASE_MODEL_PATH = ""  # alternative base model
ASSIST_MODEL_PATH = ""  # set via --assist_model_path (e.g. /...)

DATASET_PATH = ""  # set via --dataset_path
TEXT_COLUMN = "text"
SOURCE_FILTER = ""
GROUP_BY_FIELD = "book_name"
OUTPUT_JSON = "out/top1_adaptive_fusion_aggregate.json"
NUM_SAMPLES = 0

# ---- Adaptive Fusion parameters ----
ALPHA_MIN = float(os.environ.get("ALPHA_MIN", "0.17"))
ALPHA_MAX = float(os.environ.get("ALPHA_MAX", "0.56"))
ALPHA_S = float(os.environ.get("ALPHA_S", "1.0"))

# ---- Watermark parameters ----
K = 4
GAMMA = 0.25
SEEDING_SCHEME = os.environ.get("WATERMARK_SEEDING_SCHEME", "ff-additive_prf-1-False-replace-with-private-key")


# -------------------------
# Data loading and aggregation helpers
# -------------------------
def load_records(dataset_path: str, text_column: str = TEXT_COLUMN) -> list[dict]:
    if dataset_path.endswith(".jsonl"):
        records = []
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if dataset_path.endswith(".xlsx"):
        df = pd.read_excel(dataset_path)
        return [{text_column: text} for text in df[text_column].fillna("").astype(str).tolist()]

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def filter_records(
    records: list[dict],
    source_filter: str | None = None,
    text_column: str = TEXT_COLUMN,
) -> list[dict]:
    filtered = []
    for record in records:
        text = str(record.get(text_column, "")).strip()
        if not text:
            continue
        if source_filter is not None and record.get("source") != source_filter:
            continue
        filtered.append(record)
    return filtered


def group_records(records: list[dict], group_by_field: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        group_name = str(record.get(group_by_field) or f"<missing-{group_by_field}>")
        groups[group_name].append(record)
    return dict(sorted(groups.items()))


def slice_group_items(
    grouped_records: dict[str, list[dict]],
    start_group_index: int = 1,
    end_group_index: int = 0,
) -> list[tuple[str, list[dict]]]:
    items = list(grouped_records.items())
    if not items:
        return []
    start_idx = max(1, start_group_index)
    end_idx = len(items) if end_group_index <= 0 else min(end_group_index, len(items))
    if start_idx > end_idx:
        return []
    return items[start_idx - 1 : end_idx]


def compute_group_stats(
    group_name: str,
    sample_count: int,
    total_hits: int,
    total_tokens: int,
    total_margin_sum: float,
    total_alpha_sum: float,
    total_steps: int,
    gamma: float,
) -> dict:
    green_ratio = total_hits / total_tokens if total_tokens > 0 else 0.0
    std = math.sqrt(gamma * (1 - gamma) / total_tokens) if total_tokens > 0 else 0.0
    z_score = (green_ratio - gamma) / std if std > 0 else 0.0
    p_value = 1 - norm.cdf(z_score) if total_tokens > 0 else 1.0
    avg_margin = total_margin_sum / total_steps if total_steps > 0 else 0.0
    avg_alpha = total_alpha_sum / total_steps if total_steps > 0 else 0.0
    return {
        "group_name": group_name,
        "sample_count": int(sample_count),
        "valid_token_count": int(total_tokens),
        "green_hit_count": int(total_hits),
        "red_miss_count": int(total_tokens - total_hits),
        "green_ratio": float(green_ratio),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "avg_margin": float(avg_margin),
        "avg_alpha": float(avg_alpha),
        "step_count": int(total_steps),
        "detected": bool(p_value < 0.05),
    }


def write_group_results_json(
    output_json: str,
    dataset_path: str,
    source_filter: str | None,
    group_by_field: str,
    results: list[dict],
    base_model_path: str,
    assist_model_path: str,
    alpha_min: float,
    alpha_max: float,
    alpha_s: float,
    seeding_scheme: str,
) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": dataset_path,
        "source_filter": source_filter,
        "group_by_field": group_by_field,
        "base_model_path": base_model_path,
        "assist_model_path": assist_model_path,
        "alpha_min": float(alpha_min),
        "alpha_max": float(alpha_max),
        "alpha_s": float(alpha_s),
        "seeding_scheme": seeding_scheme,
        "gamma": GAMMA,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def print_group_result(result: dict, prefix: str = "") -> None:
    status = "✅ watermarked" if result["detected"] else "❌ not-detected"
    print(
        f'{prefix}{result["group_name"]:<60} '
        f'z={result["z_score"]:.4f} '
        f'p={result["p_value"]:.6g} '
        f'samples={result["sample_count"]:<3d} '
        f'tokens={result["valid_token_count"]:<6d} '
        f'hits={result["green_hit_count"]:<6d} '
        f'avg_margin={result["avg_margin"]:.4f} '
        f'avg_alpha={result["avg_alpha"]:.4f} '
        f'{status}'
    )


# -------------------------
# Model loading
# -------------------------
def load_models():
    print("🚀 Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    assist_model = AutoModelForCausalLM.from_pretrained(
        ASSIST_MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model loaded successfully\n")
    return base_model, assist_model, tokenizer


def load_watermark_detector(tokenizer):
    print("🔍 Loading watermark detector...")
    meta_cfg = MetaConfig(seed=42, device=DEVICE, out_root_dir="out/")
    wm_cfg = SimpleNamespace(
        scheme=WatermarkScheme.KGW,
        generation=SimpleNamespace(seeding_scheme=SEEDING_SCHEME, gamma=0.25, delta=3.5),
        detection=SimpleNamespace(normalizers=[], ignore_repeated_ngrams=True, z_threshold=4.0),
    )
    wm = get_watermark(meta_cfg, wm_cfg, tokenizer, BASE_MODEL_PATH)
    print("✅ Watermark detector loaded successfully\n")
    return wm


# -------------------------
# Adaptive alpha computation
# -------------------------
def compute_adaptive_alpha(base_logits_last: torch.Tensor) -> tuple[float, float]:
    top2_val = torch.topk(base_logits_last.float(), k=2).values
    margin_t = (top2_val[0] - top2_val[1]).item()
    alpha_t = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * margin_t / (margin_t + ALPHA_S)
    return alpha_t, margin_t


# -------------------------
# Per-text: count green-word hits
# -------------------------
@torch.inference_mode()
def count_green_hits(base_model, assist_model, tokenizer, wm, text: str):
    """Return (hits, total, sum_margin, sum_alpha, n_steps)."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    hits, total = 0, 0
    sum_margin, sum_alpha, n_steps = 0.0, 0.0, 0
    if input_ids.size(1) <= K:
        return hits, total, sum_margin, sum_alpha, n_steps

    for i in range(K, input_ids.size(1) - 1):
        prefix_ctx = input_ids[0, i - K : i].tolist()
        prefix_ids = input_ids[:, :i]
        prefix_attn = attention_mask[:, :i]

        base_last = base_model(
            input_ids=prefix_ids,
            attention_mask=prefix_attn,
            return_dict=True,
        ).logits[0, -1, :]
        alpha_t, margin_t = compute_adaptive_alpha(base_last)

        assist_last = assist_model(
            input_ids=prefix_ids,
            attention_mask=prefix_attn,
            return_dict=True,
        ).logits[0, -1, :]

        fused = base_last - alpha_t * assist_last
        top1_id = torch.argmax(fused).item()

        sum_margin += margin_t
        sum_alpha += alpha_t
        n_steps += 1

        green_dict = wm.get_greenness_dict_ints(prefix_ctx, [top1_id])
        if green_dict is None:
            continue

        total += 1
        hits += int(green_dict.get(top1_id, False))

    return hits, total, sum_margin, sum_alpha, n_steps


# -------------------------
# Main
# -------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-1 Adaptive Fusion aggregate detection")
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH, help="Path to base model")
    parser.add_argument("--assist_model_path", type=str, default=ASSIST_MODEL_PATH, help="Path to assistant model")
    parser.add_argument("--seeding_scheme", type=str, default=SEEDING_SCHEME)
    parser.add_argument("--alpha_min", type=float, default=ALPHA_MIN)
    parser.add_argument("--alpha_max", type=float, default=ALPHA_MAX)
    parser.add_argument("--alpha_s", type=float, default=ALPHA_S)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--text_column", type=str, default=TEXT_COLUMN)
    parser.add_argument("--source_filter", type=str, default=SOURCE_FILTER)
    parser.add_argument("--group_by_field", type=str, default=GROUP_BY_FIELD)
    parser.add_argument("--output_json", type=str, default=OUTPUT_JSON)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--start_group_index", type=int, default=1)
    parser.add_argument("--end_group_index", type=int, default=0)
    return parser


def main():
    global ALPHA_MIN, ALPHA_MAX, ALPHA_S, NUM_SAMPLES, BASE_MODEL_PATH, ASSIST_MODEL_PATH, SEEDING_SCHEME

    parser = build_arg_parser()
    args = parser.parse_args()

    BASE_MODEL_PATH = args.base_model_path
    ASSIST_MODEL_PATH = args.assist_model_path
    SEEDING_SCHEME = args.seeding_scheme
    ALPHA_MIN = args.alpha_min
    ALPHA_MAX = args.alpha_max
    ALPHA_S = args.alpha_s
    NUM_SAMPLES = args.num_samples

    print("Top-1 Adaptive Fusion Aggregate Detection")
    print(f"   alpha_min={ALPHA_MIN}  alpha_max={ALPHA_MAX}  S={ALPHA_S}")
    print(f"   base_model_path={BASE_MODEL_PATH}")
    print(f"   assist_model_path={ASSIST_MODEL_PATH}")
    print(f"   seeding_scheme={SEEDING_SCHEME}")
    print("   formula: alpha_t = alpha_min + (alpha_max - alpha_min) * margin_t / (margin_t + S)")
    print(f"   dataset_path={args.dataset_path}")
    print(f"   source_filter={args.source_filter}")
    print(f"   group_by_field={args.group_by_field}")
    print(f"   num_samples={'ALL' if NUM_SAMPLES <= 0 else NUM_SAMPLES}")
    print(f"   start_group_index={args.start_group_index}")
    print(
        f"   end_group_index={'ALL' if args.end_group_index <= 0 else args.end_group_index}"
    )
    print("=" * 60)

    records = load_records(args.dataset_path, args.text_column)
    filtered_records = filter_records(records, args.source_filter, args.text_column)
    if NUM_SAMPLES > 0:
        filtered_records = filtered_records[:NUM_SAMPLES]
    grouped_records = group_records(filtered_records, args.group_by_field)

    print(f"📚 Filtered records: {len(filtered_records)}")
    print(f"📚 Groups: {len(grouped_records)}")
    print("=" * 60)

    base_model, assist_model, tokenizer = load_models()
    wm = load_watermark_detector(tokenizer)

    selected_group_items = slice_group_items(
        grouped_records,
        start_group_index=args.start_group_index,
        end_group_index=args.end_group_index,
    )
    if not selected_group_items:
        raise ValueError("Selected group range is empty. Check start_group_index / end_group_index.")

    results = []
    total_groups = len(grouped_records)
    for idx, (group_name, group_items) in enumerate(
        selected_group_items, start=args.start_group_index
    ):
        print(f"[{idx}/{total_groups}] Processing group: {group_name} (samples={len(group_items)})")
        total_hits = 0
        total_tokens = 0
        total_margin_sum = 0.0
        total_alpha_sum = 0.0
        total_steps = 0
        progress_desc = f"[{idx}/{total_groups}] {group_name[:48]}"
        for item in tqdm(
            group_items,
            desc=progress_desc,
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        ):
            h, t, sm, sa, ns = count_green_hits(
                base_model,
                assist_model,
                tokenizer,
                wm,
                item[args.text_column],
            )
            total_hits += h
            total_tokens += t
            total_margin_sum += sm
            total_alpha_sum += sa
            total_steps += ns

        result = compute_group_stats(
            group_name=group_name,
            sample_count=len(group_items),
            total_hits=total_hits,
            total_tokens=total_tokens,
            total_margin_sum=total_margin_sum,
            total_alpha_sum=total_alpha_sum,
            total_steps=total_steps,
            gamma=GAMMA,
        )
        results.append(result)
        print_group_result(result, prefix="RESULT ")

    write_group_results_json(
        args.output_json,
        args.dataset_path,
        args.source_filter,
        args.group_by_field,
        results,
        BASE_MODEL_PATH,
        ASSIST_MODEL_PATH,
        ALPHA_MIN,
        ALPHA_MAX,
        ALPHA_S,
        SEEDING_SCHEME,
    )

    print("\n" + "=" * 60)
    print("📊 Aggregating results by book")
    print("=" * 60)
    for result in results:
        print_group_result(result)
    print("=" * 60)
    print(f"📝 JSON results written to: {args.output_json}")


if __name__ == "__main__":
    main()
