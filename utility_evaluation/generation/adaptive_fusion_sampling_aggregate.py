#!/usr/bin/env python3
"""
Adaptive Fusion Sampling — Aggregate Detection
-----------------------------------------------
In the adaptive fusion setting, selects the next token via true sampling,
then aggregates green-hit / z-score / p-value at the book level.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm

MODULE_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "ward_core"))

from src.config.watermark_config import MetaConfig, WatermarkScheme
from src.watermarks.watermark_factory import get_watermark

# ======================
# Configuration
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12

BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "")  # set via --base_model_path or $BASE_MODEL_ROOT
ASSIST_MODEL_PATH = os.environ.get("ASSIST_MODEL_PATH", "")  # set via --assist_model_path or $ASSISTANT_MODEL_ROOT

DATASET_PATH = os.environ.get("UTILITY_TEXT_DATASET") or os.environ.get("UTILITY_DATA", "")
TEXT_COLUMN = "text"
SOURCE_FILTER = None
GROUP_BY_FIELD = "id"
OUTPUT_JSON = str(MODULE_ROOT / "outputs" / "adaptive_fusion_sampling_aggregate.json")
NUM_SAMPLES = 0

ALPHA_MIN = float(os.environ.get("ALPHA_MIN", "0.17"))
ALPHA_MAX = float(os.environ.get("ALPHA_MAX", "0.56"))
ALPHA_S = float(os.environ.get("ALPHA_S", "1.0"))

K = 4
GAMMA = 0.25
SEEDING_SCHEME = os.environ.get("WATERMARK_SEEDING_SCHEME", "ff-additive_prf-1-False-replace-with-private-key")

DEFAULT_DECODE_SPECS = ["greedy", "topk:40:0.9", "topp:0.9:0.9"]
DEFAULT_SEED = 42


def normalize_distribution(values: list[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("distribution mass must be positive")
    return [float(v) / total for v in values]


def parse_decode_specs(raw_specs: list[str]) -> list[dict]:
    if not raw_specs:
        raise ValueError("decode_specs must be non-empty")

    parsed = []
    seen = set()
    for raw in raw_specs:
        if raw in seen:
            raise ValueError(f"duplicate decode spec: {raw}")
        seen.add(raw)

        if raw == "greedy":
            parsed.append(
                {
                    "decode_spec": raw,
                    "decode_mode": "greedy",
                    "top_k": 0,
                    "top_p": 1.0,
                    "temperature": 1.0,
                }
            )
            continue

        if raw.startswith("topk:"):
            try:
                _, k, temperature = raw.split(":")
            except ValueError as exc:
                raise ValueError(f"invalid topk decode spec: {raw}") from exc
            parsed.append(
                {
                    "decode_spec": raw,
                    "decode_mode": "topk",
                    "top_k": int(k),
                    "top_p": 1.0,
                    "temperature": float(temperature),
                }
            )
            continue

        if raw.startswith("topp:"):
            try:
                _, p, temperature = raw.split(":")
            except ValueError as exc:
                raise ValueError(f"invalid topp decode spec: {raw}") from exc
            parsed.append(
                {
                    "decode_spec": raw,
                    "decode_mode": "topp",
                    "top_k": 0,
                    "top_p": float(p),
                    "temperature": float(temperature),
                }
            )
            continue

        raise ValueError(f"unsupported decode spec: {raw}")

    return parsed


def build_sampling_distribution_from_probs(
    probs: list[float],
    decode_mode: str,
    top_k: int,
    top_p: float,
) -> list[float]:
    if not probs:
        raise ValueError("probs must be non-empty")

    decode_mode = decode_mode.lower()
    base_probs = normalize_distribution([float(v) for v in probs])

    if decode_mode == "greedy":
        return base_probs

    indexed = list(enumerate(base_probs))
    indexed.sort(key=lambda item: item[1], reverse=True)

    if decode_mode == "topk":
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be > 0 for decode_mode=topk, got {top_k}")
        keep_ids = {idx for idx, _ in indexed[: int(top_k)]}
    elif decode_mode == "topp":
        if not 0.0 < float(top_p) <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        keep_ids = set()
        cumulative = 0.0
        for idx, prob in indexed:
            keep_ids.add(idx)
            cumulative += prob
            if cumulative + EPS >= float(top_p):
                break
    else:
        raise ValueError(f"unsupported decode mode: {decode_mode}")

    kept = [base_probs[i] if i in keep_ids else 0.0 for i in range(len(base_probs))]
    return [round(v, 12) for v in normalize_distribution(kept)]


def sample_index_with_uniform(probs: list[float], uniform_value: float) -> int:
    threshold = min(max(float(uniform_value), 0.0), 1.0 - EPS)
    cumulative = 0.0
    for idx, prob in enumerate(normalize_distribution([float(v) for v in probs])):
        cumulative += prob
        if threshold <= cumulative + EPS:
            return idx
    return len(probs) - 1


def build_sampling_distribution_tensor(
    logits: Any,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[Any, Any]:
    scores = logits.float() / max(float(temperature), EPS)
    probs = torch.softmax(scores, dim=-1)
    keep_mask = torch.ones_like(probs, dtype=torch.bool)

    if int(top_k) > 0 and int(top_k) < int(probs.numel()):
        topk_indices = torch.topk(probs, k=int(top_k)).indices
        keep_mask = torch.zeros_like(probs, dtype=torch.bool)
        keep_mask[topk_indices] = True

    if float(top_p) < 1.0:
        current_probs = probs * keep_mask.float()
        current_probs = current_probs / current_probs.sum().clamp_min(EPS)
        sorted_probs, sorted_indices = torch.sort(current_probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        threshold = torch.tensor(float(top_p), dtype=sorted_probs.dtype, device=sorted_probs.device)
        prefix_len = int(torch.searchsorted(cumulative, threshold, right=False).item()) + 1
        prefix_len = max(1, min(prefix_len, int(probs.numel())))
        top_p_mask = torch.zeros_like(probs, dtype=torch.bool)
        top_p_mask[sorted_indices[:prefix_len]] = True
        keep_mask = keep_mask & top_p_mask

    final_probs = probs * keep_mask.float()
    final_probs = final_probs / final_probs.sum().clamp_min(EPS)
    return final_probs, keep_mask


def sample_token_id_from_logits(
    logits: Any,
    decode_mode: str,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Any,
) -> int:
    if decode_mode == "greedy":
        return int(torch.argmax(logits).item())
    probs, _ = build_sampling_distribution_tensor(
        logits=logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


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
        import pandas as pd

        df = pd.read_excel(dataset_path)
        return [{text_column: text} for text in df[text_column].fillna("").astype(str).tolist()]

    raise ValueError(f"unsupported dataset format: {dataset_path}")


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
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0)) if total_tokens > 0 else 1.0
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


def write_sampling_results_json(
    output_json: str,
    dataset_path: str,
    source_filter: str | None,
    group_by_field: str,
    decode_runs: list[dict],
    base_model_path: str,
    assist_model_path: str,
    alpha_min: float,
    alpha_max: float,
    alpha_s: float,
    seeding_scheme: str,
    seed: int,
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
        "seed": int(seed),
        "gamma": GAMMA,
        "decode_runs": decode_runs,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_group_result(result: dict, decode_spec: str, prefix: str = "") -> None:
    status = "✅ watermarked" if result["detected"] else "❌ not-detected"
    print(
        f'{prefix}[{decode_spec}] {result["group_name"]:<50} '
        f'z={result["z_score"]:.4f} '
        f'p={result["p_value"]:.6g} '
        f'samples={result["sample_count"]:<3d} '
        f'tokens={result["valid_token_count"]:<6d} '
        f'hits={result["green_hit_count"]:<6d} '
        f'avg_margin={result["avg_margin"]:.4f} '
        f'avg_alpha={result["avg_alpha"]:.4f} '
        f'{status}'
    )


def load_models(base_model_path: str, assist_model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("🚀 Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    assist_model = AutoModelForCausalLM.from_pretrained(
        assist_model_path, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Models loaded\n")
    return base_model, assist_model, tokenizer


def load_watermark_detector(tokenizer, base_model_path: str, seeding_scheme: str):
    print("🔍 Loading watermark detector...")
    meta_cfg = MetaConfig(seed=42, device=DEVICE, out_root_dir="out/")
    wm_cfg = SimpleNamespace(
        scheme=WatermarkScheme.KGW,
        generation=SimpleNamespace(seeding_scheme=seeding_scheme, gamma=0.25, delta=3.5),
        detection=SimpleNamespace(normalizers=[], ignore_repeated_ngrams=True, z_threshold=4.0),
    )
    wm = get_watermark(meta_cfg, wm_cfg, tokenizer, base_model_path)
    print("✅ Watermark detector loaded\n")
    return wm


def compute_adaptive_alpha(base_logits_last: Any, alpha_min: float, alpha_max: float, alpha_s: float) -> tuple[float, float]:
    top2_val = torch.topk(base_logits_last.float(), k=2).values
    margin_t = (top2_val[0] - top2_val[1]).item()
    alpha_t = alpha_min + (alpha_max - alpha_min) * margin_t / (margin_t + alpha_s)
    return alpha_t, margin_t


def make_generators(decode_specs: list[dict], seed: int) -> dict[str, Any]:
    generators = {}
    for offset, decode_cfg in enumerate(decode_specs):
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(int(seed) + offset)
        generators[decode_cfg["decode_spec"]] = generator
    return generators


@torch.inference_mode()
def count_green_hits_with_sampling(
    base_model,
    assist_model,
    tokenizer,
    wm,
    text: str,
    decode_specs: list[dict],
    generators: dict[str, Any],
    alpha_min: float,
    alpha_max: float,
    alpha_s: float,
) -> dict[str, dict[str, float]]:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    stats = {
        decode_cfg["decode_spec"]: {
            "hits": 0,
            "total": 0,
            "sum_margin": 0.0,
            "sum_alpha": 0.0,
            "n_steps": 0,
        }
        for decode_cfg in decode_specs
    }
    if input_ids.size(1) <= K:
        return stats

    for i in range(K, input_ids.size(1) - 1):
        prefix_ctx = input_ids[0, i - K : i].tolist()
        prefix_ids = input_ids[:, :i]
        prefix_attn = attention_mask[:, :i]

        base_last = base_model(
            input_ids=prefix_ids,
            attention_mask=prefix_attn,
            return_dict=True,
        ).logits[0, -1, :]
        alpha_t, margin_t = compute_adaptive_alpha(base_last, alpha_min, alpha_max, alpha_s)

        assist_last = assist_model(
            input_ids=prefix_ids,
            attention_mask=prefix_attn,
            return_dict=True,
        ).logits[0, -1, :]

        fused = base_last - alpha_t * assist_last

        sampled_ids = {}
        for decode_cfg in decode_specs:
            sampled_ids[decode_cfg["decode_spec"]] = sample_token_id_from_logits(
                logits=fused,
                decode_mode=decode_cfg["decode_mode"],
                temperature=decode_cfg["temperature"],
                top_k=decode_cfg["top_k"],
                top_p=decode_cfg["top_p"],
                generator=generators[decode_cfg["decode_spec"]],
            )

        green_dict = wm.get_greenness_dict_ints(prefix_ctx, list(set(sampled_ids.values())))
        if green_dict is None:
            continue

        for decode_cfg in decode_specs:
            decode_spec = decode_cfg["decode_spec"]
            token_id = sampled_ids[decode_spec]
            stats[decode_spec]["total"] += 1
            stats[decode_spec]["hits"] += int(green_dict.get(token_id, False))
            stats[decode_spec]["sum_margin"] += margin_t
            stats[decode_spec]["sum_alpha"] += alpha_t
            stats[decode_spec]["n_steps"] += 1

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive Fusion Sampling — aggregate detection")
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--assist_model_path", type=str, default=ASSIST_MODEL_PATH)
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
    parser.add_argument("--decode_specs", nargs="+", default=DEFAULT_DECODE_SPECS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    for _path_arg in ["base_model_path", "assist_model_path", "dataset_path", "output_json"]:
        if hasattr(args, _path_arg) and getattr(args, _path_arg):
            setattr(args, _path_arg, os.path.expanduser(os.path.expandvars(getattr(args, _path_arg))))

    decode_specs = parse_decode_specs(args.decode_specs)

    print("Adaptive Fusion Sampling — aggregate detection (book-level)")
    print(f"   alpha_min={args.alpha_min}  alpha_max={args.alpha_max}  S={args.alpha_s}")
    print(f"   base_model_path={args.base_model_path}")
    print(f"   assist_model_path={args.assist_model_path}")
    print(f"   seeding_scheme={args.seeding_scheme}")
    print(f"   dataset_path={args.dataset_path}")
    print(f"   source_filter={args.source_filter}")
    print(f"   group_by_field={args.group_by_field}")
    print(f"   decode_specs={args.decode_specs}")
    print(f"   seed={args.seed}")
    print(f"   num_samples={'ALL' if args.num_samples <= 0 else args.num_samples}")
    print("=" * 60)

    records = load_records(args.dataset_path, args.text_column)
    filtered_records = filter_records(records, args.source_filter, args.text_column)
    if args.num_samples > 0:
        filtered_records = filtered_records[: args.num_samples]
    grouped_records = group_records(filtered_records, args.group_by_field)

    print(f"📚 Records after filtering: {len(filtered_records)}")
    print(f"📚 Groups: {len(grouped_records)}")
    print("=" * 60)

    base_model, assist_model, tokenizer = load_models(args.base_model_path, args.assist_model_path)
    wm = load_watermark_detector(tokenizer, args.base_model_path, args.seeding_scheme)
    generators = make_generators(decode_specs, args.seed)

    selected_group_items = slice_group_items(
        grouped_records,
        start_group_index=args.start_group_index,
        end_group_index=args.end_group_index,
    )
    if not selected_group_items:
        raise ValueError("Selected group range is empty; check start_group_index / end_group_index.")

    decode_run_results = {
        decode_cfg["decode_spec"]: {
            **decode_cfg,
            "results": [],
        }
        for decode_cfg in decode_specs
    }

    total_groups = len(grouped_records)
    for idx, (group_name, group_items) in enumerate(
        selected_group_items, start=args.start_group_index
    ):
        print(f"[{idx}/{total_groups}] Processing group: {group_name} (samples={len(group_items)})")
        group_accumulators = {
            decode_cfg["decode_spec"]: {
                "hits": 0,
                "total": 0,
                "sum_margin": 0.0,
                "sum_alpha": 0.0,
                "n_steps": 0,
            }
            for decode_cfg in decode_specs
        }
        progress_desc = f"[{idx}/{total_groups}] {group_name[:40]}"
        for item in tqdm(
            group_items,
            desc=progress_desc,
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        ):
            per_text_stats = count_green_hits_with_sampling(
                base_model=base_model,
                assist_model=assist_model,
                tokenizer=tokenizer,
                wm=wm,
                text=item[args.text_column],
                decode_specs=decode_specs,
                generators=generators,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                alpha_s=args.alpha_s,
            )
            for decode_cfg in decode_specs:
                decode_spec = decode_cfg["decode_spec"]
                group_accumulators[decode_spec]["hits"] += per_text_stats[decode_spec]["hits"]
                group_accumulators[decode_spec]["total"] += per_text_stats[decode_spec]["total"]
                group_accumulators[decode_spec]["sum_margin"] += per_text_stats[decode_spec]["sum_margin"]
                group_accumulators[decode_spec]["sum_alpha"] += per_text_stats[decode_spec]["sum_alpha"]
                group_accumulators[decode_spec]["n_steps"] += per_text_stats[decode_spec]["n_steps"]

        for decode_cfg in decode_specs:
            decode_spec = decode_cfg["decode_spec"]
            acc = group_accumulators[decode_spec]
            result = compute_group_stats(
                group_name=group_name,
                sample_count=len(group_items),
                total_hits=acc["hits"],
                total_tokens=acc["total"],
                total_margin_sum=acc["sum_margin"],
                total_alpha_sum=acc["sum_alpha"],
                total_steps=acc["n_steps"],
                gamma=GAMMA,
            )
            decode_run_results[decode_spec]["results"].append(result)
            print_group_result(result, decode_spec=decode_spec, prefix="RESULT ")

    decode_runs = [decode_run_results[decode_cfg["decode_spec"]] for decode_cfg in decode_specs]
    write_sampling_results_json(
        output_json=args.output_json,
        dataset_path=args.dataset_path,
        source_filter=args.source_filter,
        group_by_field=args.group_by_field,
        decode_runs=decode_runs,
        base_model_path=args.base_model_path,
        assist_model_path=args.assist_model_path,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_s=args.alpha_s,
        seeding_scheme=args.seeding_scheme,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("📊 Book-level aggregate results")
    print("=" * 60)
    for decode_cfg in decode_specs:
        decode_spec = decode_cfg["decode_spec"]
        for result in decode_run_results[decode_spec]["results"]:
            print_group_result(result, decode_spec=decode_spec)
    print("=" * 60)
    print(f"📝 JSON results written to: {args.output_json}")


if __name__ == "__main__":
    main()
