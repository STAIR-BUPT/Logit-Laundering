#!/usr/bin/env python3
"""
Top-1 watermark detection — ControlLLM aggregation.
Aggregates all token green-word hits across the dataset,
computes overall green-word ratio / z-score / p-value, and prints a summary.

Fusion: logits = base + weight * assist (fixed weight).
"""

import sys
import os
import json
import math
import torch
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import norm
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "ward_core"))

from src.models.ControlLLM import ControlLLM
from src.watermarks.watermark_factory import get_watermark
from src.config.watermark_config import MetaConfig, WatermarkScheme
from types import SimpleNamespace

# ======================
# Configuration
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_PATH   = ""  # set via --base_model_path or env var (e.g. /...)

# BASE_MODEL_PATH = ""  # alternative base model
ASSIST_MODEL_PATH = ""  # set via --assist_model_path or env var (e.g. /...)

DATASET_PATH = ""  # set via --dataset_path
TEXT_COLUMN  = "text"
SOURCE_FILTER = ""
GROUP_BY_FIELD = "book_name"
OUTPUT_JSON = "out/top1_control_llm_aggregate.json"
NUM_SAMPLES  = 0

MODE           = "standard"        # standard / fake
CONTROL_WEIGHT = -0.6
TOP_LOGIT_FILTER = 0
FAKE_PROB      = 0.3
FAKE_K         = 5
FAKE_MODE      = "semantic_topk"   # top2 / synonym / semantic_topk
SEEDING_SCHEME = os.environ.get("WATERMARK_SEEDING_SCHEME", "ff-additive_prf-1-False-replace-with-private-key")

K     = 4
GAMMA = 0.25


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


def filter_records(records: list[dict], source_filter: str | None = None, text_column: str = TEXT_COLUMN) -> list[dict]:
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
    gamma: float,
) -> dict:
    green_ratio = total_hits / total_tokens if total_tokens > 0 else 0.0
    std = math.sqrt(gamma * (1 - gamma) / total_tokens) if total_tokens > 0 else 0.0
    z_score = (green_ratio - gamma) / std if std > 0 else 0.0
    p_value = 1 - norm.cdf(z_score) if total_tokens > 0 else 1.0
    return {
        "group_name": group_name,
        "sample_count": int(sample_count),
        "valid_token_count": int(total_tokens),
        "green_hit_count": int(total_hits),
        "red_miss_count": int(total_tokens - total_hits),
        "green_ratio": float(green_ratio),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "detected": bool(p_value < 0.05),
    }


def write_group_results_json(
    output_json: str,
    dataset_path: str,
    source_filter: str | None,
    group_by_field: str,
    mode: str,
    fake_mode: str,
    results: list[dict],
    base_model_path: str,
    assist_model_path: str,
    control_weight: float,
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
        "control_weight": float(control_weight),
        "seeding_scheme": seeding_scheme,
        "gamma": GAMMA,
        "mode": mode,
        "fake_mode": fake_mode,
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
        f'{status}'
    )


# -------------------------
# Load ControlLLM
# -------------------------
def load_control_llm_model(mode=MODE, fake_mode=FAKE_MODE):
    print(f"🚀 Loading ControlLLM (mode={mode}, fake_mode={fake_mode})...")

    basellm = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    assist_llm = AutoModelForCausalLM.from_pretrained(
        ASSIST_MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    if mode == "standard":
        model = ControlLLM(
            basellm=basellm,
            assist_llm=assist_llm,
            weight=CONTROL_WEIGHT,
            top_logit_filter=TOP_LOGIT_FILTER,
            mode="standard",
        ).to(DEVICE)
    else:
        model = ControlLLM(
            basellm=basellm,
            assist_llm=assist_llm,
            weight=CONTROL_WEIGHT,
            top_logit_filter=TOP_LOGIT_FILTER,
            mode="fake",
            fake_prob=FAKE_PROB,
            fake_k=FAKE_K,
            fake_mode=fake_mode,
        ).to(DEVICE)

    model.tokenizer = tokenizer
    print("✅ ControlLLM loaded\n")
    return model, tokenizer


# -------------------------
# Watermark detector
# -------------------------
def load_watermark_detector(tokenizer):
    print("🔍 Loading watermark detector...")
    meta_cfg = MetaConfig(seed=42, device=DEVICE, out_root_dir="out/")
    wm_cfg = SimpleNamespace(
        scheme=WatermarkScheme.KGW,
        generation=SimpleNamespace(seeding_scheme=SEEDING_SCHEME, gamma=0.25, delta=3.5),
        detection=SimpleNamespace(normalizers=[], ignore_repeated_ngrams=True, z_threshold=4.0)
    )
    wm = get_watermark(meta_cfg, wm_cfg, tokenizer, BASE_MODEL_PATH)
    print("✅ Watermark detector loaded successfully\n")
    return wm


# -------------------------
# Per-text: count green-word hits
# -------------------------
@torch.inference_mode()
def count_green_hits(model, tokenizer, wm, text: str) -> tuple[int, int]:
    """Return (hits, total)."""
    tokens    = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = tokens["input_ids"]

    hits, total = 0, 0
    if input_ids.size(1) <= K:
        return hits, total

    for i in range(K, input_ids.size(1) - 1):
        prefix_ctx = input_ids[:, i - K : i]
        prefix_ids = input_ids[:, :i]

        logits  = model(input_ids=prefix_ids).logits[:, -1, :]
        top1_id = torch.argmax(logits, dim=-1).item()

        green_dict = wm.get_greenness_dict_ints(
            prefix_ctx.squeeze().tolist(), [top1_id])
        if green_dict is None:
            continue

        total += 1
        hits  += int(green_dict.get(top1_id, False))

    return hits, total


# -------------------------
# Main
# -------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-1 ControlLLM aggregate detection")
    parser.add_argument("--mode", type=str, default=MODE)
    parser.add_argument("--fake_mode", type=str, default=FAKE_MODE)
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH, help="Path to base model")
    parser.add_argument("--assist_model_path", type=str, default=ASSIST_MODEL_PATH, help="Path to assistant model")
    parser.add_argument("--seeding_scheme", type=str, default=SEEDING_SCHEME)
    parser.add_argument("--control_weight", type=float, default=CONTROL_WEIGHT)
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
    global MODE, FAKE_MODE, NUM_SAMPLES, CONTROL_WEIGHT, BASE_MODEL_PATH, ASSIST_MODEL_PATH, SEEDING_SCHEME

    parser = build_arg_parser()
    args = parser.parse_args()

    MODE = args.mode
    FAKE_MODE = args.fake_mode
    BASE_MODEL_PATH = args.base_model_path
    ASSIST_MODEL_PATH = args.assist_model_path
    SEEDING_SCHEME = args.seeding_scheme
    CONTROL_WEIGHT = args.control_weight
    NUM_SAMPLES = args.num_samples

    print("Top-1 ControlLLM Aggregate Detection")
    print(f"   mode={MODE}  fake_mode={FAKE_MODE}  weight={CONTROL_WEIGHT}")
    print(f"   base_model_path={BASE_MODEL_PATH}")
    print(f"   assist_model_path={ASSIST_MODEL_PATH}")
    print(f"   seeding_scheme={SEEDING_SCHEME}")
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

    model, tokenizer = load_control_llm_model(MODE, FAKE_MODE)
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
        progress_desc = f"[{idx}/{total_groups}] {group_name[:48]}"
        for item in tqdm(
            group_items,
            desc=progress_desc,
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        ):
            h, t = count_green_hits(model, tokenizer, wm, item[args.text_column])
            total_hits += h
            total_tokens += t
        result = compute_group_stats(group_name, len(group_items), total_hits, total_tokens, GAMMA)
        results.append(result)
        print_group_result(result, prefix="RESULT ")

    write_group_results_json(
        args.output_json,
        args.dataset_path,
        args.source_filter,
        args.group_by_field,
        MODE,
        FAKE_MODE,
        results,
        BASE_MODEL_PATH,
        ASSIST_MODEL_PATH,
        CONTROL_WEIGHT,
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
