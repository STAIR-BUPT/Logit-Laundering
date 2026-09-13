#!/usr/bin/env python3
"""
lambda_max / alpha_max estimation script
-----------------------------------------
Aligns with the current Top-1 Adaptive Fusion experiment setup.
Under teacher-forcing, computes the top-K JS divergence curve between
the base distribution and the fused distribution
(softmax(base_logits - lambda * assist_logits)) for a set of candidate
lambda values, then selects the maximum lambda satisfying JS <= delta (budget).

Alignment details:
  1. Uses the same BASE/ASSIST models as Top-1_adaptive_fusion_aggregate.py
  2. Uses the configured calibration JSONL, optionally filtered by source
  3. Uses the same position definition: only prediction positions that the
     adaptive script actually traverses, i.e. i in [K, seq_len - 2]
     corresponding to teacher-forcing logits rows
  4. Uses the same subtraction fusion direction: base_logits - lambda * assist_logits

Output:
  - Average top-K JS per candidate lambda
  - Final selected lambda_max
  - Total number of valid positions used
  - Conservative suggestion: alpha_max can start at lambda_max or slightly below
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "ward_core"))

# ======================
# Configurable parameters
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set via --base_model_path or env var; no default path
BASE_MODEL_PATH = ""  # set via --base_model_path or env var

# Set via --assist_model_path or env var; no default path
ASSIST_MODEL_PATH = ""  # set via --assist_model_path or env var

DATASET_PATH = ""  # set via --dataset_path or env var
TEXT_COLUMN = "text"
SOURCE_FILTER = ""

# Lambda candidate grid: [0.00, 0.05, 0.10, ..., 1.00]
LAMBDA_GRID = np.linspace(0.0, 1.0, 21)

# JS budget delta (global threshold)
JS_BUDGET = 0.05

# Top-k set size
TOP_K = 5

# Context start point aligned with adaptive script
CONTEXT_K = 4

# Max tokens per sample (to limit GPU memory)
MAX_SEQ_LEN = 2048

# Number of samples to use (0 or negative means all)
NUM_SAMPLES = 0


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


def get_valid_row_slice(seq_len: int, context_k: int = CONTEXT_K) -> tuple[int, int] | None:
    """
    Map the loop in the adaptive script:
      for i in range(K, seq_len - 1)
    to a teacher-forcing logits row slice.

    For a sequence of N input tokens:
      - adaptive uses prediction positions i in [K, N-2]
      - corresponding teacher-forcing logits rows are [K-1, N-3]
      - Python slice: [K-1 : N-2)
    """
    if seq_len <= context_k + 1:
        return None
    return context_k - 1, seq_len - 2


def load_models():
    print("Loading base and assist models...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    assist_model = AutoModelForCausalLM.from_pretrained(
        ASSIST_MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    print("Models loaded.\n")
    return base_model, assist_model, tokenizer


@torch.inference_mode()
def get_logits_teacher_forcing(
    base_model, assist_model, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single batched forward pass (teacher-forcing) over the full token sequence.
    Returns:
      (base_logits, assist_logits)  shape [seq_len-1, vocab_size]
      where position j predicts input_ids[j+1]
    """
    base_out = base_model(input_ids=input_ids, return_dict=True)
    base_logits = base_out.logits[0, :-1, :].float()

    assist_out = assist_model(input_ids=input_ids, return_dict=True)
    assist_logits = assist_out.logits[0, :-1, :].float()

    return base_logits, assist_logits


def compute_topk_js_vectorized(
    base_logits: torch.Tensor,      # [T, V]
    assist_logits: torch.Tensor,    # [T, V]
    lambda_grid: np.ndarray,        # [L]
    top_k: int = 5,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute top-K JS divergence in batch over all time steps and lambda values.
    The top-K set is determined by the base distribution; both P and Q are
    re-normalized over that set.

    Returns:
      js_matrix  shape [T, L]
    """
    time_steps, _ = base_logits.shape
    grid_size = len(lambda_grid)

    base_probs = torch.softmax(base_logits, dim=-1)
    topk_indices = torch.topk(base_probs, k=top_k, dim=-1).indices

    base_topk = base_probs.gather(1, topk_indices)
    base_topk = base_topk / base_topk.sum(dim=-1, keepdim=True).clamp(min=eps)

    lambda_tensor = torch.tensor(lambda_grid, dtype=torch.float32, device=base_logits.device)
    base_l = base_logits.unsqueeze(1)      # [T, 1, V]
    assist_l = assist_logits.unsqueeze(1)  # [T, 1, V]
    lam = lambda_tensor.view(1, grid_size, 1)

    fused_logits = base_l - lam * assist_l
    fused_probs = torch.softmax(fused_logits, dim=-1)

    topk_exp = topk_indices.unsqueeze(1).expand(time_steps, grid_size, top_k)
    fused_topk = fused_probs.gather(2, topk_exp)
    fused_topk = fused_topk / fused_topk.sum(dim=-1, keepdim=True).clamp(min=eps)

    p = base_topk.unsqueeze(1)
    q = fused_topk
    m = 0.5 * (p + q)

    kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum(dim=-1)
    kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum(dim=-1)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js.cpu().numpy()


def main():
    global LAMBDA_GRID, JS_BUDGET, TOP_K, CONTEXT_K, MAX_SEQ_LEN, NUM_SAMPLES, DEVICE
    global BASE_MODEL_PATH, ASSIST_MODEL_PATH, DATASET_PATH

    parser = argparse.ArgumentParser(
        description="Estimate lambda_max / alpha_max (aligned with adaptive fusion experiment)"
    )
    parser.add_argument("--lambda_min", type=float, default=0.0)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--lambda_steps", type=int, default=21)
    parser.add_argument("--js_budget", type=float, default=JS_BUDGET)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--context_k", type=int, default=CONTEXT_K)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH,
                        help="Path to base model (e.g. $MODEL_ROOT/Llama-3.2-1B-Instruct-merged)")
    parser.add_argument("--assist_model_path", type=str, default=ASSIST_MODEL_PATH,
                        help="Path to assistant model (e.g. $ASSISTANT_MODEL_ROOT/Llama-3.2-1B-Instruct-8layers)")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
                        help="Path to input dataset (.jsonl or .xlsx)")
    parser.add_argument("--text_column", type=str, default=TEXT_COLUMN)
    parser.add_argument("--source_filter", type=str, default=SOURCE_FILTER)
    args = parser.parse_args()

    LAMBDA_GRID = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)
    JS_BUDGET = args.js_budget
    TOP_K = args.top_k
    CONTEXT_K = args.context_k
    MAX_SEQ_LEN = args.max_seq_len
    NUM_SAMPLES = args.num_samples
    DEVICE = args.device
    BASE_MODEL_PATH = args.base_model_path
    ASSIST_MODEL_PATH = args.assist_model_path
    DATASET_PATH = args.dataset_path

    print("=" * 70)
    print("lambda_max / alpha_max estimation (aligned with adaptive fusion experiment)")
    print("=" * 70)
    print(f"  lambda grid:      {LAMBDA_GRID}")
    print(f"  JS budget delta:  {JS_BUDGET}")
    print(f"  top-K:            {TOP_K}")
    print(f"  context_k:        {CONTEXT_K}")
    print(f"  max_seq_len:      {MAX_SEQ_LEN}")
    print(f"  num_samples:      {'all' if NUM_SAMPLES <= 0 else NUM_SAMPLES}")
    print(f"  device:           {DEVICE}")
    print(f"  dataset_path:     {args.dataset_path}")
    print(f"  source_filter:    {args.source_filter}")
    print("  fusion form:      softmax(base_logits - lambda * assist_logits)")
    print("=" * 70)

    records = load_records(args.dataset_path, args.text_column)
    filtered_records = filter_records(records, args.source_filter, args.text_column)
    if NUM_SAMPLES > 0:
        filtered_records = filtered_records[:NUM_SAMPLES]
    texts = [record[args.text_column] for record in filtered_records]

    print(f"\nTexts after filtering: {len(texts)}\n")

    base_model, assist_model, tokenizer = load_models()

    all_js = []
    for text in tqdm(texts, desc="Processing samples"):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        seq_len = input_ids.size(1)

        row_slice = get_valid_row_slice(seq_len, CONTEXT_K)
        if row_slice is None:
            continue

        base_logits, assist_logits = get_logits_teacher_forcing(
            base_model, assist_model, input_ids
        )

        row_start, row_end = row_slice
        selected_base = base_logits[row_start:row_end, :]
        selected_assist = assist_logits[row_start:row_end, :]
        if selected_base.size(0) == 0:
            continue

        js_mat = compute_topk_js_vectorized(
            selected_base,
            selected_assist,
            LAMBDA_GRID,
            top_k=TOP_K,
        )
        all_js.append(js_mat)

    if not all_js:
        print("Error: no valid positions found. Check data and configuration.")
        return

    all_js_flat = np.concatenate(all_js, axis=0)
    pos_total = len(all_js_flat)
    mean_js = all_js_flat.mean(axis=0)

    print(f"\nTotal valid positions: {pos_total}\n")
    print("=" * 70)
    print(f"lambda vs. avg top-{TOP_K} JS ({pos_total} positions)")
    print("=" * 70)
    print(f"{'lambda':>10}  {'avg top-K JS':>16}")
    print("-" * 30)
    for lam, js in zip(LAMBDA_GRID, mean_js):
        print(f"{lam:>10.4f}  {js:>16.6f}")

    valid_mask = mean_js <= JS_BUDGET
    if valid_mask.any():
        best_idx = np.where(valid_mask)[0].max()
        lambda_max = float(LAMBDA_GRID[best_idx])
        best_js = float(mean_js[best_idx])
    else:
        lambda_max = float(LAMBDA_GRID[0])
        best_js = float(mean_js[0])
        print(
            f"\nWarning: all lambda values exceed budget delta={JS_BUDGET}. "
            f"Using minimum lambda={lambda_max:.4f} (JS={best_js:.6f})"
        )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  JS budget delta:       {JS_BUDGET}")
    print(f"  total valid positions: {pos_total}")
    print(f"  lambda_max:            {lambda_max:.4f}")
    print(f"  avg top-{TOP_K} JS:       {best_js:.6f}")
    print(f"  suggested alpha_max:   <= {lambda_max:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
