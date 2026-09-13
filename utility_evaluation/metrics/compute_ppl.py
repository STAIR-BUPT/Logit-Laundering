#!/usr/bin/env python3
"""
Compute perplexity (fluency) of all model-generated outputs using GPT2-XL as the reference model.
PPL is evaluated on the generated text with the prompt as prefix context
(loss is computed only on the generated tokens).

Outputs:
  ppl_results.jsonl  -- per-sample PPL details
  ppl_summary.csv    -- summary statistics (mean / median / std) per (model, mode)
"""

import json
import math
import os
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GPT2_XL_PATH = os.environ.get("GPT2_XL_PATH", "${MODEL_ROOT}/gpt2-xl")

DEFAULT_OUTPUT_ROOT = Path(os.environ.get("GENERATION_OUTPUT_ROOT", Path(__file__).resolve().parents[1] / "generation" / "outputs"))
OUTPUT_DIRS = json.loads(os.environ["PPL_OUTPUT_DIRS_JSON"]) if os.environ.get("PPL_OUTPUT_DIRS_JSON") else {
    "llama32_1b": str(DEFAULT_OUTPUT_ROOT / "llama32_1b"),
    "qwen25_1b5": str(DEFAULT_OUTPUT_ROOT / "qwen25_1b5"),
    "stablelm2_1b6": str(DEFAULT_OUTPUT_ROOT / "stablelm2_1b6"),
    "gemma2_2b": str(DEFAULT_OUTPUT_ROOT / "gemma2_2b"),
    "llama32_3b": str(DEFAULT_OUTPUT_ROOT / "llama32_3b"),
}

MODES = ["base", "rankswap_2s", "rankswap_ss", "rankswap_sk", "logit_laundering"]

BASE_DIR      = Path(os.environ.get("PPL_RESULT_DIR", Path(__file__).resolve().parents[1] / "metrics_outputs"))
RESULTS_PATH  = BASE_DIR / "ppl_results.jsonl"
SUMMARY_PATH  = BASE_DIR / "ppl_summary.csv"

MAX_LENGTH = 1024  # GPT2-XL context window


# ── PPL computation ───────────────────────────────────────────────────────────

@torch.inference_mode()
def compute_ppl(model, tokenizer, prompt: str, generated: str) -> float:
    """
    Compute perplexity of `generated` using `prompt` as prefix context.
    Loss is computed only on generated tokens (prompt tokens get label=-100).
    Returns NaN if generated is empty.
    """
    generated = generated.strip()
    if not generated:
        return float("nan")

    prompt_ids    = tokenizer.encode(prompt,    add_special_tokens=False)
    generated_ids = tokenizer.encode(generated, add_special_tokens=False)

    if len(generated_ids) == 0:
        return float("nan")

    # Fit within GPT2-XL's 1024-token window: trim prompt if necessary
    total = len(prompt_ids) + len(generated_ids)
    if total > MAX_LENGTH:
        keep = max(0, MAX_LENGTH - len(generated_ids))
        prompt_ids = prompt_ids[-keep:] if keep > 0 else []

    all_ids   = prompt_ids + generated_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=DEVICE)

    # Mask out prompt tokens from loss
    labels = input_ids.clone()
    labels[0, : len(prompt_ids)] = -100

    loss = model(input_ids=input_ids, labels=labels).loss.item()
    return math.exp(loss)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_stats(values: list[float]) -> dict:
    """Return mean / median / std, ignoring NaN."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean":   float(np.mean(clean)),
        "median": float(np.median(clean)),
        "std":    float(np.std(clean)),
        "n":      len(clean),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading GPT2-XL from {GPT2_XL_PATH} ...")
    tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_XL_PATH)
    model     = GPT2LMHeadModel.from_pretrained(GPT2_XL_PATH, torch_dtype=torch.float16).to(DEVICE).eval()
    print("GPT2-XL loaded.\n")

    results_file = open(RESULTS_PATH, "w", encoding="utf-8")
    summary_rows = []  # (model_name, mode, mean, median, std, n)

    for model_name, out_dir in OUTPUT_DIRS.items():
        out_path = Path(out_dir)
        print(f"{'='*60}")
        print(f"Model: {model_name}  ({out_dir})")

        for mode in MODES:
            jsonl_path = out_path / f"{mode}.jsonl"
            if not jsonl_path.exists():
                print(f"  [{mode}] MISSING — skip")
                continue

            records = load_jsonl(str(jsonl_path))
            ppls = []

            for rec in tqdm(records, desc=f"  {model_name}/{mode}", leave=False):
                if rec.get("skipped"):
                    ppls.append(float("nan"))
                    continue
                ppl = compute_ppl(
                    model, tokenizer,
                    prompt=rec.get("prompt", ""),
                    generated=rec.get("generated", ""),
                )
                ppls.append(ppl)

                # write per-sample result
                results_file.write(json.dumps({
                    "model": model_name,
                    "mode":  mode,
                    "ppl":   ppl,
                    "prompt_preview":    rec.get("prompt",    "")[:80],
                    "generated_preview": rec.get("generated", "")[:80],
                }, ensure_ascii=False) + "\n")

            stats = safe_stats(ppls)
            print(f"  [{mode}]  mean={stats['mean']:.2f}  median={stats['median']:.2f}  std={stats['std']:.2f}  n={stats['n']}")
            summary_rows.append((model_name, mode, stats["mean"], stats["median"], stats["std"], stats["n"]))

    results_file.close()

    # Write CSV summary
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mode", "ppl_mean", "ppl_median", "ppl_std", "n"])
        writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print(f"Per-sample results → {RESULTS_PATH}")
    print(f"Summary CSV        → {SUMMARY_PATH}")

    # Pretty-print final table
    print(f"\n{'model':<16} {'mode':<22} {'mean':>8} {'median':>8} {'std':>8}")
    print("-" * 66)
    for row in summary_rows:
        print(f"{row[0]:<16} {row[1]:<22} {row[2]:>8.2f} {row[3]:>8.2f} {row[4]:>8.2f}")


if __name__ == "__main__":
    main()
