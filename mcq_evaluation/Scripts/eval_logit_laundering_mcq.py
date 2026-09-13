from __future__ import annotations

import argparse
import os
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.eval_local_mcq import (
    build_model_input,
    build_prompt,
    normalize_answer,
    normalize_options,
    sanitize_name,
    strip_prompt_echo,
)


DEFAULT_DATA_JSON = Path("${MCQ_DATA}")
DEFAULT_BASE_MODEL_DIR = "${TRAINED_BASE_MODEL}"
DEFAULT_ASSIST_MODEL_DIR = "${TRAINED_ASSISTANT}"
DEFAULT_OUTPUT_ROOT = Path("${OUTPUT_ROOT}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a base+assist adaptive logit-laundering model on authorized multiple-choice QA."
    )
    parser.add_argument("--base-model-dir", default=DEFAULT_BASE_MODEL_DIR, help="base model directory")
    parser.add_argument("--assist-model-dir", default=DEFAULT_ASSIST_MODEL_DIR, help="assistant model directory")
    parser.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON, help="path to evaluation QA json")
    parser.add_argument("--out-file", type=Path, default=None, help="path to output json")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N valid questions")
    parser.add_argument("--max-input-length", type=int, default=1024, help="tokenizer truncation length")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="generation length")
    parser.add_argument("--log-every", type=int, default=20, help="print progress every N evaluated samples")
    parser.add_argument("--trust-remote-code", action="store_true", help="pass trust_remote_code to model/tokenizer")
    parser.add_argument("--alpha-min", type=float, default=float(os.environ.get("ALPHA_MIN", "0.17")))
    parser.add_argument("--alpha-max", type=float, default=float(os.environ.get("ALPHA_MAX", "0.56")))
    parser.add_argument("--alpha-s", type=float, default=float(os.environ.get("ALPHA_S", "1.0")))
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def infer_output_path(
    data_json: Path,
    base_model_dir: str,
    alpha_min: float,
    alpha_max: float,
    alpha_s: float,
) -> Path:
    model_label = sanitize_name(Path(base_model_dir).name)
    return (
        DEFAULT_OUTPUT_ROOT
        / (
            f"{data_json.stem}.{model_label}."
            f"logit_laundering_amin{alpha_min:.2f}_amax{alpha_max:.2f}_s{alpha_s:.1f}.mcq_eval.json"
        )
    )


def compute_adaptive_alpha(
    base_logits_last: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    alpha_s: float,
) -> tuple[float, float]:
    top2_val = torch.topk(base_logits_last.float(), k=2).values
    margin = (top2_val[0] - top2_val[1]).item()
    alpha = alpha_min + (alpha_max - alpha_min) * margin / (margin + alpha_s)
    return alpha, margin


def greedy_generate_with_logit_laundering(
    base_model,
    assist_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    alpha_min: float,
    alpha_max: float,
    alpha_s: float,
    eos_token_id: int | None,
) -> torch.Tensor:
    generated_ids = input_ids
    generated_attn = attention_mask
    new_tokens = []

    for _ in range(max_new_tokens):
        base_out = base_model(input_ids=generated_ids, attention_mask=generated_attn, return_dict=True)
        assist_out = assist_model(input_ids=generated_ids, attention_mask=generated_attn, return_dict=True)

        base_last = base_out.logits[:, -1, :].float()
        assist_last = assist_out.logits[:, -1, :].float()

        alpha_t, _ = compute_adaptive_alpha(
            base_last[0],
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_s=alpha_s,
        )
        fused_last = base_last - alpha_t * assist_last
        next_token = fused_last.argmax(dim=-1)
        new_tokens.append(next_token)

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
        if generated_attn is not None:
            next_attn = torch.ones_like(next_token.unsqueeze(-1), device=generated_attn.device)
            generated_attn = torch.cat([generated_attn, next_attn], dim=-1)

        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
            break

    if not new_tokens:
        return torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat(new_tokens, dim=0)


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_models_and_tokenizer(
    base_model_dir: str,
    assist_model_dir: str,
    trust_remote_code: bool,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    assist_model = AutoModelForCausalLM.from_pretrained(
        assist_model_dir,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    base_model.eval()
    assist_model.eval()
    return tokenizer, base_model, assist_model, device


def main() -> int:
    args = parse_args()
    out_file = args.out_file or infer_output_path(
        data_json=args.data_json,
        base_model_dir=args.base_model_dir,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_s=args.alpha_s,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.data_json, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    tokenizer, base_model, assist_model, device = load_models_and_tokenizer(
        args.base_model_dir,
        args.assist_model_dir,
        args.trust_remote_code,
    )

    total = 0
    correct = 0
    cases = []
    per_book = defaultdict(lambda: {"total": 0, "correct": 0})

    for idx, item in enumerate(raw_dataset, 1):
        question = str(item.get("Question", "")).strip()
        options = normalize_options(item.get("Options", {}))
        gold_letter = normalize_answer(item.get("Gold", ""))
        book_name = str(item.get("book_name", "")).strip() or "Unknown"

        if not question or gold_letter not in {"A", "B", "C", "D"} or not all(options.values()):
            continue

        prompt = build_prompt(book_name, question, options)
        model_input = build_model_input(tokenizer, prompt)
        inputs = tokenizer(
            model_input,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_ids = greedy_generate_with_logit_laundering(
                base_model=base_model,
                assist_model=assist_model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                alpha_s=args.alpha_s,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_token_ids = generated_ids.detach().cpu().tolist() if generated_ids.numel() else []
        generated_raw = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
        generated = strip_prompt_echo(model_input, generated_raw)
        pred = normalize_answer(generated, options=options)
        if not pred:
            match = re.search(r"[ABCD]", generated.upper())
            if match:
                pred = match.group(0)

        is_correct = pred == gold_letter
        total += 1
        correct += int(is_correct)
        per_book[book_name]["total"] += 1
        per_book[book_name]["correct"] += int(is_correct)

        cases.append(
            {
                "idx": idx,
                "book_name": book_name,
                "question": question,
                "options": options,
                "gold": gold_letter,
                "gold_text": options.get(gold_letter, ""),
                "pred": pred,
                "pred_text": options.get(pred, "") if pred in options else "",
                "is_correct": is_correct,
                "generated": generated,
                "generated_raw": generated_raw,
                "answer": item.get("Answer", ""),
                "evidences": item.get("Evidences", []),
            }
        )

        if total % max(args.log_every, 1) == 0:
            acc = correct / total
            print(f"[{total}] acc={acc:.4f} latest book={book_name} gold={gold_letter} pred={pred}")

        if args.limit is not None and total >= args.limit:
            break

    per_book_summary = {
        book: {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
        }
        for book, stats in sorted(per_book.items())
    }

    summary = {
        "base_model_dir": args.base_model_dir,
        "assist_model_dir": args.assist_model_dir,
        "data_json": str(args.data_json),
        "out_file": str(out_file),
        "limit": args.limit,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_s": args.alpha_s,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "per_book": per_book_summary,
        "cases": cases,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluation summary to {out_file}")
    print(f"Accuracy: {summary['accuracy']:.4f} ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
