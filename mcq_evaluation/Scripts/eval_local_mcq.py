from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise RuntimeError("Please install torch and transformers before running this script.") from exc


DEFAULT_DATA_JSON = Path("${MCQ_DATA}")
DEFAULT_MODEL_DIR = "${TRAINED_BASE_MODEL}"
DEFAULT_OUTPUT_ROOT = Path("${OUTPUT_ROOT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local causal LM on authorized multiple-choice QA.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="local or remote model directory")
    parser.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON, help="path to evaluation QA json")
    parser.add_argument("--out-file", type=Path, default=None, help="path to output json")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N valid questions")
    parser.add_argument("--max-input-length", type=int, default=1024, help="tokenizer truncation length")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="generation length")
    parser.add_argument("--log-every", type=int, default=20, help="print progress every N evaluated samples")
    parser.add_argument("--trust-remote-code", action="store_true", help="pass trust_remote_code to model/tokenizer")
    return parser.parse_args()


def sanitize_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "model"


def infer_output_path(data_json: Path, model_dir: str) -> Path:
    model_label = sanitize_name(Path(model_dir).name)
    return DEFAULT_OUTPUT_ROOT / f"{data_json.stem}.{model_label}.mcq_eval.json"


def normalize_options(options: dict) -> dict[str, str]:
    normalized = {}
    for letter in ("A", "B", "C", "D"):
        normalized[letter] = str(options.get(letter, "")).strip()
    return normalized


def normalize_answer(text, options: dict[str, str] | None = None) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()
    upper = text.upper()

    for marker in ("OUTPUT:", "FINAL ANSWER:", "ANSWER:"):
        idx = upper.find(marker)
        if idx != -1:
            text = text[idx + len(marker) :].strip()
            upper = text.upper()
            break

    match = re.search(r"\b([ABCD])\b", upper)
    if match:
        return match.group(1)

    match = re.search(r"THE CORRECT ANSWER IS[:\s]*([ABCD])", upper)
    if match:
        return match.group(1)

    if options:
        text_lower = text.lower()
        for letter, option_text in options.items():
            option_lower = option_text.lower().strip()
            if option_lower and (text_lower == option_lower or text_lower.startswith(option_lower)):
                return letter

    return ""


def strip_prompt_echo(prompt: str, generated: str) -> str:
    if not isinstance(generated, str):
        return ""

    result = generated.strip()
    if not result:
        return ""

    prompt = prompt.strip()
    if prompt and result.startswith(prompt):
        result = result[len(prompt) :].lstrip()

    result = re.sub(r"^(?:Human|User|Prompt|Assistant)\s*:\s*", "", result, flags=re.IGNORECASE)
    return result.strip()


def build_prompt(book_title: str, question: str, options: dict[str, str]) -> str:
    return (
        f'Novel: "{book_title}"\n'
        "Task: answer the multiple-choice question using exactly one option letter.\n"
        "Rules:\n"
        "1. Choose exactly one of A, B, C, or D.\n"
        "2. Even if an option is a phrase, output only its letter.\n"
        "3. Do not output explanation or any extra words.\n\n"
        f"Question: {question}\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n\n"
        "Final answer:"
    )


def build_model_input(tokenizer, prompt: str) -> str:
    system_prompt = (
        "You are a careful multiple-choice solver for novel comprehension. "
        "Return exactly one uppercase letter: A, B, C, or D."
    )
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"


def load_model_and_tokenizer(model_dir: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def main() -> int:
    args = parse_args()
    out_file = args.out_file or infer_output_path(args.data_json, args.model_dir)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.data_json, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    tokenizer, model = load_model_and_tokenizer(args.model_dir, args.trust_remote_code)
    device = next(model.parameters()).device

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
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
            )

        generated_raw = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
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
        "model_dir": args.model_dir,
        "data_json": str(args.data_json),
        "out_file": str(out_file),
        "limit": args.limit,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "per_book": per_book_summary,
        "cases": cases,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Summary ===")
    print(f"total={total}, correct={correct}, accuracy={summary['accuracy']:.4f}")
    print(f"results saved to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
