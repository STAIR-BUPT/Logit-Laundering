#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAUVE metric computation script (JSONL version).

Default setup:
- Input: JSONL containing `prompt` / `generated` / `text` fields
- P distribution: `generated`
- Q distribution: human continuation (the `text` field with the `prompt` prefix stripped)

Also supports:
- Using another field in the same file as reference
- Passing a separate JSONL as the reference distribution
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute MAUVE for generated JSONL outputs.")
    parser.add_argument("--input_jsonl", required=True, help="input JSONL with generated texts")
    parser.add_argument(
        "--output_json",
        default=None,
        help="output summary JSON path; default is <input>.mauve.json",
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="optional experiment name for the output payload",
    )
    parser.add_argument(
        "--generated_field",
        default="generated",
        help="field used as the model output text",
    )
    parser.add_argument(
        "--reference_mode",
        default="human_continuation",
        choices=["human_continuation", "same_jsonl_field", "other_jsonl"],
        help="how to build reference texts",
    )
    parser.add_argument(
        "--reference_field",
        default="text",
        help="reference field for same_jsonl_field or other_jsonl modes",
    )
    parser.add_argument(
        "--prompt_field",
        default="prompt",
        help="prompt field used for human_continuation mode",
    )
    parser.add_argument(
        "--reference_jsonl",
        default=None,
        help="optional reference JSONL for other_jsonl mode",
    )
    parser.add_argument(
        "--skip_field",
        default="skipped",
        help="rows with this field set truthy will be skipped",
    )
    parser.add_argument(
        "--num_buckets",
        default="auto",
        help="MAUVE num_buckets, use 'auto' or an integer",
    )
    parser.add_argument(
        "--featurize_model_name",
        default="gpt2-large",
        help="transformers feature model name for MAUVE",
    )
    parser.add_argument("--device_id", type=int, default=-1, help="GPU id for MAUVE, or -1 for CPU")
    parser.add_argument("--max_text_length", type=int, default=1024, help="max token length for MAUVE")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for MAUVE featurization")
    parser.add_argument("--kmeans_num_redo", type=int, default=5, help="MAUVE kmeans_num_redo")
    parser.add_argument("--kmeans_max_iter", type=int, default=500, help="MAUVE kmeans_max_iter")
    parser.add_argument("--seed", type=int, default=25, help="random seed for MAUVE")
    parser.add_argument("--verbose", action="store_true", help="enable verbose MAUVE logging")
    return parser


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def strip_prompt_prefix(full_text: str, prompt: str) -> str:
    full_text = (full_text or "").strip()
    prompt = (prompt or "").strip()

    if not full_text:
        return ""
    if not prompt:
        return full_text

    if full_text.startswith(prompt):
        stripped = full_text[len(prompt) :].lstrip()
        return stripped or full_text

    nearby_index = full_text[: min(len(prompt) + 64, len(full_text))].find(prompt)
    if 0 <= nearby_index <= 16:
        stripped = full_text[nearby_index + len(prompt) :].lstrip()
        return stripped or full_text

    return full_text


def extract_human_continuation(
    record: dict[str, Any],
    reference_field: str = "text",
    prompt_field: str = "prompt",
) -> str:
    return strip_prompt_prefix(
        str(record.get(reference_field, "") or ""),
        str(record.get(prompt_field, "") or ""),
    )


def extract_reference_text(
    record: dict[str, Any],
    *,
    reference_mode: str,
    reference_field: str,
    prompt_field: str,
) -> tuple[str, bool]:
    if reference_mode == "human_continuation":
        full_text = str(record.get(reference_field, "") or "").strip()
        continuation = extract_human_continuation(
            record,
            reference_field=reference_field,
            prompt_field=prompt_field,
        )
        return continuation, continuation == full_text

    return str(record.get(reference_field, "") or "").strip(), False


def build_text_pairs(
    records: list[dict[str, Any]],
    *,
    generated_field: str = "generated",
    reference_mode: str = "human_continuation",
    reference_field: str = "text",
    prompt_field: str = "prompt",
    skip_field: str = "skipped",
    reference_records: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], dict[str, int]]:
    if reference_mode == "other_jsonl":
        if reference_records is None:
            raise ValueError("reference_mode=other_jsonl requires reference_records")
        if len(reference_records) != len(records):
            raise ValueError("reference_jsonl and input_jsonl must have the same number of rows")

    generated_texts: list[str] = []
    reference_texts: list[str] = []
    stats = {
        "total_records": len(records),
        "kept_records": 0,
        "skipped_flagged": 0,
        "skipped_empty_generated": 0,
        "skipped_empty_reference": 0,
        "human_continuation_fallback_full_text": 0,
    }

    for index, record in enumerate(records):
        if record.get(skip_field):
            stats["skipped_flagged"] += 1
            continue

        generated = str(record.get(generated_field, "") or "").strip()
        if not generated:
            stats["skipped_empty_generated"] += 1
            continue

        if reference_mode == "other_jsonl":
            reference_record = reference_records[index]
            reference = str(reference_record.get(reference_field, "") or "").strip()
            used_fallback = False
        else:
            reference, used_fallback = extract_reference_text(
                record,
                reference_mode=reference_mode,
                reference_field=reference_field,
                prompt_field=prompt_field,
            )

        if not reference:
            stats["skipped_empty_reference"] += 1
            continue

        if used_fallback:
            stats["human_continuation_fallback_full_text"] += 1

        generated_texts.append(generated)
        reference_texts.append(reference)
        stats["kept_records"] += 1

    return generated_texts, reference_texts, stats


def summarize_text_lengths(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {
            "count": 0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "mean_words": 0.0,
            "median_words": 0.0,
        }

    char_lengths = [len(text) for text in texts]
    word_lengths = [len(text.split()) for text in texts]
    return {
        "count": len(texts),
        "mean_chars": float(sum(char_lengths) / len(char_lengths)),
        "median_chars": float(statistics.median(char_lengths)),
        "mean_words": float(sum(word_lengths) / len(word_lengths)),
        "median_words": float(statistics.median(word_lengths)),
    }


def parse_num_buckets(value: str) -> str | int:
    if value == "auto":
        return value
    return int(value)


def compute_mauve_result(
    generated_texts: list[str],
    reference_texts: list[str],
    *,
    num_buckets: str | int,
    featurize_model_name: str,
    device_id: int,
    max_text_length: int,
    batch_size: int,
    kmeans_num_redo: int,
    kmeans_max_iter: int,
    seed: int,
    verbose: bool,
) -> dict[str, Any]:
    try:
        import mauve
    except ImportError as exc:
        raise ImportError("mauve is not installed in the current environment") from exc

    out = mauve.compute_mauve(
        p_text=generated_texts,
        q_text=reference_texts,
        num_buckets=num_buckets,
        featurize_model_name=featurize_model_name,
        device_id=device_id,
        max_text_length=max_text_length,
        batch_size=batch_size,
        kmeans_num_redo=kmeans_num_redo,
        kmeans_max_iter=kmeans_max_iter,
        seed=seed,
        verbose=verbose,
    )
    return {
        "mauve": float(out.mauve),
        "frontier_integral": float(out.frontier_integral),
        "mauve_star": float(out.mauve_star),
        "frontier_integral_star": float(out.frontier_integral_star),
        "num_buckets": int(out.num_buckets),
    }


def build_result_payload(
    *,
    experiment_name: str,
    input_jsonl: str,
    reference_mode: str,
    pair_stats: dict[str, int],
    generated_summary: dict[str, float],
    reference_summary: dict[str, float],
    mauve_result: dict[str, Any],
    mauve_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "input_jsonl": input_jsonl,
        "reference_mode": reference_mode,
        "pair_stats": pair_stats,
        "generated_summary": generated_summary,
        "reference_summary": reference_summary,
        "mauve": mauve_result,
        "mauve_kwargs": mauve_kwargs,
    }


def main() -> int:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_json) if args.output_json else input_path.with_suffix(".mauve.json")
    experiment_name = args.experiment_name or input_path.stem

    records = load_jsonl(args.input_jsonl)
    reference_records = load_jsonl(args.reference_jsonl) if args.reference_jsonl else None

    generated_texts, reference_texts, pair_stats = build_text_pairs(
        records,
        generated_field=args.generated_field,
        reference_mode=args.reference_mode,
        reference_field=args.reference_field,
        prompt_field=args.prompt_field,
        skip_field=args.skip_field,
        reference_records=reference_records,
    )
    if not generated_texts:
        raise ValueError("No valid text pairs were collected for MAUVE")

    generated_summary = summarize_text_lengths(generated_texts)
    reference_summary = summarize_text_lengths(reference_texts)
    num_buckets = parse_num_buckets(args.num_buckets)
    mauve_kwargs = {
        "num_buckets": num_buckets,
        "featurize_model_name": args.featurize_model_name,
        "device_id": args.device_id,
        "max_text_length": args.max_text_length,
        "batch_size": args.batch_size,
        "kmeans_num_redo": args.kmeans_num_redo,
        "kmeans_max_iter": args.kmeans_max_iter,
        "seed": args.seed,
    }

    mauve_result = compute_mauve_result(
        generated_texts,
        reference_texts,
        num_buckets=num_buckets,
        featurize_model_name=args.featurize_model_name,
        device_id=args.device_id,
        max_text_length=args.max_text_length,
        batch_size=args.batch_size,
        kmeans_num_redo=args.kmeans_num_redo,
        kmeans_max_iter=args.kmeans_max_iter,
        seed=args.seed,
        verbose=args.verbose,
    )

    payload = build_result_payload(
        experiment_name=experiment_name,
        input_jsonl=args.input_jsonl,
        reference_mode=args.reference_mode,
        pair_stats=pair_stats,
        generated_summary=generated_summary,
        reference_summary=reference_summary,
        mauve_result=mauve_result,
        mauve_kwargs=mauve_kwargs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"kept_pairs={pair_stats['kept_records']} / total={pair_stats['total_records']}")
    print(f"mauve={mauve_result['mauve']:.6f}")
    print(f"mauve_star={mauve_result['mauve_star']:.6f}")
    print(f"frontier_integral={mauve_result['frontier_integral']:.6f}")
    print(f"output_json={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
