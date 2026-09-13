#!/usr/bin/env python3
"""Extract a prefix of flagged samples plus other samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_samples(
    input_file: str,
    output_file: str,
    flag_field: str,
    flagged_count: int,
    other_count: int,
) -> None:
    with open(input_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    flagged = [item for item in data if item.get(flag_field) is True]
    others = [item for item in data if item.get(flag_field) is not True]

    selected = flagged[:flagged_count] + others[:other_count]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)

    print(f"Selected {len(selected)} samples "
          f"({min(flagged_count, len(flagged))} flagged + "
          f"{min(other_count, len(others))} other)")
    print(f"Wrote: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--flag-field",
        required=True,
        help="Boolean field used to identify the target subset",
    )
    parser.add_argument("--flagged-count", type=int, required=True)
    parser.add_argument("--other-count", type=int, required=True)
    args = parser.parse_args()
    extract_samples(
        args.input,
        args.output,
        args.flag_field,
        args.flagged_count,
        args.other_count,
    )


if __name__ == "__main__":
    main()
