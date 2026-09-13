#!/usr/bin/env python3
"""Move samples with a boolean flag to the front of a JSON dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def reorganize_by_flag(input_file: str, output_file: str, flag_field: str) -> None:
    with open(input_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    flagged = [item for item in data if item.get(flag_field) is True]
    others = [item for item in data if item.get(flag_field) is not True]
    reorganized = flagged + others

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(reorganized, handle, ensure_ascii=False, indent=2)

    print(f"Total: {len(data)}")
    print(f"{flag_field}=true: {len(flagged)}")
    print(f"other: {len(others)}")
    print(f"Wrote: {output_file}")
    print(f"Set EVASION_SIZE={len(flagged)} for assistant training.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--flag-field",
        required=True,
        help="Boolean field used to identify the target subset placed first",
    )
    args = parser.parse_args()
    reorganize_by_flag(args.input, args.output, args.flag_field)


if __name__ == "__main__":
    main()
