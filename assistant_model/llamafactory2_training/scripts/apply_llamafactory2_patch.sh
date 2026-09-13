#!/usr/bin/env bash
# Apply Logit Laundering assistant-training changes onto a clean LLaMA-Factory checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH="$PACK_ROOT/patches/LLaMA-Factory2_training.patch"
BASE_HEAD_FILE="$PACK_ROOT/patches/LLaMA-Factory2_BASE_HEAD.txt"
EXPECTED_HEAD="$(tr -d '[:space:]' < "$BASE_HEAD_FILE")"

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "Usage: $0 /path/to/LLaMA-Factory" >&2
  exit 2
fi
TARGET="$(cd "$TARGET" && pwd)"

if [[ ! -d "$TARGET/.git" ]]; then
  echo "error: $TARGET is not a git checkout" >&2
  exit 1
fi

HEAD="$(git -C "$TARGET" rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "error: expected revision $EXPECTED_HEAD, found $HEAD" >&2
  echo "hint: git -C \"$TARGET\" checkout $EXPECTED_HEAD" >&2
  exit 1
fi

git -C "$TARGET" apply "$PATCH"
echo "Applied $PATCH to $TARGET"
echo "Set EVASION_SIZE and SEGMENT_RATIO before training."
