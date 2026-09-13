#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
for v in WATERMARKED_DATA ATTACKED_DATA WATERMARKED_TEXT_COLUMN ATTACKED_TEXT_COLUMN ATTACK_METHOD; do require_var "$v"; done
require_file "$WATERMARKED_DATA"
mkdir -p "$(dirname "$ATTACKED_DATA")"
extra=()
if [[ "$ATTACK_METHOD" == swap ]]; then require_var ATTACK_MODEL_PATH; extra+=(--model-path "$ATTACK_MODEL_PATH"); fi
if [[ -n "${SOURCE_FILTER:-}" ]]; then extra+=(--filter-field "$SOURCE_FILTER"); fi
run "$PYTHON_BIN" "$REPO_ROOT/attacks/data_level/text/text_attack.py" \
  --method "$ATTACK_METHOD" \
  --input "$WATERMARKED_DATA" \
  --output "$ATTACKED_DATA" \
  --input-field "$WATERMARKED_TEXT_COLUMN" \
  --output-field "$ATTACKED_TEXT_COLUMN" \
  "${extra[@]}"
