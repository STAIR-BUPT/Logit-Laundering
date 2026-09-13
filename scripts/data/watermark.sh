#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
for v in SOURCE_DATA WATERMARKED_DATA SOURCE_TEXT_COLUMN WATERMARKED_TEXT_COLUMN WATERMARK_KEY WATERMARK_DELTA WATERMARK_MODEL_YAML; do require_var "$v"; done
require_file "$SOURCE_DATA"
require_file "$WATERMARK_MODEL_YAML"
mkdir -p "$(dirname "$WATERMARKED_DATA")"
filter_args=()
if [[ -n "${SOURCE_FILTER:-}" ]]; then
  filter_args=(--filter_field "$SOURCE_FILTER")
fi
run "$PYTHON_BIN" "$REPO_ROOT/watermarking/paraphrase_with_watermark.py" \
  --mode mode1 \
  --input "$SOURCE_DATA" \
  --output "$WATERMARKED_DATA" \
  --input_col "$SOURCE_TEXT_COLUMN" \
  --output_col "$WATERMARKED_TEXT_COLUMN" \
  --watermark_key "$WATERMARK_KEY" \
  --delta "$WATERMARK_DELTA" \
  --model_yaml "$WATERMARK_MODEL_YAML" \
  "${filter_args[@]}" \
  ${WATERMARK_NUM_SAMPLES:+--num_samples "$WATERMARK_NUM_SAMPLES"}
