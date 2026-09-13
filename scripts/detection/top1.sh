#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
for v in TRAINED_BASE_MODEL TRAINED_ASSISTANT AUDIT_DATA WATERMARKED_TEXT_COLUMN GROUP_BY_FIELD WATERMARK_SEEDING_SCHEME ALPHA_MIN ALPHA_MAX ALPHA_S; do require_var "$v"; done
require_dir "$TRAINED_BASE_MODEL"; require_dir "$TRAINED_ASSISTANT"; require_file "$AUDIT_DATA"
run "$PYTHON_BIN" "$REPO_ROOT/detection/top1/Top-1_adaptive_fusion_aggregate.py" --base_model_path "$TRAINED_BASE_MODEL" --assist_model_path "$TRAINED_ASSISTANT" --dataset_path "$AUDIT_DATA" --text_column "$WATERMARKED_TEXT_COLUMN" --source_filter "${SOURCE_FILTER:-}" --group_by_field "$GROUP_BY_FIELD" --seeding_scheme "$WATERMARK_SEEDING_SCHEME" --alpha_min "$ALPHA_MIN" --alpha_max "$ALPHA_MAX" --alpha_s "$ALPHA_S" --num_samples "${NUM_SAMPLES:-0}" --output_json "$OUTPUT_ROOT/top1-adaptive.json"

