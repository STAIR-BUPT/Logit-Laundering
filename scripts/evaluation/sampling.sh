#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
for v in TRAINED_BASE_MODEL TRAINED_ASSISTANT UTILITY_DATA WATERMARK_SEEDING_SCHEME ALPHA_MIN ALPHA_MAX ALPHA_S; do require_var "$v"; done
require_dir "$TRAINED_BASE_MODEL"; require_dir "$TRAINED_ASSISTANT"; require_file "$UTILITY_DATA"
read -r -a decode_specs <<< "${DECODE_SPECS:-greedy topk:40:0.9 topp:0.9:0.9}"
run "$PYTHON_BIN" "$REPO_ROOT/utility_evaluation/generation/adaptive_fusion_sampling_aggregate.py" --base_model_path "$TRAINED_BASE_MODEL" --assist_model_path "$TRAINED_ASSISTANT" --dataset_path "$UTILITY_DATA" --text_column "$SOURCE_TEXT_COLUMN" --group_by_field id --seeding_scheme "$WATERMARK_SEEDING_SCHEME" --alpha_min "$ALPHA_MIN" --alpha_max "$ALPHA_MAX" --alpha_s "$ALPHA_S" --num_samples "${NUM_SAMPLES:-0}" --decode_specs "${decode_specs[@]}" --output_json "$OUTPUT_ROOT/sampling-adaptive.json"
