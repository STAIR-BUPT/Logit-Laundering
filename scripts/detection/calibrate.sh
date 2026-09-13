#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
for v in TRAINED_BASE_MODEL TRAINED_ASSISTANT CALIBRATION_DATA WATERMARKED_TEXT_COLUMN JS_BUDGET; do require_var "$v"; done
require_dir "$TRAINED_BASE_MODEL"; require_dir "$TRAINED_ASSISTANT"; require_file "$CALIBRATION_DATA"
run "$PYTHON_BIN" "$REPO_ROOT/budget_estimation/estimate_lambda_max.py" --base_model_path "$TRAINED_BASE_MODEL" --assist_model_path "$TRAINED_ASSISTANT" --dataset_path "$CALIBRATION_DATA" --text_column "$WATERMARKED_TEXT_COLUMN" --source_filter "${SOURCE_FILTER:-}" --js_budget "$JS_BUDGET" --num_samples "${NUM_SAMPLES:-0}"

