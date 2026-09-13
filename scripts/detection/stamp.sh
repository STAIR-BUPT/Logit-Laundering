#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
for v in TRAINED_BASE_MODEL TRAINED_ASSISTANT PUBLIC_AUDIT_DATA PRIVATE_AUDIT_DATA WATERMARKED_TEXT_COLUMN GROUP_BY_FIELD ALPHA_MIN ALPHA_MAX ALPHA_S; do require_var "$v"; done
require_dir "$TRAINED_BASE_MODEL"; require_dir "$TRAINED_ASSISTANT"; require_file "$PUBLIC_AUDIT_DATA"; require_file "$PRIVATE_AUDIT_DATA"
run "$PYTHON_BIN" "$REPO_ROOT/detection/stamp/stamp_statistical_test_aggregate_fast.py" --base_model_path "$TRAINED_BASE_MODEL" --assist_model_path "$TRAINED_ASSISTANT" --public_dataset_file "$PUBLIC_AUDIT_DATA" --private_dataset_files "$PRIVATE_AUDIT_DATA" --text_key "$WATERMARKED_TEXT_COLUMN" --group_by_field "$GROUP_BY_FIELD" --alpha_min "$ALPHA_MIN" --alpha_max "$ALPHA_MAX" --alpha_s "$ALPHA_S"

