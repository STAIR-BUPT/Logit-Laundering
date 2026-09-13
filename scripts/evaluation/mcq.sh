#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
for v in TRAINED_BASE_MODEL TRAINED_ASSISTANT MCQ_DATA ALPHA_MIN ALPHA_MAX ALPHA_S; do require_var "$v"; done
require_dir "$TRAINED_BASE_MODEL"; require_dir "$TRAINED_ASSISTANT"; require_file "$MCQ_DATA"
run "$PYTHON_BIN" "$REPO_ROOT/mcq_evaluation/Scripts/eval_logit_laundering_mcq.py" --base-model-dir "$TRAINED_BASE_MODEL" --assist-model-dir "$TRAINED_ASSISTANT" --data-json "$MCQ_DATA" --out-file "$OUTPUT_ROOT/mcq-logit-laundering.json" --alpha-min "$ALPHA_MIN" --alpha-max "$ALPHA_MAX" --alpha-s "$ALPHA_S"

