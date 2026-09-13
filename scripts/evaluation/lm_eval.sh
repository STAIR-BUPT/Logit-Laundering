#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
require_var LM_EVAL_CONFIG
for v in BASE_MODEL_ROOT ASSISTANT_MODEL_ROOT HF_DATASETS_CACHE_DIR; do require_var "$v"; done
require_dir "$BASE_MODEL_ROOT"; require_dir "$ASSISTANT_MODEL_ROOT"; require_dir "$HF_DATASETS_CACHE_DIR"
config="$LM_EVAL_CONFIG"; [[ "$config" = /* ]] || config="$REPO_ROOT/$config"
require_file "$config"
args=(--config "$config" --all --batch-size "${LM_EVAL_BATCH_SIZE:-1}" --output-dir "$OUTPUT_ROOT/lm-eval")
if [[ "${LM_EVAL_LIMIT:-0}" -gt 0 ]]; then args+=(--limit "$LM_EVAL_LIMIT"); fi
(cd "$REPO_ROOT/utility_evaluation/lm_eval_harness" && run "$PYTHON_BIN" evaluate_models_from_config.py "${args[@]}")
