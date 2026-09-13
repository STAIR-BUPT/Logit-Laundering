#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
for v in BASE_MODEL ASSISTANT_INIT ASSISTANT_LAYERS; do require_var "$v"; done
require_dir "$BASE_MODEL"
mkdir -p "$(dirname "$ASSISTANT_INIT")"
run env BASE_MODEL_PATH="$BASE_MODEL" SAVE_PATH="$ASSISTANT_INIT" NUM_LAYERS="$ASSISTANT_LAYERS" "$PYTHON_BIN" "$REPO_ROOT/assistant_model/build_small_llm.py"

