#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
for v in LLAMA_FACTORY2_ROOT ASSISTANT_TRAIN_CONFIG ASSISTANT_MODEL_ROOT; do require_var "$v"; done
require_dir "$LLAMA_FACTORY2_ROOT"; require_file "$ASSISTANT_TRAIN_CONFIG"
command -v llamafactory-cli >/dev/null || die 'llamafactory-cli is not installed'
(cd "$LLAMA_FACTORY2_ROOT" && run llamafactory-cli train "$ASSISTANT_TRAIN_CONFIG")
