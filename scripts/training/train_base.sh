#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_experiment_config "${1:-}"
require_var BASE_TRAIN_CONFIG
require_var OUTPUT_MODEL_ROOT
config="$BASE_TRAIN_CONFIG"; [[ "$config" = /* ]] || config="$REPO_ROOT/$config"
require_file "$config"
command -v llamafactory-cli >/dev/null || die 'llamafactory-cli is not installed'
run llamafactory-cli train "$config"
