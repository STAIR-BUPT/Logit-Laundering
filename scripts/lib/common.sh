#!/usr/bin/env bash
set -euo pipefail

SCRIPT_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_LIB_DIR/../.." && pwd)"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '==> %s\n' "$*"; }

load_experiment_config() {
  local config_path="${1:-$REPO_ROOT/configs/experiment.env}"
  [[ -f "$config_path" ]] || die "configuration not found: $config_path (copy configs/experiment.env.example first)"
  # shellcheck disable=SC1090
  set -a
  source "$config_path"
  set +a
  export PYTHON_BIN="${PYTHON_BIN:-python3}"
  export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
  require_var OUTPUT_ROOT
  mkdir -p "$OUTPUT_ROOT"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then export CUDA_VISIBLE_DEVICES; fi
  info "repository: $REPO_ROOT"
  info "configuration: $config_path"
  info "outputs: $OUTPUT_ROOT"
}

require_var() {
  local name="$1" value="${!1:-}"
  [[ -n "$value" ]] || die "required configuration variable is empty: $name"
  [[ "$value" != /absolute/path/* ]] || die "replace placeholder value for $name: $value"
  [[ "$value" != replace-with-* ]] || die "replace placeholder value for $name"
}

require_file() { [[ -f "$1" ]] || die "file not found: $1"; }
require_dir() { [[ -d "$1" ]] || die "directory not found: $1"; }

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}
