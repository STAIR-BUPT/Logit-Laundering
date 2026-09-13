#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo 'Checking shell syntax...'
while IFS= read -r script; do bash -n "$script"; done < <(find scripts -type f -name '*.sh' | sort)

echo 'Checking Python syntax for project entry points...'
python3 -m py_compile \
  src/logit_laundering/__init__.py \
  src/logit_laundering/cli.py \
  assistant_model/build_small_llm.py \
  attacks/data_level/character/char_attack.py \
  attacks/data_level/text/text_attack.py \
  budget_estimation/estimate_lambda_max.py \
  detection/stamp/stamp_statistical_test_aggregate.py \
  detection/stamp/stamp_statistical_test_aggregate_fast.py \
  detection/stamp/stamp_statistical_test_base_aggregate.py \
  detection/top1/Top-1_adaptive_fusion_aggregate.py \
  detection/top1/Top-1_control_llm_aggregate.py \
  mcq_evaluation/Scripts/eval_local_mcq.py \
  mcq_evaluation/Scripts/eval_logit_laundering_mcq.py \
  utility_evaluation/generation/adaptive_fusion_sampling_aggregate.py \
  utility_evaluation/generation/fakebase_sampling_aggregate.py \
  utility_evaluation/lm_eval_harness/evaluate_models_from_config.py \
  utility_evaluation/metrics/calculate_mauve_json.py \
  utility_evaluation/metrics/compute_ppl.py \
  watermarking/paraphrase_with_watermark.py

echo 'Checking JSON configuration files...'
python3 - <<'PY'
import json
from pathlib import Path

for path in sorted(Path('.').rglob('*.json')):
    if '.git' in path.parts:
        continue
    json.loads(path.read_text(encoding='utf-8'))
print('JSON OK')
PY

echo 'Checking packaging entry points...'
required_paths=(
  LICENSE
  docs/index.html
  docs/static/css/style.css
  assistant_model/llamafactory2_training/scripts/apply_llamafactory2_patch.sh
  assistant_model/llamafactory2_training/new_files/data/reorganize_by_flag.py
  assistant_model/llamafactory2_training/new_files/src/llamafactory/train/pt/assistant_loss.py
  assistant_model/llamafactory2_training/patches/LLaMA-Factory2_training.patch
  assistant_model/llamafactory2_training/patches/LLaMA-Factory2_BASE_HEAD.txt
)
for path in "${required_paths[@]}"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 1; }
done
if grep -E -- '--filter_illegal_only' scripts/data/*.sh >/dev/null 2>&1; then
  echo 'data stage scripts still reference removed --filter_illegal_only' >&2
  exit 1
fi

echo 'Checking restricted paths...'
restricted=(
  'mcq_evaluation/restricted_mcq'
  'base_model_training/llamafactory/data/authorized_sft_qa.json'
  'base_model_training/llamafactory/data/authorized_wm_corpus.jsonl'
)
for path in "${restricted[@]}"; do
  [[ ! -e "$path" ]] || { echo "restricted path exists: $path" >&2; exit 1; }
done
shopt -s nullglob
for path in utility_evaluation/data/*.jsonl; do
  echo "restricted path exists: $path" >&2
  exit 1
done
shopt -u nullglob

echo 'Checking CLI dry-run commands...'
python3 -m logit_laundering.cli list-stages >/dev/null
python3 -m logit_laundering.cli run calibrate --dry-run >/dev/null

echo 'Repository validation passed.'
