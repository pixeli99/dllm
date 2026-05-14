#!/usr/bin/env bash
# scripts/phase3/eval_sweep.sh [num_gpu]
#
# Eval every Phase 3 cell on GSM8K + MATH-500 (WODD_PLAN §3.3 says only
# these two for the headline curve). HumanEval/MBPP would tell us about
# code generalisation but are not part of the Phase 3 scaling result.

set -euo pipefail
NUM_GPU="${1:-8}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

run_eval() {
  local NAME=$1     # "metastate_M64" / "wodd_d1024_T8"
  local ARM=$2
  local CKPT=".models/phase3/${NAME}/checkpoint-final"
  if [[ ! -d "$CKPT" ]]; then
    echo "[skip] $NAME: $CKPT not found"
    return
  fi
  local OUT=".eval/phase3/${NAME}"
  mkdir -p "$OUT"

  export PYTHONPATH=".:${PYTHONPATH:-}"
  export HF_ALLOW_CODE_EVAL=1
  export HF_DATASETS_TRUST_REMOTE_CODE=True
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

  local EVAL_ENTRY="dllm/pipelines/llada_${ARM}/eval.py"
  local MODEL_NAME="llada_${ARM}"
  local MA="pretrained=${CKPT},max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

  # GSM8K
  accelerate launch --num_processes "${NUM_GPU}" "${EVAL_ENTRY}" \
    --tasks gsm8k_cot --num_fewshot 5 \
    --model "${MODEL_NAME}" --apply_chat_template \
    --model_args "${MA}" \
    --output_path "${OUT}/gsm8k_cot.json" --seed 0

  # MATH-500 (using minerva_math as proxy)
  accelerate launch --num_processes "${NUM_GPU}" "${EVAL_ENTRY}" \
    --tasks minerva_math --num_fewshot 4 \
    --model "${MODEL_NAME}" --apply_chat_template \
    --model_args "${MA}" \
    --output_path "${OUT}/minerva_math.json" --seed 0
}

for M in 64 128 256 512 1024; do
  run_eval "metastate_M${M}" metastate
done
for DT in "256 4" "512 4" "1024 4" "1024 8" "1024 16"; do
  D="${DT% *}"; T="${DT#* }"
  run_eval "wodd_d${D}_T${T}" wodd
done

echo "[phase3 eval] done; jsons under .eval/phase3/"
