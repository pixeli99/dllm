#!/usr/bin/env bash
# scripts/phase6/eval_falsification.sh
#
# Phase 6 — three pre-registered settings where WoDD is predicted to
# *not* help (WODD_PLAN.md §3.6). We eval the converged Phase 2 / 3
# checkpoints zero-shot on each.
#
# (a) short-output GLUE classification    -- predicted: WoDD ≈ vanilla
# (b) TriviaQA closed-book                -- predicted: WoDD ≈ vanilla
# (c) 2-digit addition (synthetic)        -- predicted: WoDD ≈ MetaState
#                                            below bit-budget threshold,
#                                            separates above
#
# This script runs (a) and (b) via lm-eval and (c) via the local
# `examples/phase6/eval_two_digit_add.py`. Per the plan, we only test
# the THREE arms that the prediction names: vanilla / metastate / wodd.
# LATTS and DPad are not part of the §3.6 falsification battery.

set -euo pipefail

NUM_GPU="${NUM_GPU:-8}"
# Paths to converged checkpoints. Override via env.
VANILLA_CKPT="${VANILLA_CKPT:-GSAI-ML/LLaDA-8B-Base}"
METASTATE_CKPT="${METASTATE_CKPT:-.models/phase2/metastate/seed0/checkpoint-final}"
WODD_CKPT="${WODD_CKPT:-.models/phase2/wodd/seed0/checkpoint-final}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

run_lm_eval() {
  local ARM=$1
  local CKPT=$2
  local TASK=$3
  local NSHOT=$4
  local MAXN=$5
  local TAG=$6      # filename tag
  local OUT=".eval/phase6/${TAG}/${ARM}"
  mkdir -p "$OUT"

  case "$ARM" in
    vanilla)   ENTRY="dllm/pipelines/llada/eval.py";        MODEL="llada" ;;
    metastate) ENTRY="dllm/pipelines/llada_metastate/eval.py"; MODEL="llada_metastate" ;;
    wodd)      ENTRY="dllm/pipelines/llada_wodd/eval.py";   MODEL="llada_wodd" ;;
    *) echo "unknown arm $ARM"; return 1 ;;
  esac

  accelerate launch --num_processes "${NUM_GPU}" "${ENTRY}" \
    --tasks "${TASK}" --num_fewshot "${NSHOT}" \
    --model "${MODEL}" --apply_chat_template \
    --model_args "pretrained=${CKPT},max_new_tokens=${MAXN},steps=${MAXN},block_size=${MAXN},cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]" \
    --output_path "${OUT}/${TASK}.json" \
    --seed 0
}

# (a) GLUE: cola is the canonical "single-sentence classification".
for ARM_CKPT in "vanilla:${VANILLA_CKPT}" "metastate:${METASTATE_CKPT}" "wodd:${WODD_CKPT}"; do
  ARM="${ARM_CKPT%%:*}"
  CKPT="${ARM_CKPT#*:}"
  run_lm_eval "$ARM" "$CKPT" cola 0 8 "glue"
done

# (b) TriviaQA closed-book.
for ARM_CKPT in "vanilla:${VANILLA_CKPT}" "metastate:${METASTATE_CKPT}" "wodd:${WODD_CKPT}"; do
  ARM="${ARM_CKPT%%:*}"
  CKPT="${ARM_CKPT#*:}"
  run_lm_eval "$ARM" "$CKPT" triviaqa 5 64 "triviaqa"
done

# (c) 2-digit addition (synthetic, local). 8 GPUs are wasted on this
# tiny task; run on 1 GPU.
for ARM_CKPT in "vanilla:${VANILLA_CKPT}" "metastate:${METASTATE_CKPT}" "wodd:${WODD_CKPT}"; do
  ARM="${ARM_CKPT%%:*}"
  CKPT="${ARM_CKPT#*:}"
  CUDA_VISIBLE_DEVICES=0 python examples/phase6/eval_two_digit_add.py \
    --arm "${ARM}" --checkpoint "${CKPT}" \
    --n_problems 400 --seed 0
done

echo "[phase6] all falsification evals done; jsons under .eval/phase6/"
