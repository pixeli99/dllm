#!/usr/bin/env bash
# scripts/phase2/eval_arm.sh ARM CHECKPOINT_PATH [num_gpu] [seed]
#
# Runs the four WODD_PLAN.md §3.2 evals for ONE Phase 2 cell:
#   - gsm8k_cot         (math, in-distribution)
#   - minerva_math      (proxy for MATH-500; in-distribution)
#   - humaneval         (code, out-of-distribution)
#   - mbpp              (code, out-of-distribution)
#
# ARM is one of {vanilla, latts, metastate, dpad, wodd}. For the
# non-vanilla arms we dispatch to the pipeline-specific eval entrypoint
# (dllm/pipelines/llada_${ARM}/eval.py) which registers a custom
# lm_eval --model name and wires the right HF AutoModel class via
# AutoConfig.register on import.
#
# Per WODD_PLAN.md §3.2: pass@1 with greedy decoding (cfg_scale=0.0).
# Self-consistency / N=16 sampling is a separate run; not in this script.

set -euo pipefail

ARM="${1:?arm required (vanilla|latts|metastate|dpad|wodd)}"
CKPT="${2:?checkpoint path required}"
NUM_GPU="${3:-8}"
SEED="${4:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Resolve eval entrypoint per arm.
case "$ARM" in
  vanilla)
    EVAL_ENTRY="dllm/pipelines/llada/eval.py"
    MODEL_NAME="llada"
    ;;
  latts)
    EVAL_ENTRY="dllm/pipelines/llada_latts/eval.py"
    MODEL_NAME="llada_latts"
    ;;
  metastate)
    EVAL_ENTRY="dllm/pipelines/llada_metastate/eval.py"
    MODEL_NAME="llada_metastate"
    ;;
  dpad)
    EVAL_ENTRY="dllm/pipelines/llada_dpad/eval.py"
    MODEL_NAME="llada_dpad"
    ;;
  wodd)
    EVAL_ENTRY="dllm/pipelines/llada_wodd/eval.py"
    MODEL_NAME="llada_wodd"
    ;;
  *)
    echo "Unknown ARM=$ARM"; exit 1
    ;;
esac

# Output dir for eval results.
OUTPUT_DIR=".eval/phase2/${ARM}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

COMMON_ARGS="--model ${MODEL_NAME}"
BATCH_SIZE="${BATCH_SIZE:-4}"
COMMON_MODEL_ARGS_GREEDY="pretrained=${CKPT},max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"
COMMON_MODEL_ARGS_CODE="pretrained=${CKPT},max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[126081],begin_suppress_tokens=[]"

# 1. GSM8K (CoT)
accelerate launch --num_processes "${NUM_GPU}" "${EVAL_ENTRY}" \
  --tasks gsm8k_cot --num_fewshot 0 ${COMMON_ARGS} \
  --batch_size "${BATCH_SIZE}" \
  --model_args "${COMMON_MODEL_ARGS_GREEDY}" \
  --output_path "${OUTPUT_DIR}/gsm8k_cot.json" \
  --seed "${SEED}"

# 2. MATH-500 (proxy via minerva_math)
accelerate launch --num_processes "${NUM_GPU}" "${EVAL_ENTRY}" \
  --tasks minerva_math --num_fewshot 4 ${COMMON_ARGS} \
  --batch_size "${BATCH_SIZE}" \
  --model_args "${COMMON_MODEL_ARGS_GREEDY}" \
  --output_path "${OUTPUT_DIR}/minerva_math.json" \
  --seed "${SEED}"

# 3. HumanEval (instruct, needs unsafe-code)
accelerate launch --num_processes "${NUM_GPU}" "${EVAL_ENTRY}" \
  --tasks humaneval_instruct_llada --num_fewshot 0 ${COMMON_ARGS} \
  --batch_size "${BATCH_SIZE}" \
  --model_args "${COMMON_MODEL_ARGS_CODE}" \
  --output_path "${OUTPUT_DIR}/humaneval.json" \
  --confirm_run_unsafe_code \
  --seed "${SEED}"

# 4. MBPP
accelerate launch --num_processes "${NUM_GPU}" "${EVAL_ENTRY}" \
  --tasks mbpp_instruct_llada --num_fewshot 3 ${COMMON_ARGS} \
  --batch_size "${BATCH_SIZE}" \
  --model_args "pretrained=${CKPT},max_new_tokens=256,steps=256,block_size=256,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
  --output_path "${OUTPUT_DIR}/mbpp.json" \
  --confirm_run_unsafe_code \
  --seed "${SEED}"

echo "[eval] $ARM seed=$SEED results under $OUTPUT_DIR"
