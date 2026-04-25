#!/usr/bin/env bash
# Evaluate the looped model and/or vanilla baseline on MATH-500.
# Run from anywhere:
#   bash /lustre/projects/polyullm/lipengxiang_tmp/dllm/scripts/looped_llada/eval_math500.sh --target both

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

target="both"
checkpoint="latest"
loop_root=".models/loop/full-openmath2-500k"
baseline_root=".models/baseline/full-openmath2-500k"
num_gpu=8
batch_size=4
num_fewshot=0
# MATH problems are typically longer than GSM8K — bump generation budget.
max_new_tokens=512
steps=512
block_size=64
cfg_scale=0.0
limit=""
log_samples=True
mu_rec_eval=""

usage() {
  cat <<'EOF'
Usage:
  eval_math500.sh [options]

Options:
  --target loop|baseline|both     Which model(s) to evaluate. Default: both
  --checkpoint latest|final|auto|PATH
                                  auto prefers checkpoint-final, then latest checkpoint-*.
                                  Default: latest
  --loop_root PATH                Looped model output root.
  --baseline_root PATH            Baseline model output root.
  --num_gpu N                     Number of accelerate processes. Default: 8
  --batch_size N                  Eval generation batch size per process. Default: 1
  --num_fewshot N                 MATH-500 few-shot count. Default: 0
  --max_new_tokens N              Generated token budget. Default: 512
  --steps N                       Diffusion sampling steps. Default: 512
  --block_size N                  Diffusion block size. Default: 64
  --cfg_scale X                   CFG scale. Default: 0.0
  --limit N                       Optional lm-eval limit for smoke tests.
  --mu_rec_eval N                 Override looped checkpoint test-time recurrence.
  --log_samples                   Save per-sample prompts, generations, and metrics. Default
  --no_log_samples                Disable per-sample logging.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="$2"; shift 2 ;;
    --checkpoint)
      checkpoint="$2"; shift 2 ;;
    --loop_root)
      loop_root="$2"; shift 2 ;;
    --baseline_root)
      baseline_root="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --batch_size)
      batch_size="$2"; shift 2 ;;
    --num_fewshot)
      num_fewshot="$2"; shift 2 ;;
    --max_new_tokens)
      max_new_tokens="$2"; shift 2 ;;
    --steps)
      steps="$2"; shift 2 ;;
    --block_size)
      block_size="$2"; shift 2 ;;
    --cfg_scale)
      cfg_scale="$2"; shift 2 ;;
    --limit)
      limit="$2"; shift 2 ;;
    --mu_rec_eval)
      mu_rec_eval="$2"; shift 2 ;;
    --log_samples)
      log_samples=True; shift ;;
    --no_log_samples)
      log_samples=False; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

latest_checkpoint() {
  local root="$1"
  find "${root}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | awk -F'checkpoint-' '/checkpoint-[0-9]+$/ {print $2 "\t" $0}' \
    | sort -n \
    | tail -n 1 \
    | cut -f2-
}

resolve_checkpoint() {
  local root="$1"
  case "${checkpoint}" in
    auto)
      if [[ -d "${root}/checkpoint-final" ]]; then
        printf '%s\n' "${root}/checkpoint-final"
      else
        latest_checkpoint "${root}"
      fi
      ;;
    latest)
      latest_checkpoint "${root}"
      ;;
    final)
      printf '%s\n' "${root}/checkpoint-final"
      ;;
    *)
      printf '%s\n' "${checkpoint}"
      ;;
  esac
}

run_one() {
  local name="$1"
  local root="$2"
  local model_path
  model_path="$(resolve_checkpoint "${root}")"

  if [[ -z "${model_path}" || ! -d "${model_path}" ]]; then
    echo "No checkpoint found for ${name}. Checked root: ${root}" >&2
    exit 1
  fi

  local checkpoint_name
  checkpoint_name="$(basename "${model_path}")"
  local output_path=".eval/math500/${name}/${checkpoint_name}"
  if [[ -n "${mu_rec_eval}" && "${name}" = "loop" ]]; then
    output_path="${output_path}/mu${mu_rec_eval}"
  fi
  mkdir -p "${output_path}"

  echo ">>> Evaluating ${name}"
  echo ">>> checkpoint: ${model_path}"
  echo ">>> output_path: ${output_path}"

  local model_args
  model_args="pretrained=${model_path},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},cfg_scale=${cfg_scale},suppress_tokens=[],begin_suppress_tokens=[126081;126348]"
  if [[ -n "${mu_rec_eval}" && "${name}" = "loop" ]]; then
    model_args="${model_args},mu_rec_eval=${mu_rec_eval}"
  fi

  local extra_args=()
  if [[ -n "${limit}" ]]; then
    extra_args+=(--limit "${limit}")
  fi
  if [[ "${log_samples}" = "True" ]]; then
    extra_args+=(--log_samples)
  fi

  accelerate launch --num_processes "${num_gpu}" "${repo_root}/dllm/pipelines/llada/eval.py" \
    --tasks hendrycks_math500 \
    --num_fewshot "${num_fewshot}" \
    --model llada \
    --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "${model_args}" \
    --output_path "${output_path}" \
    "${extra_args[@]}"
}

case "${target}" in
  loop)
    run_one "loop" "${loop_root}" ;;
  baseline)
    run_one "baseline" "${baseline_root}" ;;
  both)
    run_one "loop" "${loop_root}"
    run_one "baseline" "${baseline_root}" ;;
  *)
    echo "Invalid --target: ${target}" >&2
    usage >&2
    exit 2 ;;
esac
