#!/usr/bin/env bash
# Evaluate a (looped or vanilla) LLaDA checkpoint on GSM8K-CoT.
#
# For looped checkpoints, the inference-time T_rec is taken from the
# saved config.json's `mu_rec_eval` field. Use --t_rec to patch the
# config in-place before evaluation (does NOT modify any other field).
#
# Run from the repo root:
#   bash scripts/looped_llada/eval_gsm8k.sh --checkpoint .models/loop_belief/openmath2-v2/checkpoint-final --t_rec 4

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

checkpoint=""
t_rec=""
num_gpu=8
batch_size=4
num_fewshot=0
max_new_tokens=256
steps=256
block_size=64
cfg_scale=0.0
limit=""
log_samples=True
chat_template=True

usage() {
  cat <<'EOF'
Usage:
  eval_gsm8k.sh --checkpoint PATH [options]

Options:
  --checkpoint PATH       Path to the model checkpoint to evaluate. (required)
  --t_rec N               If set, patch config.json's mu_rec_eval to N before eval.
  --num_gpu N             Number of accelerate processes. Default: 4
  --batch_size N          Eval generation batch size per process. Default: 1
  --num_fewshot N         GSM8K CoT few-shot count. Default: 0
  --max_new_tokens N      Generated token budget. Default: 256
  --steps N               Diffusion outer-T steps. Default: 256
  --block_size N          Diffusion block size. Default: 64
  --cfg_scale X           CFG scale. Default: 0.0
  --limit N               Optional lm-eval limit for smoke tests.
  --log_samples           Save per-sample prompts/generations. Default on.
  --no_log_samples        Disable per-sample logging.
  --no_chat_template      Skip --apply_chat_template (use for vanilla
                          LLaDA-8B-Base, which was NOT trained on chat
                          format -- applying a chat template tanks the
                          score to ~0). Default: chat template enabled
                          (correct for SFT'd checkpoints + Instruct).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)         checkpoint="$2"; shift 2 ;;
    --t_rec)              t_rec="$2"; shift 2 ;;
    --num_gpu)            num_gpu="$2"; shift 2 ;;
    --batch_size)         batch_size="$2"; shift 2 ;;
    --num_fewshot)        num_fewshot="$2"; shift 2 ;;
    --max_new_tokens)     max_new_tokens="$2"; shift 2 ;;
    --steps)              steps="$2"; shift 2 ;;
    --block_size)         block_size="$2"; shift 2 ;;
    --cfg_scale)          cfg_scale="$2"; shift 2 ;;
    --limit)              limit="$2"; shift 2 ;;
    --log_samples)        log_samples=True; shift ;;
    --no_log_samples)     log_samples=False; shift ;;
    --no_chat_template)   chat_template=False; shift ;;
    -h|--help)            usage; exit 0 ;;
    *)                    echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${checkpoint}" ]]; then
  echo "Error: --checkpoint is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "${checkpoint}" ]]; then
  echo "Error: checkpoint dir not found: ${checkpoint}" >&2
  exit 2
fi

# Optionally patch config.json's mu_rec_eval. Idempotent.
if [[ -n "${t_rec}" ]]; then
  python - "${checkpoint}" "${t_rec}" <<'PY'
import json, sys
ckpt, t = sys.argv[1], int(sys.argv[2])
cfg_path = f"{ckpt}/config.json"
with open(cfg_path) as f:
    cfg = json.load(f)
old = cfg.get("mu_rec_eval", "<absent>")
cfg["mu_rec_eval"] = t
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"[eval_gsm8k] Patched {cfg_path}: mu_rec_eval {old} -> {t}")
PY
fi

model_args="pretrained=${checkpoint},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},cfg_scale=${cfg_scale}"

extra_args=()
if [[ -n "${limit}" ]]; then extra_args+=( --limit "${limit}" ); fi
if [[ "${log_samples}" == "True" ]]; then
  output_dir="${checkpoint}/eval_gsm8k${t_rec:+_T${t_rec}}"
  mkdir -p "${output_dir}"
  extra_args+=( --log_samples --output_path "${output_dir}" )
fi
if [[ "${chat_template}" == "True" ]]; then
  extra_args+=( --apply_chat_template )
fi

accelerate launch --num_processes "${num_gpu}" \
  dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --model llada \
  --num_fewshot "${num_fewshot}" \
  --batch_size "${batch_size}" \
  --model_args "${model_args}" \
  "${extra_args[@]}"
