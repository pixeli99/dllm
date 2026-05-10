#!/usr/bin/env bash
# TD distillation P0 sanity (REENTRY_TD_SPEC.md Phase 2).
#
# Verifies that progressive distillation works on LLaDA at all:
# vanilla student initialised from the teacher SFT checkpoint, fine-tuned
# against the cached teacher trajectory at stride K (default
# teacher_steps / student_steps). No inner loop, no re-entry.
#
# Required env vars (or override via flags):
#   TEACHER_CKPT  -- the SFT teacher checkpoint used to build the cache
#   CACHE_DIR     -- output of `python -m dllm.tools.build_teacher_cache`
#
# Run from the repo root: `bash scripts/looped_llada/td_p0.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

TEACHER_CKPT="${TEACHER_CKPT:?set TEACHER_CKPT to the SFT teacher checkpoint path}"
CACHE_DIR="${CACHE_DIR:?set CACHE_DIR to the teacher trajectory cache path}"
OUTPUT_DIR="${OUTPUT_DIR:-.models/loop_belief/td-p0-128}"
STUDENT_STEPS="${STUDENT_STEPS:-128}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft_td.py \
  --model_name_or_path "${TEACHER_CKPT}" \
  --cache_dir "${CACHE_DIR}" \
  --student_steps "${STUDENT_STEPS}" \
  --teacher_stride 0 \
  --val_fraction 0.05 \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 30000 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --warmup_steps 500 \
  --lr_scheduler_type constant_with_warmup \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --report_to tensorboard \
  --logging_dir "${OUTPUT_DIR}/tb" \
  --logging_steps 50 \
  --save_steps 0.1 \
  --save_total_limit 3 \
  --eval_strategy steps \
  --eval_steps 0.1 \
  --bf16 True \
  --dataloader_num_workers 4
