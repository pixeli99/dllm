#!/usr/bin/env bash
# scripts/phase2/launch_arm.sh ARM SEED [extra HF flags...]
#
# Launch one Phase 2 training run for one arm × one seed on this node.
# Defaults to 8×H800 FSDP via accelerate. The 5-arm × 5-seed grid is
# 25 launches; cluster them via launch_all.sh or your SLURM array
# scheduler (preferred).

set -euo pipefail

ARM="${1:-wodd}"
SEED="${2:-0}"
shift 2 || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# WODD_PLAN.md §3.2 used the WoDD trainer defaults; mitigation from
# Phase 1b for gate-collapse is encoded here for the wodd arm only.
EXTRA_WODD_FLAGS=""
if [[ "$ARM" == "wodd" ]]; then
  EXTRA_WODD_FLAGS=" --gate_entropy_lambda 1e-2 --write_sparsity_lambda 1e-2 "
fi

# Where to write the checkpoint / logs for this (arm, seed) cell.
OUTPUT_DIR=".models/phase2/${ARM}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

accelerate launch \
  --config_file scripts/accelerate_configs/fsdp.yaml \
  examples/phase2_sft.py \
  --arm "$ARM" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path GSAI-ML/LLaDA-8B-Base \
  --dataset_args ".data/sft/llada/openmath2-500k[train:500000,test:5000]" \
  --load_preprocessed_data True \
  --remove_unused_columns True \
  --bf16 True \
  --gradient_checkpointing True \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.0 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps --save_steps 10000 --save_total_limit 2 \
  --eval_strategy no \
  --logging_steps 50 \
  --diag_log_every 50 \
  --report_to none \
  $EXTRA_WODD_FLAGS \
  "$@"
