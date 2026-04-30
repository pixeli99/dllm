#!/usr/bin/env bash
# V2 (latent feedback): main experiment.
# Loop applies a 2-layer residual MLP (RecursiveLink) to h_r every iter,
# alpha-gated at mask positions.
# Run from the repo root: `bash scripts/looped_llada/run.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop_belief/openmath2-v2 \
  --dataset_args "nvidia/OpenMathInstruct-2[train:500000,test:5000]" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --report_to tensorboard \
  --logging_dir .models/loop_belief/openmath2-v2/tb \
  --save_steps 0.05 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no \
  --use_latent_feedback True \
  --alpha_init 0.0 \
  --t_rec_min 1 --t_rec_max 6 --t_rec_eval 4 \
  --mu_rec_eval 4 \
  --loop_lr_mult 10.0 \
  --diag_log_every 50
