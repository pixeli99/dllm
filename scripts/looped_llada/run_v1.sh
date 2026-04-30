#!/usr/bin/env bash
# V1 (pure h-recurrence ablation): no latent feedback.
# Used to isolate the contribution of RecursiveLink over plain looping.
# V1 has no recursive_link, so the V2 default freeze profile (which trains
# only recursive_link) is overridden here -- R blocks + ln_f + wte must
# stay trainable.
# Run from the repo root: `bash scripts/looped_llada/run_v1.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop_belief/openmath2-v1 \
  --dataset_args ".data/sft/llada/openmath2-500k" \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --report_to tensorboard \
  --logging_dir .models/loop_belief/openmath2-v1/tb \
  --save_steps 0.05 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no \
  --use_latent_feedback False \
  --t_rec_min 1 --t_rec_max 6 --t_rec_eval 4 \
  --mu_rec_eval 4 \
  --loop_lr_mult 1.0 \
  --freeze_recurrent_blocks False \
  --freeze_ln_f False \
  --freeze_wte False \
  --diag_log_every 50
