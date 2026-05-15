#!/usr/bin/env bash
# V2.1-a + damped recurrent update ablation.
#
# Same feedback-bridge architecture as scripts/looped_llada/run.sh, but each
# feedback iteration commits the proposed recurrent state with:
#   h_next = h_prev + alpha * (R(RecursiveLink(h_prev)) - h_prev)
# The first pass is unchanged, so T_rec=1 still matches vanilla split LLaDA.
#
# Run from the repo root: `bash scripts/looped_llada/run_damped.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop_belief/openmath2-v2.1a-damped-a0.5 \
  --dataset_args ".data/sft/llada/openmath2-500k" \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 5e-4 \
  --report_to tensorboard \
  --logging_dir .models/loop_belief/openmath2-v2.1a-damped-a0.5/tb \
  --save_steps 0.05 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no \
  --use_latent_feedback True \
  --use_damped_update True \
  --damping_alpha 0.5 \
  --t_rec_min 2 --t_rec_max 6 --t_rec_eval 4 \
  --mu_rec_eval 4 \
  --loop_lr_mult 1.0 \
  --diag_log_every 50
