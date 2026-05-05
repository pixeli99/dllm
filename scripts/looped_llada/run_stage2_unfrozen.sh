#!/usr/bin/env bash
# Stage 2 of the two-stage recipe: warm-start from Stage 1's frozen-R
# checkpoint, then unfreeze R + ln_f + wte for the main fine-tune.
#
# Stage 1 should have run scripts/looped_llada/run_stage1_frozen.sh
# first, leaving a checkpoint at:
#   .models/loop_belief/openmath2-v2.1a-stage1/checkpoint-final
#
# Stage 2 starts a fresh LR scheduler / optimizer / step counter --
# only the model weights are carried over. This means Stage 2 has its
# own warmup, which is what we want when the trainable surface changes
# (link is now joined by 4B backbone params with very different update
# magnitudes).
#
# Run from the repo root: `bash scripts/looped_llada/run_stage2_unfrozen.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

stage1_ckpt=".models/loop_belief/openmath2-v2.1a-stage1/checkpoint-final"

if [[ ! -d "${stage1_ckpt}" ]]; then
  echo "Error: Stage 1 checkpoint not found at ${stage1_ckpt}" >&2
  echo "Run scripts/looped_llada/run_stage1_frozen.sh first." >&2
  exit 2
fi

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop_belief/openmath2-v2.1a-two-stage \
  --looped_init_from "${stage1_ckpt}" \
  --dataset_args ".data/sft/llada/openmath2-500k" \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --report_to tensorboard \
  --logging_dir .models/loop_belief/openmath2-v2.1a-two-stage/tb \
  --save_steps 0.05 \
  --group_by_length False \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no \
  --use_latent_feedback True \
  --t_rec_min 1 --t_rec_max 4 --t_rec_eval 3 \
  --mu_rec_eval 3 \
  --loop_lr_mult 10.0 \
  --freeze_recurrent_blocks False \
  --freeze_ln_f False \
  --freeze_wte False \
  --diag_log_every 50
