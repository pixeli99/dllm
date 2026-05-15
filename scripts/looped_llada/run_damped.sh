#!/usr/bin/env bash
# V2.1-a + damped recurrent update ablation.
#
# Same frozen-stage Instruct defaults as scripts/looped_llada/run_stage1_frozen.sh,
# but each feedback iteration commits the proposed recurrent state with:
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
  --output_dir .models/loop_belief/openmath2-v2.1a-stage1-ins-damped-learnalpha-a0.5 \
  --dataset_args ".data/sft/llada/openmath2-500k" \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --max_steps 1000 \
  --model_name_or_path /lustre/projects/polyullm/lipengxiang_tmp/LLaDA-8B-Instruct \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --group_by_length False \
  --max_length 1024 \
  --learning_rate 5e-4 \
  --report_to tensorboard \
  --logging_dir .models/loop_belief/openmath2-v2.1a-stage1-ins-damped-learnalpha-a0.5/tb \
  --save_steps 0.25 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no \
  --use_latent_feedback True \
  --use_damped_update True \
  --damping_alpha 0.5 \
  --learn_damping_alpha True \
  --t_rec_min 2 --t_rec_max 4 --t_rec_eval 3 \
  --mu_rec_eval 3 \
  --loop_lr_mult 1.0 \
  --diag_log_every 50
