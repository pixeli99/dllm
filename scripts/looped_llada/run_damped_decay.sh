#!/usr/bin/env bash
# V2.1-a + decayed normalized-budget damped recurrent update ablation.
#
# Same frozen-stage Instruct defaults as scripts/looped_llada/run_stage1_frozen.sh,
# but feedback steps share a fixed total refinement budget with exponentially
# decayed step sizes:
#   alpha_r = tau * softmax(-beta * r)
# Run with BETA=0.5 for mild decay or BETA=1.0 for stronger decay.
#
# Run from the repo root: `bash scripts/looped_llada/run_damped_decay.sh`
# Sweep knobs with:
#   TAU=1.25 BETA=1.0 bash scripts/looped_llada/run_damped_decay.sh

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

tau="${TAU:-1.0}"
beta="${BETA:-0.5}"
run_name="openmath2-v2.1a-stage1-ins-damped-decay-tau${tau}-b${beta}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir ".models/loop_belief/${run_name}" \
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
  --logging_dir ".models/loop_belief/${run_name}/tb" \
  --save_steps 0.25 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no \
  --use_latent_feedback True \
  --use_damped_update True \
  --damping_schedule decay \
  --damping_tau "${tau}" \
  --damping_decay_beta "${beta}" \
  --t_rec_min 2 --t_rec_max 4 --t_rec_eval 3 \
  --mu_rec_eval 3 \
  --loop_lr_mult 1.0 \
  --diag_log_every 50
