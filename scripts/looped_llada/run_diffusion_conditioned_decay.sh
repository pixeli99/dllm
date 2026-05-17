#!/usr/bin/env bash
# Closed-loop denoising with diffusion-conditioned decayed damping.
#
# This is the next main method after the fixed decay beta=0.5 result:
#   alpha_{b,r,i} = tau_b * softmax(-beta * r) * gate_i
#   tau_b = softplus(tau_logit + slope * (mask_ratio_b - ref))
#
# Masked tokens receive the full update gate, while unmasked/prompt tokens
# receive a small gate to reduce context drift. T_rec=1 remains the vanilla
# split LLaDA path because all conditioning is applied only on feedback passes.
#
# Run from the repo root:
#   bash scripts/looped_llada/run_diffusion_conditioned_decay.sh
#
# Useful knobs:
#   BETA=0.5 TAU=1.0 UNMASKED_GATE=0.1 T_REC_MAX=6 \
#     bash scripts/looped_llada/run_diffusion_conditioned_decay.sh

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

tau="${TAU:-1.0}"
beta="${BETA:-0.5}"
masked_gate="${MASKED_GATE:-1.0}"
unmasked_gate="${UNMASKED_GATE:-0.1}"
mask_tau_ref="${MASK_TAU_REF:-0.5}"
mask_tau_slope_init="${MASK_TAU_SLOPE_INIT:-0.0}"
learn_mask_tau="${LEARN_MASK_TAU:-True}"
t_rec_min="${T_REC_MIN:-2}"
t_rec_max="${T_REC_MAX:-6}"
t_rec_eval="${T_REC_EVAL:-6}"
max_steps="${MAX_STEPS:-1000}"

run_name="openmath2-v2.1a-stage1-ins-diffcond-decay-tau${tau}-b${beta}-ug${unmasked_gate}-t${t_rec_min}-${t_rec_max}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir ".models/loop_belief/${run_name}" \
  --dataset_args ".data/sft/llada/openmath2-500k" \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --max_steps "${max_steps}" \
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
  --learn_damping_tau True \
  --damping_decay_beta "${beta}" \
  --use_diffusion_conditioned_damping True \
  --damping_masked_gate "${masked_gate}" \
  --damping_unmasked_gate "${unmasked_gate}" \
  --learn_mask_tau "${learn_mask_tau}" \
  --mask_tau_ref "${mask_tau_ref}" \
  --mask_tau_slope_init "${mask_tau_slope_init}" \
  --t_rec_min "${t_rec_min}" --t_rec_max "${t_rec_max}" --t_rec_eval "${t_rec_eval}" \
  --mu_rec_eval "${t_rec_eval}" \
  --loop_lr_mult 1.0 \
  --diag_log_every 50
