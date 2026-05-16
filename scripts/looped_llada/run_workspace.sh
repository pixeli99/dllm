#!/usr/bin/env bash
# V2.1-a + feedback-only recurrent workspace slots.
#
# Same frozen-stage Instruct defaults as scripts/looped_llada/run_stage1_frozen.sh,
# with normalized damping and coda-aware deep supervision enabled by default.
# Workspace slots are appended only inside feedback recurrent passes (r >= 1),
# so T_rec=1 keeps the vanilla split path.
#
# Run from the repo root: `bash scripts/looped_llada/run_workspace.sh`
# Sweep slots/budget with:
#   SLOTS=8 TAU=1.25 bash scripts/looped_llada/run_workspace.sh

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

slots="${SLOTS:-16}"
tau="${TAU:-1.0}"
deep_sup="${ENABLE_DEEP_SUP:-True}"
lambda_deep="${LAMBDA_DEEP:-0.3}"
deep_sup_k="${DEEP_SUP_K:-2}"
workspace_init_std="${WORKSPACE_INIT_STD:-0.02}"

# LLaDA configs commonly cache RoPE up to 1024 positions. Feedback workspace
# uses token positions plus workspace positions, so leave room by default.
max_length="${MAX_LENGTH:-$((1024 - slots))}"
if (( max_length < 1 )); then
  echo "MAX_LENGTH must be positive after reserving SLOTS=${slots}." >&2
  exit 1
fi

run_name="openmath2-v2.1a-stage1-ins-workspace-s${slots}-norm-tau${tau}"

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
  --max_length "${max_length}" \
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
  --damping_schedule normalized \
  --damping_tau "${tau}" \
  --learn_damping_tau True \
  --workspace_size "${slots}" \
  --workspace_init_std "${workspace_init_std}" \
  --enable_deep_sup "${deep_sup}" \
  --lambda_deep "${lambda_deep}" \
  --deep_sup_k_subset "${deep_sup_k}" \
  --t_rec_min 2 --t_rec_max 4 --t_rec_eval 3 \
  --mu_rec_eval 3 \
  --loop_lr_mult 1.0 \
  --diag_log_every 1
