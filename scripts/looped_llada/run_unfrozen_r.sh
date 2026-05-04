#!/usr/bin/env bash
# V2.1-a + R unfreeze (P2a): tests whether 16 frozen R blocks have enough
# capacity to be iterated, vs. learning R as an iterable operator.
#
# Same architecture as run.sh (V2.1-a feedback bridge, first-pass bypass)
# BUT: --freeze_recurrent_blocks False --freeze_ln_f False --freeze_wte False
# so the trainable surface is { recursive_link.* + R blocks + ln_f + wte },
# roughly the central 4B parameters of LLaDA-8B.
#
# Lower base learning rate (2e-5) since we're training pretrained backbone
# weights, not just an adapter. recursive_link still gets 10x via
# loop_lr_mult since it's the new module.
#
# Compares against P1 (run.sh, frozen R) on the same data + T_rec range:
#   - If P2a >> P1 -> R needs to be trained for iterability (paper story
#     becomes "loop training teaches middle blocks to be iterable")
#   - If P2a ~ P1 -> 33M adapter is sufficient, attribution stays clean
#
# Run from the repo root: `bash scripts/looped_llada/run_unfrozen_r.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop_belief/openmath2-v2.1a-unfrozen \
  --dataset_args ".data/sft/llada/openmath2-500k" \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --report_to tensorboard \
  --logging_dir .models/loop_belief/openmath2-v2.1a-unfrozen/tb \
  --save_steps 0.05 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no \
  --use_latent_feedback True \
  --t_rec_min 2 --t_rec_max 6 --t_rec_eval 4 \
  --mu_rec_eval 4 \
  --loop_lr_mult 10.0 \
  --freeze_recurrent_blocks False \
  --freeze_ln_f False \
  --freeze_wte False \
  --diag_log_every 50
