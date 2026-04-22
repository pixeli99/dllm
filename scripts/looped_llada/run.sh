#!/usr/bin/env bash
# Main experiment: LLaDA-Looped full method (Poisson mu_rec + t-adaptive
# schedule + cycle-consistency) on nvidia/OpenMathInstruct-2, 500k slice.
# Run from the repo root: `bash scripts/looped_llada/run.sh`

accelerate launch \
  --config_file scripts/accelerate_configs/fsdp.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop/full-openmath2-500k \
  --dataset_args .data/sft/llada/openmath2-500k \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --recurrence_dist poisson \
  --mu_rec_train_mean 2 --mu_rec_train_max 4 \
  --t_adaptive True \
  --t_bump_center 0.5 --t_bump_width 0.25 --t_bump_scale 1.0 \
  --cycle_consistency_weight 0.1
