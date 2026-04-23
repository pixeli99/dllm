#!/usr/bin/env bash
# Main experiment: LLaDA-Looped (Poisson mu_rec + t-adaptive schedule)
# on nvidia/OpenMathInstruct-2, 500k slice.
# Run from the repo root: `bash scripts/looped_llada/run.sh`

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop/full-openmath2-500k \
  --dataset_args .data/sft/llada/openmath2-500k \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --report_to tensorboard \
  --logging_dir .models/loop/full-openmath2-500k/tb \
  --save_steps 0.05 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --recurrence_dist poisson \
  --eval_strategy no \
  --mu_rec_train_mean 2 --mu_rec_train_min 2 --mu_rec_train_max 4 \
  --mu_bwd_ratio 1.0 \
  --loop_lr_mult 10.0 \
  --t_adaptive True \
  --t_bump_center 0.5 --t_bump_width 0.25 --t_bump_scale 1.0
