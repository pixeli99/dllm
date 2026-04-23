#!/usr/bin/env bash
# Vanilla LLaDA baseline for the same OpenMath-2 500k setup as run.sh.
# Matched hyperparameters: data, batching, lr, save cadence, logging, and
# no-eval setting. Only loop-specific model/trainer args are removed.
# Run from the repo root: `bash scripts/looped_llada/run_baseline.sh`

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada/sft.py \
  --output_dir .models/baseline/full-openmath2-500k \
  --dataset_args .data/sft/llada/openmath2-500k \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --report_to tensorboard \
  --logging_dir .models/baseline/full-openmath2-500k/tb \
  --save_steps 0.05 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no
