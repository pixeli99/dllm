#!/usr/bin/env bash
# Sanity check: LLaDA-Looped with μ_rec = 1 should be numerically identical
# to vanilla LLaDA (h_0=0, B=I, A≈0, e=x_prelude -> h_1 = R(x_prelude)).
# If the vanilla baseline gave loss ≈ 3 and this run diverges, the bug is
# in the looped forward or MDLMLoopedTrainer, NOT in data/trainer/FSDP.
# Run from repo root: `bash scripts/looped_llada/run.sh`

accelerate launch \
  --config_file scripts/accelerate_configs/fsdp.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop/mu1-sanity \
  --dataset_args .data/sft/llada/openmath2-500k \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --recurrence_dist fixed \
  --mu_rec_train_mean 1 --mu_rec_train_min 1 --mu_rec_train_max 1 \
  --mu_rec_eval 1
