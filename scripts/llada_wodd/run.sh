#!/bin/bash
# LLaDA-WoDD SFT run script.
#
# WoDD = Write-on-Demand Diffusion. Trains write_head + tape_read on top of
# a frozen LLaDA-8B backbone, with the loss
#     L = L_mdlm + λ_s * mean(g_k) - λ_e * H(Bern(g_k))
#
# The W1-W4 priority is to verify:
#   (a) mean_gate trends nontrivially > 0 (model actually writes)
#   (b) gate_entropy stays positive (no collapse)
#   (c) MDLM eval loss matches frozen LLaDA at T_step=1 (zero-init check
#       AT INIT) and improves with T_step at convergence (capacity claim)
#
# Default: 1-node FSDP, frozen backbone, T_step ~ Uniform{1..4}.

set -euo pipefail

accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/llada_wodd/sft.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --prelude_layers 8 \
    --recurrent_layers 16 \
    --coda_layers 8 \
    --tape_dim 1024 \
    --tape_max_writes 16 \
    --tape_n_heads 8 \
    --t_step_min 1 \
    --t_step_max 4 \
    --t_step_eval 4 \
    --write_sparsity_lambda 1e-2 \
    --gate_entropy_lambda 1e-3 \
    --wodd_lr_mult 10.0 \
    --freeze_prelude True \
    --freeze_recurrent_blocks True \
    --freeze_coda True \
    --freeze_ln_f True \
    --freeze_wte True \
    --output_dir .models/LLaDA-8B-WoDD/openmath2-v0 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_checkpointing True \
    --diag_log_every 50
