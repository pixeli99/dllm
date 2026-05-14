#!/bin/bash
# Phase 1: zero-init drop-in sanity check.
#
# Verifies that LLaDAWoDDModelLM with zero_init_wodd=True and T_step=1
# produces logits IDENTICAL to vanilla LLaDAModelLM within fp32 1e-4.
# If this fails, the training (Phase 2+) is unsafe — DO NOT proceed.

set -euo pipefail

python examples/llada_wodd/phase1_sanity.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --n_samples 64 \
    --seq_len 512 \
    --tolerance 1e-4 \
    --dtype float32 \
    --device cuda
