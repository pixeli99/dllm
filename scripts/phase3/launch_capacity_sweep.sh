#!/usr/bin/env bash
# scripts/phase3/launch_capacity_sweep.sh
#
# Phase 3 (headline) capacity sweep per WODD_PLAN.md §3.3.
# Five MetaState points × five WoDD points, holding partition + training
# hparams fixed, varying ONLY the capacity knob:
#
#   MetaState: state_slots ∈ {64, 128, 256, 512, 1024}, state_dim=1024
#   WoDD:      (tape_dim, T_step) ∈
#                {(256,4),(512,4),(1024,4),(1024,8),(1024,16)}
#
# Each cell is one full Phase-2-cell-sized training run (no seed multi-
# plicity in Phase 3; the curve shape is the science result). Total
# 10 runs × ~6-10 GPU-h ≈ 60-100 GPU-h.
#
# Output: .models/phase3/{metastate_M, wodd_DxT}/checkpoint-final.
# Eval (Phase 3 reads GSM8K + MATH-500 only) is done by
# scripts/phase3/eval_sweep.sh after training completes.
#
# Usage:
#   ./scripts/phase3/launch_capacity_sweep.sh          # full 10 runs
#   ARM=metastate ./scripts/phase3/launch_capacity_sweep.sh  # MS curve only
#   ARM=wodd      ./scripts/phase3/launch_capacity_sweep.sh  # WoDD curve only

set -euo pipefail

ARM_FILTER="${ARM:-both}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

run_cell() {
  local NAME=$1     # eg "metastate_M64" or "wodd_d1024_T8"
  local ARM=$2      # "metastate" or "wodd"
  local KW=$3       # JSON arm_kwargs override
  local OUTPUT_DIR=".models/phase3/${NAME}"
  mkdir -p "$OUTPUT_DIR"
  echo "=== Phase 3 cell: $NAME (arm=$ARM, kw=$KW) ==="

  EXTRA_WODD=""
  if [[ "$ARM" == "wodd" ]]; then
    EXTRA_WODD=" --gate_entropy_lambda 1e-2 --write_sparsity_lambda 1e-2 "
  fi

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/phase2_sft.py \
    --arm "$ARM" --seed 0 \
    --arm_kwargs_json "$KW" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path GSAI-ML/LLaDA-8B-Base \
    --dataset_args ".data/sft/llada/openmath2-500k[train:500000,test:5000]" \
    --load_preprocessed_data True --remove_unused_columns True \
    --bf16 True --gradient_checkpointing True \
    --num_train_epochs 3 \
    --learning_rate 2e-5 --warmup_ratio 0.01 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps --save_steps 5000 --save_total_limit 2 \
    --eval_strategy no --logging_steps 50 \
    --diag_log_every 50 --report_to none \
    $EXTRA_WODD
}

# ----- MetaState curve: slots ∈ {64, 128, 256, 512, 1024}, dim fixed -----
METASTATE_KW_TEMPLATE='{"prelude_layers":8,"recurrent_layers":16,"coda_layers":8,"t_step_eval":4,"state_slots":__M__,"state_dim":1024,"state_n_heads":8,"zero_init_metastate":true}'

if [[ "$ARM_FILTER" == "both" || "$ARM_FILTER" == "metastate" ]]; then
  for M in 64 128 256 512 1024; do
    KW="${METASTATE_KW_TEMPLATE/__M__/$M}"
    run_cell "metastate_M${M}" metastate "$KW"
  done
fi

# ----- WoDD curve: (tape_dim, T_step) per the plan -----
if [[ "$ARM_FILTER" == "both" || "$ARM_FILTER" == "wodd" ]]; then
  for DT in "256 4" "512 4" "1024 4" "1024 8" "1024 16"; do
    D="${DT% *}"; T="${DT#* }"
    KW=$(cat <<EOF
{"prelude_layers":8,"recurrent_layers":16,"coda_layers":8,"t_step_eval":${T},"tape_dim":${D},"tape_max_writes":$((T > 16 ? T : 16)),"tape_n_heads":8,"zero_init_wodd":true}
EOF
)
    KW="$(echo "$KW" | tr -d '\n')"
    run_cell "wodd_d${D}_T${T}" wodd "$KW"
  done
fi

echo "[phase3] all cells launched"
