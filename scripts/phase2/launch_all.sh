#!/usr/bin/env bash
# scripts/phase2/launch_all.sh
#
# Sequentially run every (arm, seed) cell of the Phase 2 head-to-head.
# 5 arms × 5 seeds = 25 runs. Best used as a SLURM array (one cell per
# array task) but this shell loop is fine for a single 8-GPU node.
#
# Usage:
#   ./scripts/phase2/launch_all.sh            # run all 25 cells
#   ARMS="metastate wodd" SEEDS="0 1" ./scripts/phase2/launch_all.sh

set -euo pipefail

ARMS="${ARMS:-vanilla latts metastate dpad wodd}"
SEEDS="${SEEDS:-0 1 2 3 4}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for arm in $ARMS; do
  for seed in $SEEDS; do
    echo "=== arm=$arm seed=$seed ==="
    bash scripts/phase2/launch_arm.sh "$arm" "$seed"
    # Crude wall-time per cell ~ depends on cluster; one 500K-sample,
    # 3-epoch, B=2, 8xH800 run is on the order of 6-10 hours.
  done
done
