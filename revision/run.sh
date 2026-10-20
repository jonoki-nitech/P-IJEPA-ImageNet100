#!/usr/bin/env bash
# =============================================================================
# revision/run.sh
#
#   Repeat the full single-seed sweep (revision/run_seed.sh; default SEED=0)
#   over several seeds, one after another. This is the intended one-command
#   entry point for the receiver-side multi-seed experiment.
#
#     selection JSONs : selected_patch_index/seed_<SEED>/block_*.json
#     stats files     : selected_patch_index/seed_<SEED>/stats_block_*.json
#     MAE error curves: images/rec_images_full/seed_<SEED>/b<b>/
#     package         : result_seed_<SEED>.zip
#
#   Seed 0 artefacts (selected_patch_index/, images/rec_images_full/b<b>/,
#   result.zip and images/calibration/) stay exactly where they are: the
#   per-seed runs neither read nor overwrite them. images/calibration is
#   seed-independent (sigma-predictor calibration) and is simply re-produced
#   at each run, identically for every seed.
#
#   NOTE: sequential by construction (~7-14 h per seed on the reference GPU);
#   run it under nohup / tmux so it survives disconnections, e.g.
#       nohup bash revision/run.sh > run.log 2>&1 &
#
# Usage:
#   PYTHON=/path/to/python bash revision/run.sh
#   BATCH=<n> DEVICE=cuda:0 CONFIG=<yaml>   optional, forwarded to run_seed.sh
#   SEEDS="1 2"      override the default seed list ("1 2 3 4")
# =============================================================================

set -e
set -u

SEEDS=${SEEDS:-"1 2 3 4"}

for SEED in $SEEDS; do
    echo ""
    echo "#################################################################"
    echo "###  revision/run_seed.sh with SEED=$SEED"
    echo "#################################################################"
    SEED="$SEED" \
    PYTHON="${PYTHON:-python}" \
    BATCH="${BATCH:-64}" \
    DEVICE="${DEVICE:-cuda:0}" \
    CONFIG="${CONFIG:-configs/configs.yaml}" \
    bash revision/run_seed.sh
done

echo ""
echo "=== run.sh: done ($SEEDS) ========================================="
ls -d selected_patch_index/seed_* 2>/dev/null || echo "(no seed subdirectories)"
