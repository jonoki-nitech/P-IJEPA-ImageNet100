#!/usr/bin/env bash
# =============================================================================
# revision/archive/run.sh
#
#   [ARCHIVED] moved from revision/run.sh (internal paths updated to the
#   archived location so the script still runs as-is). Kept for reference
#   and re-runs of the multi-seed sweep. NOTE: the name revision/run.sh is
#   now taken by the deterministic-baseline training script.
#
#   Repeat the full single-seed sweep (revision/archive/run_seed.sh; default SEED=0)
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
#   Once ALL seeds have finished, the per-seed artefacts are additionally
#   consolidated into a single result.zip (per-seed subdirectories preserved,
#   directory layout identical to the repo; the seed 0 products and the
#   seed zips themselves are NOT included). result.zip is therefore the
#   one file to share for the whole multi-seed experiment. Contents can be
#   previewed without full extraction:
#       python -c "import zipfile; print(zipfile.ZipFile('result.zip').namelist())"
#   NOTE: an existing result.zip is OVERWRITTEN at this step (it no longer
#   means the seed 0 package once the multi-seed run has completed; the
#   seed 0 zip, if still needed, can be re-created with
#   bash revision/archive/run_seed.sh), and each per-seed result_seed_<SEED>.zip
#   is kept as well.
#   SKIP_ZIP=1 skips both the per-seed packaging and the final consolidation.
#
#   NOTE: sequential by construction (~7-14 h per seed on the reference GPU);
#   run it under nohup / tmux so it survives disconnections, e.g.
#       nohup bash revision/archive/run.sh > run.log 2>&1 &
#
# Usage:
#   PYTHON=/path/to/python bash revision/archive/run.sh
#   BATCH=<n> DEVICE=cuda:0 CONFIG=<yaml>   optional, forwarded to run_seed.sh
#   SEEDS="1 2"      override the default seed list ("1 2 3 4")
# =============================================================================

set -e
set -u

SEEDS=${SEEDS:-"1 2 3 4"}

for SEED in $SEEDS; do
    echo ""
    echo "#################################################################"
    echo "###  revision/archive/run_seed.sh with SEED=$SEED"
    echo "#################################################################"
    SEED="$SEED" \
    PYTHON="${PYTHON:-python}" \
    BATCH="${BATCH:-64}" \
    DEVICE="${DEVICE:-cuda:0}" \
    CONFIG="${CONFIG:-configs/configs.yaml}" \
    bash revision/archive/run_seed.sh
done

# -----------------------------------------------------------------------------
# Consolidate: once every seed finished, pack the per-seed artefacts into a
#   single result.zip (per-seed directory layout preserved; seed 0 excluded,
#   per-seed zips kept as before). Skipped with SKIP_ZIP=1.
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    ZIP=${COMBINED_ZIP:-result.zip}
    echo ""
    echo "=== Consolidate per-seed artefacts into $ZIP ===================="
    rm -f "$ZIP"
    SEEDS_LIST="$SEEDS" "$PY" - "$ZIP" <<'PYEOF'
import os, sys, zipfile
zip_name = sys.argv[1]
seeds = [s.strip() for s in os.environ.get("SEEDS_LIST", "1 2 3 4").split()
         if s.strip() and s.strip() != "0"]
roots = ["images/calibration"]
for s in seeds:
    roots.append(f"selected_patch_index/seed_{s}")
    roots.append(f"images/rec_images_full/seed_{s}")
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in roots:
        if not os.path.isdir(root):
            print(f"(missing: {root} — skipped)")
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                zf.write(os.path.join(dirpath, f))
print(f"created: {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB)")
PYEOF
fi

echo ""
echo "=== run.sh: done ($SEEDS) ========================================="
ls -d selected_patch_index/seed_* 2>/dev/null || echo "(no seed subdirectories)"
