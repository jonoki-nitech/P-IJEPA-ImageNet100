#!/usr/bin/env bash
# =============================================================================
# revision/run.sh
#
#   Two experiments, end to end, for the hand-off run:
#     Step 0: uncertainty (sigma) calibration           -> images/calibration/
#     Steps 1-4: 7 selection strategies x block sizes b=1..4
#               (patch selection until saturation, capped at T steps)
#               -> per-variant index JSONs + MAE reconstruction-error vs t
#
#   7 strategies generated per block size:
#     selected            : sigma-based selection, random initial block
#     start_center        : sigma-based selection, center initial block
#     random              : random selection, random initial block
#     random_center       : random selection, center initial block
#     fixed_center_out    : fixed, center-outward order
#     fixed_raster        : fixed, raster order (center initial block)
#     fixed_raster_no_center : fixed, raster order (scan order initial)
#
#   MAE evaluation automatically truncates each variant at its observable
#   budget (len(JSON) // b^2), aligning all strategies at the smallest
#   budget, i.e. plotting stops where duplicated observations would begin.
#   For the first 10 samples a gallery PNG (original / observed /
#   reconstruction per strategy at the final t) is also saved.
#   These outputs go to images/rec_images_tmax32/b{b}/ so the earlier
#   T=10 results in images/rec_images/b{b}/ are left untouched.
#
# Usage:
#   PYTHON=/path/to/python bash revision/run.sh
#   (PY defaults to "python". Scripts auto-fall back to CPU when CUDA is
#    unavailable, so no device flag is needed.)
#
#   Requires: torch, torchvision, numpy, matplotlib, pyyaml,
#             transformers==4.47.1, einops, tqdm, Pillow.
#   The MAE model is downloaded from Hugging Face ("facebook/vit-mae-base")
#   on first use (or bundled in the HF cache). The JEPA checkpoint must be
#   at results/pretrained/jepa-latest.pth.tar and the images under
#   data/imagenet_single/ (see configs/configs_local.yaml).
#
#   NOTE: this script overwrites images/calibration/ outputs. Everything
#   from the earlier T=10 experiments (block_*_tmax_10_*.json,
#   images/rec_images/b{1..4}/) remains untouched; new results use
#   block_*_tmax_32_*.json and images/rec_images_tmax32/b{b}/.
# =============================================================================

set -e
set -u

PY=${PYTHON:-python}
CFG=${CONFIG:-configs/configs.yaml}   # dataset paths; receiver default = the original configs.yaml
T=32          # cap on selection steps; sigma/random saturate earlier for b=3,4
SEED=0        # seed for the stochastic strategies (sigma & random variants)
DEVICE=${DEVICE:-cuda:0}   # override with e.g. DEVICE=cpu

echo "python : $($PY --version 2>&1)"
echo "config : $CFG   T=$T   seed=$SEED   device=$DEVICE"

# -----------------------------------------------------------------------------
# Step 0: calibration of the sigma predictor
#   default: b=4, one 9x9-patch evaluation position per image ("5:5"),
#   context patches excluded from the calibration points.
#   Outputs (images/calibration/): reliability diagram + metrics CSV +
#   calibration_points.pt (per-point sigma / prediction error + ids,
#   for later re-analysis: python calibration_eval.py --points_pt <path>)
# -----------------------------------------------------------------------------
echo ""
echo "=== Step 0: calibration ==========================================="
$PY calibration_eval.py --fname "$CFG" --b 4 --device "$DEVICE"

# -----------------------------------------------------------------------------
# Steps 1-4: block sizes b = 1..4
# -----------------------------------------------------------------------------
for b in 1 2 3 4; do

    echo ""
    echo "=== block size b=$b: generating 7 selection JSONs ================="

    $PY select_and_save_patch_index.py --fname "$CFG" --T "$T" --b "$b" --seed "$SEED" --device "$DEVICE"
    $PY select_and_save_patch_index.py --fname "$CFG" --T "$T" --b "$b" --start center --seed "$SEED" --device "$DEVICE"
    $PY select_and_save_patch_index_random.py --fname "$CFG" --T "$T" --b "$b" --seed "$SEED" --device "$DEVICE"
    $PY select_and_save_patch_index_random_center.py --fname "$CFG" --T "$T" --b "$b" --seed "$SEED"
    $PY select_and_save_patch_index_fixed.py --fname "$CFG" --T "$T" --b "$b" --mode center_out
    $PY select_and_save_patch_index_fixed.py --fname "$CFG" --T "$T" --b "$b" --mode raster
    $PY select_and_save_patch_index_fixed.py --fname "$CFG" --T "$T" --b "$b" --mode raster --init order

    echo ""
    echo "=== block size b=$b: MAE reconstruction error vs t ================"
    $PY test_mae_vs_t.py --fname "$CFG" --T "$T" --b "$b" --device "$DEVICE" \
        --out_dir "images/rec_images_tmax32/b$b"

done

# -----------------------------------------------------------------------------
# Summary of the produced artefacts
# -----------------------------------------------------------------------------
echo ""
echo "=== Done. Produced artefacts ======================================"
echo "--- calibration -------------------------------------------------"
ls -d images/calibration 2>/dev/null || echo "(missing: images/calibration)"
echo "--- selection index JSONs (block_b_tmax_32_*) --------------------"
ls selected_patch_index/*tmax_32_test*.json 2>/dev/null || echo "(missing)"
echo "--- MAE error-vs-t results ---------------------------------------"
ls -d images/rec_images_tmax32/b[1-4] 2>/dev/null
for b in 1 2 3 4; do
    ls -d "images/rec_images_tmax32/b$b/gallery" 2>/dev/null || true
done

# -----------------------------------------------------------------------------
# Package the artefacts into a single zip (path overridable with ZIP=<name>)
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    ZIP=${ZIP:-handoff_artifacts_tmax32.zip}
    echo ""
    echo "=== Pack artefacts into $ZIP ===================================="
    rm -f "$ZIP"
    "$PY" - "$ZIP" <<'PYEOF'
import os, sys, zipfile
zip_name = sys.argv[1]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in ("images/calibration", "images/rec_images_tmax32"):
        for dirpath, _, files in os.walk(root):
            for f in files:
                zf.write(os.path.join(dirpath, f))
    for f in os.listdir("selected_patch_index"):
        if "tmax_32_test" in f:
            zf.write(os.path.join("selected_patch_index", f))
print(f"created: {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB)")
PYEOF
fi

exit 0
