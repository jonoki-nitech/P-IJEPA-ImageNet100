#!/usr/bin/env bash
# =============================================================================
# revision/run.sh
#
#   Two experiments, end to end, for the hand-off run:
#     Step 0: uncertainty (sigma) calibration           -> images/calibration/
#     Steps 1-4: 9 selection strategies x block sizes b=1..4
#               -> per-variant index JSONs + MAE reconstruction-error vs t
#
#   9 strategies generated per block size:
#     selected            : sigma-based selection, random initial block
#     start_center        : sigma-based selection, center initial block
#     random              : random selection, random initial block
#     random_center       : random selection, center initial block
#     fixed_center_out    : fixed, center-outward order
#     fixed_raster        : fixed, raster order (center initial block)
#     fixed_raster_no_center : fixed, raster order (scan order initial)
#     ame                 : external baseline (Attention-Map Entropy;
#                           Pardyl et al., IJCAI 2023) using the MAE decoder
#                           attention; random initial block (seed = SEED);
#                           runs to natural saturation (no T cap)
#     lookahead           : 1-step lookahead selection (mu-predicted
#                           rollout); pre-screens candidates with the current
#                           sigma map (default --topk 16) and simulates only
#                           the best 16 per image-step; random initial block
#                           (seed = SEED). The most expensive selection
#                           (~16x the sigma selection), so it runs for
#                           b=2..4 only (skipped at b=1, where 196
#                           candidates per image-step would roughly double
#                           the total runtime for a modest gain)
#
#   The selection horizon is the MAXIMUM OBSERVABLE BLOCK COUNT per block
#   size, t_max = 196 / b^2 (196 = the 14x14 patch grid): every strategy is
#   generated (and evaluated) up to full coverage, stopping naturally at its
#   saturation point when no fully-unobserved b x b window remains
#   (duplicate-free selection). No artificial step cap is imposed.
#
#   The MAE evaluation automatically truncates each variant at its observable
#   budget (len(JSON) // b^2), aligning all strategies at the smallest
#   budget, i.e. plotting stops where duplicated observations would begin.
#   For the first 10 samples a gallery PNG (original / observed /
#   reconstruction per strategy at the final t) is also saved.
#   These outputs go to images/rec_images_full/b{b}/; the earlier
#   T=10 results in images/rec_images/b{b}/ are left untouched.
#
# Usage:
#   PYTHON=/path/to/python bash revision/run.sh
#   (PY defaults to "python". Scripts auto-fall back to CPU when CUDA is
#    unavailable, so no device flag is needed.)
#   BATCH=<n>   forwards n images together per step (default 64); set a
#               smaller value (e.g. 16) on GPUs with little memory. Affects
#               the sigma selections, the AME selection and the evaluation
#               sweep; results are independent of the batch size.
#   MAX_T=<n>   caps the selection horizon at n steps for quick smoke tests
#               (default 0 = full t_max). The capped t_max enters the product
#               filenames (e.g. block_1_tmax_2_*), so smoke-test outputs
#               never collide with full-run artefacts. NOTE: the AME
#               selection always runs to its natural saturation regardless
#               of MAX_T (its --T only labels the filename).
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
#   block_*_tmax_<196/b^2>_*.json and images/rec_images_full/b{b}/.
# =============================================================================

set -e
set -u

PY=${PYTHON:-python}
CFG=${CONFIG:-configs/configs.yaml}   # dataset paths; receiver default = the original configs.yaml
# selection horizon per block size: TMAX = 196 // b^2 (max observable b x b
# blocks over the 14x14 = 196 patch grid; 196/49/21/12 for b=1/2/3/4)
N_PATCHES=196
SEED=0        # seed for the stochastic strategies (sigma & random variants)
DEVICE=${DEVICE:-cuda:0}   # override with e.g. DEVICE=cpu
BATCH=${BATCH:-64}         # images forwarded together per step (GPU memory knob)
MAX_T=${MAX_T:-0}          # optional cap on the per-b selection horizon (smoke tests)

echo "python : $($PY --version 2>&1)"
echo "config : $CFG   patches=$N_PATCHES   seed=$SEED   batch=$BATCH   device=$DEVICE"

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

    TMAX=$((N_PATCHES / b / b))   # 196/49/21/12 for b=1/2/3/4
    if [ "$MAX_T" -gt 0 ] && [ "$MAX_T" -lt "$TMAX" ]; then TMAX="$MAX_T"; fi
    if [ "$b" = 1 ]; then NSEL=8; else NSEL=9; fi   # + lookahead from b=2
    echo ""
    echo "=== block size b=$b (t_max=$TMAX): generating $NSEL selection JSONs (7 + AME + lookahead for b>=2) ==="

    $PY select_and_save_patch_index.py --fname "$CFG" --T "$TMAX" --b "$b" --seed "$SEED" --device "$DEVICE" --preload --batch_size "$BATCH"
    $PY select_and_save_patch_index.py --fname "$CFG" --T "$TMAX" --b "$b" --start center --seed "$SEED" --device "$DEVICE" --preload --batch_size "$BATCH"
    $PY select_and_save_patch_index_random.py --fname "$CFG" --T "$TMAX" --b "$b" --seed "$SEED" --device "$DEVICE"
    $PY select_and_save_patch_index_random_center.py --fname "$CFG" --T "$TMAX" --b "$b" --seed "$SEED"
    $PY select_and_save_patch_index_fixed.py --fname "$CFG" --T "$TMAX" --b "$b" --mode center_out
    $PY select_and_save_patch_index_fixed.py --fname "$CFG" --T "$TMAX" --b "$b" --mode raster
    $PY select_and_save_patch_index_fixed.py --fname "$CFG" --T "$TMAX" --b "$b" --mode raster --init order
    # AME baseline: random initial block (--seed); runs to natural
    # saturation on its own (--T only labels the output filename
    # consistently with the in-house strategies)
    $PY select_and_save_patch_index_ame.py --fname "$CFG" --T "$TMAX" --b "$b" --seed "$SEED" --device "$DEVICE" --batch_size "$BATCH"
    # 1-step lookahead: pre-screens candidates with the current sigma map
    # (default --topk 16) and simulates only the 16 best anchors per
    # image-step; random initial block (--seed, i.e. default --start random).
    # NOTE: the most expensive selection (~16x the sigma selection), so it
    # runs for b=2..4 only; the b=1 case (196 candidates per image-step)
    # alone would add roughly half a day even on a GPU for a modest gain
    if [ "$b" != 1 ]; then
        $PY select_and_save_patch_index_lookahead.py --fname "$CFG" --T "$TMAX" --b "$b" --seed "$SEED" --device "$DEVICE" --preload --batch_size "$BATCH"
    fi

    echo ""
    echo "=== block size b=$b: MAE reconstruction error vs t ================"
    # --preload keeps the (normalized) images in CPU RAM during the sweep
    # (~0.6 MB per image; omit on low-RAM machines)
    $PY test_mae_vs_t.py --fname "$CFG" --T "$TMAX" --b "$b" --device "$DEVICE" --preload \
        --batch_size "$BATCH" --out_dir "images/rec_images_full/b$b"

done

# -----------------------------------------------------------------------------
# Summary of the produced artefacts
# -----------------------------------------------------------------------------
echo ""
echo "=== Done. Produced artefacts ======================================"
echo "--- calibration -------------------------------------------------"
ls -d images/calibration 2>/dev/null || echo "(missing: images/calibration)"
echo "--- selection index JSONs (block_b_tmax_<196/b^2>_*) --------------"
ls selected_patch_index/*tmax_*_test*.json 2>/dev/null || echo "(missing)"
echo "--- MAE error-vs-t results ---------------------------------------"
ls -d images/rec_images_full/b[1-4] 2>/dev/null
for b in 1 2 3 4; do
    ls -d "images/rec_images_full/b$b/gallery" 2>/dev/null || true
done

# -----------------------------------------------------------------------------
# Package the artefacts into a single zip (path overridable with ZIP=<name>)
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    ZIP=${ZIP:-result.zip}
    echo ""
    echo "=== Pack artefacts into $ZIP ===================================="
    rm -f "$ZIP"
    "$PY" - "$ZIP" <<'PYEOF'
import fnmatch, os, sys, zipfile
zip_name = sys.argv[1]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in ("images/calibration", "images/rec_images_full"):
        for dirpath, _, files in os.walk(root):
            for f in files:
                zf.write(os.path.join(dirpath, f))
    for f in os.listdir("selected_patch_index"):
        if fnmatch.fnmatch(f, "block_*_tmax_*_test*.json") or \
           fnmatch.fnmatch(f, "stats_block_*_tmax_*_test*.json"):
            zf.write(os.path.join("selected_patch_index", f))
print(f"created: {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB)")
PYEOF
fi

exit 0
