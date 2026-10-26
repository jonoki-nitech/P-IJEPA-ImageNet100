#!/usr/bin/env bash
# =============================================================================
# revision/run.sh — DET uncertainty experiments (MC-Dropout)
#
#   [history] revision/run.sh previously trained the DET baseline; that
#   script now lives at revision/archive/run_mcdropout_train.sh. With the
#   DET training completed (checkpoint: results/det-pretrained/), this
#   script runs the uncertainty experiments on the DET model — the MC-
#   Dropout counterpart of the sigma experiments that produced
#   images/calibration/, selected_patch_index/ and images/rec_images_full/.
#
#   Step 0: calibration of the unobserved-patch uncertainty
#           calibration_eval_mcdropout.py (b=4, position 5:5, whole val set
#           by default — the same defaults as the sigma calibration)
#           -> images/calibration_det/
#           Same metrics/points format as calibration_eval.py, so the two
#           can be compared post-hoc from the saved calibration_points.pt
#           files (--points_pt mode, inputs never overwritten).
#
#   Step 1: per-seed selection + evaluation, SEEDS="0 1 2 3 4" by default
#           (the sigma-experiment convention: seed 0 writes to the
#           repo-root directories, other seeds to seed_<s>/ subdirectories):
#           for b in 2..4 with the sigma budget TMAX = 196/b^2 (=49, 21, 12;
#           NOTE: b=1 is SKIPPED for the uncertainty-driven selections, same
#           restriction as the sigma lookahead):
#             (a) MC-Dropout patch selection, RANDOM start (the DET
#                 counterpart of the sigma-greedy strategy)
#                 -> <SEL><seed-path>/block_<b>_tmax_<T>_test_mcdropout.json
#                    (+ stats_*.json with the variance trajectory)
#             (b) ... and the CENTER start variant (sigma's
#                 selected_start_center counterpart)
#                 -> ..._test_mcdropout_start_center.json
#             (c) 1-step lookahead MC-Dropout selection (default ON; the
#                 DET counterpart of the sigma lookahead, --topk pre-screen;
#                 by far the most expensive selection)
#                 -> ..._test_lookahead_mcdropout.json
#             (d) MAE reconstruction-error-vs-t evaluation of all variants
#                 -> <EVAL><seed-path>/b<b>/
#           per-seed package: result_det_seed_<s>.zip (selection JSONs +
#           stats + eval results + the calibration artefacts, mirroring the
#           sigma experiment's result_seed_<s>.zip contents)
#
#   End: consolidate every seed into result.zip (directory layout
#        preserved; per-seed zips kept). SKIP_ZIP=1 skips both packagings.
#
#   Comparability notes:
#     * with the same --seed, the initial blocks are identical to the
#       sigma-greedy/random strategies (same RNG convention), so error
#       curves align across strategies per seed;
#     * the "_mcdropout" filename suffix is unique: copying the selection
#       JSONs into selected_patch_index[seed dirs] afterwards and re-running
#       test_mae_vs_t.py there plots them together with the sigma
#       strategies in ONE chart (post-hoc comparison, no re-selection);
#     * the DET checkpoint's sigma modules are random init and are never
#       used; NEVER feed the DET checkpoint through the sigma scripts.
#
#   Memory/time notes: selection costs mc_k predictor forwards per step
#   (default 16) — on the reference GPU this is the same order as the
#   sigma lookahead strategy; the calibration step is ~mc_k times the
#   sigma calibration cost. For quick smoke tests use MC_K=2 MAX_T=2
#   CAL_N=<small> and SKIP_ZIP to keep the artefacts out.
#
# Usage (activate the env first, e.g. `conda activate p-ijepa`; RUN IT FROM
# THE REPOSITORY ROOT):
#   bash revision/run.sh 2>&1 | tee run.log
#   (long run: nohup bash revision/run.sh > run_det_exp.log 2>&1 &)
#
#   Optional env overrides:
#     PYTHON=<path>        default "python"
#     CONFIG=<yaml>        default configs/configs.yaml (data/model-shape)
#     DEVICE=<dev>         default "cuda:0" (scripts fall back to CPU
#                          automatically when CUDA is unavailable)
#     BATCH=<n>            images per selection step (default 64)
#     SEEDS="0 1 2 3 4"    default seed list (seed 0 included this time)
#     MC_K=<k>             MC-Dropout samples per step (default 16)
#     LOOKAHEAD=0|1        run the lookahead MC selection (default 1 = on;
#                          disable with LOOKAHEAD=0 for a cheaper run)
#     TOPK=<k>             lookahead candidate pre-screen size (default 16)
#     MAX_T=<n>            cap the selection horizon for smoke tests
#                          (the actual tmax is in the JSON names, so smoke
#                          artefacts never collide with full-run ones)
#     CAL_N=<n>            cap the calibration images (default 0 = all)
#     CAL_DIR=<dir>        default images/calibration_det
#     SEL_ROOT=<dir>       default selected_patch_index_mcdropout
#     EVAL_ROOT=<dir>      default images/rec_images_mcdropout
#     SKIP_ZIP=1           skip the per-seed and consolidated packaging
#     COMBINED_ZIP=<name>  default "result.zip"
# =============================================================================

set -e
set -u

PY=${PYTHON:-python}
CONFIG=${CONFIG:-configs/configs.yaml}
DEVICE=${DEVICE:-cuda:0}
BATCH=${BATCH:-64}
SEEDS=${SEEDS:-"0 1 2 3 4"}
MC_K=${MC_K:-16}
LOOKAHEAD=${LOOKAHEAD:-1}
TOPK=${TOPK:-16}
CAL_DIR=${CAL_DIR:-images/calibration_det}
SEL_ROOT=${SEL_ROOT:-selected_patch_index_mcdropout}
EVAL_ROOT=${EVAL_ROOT:-images/rec_images_mcdropout}

echo "python : $($PY --version 2>&1)"
echo "mc_k=$MC_K  lookahead=$LOOKAHEAD(topk=$TOPK)  seeds=[$SEEDS]  device=$DEVICE  batch=$BATCH  config=$CONFIG"

# -----------------------------------------------------------------------------
# Step 0: MC-Dropout calibration (counterpart of the sigma calibration step:
# b=4, position 5:5, the whole val set unless CAL_N>0)
# -----------------------------------------------------------------------------
echo ""
echo "### Step 0: MC-Dropout calibration -> $CAL_DIR"
CAL_ARGS=(--fname "$CONFIG" --device "$DEVICE" --mc_k "$MC_K"
          --out_dir "$CAL_DIR")
if [ "${CAL_N:-0}" != "0" ]; then
    CAL_ARGS+=(--N "$CAL_N")
fi
"$PY" calibration_eval_mcdropout.py "${CAL_ARGS[@]}"

# -----------------------------------------------------------------------------
# Step 1: per-seed selection + MAE evaluation
# -----------------------------------------------------------------------------
for SEED in $SEEDS; do
    echo ""
    echo "#################################################################"
    echo "###  seed $SEED"
    echo "#################################################################"
    if [ "$SEED" = "0" ]; then
        SEL_DIR="$SEL_ROOT"                 # seed 0: repo-root directories
        EVAL_DIR="$EVAL_ROOT"               # (the sigma-experiment convention)
    else
        SEL_DIR="$SEL_ROOT/seed_$SEED"
        EVAL_DIR="$EVAL_ROOT/seed_$SEED"
    fi

    # NOTE: the uncertainty-driven selections (greedy MC / lookahead MC)
    # run for b = 2..4 only: b = 1 is skipped (same restriction as the
    # sigma lookahead — 196 candidates per window make the strategy both
    # prohibitive and ill-conditioned)
    for b in 2 3 4; do
        TMAX=$(( 196 / (b * b) ))
        if [ -n "${MAX_T:-}" ]; then
            TMAX=$MAX_T
        fi
        echo "### seed $SEED, b=$b (tmax=$TMAX)"

        # (a) MC-Dropout selection, random start
        "$PY" select_and_save_patch_index_mcdropout.py \
              --fname "$CONFIG" --T "$TMAX" --b "$b" --seed "$SEED" \
              --device "$DEVICE" --mc_k "$MC_K" --preload \
              --batch_size "$BATCH" --outdir "$SEL_DIR"

        # (b) ... and the center-start variant
        "$PY" select_and_save_patch_index_mcdropout.py \
              --fname "$CONFIG" --T "$TMAX" --b "$b" --seed "$SEED" \
              --device "$DEVICE" --mc_k "$MC_K" --preload \
              --batch_size "$BATCH" \
              --start center --outdir "$SEL_DIR"

        # (c) 1-step lookahead MC-Dropout selection (default ON; random
        #     start, same protocol as the sigma lookahead)
        if [ "$LOOKAHEAD" = "1" ]; then
            "$PY" select_and_save_patch_index_lookahead_mcdropout.py \
                  --fname "$CONFIG" --T "$TMAX" --b "$b" --seed "$SEED" \
                  --device "$DEVICE" --mc_k "$MC_K" --topk "$TOPK" \
                  --preload --batch_size "$BATCH" --outdir "$SEL_DIR"
        fi

        # (d) MAE reconstruction-error-vs-t evaluation of the variants
        "$PY" test_mae_vs_t.py \
              --fname "$CONFIG" --T "$TMAX" --b "$b" \
              --index_dir "$SEL_DIR" --out_dir "$EVAL_DIR/b$b" \
              --device "$DEVICE" --preload
    done
    echo "### seed $SEED: done"
done

echo ""
echo "=== revision/run.sh: experiments done (seeds: $SEEDS) ============="

# -----------------------------------------------------------------------------
# Packaging: per-seed zip (selection JSONs + stats + MAE-eval results + the
# calibration artefacts; same content policy as the sigma experiment's
# result_seed_<s>.zip), then the consolidated result.zip. Skipped with
# SKIP_ZIP=1. Existing zips are overwritten.
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    for SEED in $SEEDS; do
        ZIP="result_det_seed_${SEED}.zip"
        echo ""
        echo "=== package seed $SEED -> $ZIP ============== "
        rm -f "$ZIP"
        "$PY" - "$ZIP" "$SEED" "$SEL_ROOT" "$EVAL_ROOT" "$CAL_DIR" <<'PYEOF'
import os, sys, zipfile
zip_name, seed, sel_root, eval_root, cal_dir = sys.argv[1:6]
if seed == "0":
    roots = [sel_root, eval_root, cal_dir]      # seed 0 = repo-root dirs
else:
    roots = [f"{sel_root}/seed_{seed}", f"{eval_root}/seed_{seed}", cal_dir]
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
    done

    ZIP=${COMBINED_ZIP:-result.zip}
    echo ""
    echo "=== consolidate all per-seed artefacts into $ZIP ==========="
    rm -f "$ZIP"
    "$PY" - "$ZIP" "$SEEDS" "$SEL_ROOT" "$EVAL_ROOT" "$CAL_DIR" <<'PYEOF'
import os, sys, zipfile
zip_name, seeds_str, sel_root, eval_root, cal_dir = sys.argv[1:6]
seeds = [s for s in seeds_str.split() if s]
roots = []
if "0" in seeds:
    roots += [sel_root, eval_root]              # seed 0 = repo-root dirs
for s in seeds:
    if s == "0":
        continue
    roots += [f"{sel_root}/seed_{s}", f"{eval_root}/seed_{s}"]
# the calibration is seed-independent: include it once
roots.append(cal_dir)
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
echo "=== revision/run.sh: done ========================================="
ls -d selected_patch_index_mcdropout* 2>/dev/null || echo "(no selection artefacts)"