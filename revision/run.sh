#!/usr/bin/env bash
# =============================================================================
# revision/run.sh — sampled-rollout 1-step lookahead on the σ model
#
#   New σ-side strategy "lookahead_sample": identical to the σ 1-step
#   lookahead (select_and_save_patch_index_lookahead.py) except that the
#   candidate block features in the rollout are SAMPLED from the model's
#   predictive distribution (mu + sigma * eps) instead of the mu-predictor
#   MEAN; the sampled next context is scored with the sigma predictor and
#   the candidate with the smallest remaining sigma-sum is observed.
#   See select_and_save_patch_index_lookahead_sample.py for details.
#
#   [history] revision/run.sh previously drove the DET/MC-Dropout
#   uncertainty experiments; that script lives on at
#   revision/archive/run_mcdropout_uncertainty.sh (det training:
#   archive/run_mcdropout_train.sh).
#
#   Protocol (mirrors the sigma experiment conventions):
#     * b = 2..4 only (b=1 skipped, same restriction as the σ lookahead);
#       tmax = 196/b^2 (= 49 / 21 / 12)
#     * RANDOM start only — there is NO center-start variant
#     * seed 0 writes to the repo-root directories, other seeds to
#       seed_<s>/ subdirectories (the σ-experiment convention)
#     * with the same --seed the initial blocks are IDENTICAL to the other
#       random-start strategies (shared RNG convention; the sampler uses a
#       dedicated torch.Generator so the initial-block stream is fresh)
#     * NO peeking: the rollouts use only observed features + the model's
#       own predictions (mu mean + current sigma), never true features
#
#   Evaluation:
#     * after every (seed, b), test_mae_vs_t.py evaluates ONLY this
#       experiment's variants (--variants lookahead_sample[<K>]): each
#       seed's rec_error_vs_t.csv keeps one row per K (fast; the sigma
#       variants were already evaluated individually - cross-strategy
#       comparison plots are assembled post-hoc, e.g. from the saved
#       selection JSONs or the earlier per-variant CSVs)
#
#   Output directories (fully SEPARATE from the sigma/MC experiments):
#     selection JSONs + stats -> selected_patch_index_lookahead_sample/
#                                (seed 0 at the root; other seeds in
#                                seed_<s>/ subdirectories)
#     MAE evaluation          -> images/rec_image_lookahead_sample/...
#
#   Packaging: after every seed finishes, everything (selection + eval,
#   all seeds, directory layout preserved) is consolidated into a single
#   result.zip (SKIP_ZIP=1 to skip; COMBINED_ZIP=<name> to rename; no
#   per-seed zips)
#
#   Cost: per step ≈ (σ lookahead cost) + SAMPLES extra sigma-predictor
#   forwards per candidate row-chunk. For smoke tests use MAX_T=2
#   (artefacts then never collide with full-run ones).
#
# Usage (activate the env first, e.g. `conda activate p-ijepa`; RUN IT FROM
# THE REPOSITORY ROOT):
#   bash revision/run.sh 2>&1 | tee run_sample_lookahead.log
#   (long run: nohup bash revision/run.sh > run_sample_lookahead.log 2>&1 &)
#
#   Optional env overrides:
#     PYTHON=<path>        default "python"
#     CONFIG=<yaml>        default configs/configs.yaml (data/model-shape)
#     DEVICE=<dev>         default "cuda:0" (scripts fall back to CPU
#                          automatically when CUDA is unavailable)
#     BATCH=<n>            rows per selection forward (default 64)
#     SEEDS="0 1 2 3 4"    default seed list
#     SAMPLES="1"          sampled-rollout K values to RUN (K chains averaged;
#                          default "1"). Artefact names are K-tagged:
#                          K=1 -> ..._test_lookahead_sample.json,
#                          K>1 -> ..._test_lookahead_sample<K>.json
#     HORIZON="2"          cascade-rollout horizons H to RUN (default "2":
#                          look 1 step ahead in the SAMPLING sense; the
#                          simulated chain continues for H levels, every
#                          further level greedily picks its next block from
#                          the model's own simulated sigma map). H=1 gives
#                          the plain one-step names, H>1 appends h<H>
#                          (K=1, H=2 -> ..._test_lookahead_sampleh2.json)
#     TOPK=<k>             candidate pre-screen size (default 16)
#     MAX_T=<n>            cap the selection horizon for smoke tests
#                          (the actual tmax is in the JSON names, so smoke
#                          artefacts never collide with full-run ones)
#     SEL_ROOT=<dir>       default selected_patch_index_lookahead_sample
#     EVAL_ROOT=<dir>      default images/rec_image_lookahead_sample
#     SKIP_ZIP=1           skip the consolidated result.zip packaging
#     COMBINED_ZIP=<name>  default "result.zip"
# =============================================================================

set -e
set -u

PY=${PYTHON:-python}
CONFIG=${CONFIG:-configs/configs.yaml}
DEVICE=${DEVICE:-cuda:0}
BATCH=${BATCH:-64}
SEEDS=${SEEDS:-"0 1 2 3 4"}
SAMPLES=${SAMPLES:-"1"}
HORIZON=${HORIZON:-"2"}
TOPK=${TOPK:-16}
SEL_ROOT=${SEL_ROOT:-selected_patch_index_lookahead_sample}
EVAL_ROOT=${EVAL_ROOT:-images/rec_image_lookahead_sample}

echo "python : $($PY --version 2>&1)"
echo "samples=[$SAMPLES]  horizon=[$HORIZON]  topk=$TOPK  seeds=[$SEEDS]  device=$DEVICE  batch=$BATCH  config=$CONFIG"

# -----------------------------------------------------------------------
# Per seed: sampled-rollout lookahead selection (b = 2..4, random start)
# + MAE reconstruction-error-vs-t evaluation of this experiment's variants
# -----------------------------------------------------------------------
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

    # NOTE: b = 2..4 only (same restriction as the sigma lookahead: 196
    # candidate windows per image-step make b = 1 prohibitive)
    for b in 2 3 4; do
        TMAX=$(( 196 / (b * b) ))
        if [ -n "${MAX_T:-}" ]; then
            TMAX=$MAX_T
        fi
        echo "### seed $SEED, b=$b (tmax=$TMAX)"

        # (a) sampled-rollout lookahead selection (random start), one run
        #     per (K, H) pair in SAMPLES x HORIZON (defaults K="1",
        #     H="2"); the artefact name carries the K/H tags
        EVAL_VARS=()
        for K in $SAMPLES; do
            for H in $HORIZON; do
                if [ "$K" = "1" ]; then
                    VAR="lookahead_sample"
                else
                    VAR="lookahead_sample${K}"
                fi
                if [ "$H" != "1" ]; then
                    VAR="${VAR}h${H}"
                fi
                echo "### seed $SEED, b=$b, samples=$K, horizon=$H (variant: $VAR)"
                "$PY" select_and_save_patch_index_lookahead_sample.py \
                      --fname "$CONFIG" --T "$TMAX" --b "$b" --seed "$SEED" \
                      --device "$DEVICE" --samples "$K" --horizon "$H" \
                      --topk "$TOPK" --preload --batch_size "$BATCH" \
                      --outdir "$SEL_DIR"
                EVAL_VARS+=("$VAR")
            done
        done

        # (b) MAE reconstruction-error-vs-t evaluation of THIS experiment's
        #     variants ONLY (one call so the rec_error_vs_t.csv keeps a
        #     row per K; the sigma variants were already evaluated
        #     individually)
        "$PY" test_mae_vs_t.py \
              --fname "$CONFIG" --T "$TMAX" --b "$b" \
              --index_dir "$SEL_DIR" --out_dir "$EVAL_DIR/b$b" \
              --device "$DEVICE" --preload \
              --variants "${EVAL_VARS[@]}"
    done
    echo "### seed $SEED: done"
done

echo ""
echo "=== revision/run.sh: sampled-rollout lookahead done (seeds: $SEEDS) =="
ls -d "$SEL_ROOT"* 2>/dev/null || echo "(no selection artefacts)"

# ---------------------------------------------------------------------------
# Packaging: consolidate every artefact (selection JSONs + stats + MAE-eval
# results; all seeds, directory layout preserved) into ONE result.zip.
# Skipped with SKIP_ZIP=1; renamed with COMBINED_ZIP=<name>. Overwritten.
# ---------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    ZIP=${COMBINED_ZIP:-result.zip}
    echo ""
    echo "=== consolidate all artefacts into $ZIP ==========="
    rm -f "$ZIP"
    "$PY" - "$ZIP" "$SEEDS" "$SEL_ROOT" "$EVAL_ROOT" <<'PYEOF'
import os, sys, zipfile
zip_name, seeds_str, sel_root, eval_root = sys.argv[1:5]
seeds = [s for s in seeds_str.split() if s]
roots = []
for s in seeds:
    if s == "0":
        roots += [sel_root, eval_root]      # seed 0 = repo-root directories
    else:
        roots += [f"{sel_root}/seed_{s}", f"{eval_root}/seed_{s}"]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in roots:
        if not os.path.isdir(root):
            print(f"(missing: {root}  -- skipped)")
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                zf.write(os.path.join(dirpath, f))
print(f"created: {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB)")
PYEOF
fi
