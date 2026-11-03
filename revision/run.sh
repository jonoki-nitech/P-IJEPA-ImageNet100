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
#     * after every selection, test_mae_vs_t.py evaluates ONLY the new
#       variant (--variants lookahead_sample): each seed's rec_error_vs_t.csv
#       contains just "lookahead_sample" (fast; the existing sigma
#       variants were already evaluated individually - cross-strategy
#       comparison plots are assembled post-hoc, e.g. from the saved
#       selection JSONs or the earlier per-variant CSVs)
#
#   Cost: per step ≈ (σ lookahead cost) + SAMPLES extra sigma-predictor
#   forwards per candidate row-chunk. For smoke tests use MAX_T=2,
#   SAMPLES=1, SKIP-heavy defaults below; no zip packaging in this script.
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
#     SAMPLES="1 4"        sampled-rollout K values to RUN (default "1 4":
#                          both the pure sampling variant and the K=4
#                          averaged variant). Artefact names are K-tagged:
#                          K=1 -> ..._test_lookahead_sample.json,
#                          K>1 -> ..._test_lookahead_sample<K>.json
#     TOPK=<k>             candidate pre-screen size (default 16)
#     MAX_T=<n>            cap the selection horizon for smoke tests
#                          (the actual tmax is in the JSON names, so smoke
#                          artefacts never collide with full-run ones)
#     SEL_ROOT=<dir>       default selected_patch_index
#     EVAL_ROOT=<dir>      default images/rec_images_full
# =============================================================================

set -e
set -u

PY=${PYTHON:-python}
CONFIG=${CONFIG:-configs/configs.yaml}
DEVICE=${DEVICE:-cuda:0}
BATCH=${BATCH:-64}
SEEDS=${SEEDS:-"0 1 2 3 4"}
SAMPLES=${SAMPLES:-"1 4"}
TOPK=${TOPK:-16}
SEL_ROOT=${SEL_ROOT:-selected_patch_index}
EVAL_ROOT=${EVAL_ROOT:-images/rec_images_full}

echo "python : $($PY --version 2>&1)"
echo "samples=[$SAMPLES]  topk=$TOPK  seeds=[$SEEDS]  device=$DEVICE  batch=$BATCH  config=$CONFIG"

# -----------------------------------------------------------------------
# Per seed: sampled-rollout lookahead selection (b = 2..4, random start)
# + MAE reconstruction-error-vs-t evaluation of all variants in the dir
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

        # (a) sampled-rollout 1-step lookahead selection (random start),
        #     one run per K in SAMPLES (default "1 4"); the artefact name
        #     carries the K tag (see the header notes)
        EVAL_VARS=()
        for K in $SAMPLES; do
            if [ "$K" = "1" ]; then
                VAR="lookahead_sample"
            else
                VAR="lookahead_sample${K}"
            fi
            echo "### seed $SEED, b=$b, samples=$K (variant: $VAR)"
            "$PY" select_and_save_patch_index_lookahead_sample.py \
                  --fname "$CONFIG" --T "$TMAX" --b "$b" --seed "$SEED" \
                  --device "$DEVICE" --samples "$K" --topk "$TOPK" \
                  --preload --batch_size "$BATCH" \
                  --outdir "$SEL_DIR"
            EVAL_VARS+=("$VAR")
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
