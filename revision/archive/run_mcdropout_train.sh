#!/usr/bin/env bash
# =============================================================================
# revision/archive/run_mcdropout_train.sh
#
#   [moved] the DET training entry point, previously revision/run.sh. The
#   name revision/run.sh now refers to the DET uncertainty experiments
#   (calibration + selection + MAE eval), and this script remains the
#   production command for re-training the DET baseline.
#
#   Train the DETERMINISTIC I-JEPA dropout baseline (main_det.py) on the
#   GPU machine, in one command. This is the production entry point for the
#   DET experiment.
#
#   What it runs:
#     python main_det.py --fname configs/configs_det.yaml --devices cuda:0
#   (= the same I-JEPA setup as the probabilistic training, but
#    deterministic: the mu-predictor is the predictor, the loss is plain
#    MSE, the sigma modules are checkpointed but never trained, and the
#    mu-predictor carries Dropout(p=0.1) (after the attention projection and
#    in the MLPs of every block) for later MC-Dropout inference.
#    See src/train_det.py for the full design.)
#
#   Outputs (do not collide with the probabilistic checkpoint):
#     checkpoint : results/vitb16.224-bs.2048-ep.300-det/jepa-det-latest.pth.tar
#                  (+ jepa-det-ep{50,100,...}.pth.tar every 50 epochs)
#     CSV log    : results/vitb16.224-bs.2048-ep.300-det/jepa-det_r0.csv
#     params copy: results/vitb16.224-bs.2048-ep.300-det/params-ijepa.yaml
#     STL10 mon. : results/vitb16.224-bs.2048-ep.300-det/stl10_acc_monitoring.csv
#                  (every 10 epochs; same cadence/code as the probabilistic
#                  training. The default STL10 root is the GPU-machine path
#                  hardcoded in stl10_eval.py; override with STL10_ROOT.)
#     package    : result.zip — the DET artefacts for sharing, created at
#                  the end automatically: the latest checkpoint + the log
#                  files (CSV / params yaml / STL10 monitoring). The epoch
#                  snapshots (jepa-det-ep*.pth.tar) are NOT included.
#
#   Memory note: the DET run is strictly CHEAPER in VRAM than the
#   probabilistic one (the sigma branch is never forwarded), so
#   batch_size=64 that the probabilistic run fits is safe here. The exact
#   peak VRAM appears in the log's first "[1, 0] ... [mem: ...]" line.
#
# Usage (activate the env first, e.g. `conda activate p-ijepa`; NOTE: the
#   script and the paths it references are repo-root relative, so RUN IT
#   FROM THE REPOSITORY ROOT):
#   bash revision/run.sh 2>&1 | tee run_train_dropout.log
#   (600 epochs = the same runtime scale as the probabilistic training;
#    run under nohup / tmux so it survives disconnections, e.g.
#     nohup bash revision/run.sh > run_train_dropout.log 2>&1 &)
#
#   Optional 1st argument = config selection:
#     bash revision/run.sh            -> configs/configs_det.yaml
#                                        (batch 64  x backward/step 32)
#     bash revision/run.sh bs64       -> same as no argument
#     bash revision/run.sh bs32       -> configs/configs_det_bs32.yaml
#                                        (batch 32 x backward/step 64;
#                                        low-VRAM fallback when the default
#                                        does not fit — halved activation
#                                        memory, same effective 2048-image
#                                        optimizer step)
#     bash revision/run.sh <path>.yaml -> any custom config
#    (a CONFIG=<yaml> env override also works; the argument wins)
#
#   Optional env overrides:
#     PYTHON=<path>          default "python"
#     DEVICES="cuda:0 ..."   default "cuda:0" (multi-GPU: extra words)
#     CONFIG=<yaml>          default configs/configs_det.yaml (the 1st
#                            argument above takes precedence)
#     STL10_ROOT=<dir>       dir containing stl10_binary/ (default: the
#                            GPU-machine path inside stl10_eval.py)
#     ZIP=<name>             default "result.zip" (name of the final package)
#     SKIP_ZIP=1             skip the final result.zip packaging
#     DET_DIR=<dir>          override the DET logging folder (otherwise
#                            auto-detected from CONFIG's logging.folder)
# =============================================================================

set -e

PYTHON=${PYTHON:-python}
DEVICES=${DEVICES:-"cuda:0"}
CONFIG=${CONFIG:-configs/configs_det.yaml}

# -- config selection: optional 1st argument (bs32 / bs64 / a .yaml path);
#    CONFIG env still works but the argument takes precedence
case "${1:-}" in
    "" | bs64) : ;;
    bs32) CONFIG=configs/configs_det_bs32.yaml ;;
    *.yaml | *.yml) CONFIG="$1" ;;
    *) echo "unknown config argument '$1' (use bs32, bs64, or a .yaml path)" >&2
       exit 1 ;;
esac

echo "python : $($PYTHON --version 2>&1)"
echo "config : $CONFIG   devices: $DEVICES"

$PYTHON main_det.py --fname "$CONFIG" --devices $DEVICES

echo ""
echo "=== revision/run.sh (DET training): done ==========================="
ls -l results/vitb16.224-bs.2048-ep.300-det/*.pth.tar 2>/dev/null || echo "(no checkpoint found)"

# -----------------------------------------------------------------------------
# Package: pack the DET artefacts into result.zip for sharing — the latest
#   checkpoint + the log files (CSV / params yaml / STL10 monitoring); the
#   epoch snapshots (jepa-det-ep*.pth.tar) stay out of the zip. Skipped with
#   SKIP_ZIP=1. An existing result.zip is overwritten.
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    ZIP=${ZIP:-result.zip}
    DET_DIR=${DET_DIR:-$("$PYTHON" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['logging']['folder'])" "$CONFIG")}

    echo ""
    echo "=== Package DET artefacts into $ZIP ============================"
    rm -f "$ZIP"
    "$PYTHON" - "$DET_DIR" "$ZIP" <<'PYEOF'
import glob, os, sys, zipfile
folder = sys.argv[1].rstrip("/")
zip_name = sys.argv[2]
patterns = ["*-latest.pth.tar", "*_r*.csv", "params-*.yaml", "stl10_acc_monitoring.csv"]
files = []
for pat in patterns:
    files.extend(sorted(glob.glob(os.path.join(folder, pat))))
if not files:
    raise SystemExit(f"no artefacts found in {folder} (zip not created)")
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        # .pth.tar is already-compressed binary: deflate gains nothing but
        # costs minutes, so store it uncompressed (logs stay deflated)
        ct = zipfile.ZIP_STORED if f.endswith(".pth.tar") else zipfile.ZIP_DEFLATED
        zf.write(f, compress_type=ct)  # repo-root relative layout preserved
        print("added:", f)
print(f"created: {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB)")
PYEOF
fi
