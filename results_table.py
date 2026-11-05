# Final-MAE comparison table across ALL patch-selection strategies.
#
# Reads the recorded rec_error_vs_t.csv of every experiment
# (read-only inputs) and builds ONE table: rows = strategies, columns =
# b = 2/3/4, cells = final MAE after aligning every strategy (and every
# seed) to the EARLIEST saturation endpoint of that b - the same
# alignment/paper-figure convention as paper_figures.py - aggregated over
# the 5 seeds (mean ± std, ddof=1).
#
# Data sources (5-seed convention: seed 0 at the root, seeds 1-4 in
# seed_<s>/):
#   sigma strategies + random
#       <results>/ImageNet100_1/images/rec_images_full/b<b>/
#       <results>/ImageNet100_2/images/rec_images_full/seed_<s>/b<b>/
#   MC-Dropout (DET)
#       <results>/ImageNet100_MCDropout_infer/images/rec_images_mcdropout/...
#   sampled-rollout lookahead (this revision's new experiment)
#       <results>/ImageNet100_lookahead_sample/images/rec_image_lookahead_sample/...
#
# Output:
#   prints the markdown table and saves
#   images/figures_paper/results_table.md (+ .csv, machine-readable)
#
# Usage: python results_table.py

import csv
import os
import sys

import numpy as np

# the console may use a legacy code page (cp932 on this machine); make the
# stdout tolerant so the markdown table (σ, ·) prints regardless
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(HERE, "images", "figures_paper")

BLOCKS = [2, 3, 4]

# (experiment root-builder, csv variant, display name)
STRATS = [
    ("sigma", "selected", "σ-head greedy"),
    ("sigma", "lookahead", "σ-head 1-step lookahead"),
    ("sigma", "ame", "AME"),
    ("sigma", "fixed_center_out", "fixed center-out"),
    ("sigma", "random", "random"),
    ("mc", "mcdropout", "MC-Dropout greedy"),
    ("mc", "lookahead_mcdropout", "MC-Dropout 1-step lookahead"),
    ("mc", "mcdropout_start_center", "MC-Dropout center-start"),
    ("sample", "lookahead_sample", "lookahead_sample (K=1)"),
    ("sample", "lookahead_sample4", "lookahead_sample4 (K=4)"),
]


def eval_root(exp, seed, b):
    if exp == "sigma":
        base = (os.path.join(RESULTS, "ImageNet100_1", "images",
                             "rec_images_full") if seed == 0 else
                os.path.join(RESULTS, "ImageNet100_2", "images",
                             "rec_images_full", f"seed_{seed}"))
    elif exp == "mc":
        base = (os.path.join(RESULTS, "ImageNet100_MCDropout_infer", "images",
                             "rec_images_mcdropout") if seed == 0 else
                os.path.join(RESULTS, "ImageNet100_MCDropout_infer", "images",
                             "rec_images_mcdropout", f"seed_{seed}"))
    else:
        base = (os.path.join(RESULTS, "ImageNet100_lookahead_sample", "images",
                             "rec_image_lookahead_sample") if seed == 0 else
                os.path.join(RESULTS, "ImageNet100_lookahead_sample", "images",
                             "rec_image_lookahead_sample", f"seed_{seed}"))
    return os.path.join(base, f"b{b}", "rec_error_vs_t.csv")


def read_variant(path, variant):
    """-> per-seed array of the recorded (mean-over-images) curve."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:
        if row[0] == variant:
            vals = []
            for v in row[1:]:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            return np.array(vals)
    raise KeyError(f"{variant} not found in {path}")


def eff_len(a):
    idx = np.flatnonzero(np.isfinite(a))
    return 0 if len(idx) == 0 else idx[-1] + 1


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    finals = {}          # (b, variant) -> (mean, std, L)
    for b in BLOCKS:
        curves = {}
        L_all = None
        for exp, variant, _label in STRATS:
            cs = []
            for seed in range(5):
                p = eval_root(exp, seed, b)
                if not os.path.isfile(p):
                    print(f"!! missing {exp}/{variant} seed{seed} b{b}: {p}")
                    continue
                cs.append(read_variant(p, variant))
            if not cs:
                continue
            curves[variant] = cs
            L = min(eff_len(c) for c in cs)       # per-variant earliest end
            L_all = L if L_all is None else min(L_all, L)
        for variant, cs in curves.items():
            ends = np.array([c[L_all - 1] for c in cs])
            finals[(b, variant)] = (float(ends.mean()),
                                    float(ends.std(ddof=1)), L_all)

    md, csv_rows = ["| strategy | b=2 | b=3 | b=4 |",
                    "|---|---|---|---|"], \
                  [["strategy", "b=2 mean", "b=2 std",
                    "b=3 mean", "b=3 std", "b=4 mean", "b=4 std"]]
    for exp, variant, label in STRATS:
        cells = []
        for b in BLOCKS:
            if (b, variant) in finals:
                m, s, L = finals[(b, variant)]
                cells.append(f"{m:.4f} ± {s:.4f}")
            else:
                cells.append("—")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
        csv_rows.append([label] + sum(
            ([f"{finals[(b, variant)][0]:.6f}",
              f"{finals[(b, variant)][1]:.6f}"] for b in BLOCKS), []))

    Ls = {b: finals[(b, STRATS[0][1])][2] for b in BLOCKS if
          (b, STRATS[0][1]) in finals}
    md.append("")
    md.append("aligned final t (patches observed = t·b²): "
              + ", ".join(f"b={b}: t={Ls[b]} ({Ls[b] * b * b} patches)"
                          for b in BLOCKS))

    table = "\n".join(md)
    print(table)
    with open(os.path.join(OUT, "results_table.md"), "w",
              encoding="utf-8", newline="") as f:
        f.write(table + "\n")
    with open(os.path.join(OUT, "results_table.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(csv_rows)
    print("\nsaved ->", os.path.join(OUT, "results_table.md"))
