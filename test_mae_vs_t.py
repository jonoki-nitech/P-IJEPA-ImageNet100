# MAE reconstruction error vs the number of observation steps t, for every
# selection strategy (one line chart with one line per variant).
#
# The index JSONs (selected_patch_index/*.json) and all outputs of
# test_mae_on_selected_patches.py are left untouched: for a given variant the
# dataset slices the first t*b*b indices of the same JSON internally
# (input_patch_num=t in ImageNetWithMasks.__getitem__), so sweeping t is just
# a loop over dataset attribute updates, not a regeneration of the JSONs.
#
# For each variant, per (t, image): give MAE the visible patches
# (first t*b*b indices), reconstruct the unobserved region, and compute the
# pixel MSE over the unobserved region (identical formula to test_mae, so the
# t=8 value reproduces the previous single-point run).
#
# Outputs (default --out_dir images/rec_images):
#   rec_error_vs_t.png   one chart, x = t, y = mean pixel MSE, band = +/-
#                        std over images, one line per variant
#   rec_error_vs_t.csv   mean values, rows = variants, cols = t
#   rec_error_vs_t.pt    raw per-sample MSE arrays for later analysis
#
# Variant naming (same rule as test_mae_on_selected_patches.py):
#   "selected"        -> block_{b}_tmax_{T}_test.json            (sigma)
#   any other name V  -> block_{b}_tmax_{T}_test_V.json
#
# Usage:
#   python test_mae_vs_t.py --fname configs/configs_local.yaml --T 10 --b 3
#   python test_mae_vs_t.py --variants selected random          # subset
#   python test_mae_vs_t.py --log                               # log-scale y

import argparse
import csv
import glob
import json
import os
import pprint
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops.layers.torch import Rearrange
from torchvision import transforms as T
from transformers import ViTMAEForPreTraining
from tqdm import tqdm

from src.datasets.imagenet1k import ImageNet

normalize = T.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
inv = T.Normalize(
    mean=[-m/s for m, s in zip((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))],
    std=[1/s for s in (0.229, 0.224, 0.225)]
)


class ImageNetWithMasks(ImageNet):
    def __init__(self, root, image_folder, train, copy_data, index_targets, input_patch_num=10, mask_list=None, blocksize=2, transform=None):
        super().__init__(root=root,
        image_folder=image_folder,
        transform=transform,
        train=train,
        copy_data=copy_data,
        index_targets=index_targets)

        if mask_list is None:
            raise ValueError("mask_list must be provided.")
        if len(mask_list) != len(self.samples):
            raise ValueError("Length of mask_list must match ImageNet dataset length.")

        self.mask_list = mask_list
        self.input_patch_num = input_patch_num
        self.block_size = blocksize

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # first t * b * b indices of the variant's JSON (observation order)
        masks = self.mask_list[idx][:self.input_patch_num*self.block_size*self.block_size]
        target_masks = [x for x in range(196) if x not in masks ]

        return img, target, torch.tensor(masks, dtype=torch.long), \
            torch.tensor(target_masks, dtype=torch.long)


def select_tokens(x, idx):
    """Return tokens at the given indices, preserving the index order.
    x: (b, tokens, c), idx: (b, k)"""
    idx = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return x.gather(dim=1, index=idx)


def visualize_rec_image(img, rec_patches, target_mask):
    patchify = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=16,
                p2=16)
    unpatchify = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=16,
            p2=16,
            h=14, w=14)
    img_patches = patchify(img)
    img_patches[:, target_mask[0], :] = rec_patches
    img_rec = unpatchify(img_patches)

    return img_rec


def forward_with_mask_idx(model, img, mask_idx=None, target_idx=None):

    with torch.no_grad():

        idx = torch.cat([mask_idx, target_idx], dim=1)
        k = mask_idx.shape[1]

        # embedding
        pos_embs = model.vit.embeddings.position_embeddings
        embs = model.vit.embeddings.patch_embeddings(img)
        embs = embs + pos_embs[:, 1:, :]

        # mask
        embs = select_tokens(embs, mask_idx)
        ids_restore = idx.argsort(1)

        # add cls token
        cls_token = model.vit.embeddings.cls_token + pos_embs[:, :1, :]
        cls_tokens = cls_token.expand(embs.shape[0], -1, -1)
        embs = torch.cat((cls_tokens, embs), dim=1)

        # encode
        encoder_output = model.vit.encoder(embs)
        sequence_output = encoder_output[0]
        sequence_output = model.vit.layernorm(sequence_output)

        # decode
        decoder_outputs = model.decoder(sequence_output, ids_restore)
        logits = decoder_outputs.logits

        rec_patches = select_tokens(logits, idx[:, k:])

    return logits, rec_patches


def variant_json_path(variant, block_size, t_max):
    if variant == "selected":
        return f"selected_patch_index/block_{block_size}_tmax_{t_max}_test.json"
    return f"selected_patch_index/block_{block_size}_tmax_{t_max}_test_{variant}.json"


def plot_results(results, t_list, out_dir, log_scale, title):
    """One chart: x = t, y = mean pixel MSE, one line per variant (no bands)."""
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for variant, arr in results.items():
        mean = arr.mean(axis=1)
        ax.plot(t_list, mean, marker="o", ms=4, lw=1.8, label=variant)

    ax.set_xlabel("observation steps t")
    ax.set_ylabel("pixel MSE (unobserved region)")
    if len(t_list) <= 40:
        # with very long sweeps (e.g. t_max = 196 for b=1) the full tick
        # list would be unreadable; keep the default locator instead
        ax.set_xticks(t_list)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "rec_error_vs_t.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def print_summary(results, t_list):
    print("=" * 70)
    for variant, arr in results.items():
        mean = arr.mean(axis=1)
        row = "  ".join(f"t={t}:{m:.4f}" for t, m in zip(t_list, mean))
        print(f"{variant:>22}: {row}")
    print("=" * 70)


def as_display(t):
    """(1, 3, 224, 224) float tensor -> HxW numpy in [0, 1] for imshow."""
    return t.squeeze().permute(1, 2, 0).clamp(0, 1).numpy()


def save_gallery_images(gallery_cache, variants, out_dir, b, t_final):
    """Save reconstruction gallery PNGs for the first samples.

    One PNG per sample: top row = original (every column), middle row = the
    observation given to MAE (unobserved = uniform), bottom row = each
    variant's reconstruction at the final aligned t with the observed
    patches replaced by the actual pixels
    (same compositing as test_mae_on_selected_patches.py).
    gallery_cache keys: 'original' {n: tensor},
                        'observed' {n: {variant: tensor}},
                        'rec'      {n: {variant: tensor}}
    """
    samples = sorted(gallery_cache["original"].keys())
    if not samples:
        return None
    gallery_dir = os.path.join(out_dir, "gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    for n in samples:
        fig, axes = plt.subplots(3, len(variants), squeeze=False,
                                 figsize=(2 * len(variants), 6.6))
        for j, variant in enumerate(variants):
            axes[0, j].imshow(as_display(gallery_cache["original"][n]))
            axes[0, j].axis("off")
            axes[0, j].set_title(variant, fontsize=8)
            axes[1, j].imshow(as_display(gallery_cache["observed"][n][variant]))
            axes[1, j].axis("off")
            axes[2, j].imshow(as_display(gallery_cache["rec"][n][variant]))
            axes[2, j].axis("off")
        axes[1, 0].set_ylabel("observed")
        axes[2, 0].set_ylabel("reconstruction")
        fig.suptitle(f"sample {n} @ final t={t_final} (b={b})", fontsize=10)
        plt.tight_layout()
        path = os.path.join(gallery_dir, f"n{n:04d}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
    print(f"saved: {len(samples)} gallery images "
          f"(e.g. {os.path.join(gallery_dir, f'n{samples[0]:04d}.png')})")
    return gallery_dir


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/configs.yaml')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='which device to use on local machine')
parser.add_argument(
    '--T', type=int, default=10,
    help='t_max used when the JSONs were generated (also the default t sweep range)')
parser.add_argument(
    '--b', type=int, default=3,
    help='block size')
parser.add_argument(
    '--tlist', type=int, nargs='+', default=None,
    help='observation step counts to sweep (default: 1..T)')
parser.add_argument(
    '--variants', type=str, nargs='*', default=None,
    help='variants to evaluate (default: every JSON matching '
         'selected_patch_index/block_{b}_tmax_{T}_test*.json)')
parser.add_argument(
    '--log', action='store_true',
    help='use a log scale on the y axis')
parser.add_argument(
    '--N', type=int, default=0,
    help='number of test images (0 = all)')
parser.add_argument(
    '--out_dir', type=str, default='images/rec_images',
    help='output directory')
parser.add_argument(
    '--points_pt', type=str, default='',
    help='path to an existing rec_error_vs_t.pt; if set, the script does '
         'plotting only (no model loading, no dataset, no forward passes)')
parser.add_argument(
    '--gallery_n', type=int, default=10,
    help='save reconstruction gallery PNGs (original / observed / each '
         "variant's reconstruction at the final t) for the first this-many "
         'samples; 0 disables')

if __name__ == "__main__":

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  analysis-only mode: plot from previously collected arrays
    # ------------------------------------------------------------------ #
    if args.points_pt:
        data = torch.load(args.points_pt, map_location="cpu")
        t_list = [int(t) for t in data["t_list"]]
        results = {v: np.asarray(a, dtype=float) for v, a in data["results"].items()}
        b_pt = int(data.get("block_size", args.b))
        tmax_pt = int(data.get("t_max", args.T))

        # alignment: truncate at the smallest observable budget among the
        # displayed variants (re-derived from the on-disk index JSONs;
        # read-only, no model needed)
        variant_max_t = {}
        for v in results:
            p = variant_json_path(v, b_pt, tmax_pt)
            if os.path.exists(p):
                ml = json.load(open(p))
                variant_max_t[v] = min(len(x) // (b_pt ** 2) for x in ml)
        T_plot = min(t for t in [tmax_pt] + list(variant_max_t.values()))
        keep = [i for i, t in enumerate(t_list) if t <= T_plot]
        if len(keep) < len(t_list):
            t_list = [t_list[i] for i in keep]
            results = {v: a[keep] for v, a in results.items()}
            dropped = [f"{v}(budget={b})" for v, b in variant_max_t.items() if b > T_plot]
            print(f"aligned max t = {T_plot} (dropped tail of: {dropped})")

        title = (f"MAE reconstruction error vs t "
                 f"(b={b_pt}, T_max={tmax_pt}, N={data.get('n_images', '?')}, "
                 f"aligned max t={T_plot})")
        print(f"[analysis-only mode] plotting from: {args.points_pt}")
        png_path = plot_results(results, t_list, args.out_dir, args.log, title)
        print_summary(results, t_list)
        print(f"saved: {png_path}")
        sys.exit(0)

    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device

    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']
    copy_data = params['meta']['copy_data']
    patch_size = params['mask']['patch_size']

    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))]
    )

    test_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=test_transform,
        train=False,
        copy_data=False,
        index_targets=False)

    N = len(test_dataset)
    if args.N > 0:
        N = min(N, args.N)

    t_list = args.tlist if args.tlist is not None else list(range(1, args.T + 1))

    # ---- variants (auto: every JSON present for this (b, T)) -------------
    if args.variants is None or len(args.variants) == 0:
        base = f"block_{args.b}_tmax_{args.T}_test"
        found = sorted(glob.glob(f"selected_patch_index/{base}*.json"))
        variants = []
        for p in found:
            stem = os.path.basename(p)[:-5]
            if stem == base:
                variants.append("selected")
            else:
                variants.append(stem[len(base) + 1:])
    else:
        variants = args.variants

    # ---- JSON loading ----------------------------------------------------
    variant_json_lengths = {}
    for variant in variants:
        path = variant_json_path(variant, args.b, args.T)
        if not os.path.exists(path):
            print(f"[skip] {variant}: {path} not found")
            continue
        with open(path, "r") as f:
            mask_list = json.load(f)
        variant_json_lengths[variant] = mask_list

    variants = list(variant_json_lengths.keys())

    # per-variant observable budget: the largest t for which EVERY image still
    # receives fresh patches at step t (saturation of one image would otherwise
    # repeat the same visible set and show as a flat segment)
    variant_max_t = {
        v: min(len(ml) // (args.b ** 2) for ml in variant_json_lengths[v])
        for v in variants
    }
    T_plot = min(t for t in [args.T] + list(variant_max_t.values()))
    t_list = [t for t in t_list if t <= T_plot]
    dropped = []
    for v in variants:
        if variant_max_t[v] > T_plot:
            dropped.append(f"{v}(budget={variant_max_t[v]})")
    print(f"aligned max t = {T_plot} "
          f"(truncated variants beyond their saturation: {dropped})")

    # ---- build datasets ---------------------------------------------------
    datasets = {}
    for variant in variants:
        # input_patch_num=1 for construction; the sweep overrides it per t
        # blocksize is passed explicitly (the class default of 2 is a trap)
        datasets[variant] = ImageNetWithMasks(
            root=root_path,
            image_folder=image_folder,
            transform=test_transform,
            train=False,
            copy_data=copy_data,
            index_targets=False,
            mask_list=variant_json_lengths[variant],
            input_patch_num=1,
            blocksize=args.b)

    variants = list(datasets.keys())
    print(f"variants: {variants}, t sweep: {t_list}, images: {N}")

    # model
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", attn_implementation="sdpa").to(device)

    # ---- sweep ------------------------------------------------------------
    results = {v: np.zeros((len(t_list), N)) for v in variants}

    # gallery cache: filled only at the final t, for the first --gallery_n
    # samples; composed into PNGs after the sweep
    gallery_on = args.gallery_n > 0 and args.points_pt == ""
    gallery_cache = {"original": {}, "observed": {}, "rec": {}}
    is_final_t = len(t_list) - 1

    for variant in variants:
        ds = datasets[variant]
        for ti, t in enumerate(t_list):
            ds.input_patch_num = t
            for n in tqdm(range(N), desc=f"{variant} t={t}", leave=False):
                x, y, input_mask, target_mask = ds[n]
                x, input_mask, target_mask = x.unsqueeze(0).to(device), input_mask.unsqueeze(0).to(device), target_mask.unsqueeze(0).to(device)
                pred, rec_patches = forward_with_mask_idx(model, x, input_mask, target_mask)

                x_input = visualize_rec_image(x, torch.zeros_like(rec_patches), target_mask)
                mask = visualize_rec_image(torch.ones_like(x, dtype=torch.bool), torch.zeros_like(rec_patches, dtype=bool), target_mask)
                pred_image = model.unpatchify(pred.detach())

                x = inv(x)
                pred_image = inv(pred_image)
                unobserved = ~mask
                if unobserved.any():
                    mse = F.mse_loss(x[unobserved], pred_image[unobserved])
                else:
                    # full coverage: every patch is observed, so the
                    # unobserved-region error is vacuously zero
                    mse = torch.zeros(())
                results[variant][ti, n] = mse.item()

                if gallery_on and ti == is_final_t and n < args.gallery_n:
                    # bottom-row image: MAE reconstruction with the observed
                    # patches replaced by the actual pixels
                    # (same compositing as test_mae_on_selected_patches.py)
                    pred_gallery = pred_image.clone()
                    pred_gallery[mask] = inv(x_input)[mask]
                    gallery_cache["original"][n] = x.cpu()
                    gallery_cache["observed"].setdefault(n, {})[variant] = inv(x_input).cpu()
                    gallery_cache["rec"].setdefault(n, {})[variant] = pred_gallery.cpu()

    # ---- gallery PNGs (first samples, at the aligned final t) --------------
    if gallery_on:
        save_gallery_images(gallery_cache, variants, args.out_dir,
                            args.b, t_list[-1])

    # ---- plot -------------------------------------------------------------
    title = (f"MAE reconstruction error vs t "
             f"(b={args.b}, T_max={args.T}, N={N})")
    png_path = plot_results(results, t_list, args.out_dir, args.log, title)

    # ---- csv + raw arrays -------------------------------------------------
    csv_path = os.path.join(args.out_dir, "rec_error_vs_t.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + [f"t={t}" for t in t_list])
        for variant in variants:
            mean = results[variant].mean(axis=1)
            writer.writerow([variant] + [f"{m:.6f}" for m in mean])

    torch.save({
        "results": {v: torch.tensor(results[v]) for v in variants},
        "t_list": t_list,
        "block_size": args.b,
        "t_max": args.T,
        "n_images": N,
        "variant_max_t": variant_max_t,
        "aligned_max_t": T_plot,
    }, os.path.join(args.out_dir, "rec_error_vs_t.pt"))

    # ---- console summary --------------------------------------------------
    print_summary(results, t_list)
    print(f"saved: {png_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {os.path.join(args.out_dir, 'rec_error_vs_t.pt')}")

