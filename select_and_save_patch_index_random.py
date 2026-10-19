# Random block selection (lightweight rewrite of the original image loop).
#
# Selection rule (identical to the original per-image script):
#   * INITIAL block: (left, top) sampled with random.choice twice
#     (left first, then top) -> the RNG call order is the same as the
#     original, so the output for a given --seed is identical.
#   * Then, at each step, every (left, top) sliding window fully outside the
#     observed region is a candidate; one is chosen uniformly with
#     random.choice (candidates are built in le-major / to-minor order,
#     exactly like the original).
#   * The rule never looks at the image and needs no network forward, so the
#     image files are not even decoded here and the pretrained checkpoint is
#     not loaded (the original loaded both, pointlessly: same outputs, much
#     slower).
#
# Speed: the original recomputed every candidate window with einops inside
# the per-image loop (~1.9 x 10^8 calls for 5000 images at b=1, ~2.4 h);
# here the window index lists are precomputed once and the overlap test is a
# numpy gather, which turns the whole sweep into <1 min.
#
# Output: selected_patch_index/block_{b}_tmax_{T}_test_random.json
#
# Usage:
#   python select_and_save_patch_index_random.py \
#       --fname configs/configs_local.yaml --T 10 --b 3 [--seed 0]

import argparse
import json
import os
import pprint
import random

import numpy as np
import yaml
from torchvision import transforms as T
from tqdm import tqdm

from src.datasets.imagenet1k import ImageNet

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/configs.yaml')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='kept for CLI compatibility; unused (no forward pass is needed)')
parser.add_argument(
    '--T', type=int, default=10,
    help='time step')
parser.add_argument(
    '--b', type=int, default=3,
    help='block size')
parser.add_argument(
    '--seed', type=int, default=None,
    help='manual seed for the random selection sequence; if omitted, '
         'behavior is unchanged (non-deterministic)')

if __name__ == "__main__":

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"seed set: {args.seed}")

    # -- load script args
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # -- DATA (only the file list is used; N images = N selections)
    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']
    copy_data = params['meta']['copy_data']

    test_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=T.Compose([]),
        train=False,
        copy_data=copy_data,
        index_targets=False)

    N = len(test_dataset)
    T = args.T
    block_size = args.b

    # -- precompute the patch indices of every possible b x b sliding window.
    #    Rows follow the SAME (le-major, to-minor) order as the original
    #    candidate scan, so random.choice sees candidates in the same order.
    n_pos = 14 - block_size + 1
    patch_grid = np.arange(196).reshape(14, 14)
    anchor_pos = [(le, to) for le in range(n_pos) for to in range(n_pos)]
    anchor_index = {pos: i for i, pos in enumerate(anchor_pos)}
    window_matrix = np.stack([
        patch_grid[to:to + block_size, le:le + block_size].reshape(-1)
        for (le, to) in anchor_pos])
    anchor_windows = [window_matrix[i].tolist()
                      for i in range(len(anchor_pos))]

    selected_index_list = []
    n_saturated = 0

    for n in tqdm(range(N)):

        # initial block (same RNG call order as the original: left, then top)
        left = random.choice(range(n_pos))
        top = random.choice(range(n_pos))
        row = anchor_index[(left, top)]
        win = anchor_windows[row]

        context = list(win)
        observed = np.zeros(196, dtype=bool)
        observed[win] = True

        for t in range(T):
            # candidates: sliding windows fully unobserved (same order)
            overlap = observed[window_matrix].any(axis=1)
            if overlap.all():
                n_saturated += 1
                break

            valid = np.flatnonzero(~overlap)
            left, top = random.choice([anchor_pos[k] for k in valid])
            win = anchor_windows[anchor_index[(left, top)]]
            context.extend(win)
            observed[win] = True

        selected_index_list.append(context)

    if n_saturated > 0:
        print(f"end: {n_saturated}/{N} image(s) saturated "
              f"(no fully-unobserved b x b block left before t={T})")

    os.makedirs("selected_patch_index", exist_ok=True)

    with open(f"selected_patch_index/block_{block_size}_tmax_{T}_test_random.json", "w") as f:
        json.dump(selected_index_list, f)
