# Center-start random block selection (lightweight rewrite of the original
# per-image loop).
#
# Differences from select_and_save_patch_index_random.py (whose header shows
# the full selection rule):
#   1) The INITIAL block is fixed to the (near-)exact center of the 14x14
#      patch grid: left = top = (14 - b + 1) // 2
#      e.g. b=3 -> 6 (rows/cols 6..8: the block whose center patch, patch 7,
#      is the exact image center); b=4 -> 5 (rows/cols 5..8).
#   2) The T subsequent steps use the same non-overlapping uniform random
#      block choice as the random script.
#   3) No forward pass and no image decoding is needed: only the file list
#      (for N) is read. Output for a given --seed matches the original
#      script (the candidate order and the RNG call order are unchanged).
#
# Output: selected_patch_index/block_{b}_tmax_{T}_test_random_center.json
#
# Usage:
#   python select_and_save_patch_index_random_center.py \
#       --fname configs/configs_local.yaml --T 10 --b 3 [--seed 0]

import argparse
import json
import os
import pprint
import random

import numpy as np
import yaml
from tqdm import tqdm

from src.datasets.imagenet1k import ImageNet

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/configs.yaml')
parser.add_argument(
    '--T', type=int, default=10,
    help='time step')
parser.add_argument(
    '--b', type=int, default=3,
    help='block size')
parser.add_argument(
    '--seed', type=int, default=None,
    help='random seed (None = unseeded, like the original)')

if __name__ == "__main__":

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # -- load script args
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # -- DATA (only the file list is used; N images = N selections)
    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']

    test_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=None,
        train=False,
        copy_data=False,
        index_targets=False)

    N = len(test_dataset)
    T = args.T
    block_size = args.b

    # initial block = exact center of the grid (see header for the formula)
    center = (14 - block_size + 1) // 2

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

        win = anchor_windows[anchor_index[(center, center)]]
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

    with open(f"selected_patch_index/block_{block_size}_tmax_{T}_test_random_center.json", "w") as f:
        json.dump(selected_index_list, f)
