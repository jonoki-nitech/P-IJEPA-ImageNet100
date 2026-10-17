# Center-start random block selection.
# Copied from select_and_save_patch_index_random.py (the original is kept intact).
#
# Differences from the original random script:
#   1) The INITIAL block is fixed to the (near-)exact center of the 14x14 patch grid:
#         left = top = (14 - b + 1) // 2
#      e.g. b=3 -> 6 (rows/cols 6..8: the block whose center patch, patch 7,
#      is the exact image center); b=4 -> 5 (rows/cols 5..8).
#   2) The T subsequent steps use the same non-overlapping uniform random
#      block choice as the original.
#   3) The network checkpoint load is removed: this selection is purely random
#      and never reads the model, so dropping it does not change the output
#      (it just makes the run much faster).
#   4) Optional --seed for reproducibility (the original is unseeded).
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

import yaml
import torch
from einops.layers.torch import Rearrange
from tqdm import tqdm

from src.datasets.imagenet1k import ImageNet


def block_to_index(left, top, w, h):

    unpatchfy = Rearrange("(h w)->h w",
                          h=14, w=14)

    blocks = torch.arange(0, 196)
    blocks = unpatchfy(blocks)

    blocks = blocks[top:top + h, left:left + w]

    index = blocks.flatten().tolist()

    return index


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

    # -- DATA
    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']

    # dataset
    # (transform is irrelevant for index selection; keep the standard one)
    from torchvision import transforms as T
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
    T = args.T
    block_size = args.b

    # initial block = exact center of the grid (see header for the formula)
    center = (14 - block_size + 1) // 2

    selected_index_list = []

    for n in tqdm(range(N)):

        context = block_to_index(center, center, block_size, block_size)

        for t in range(T):
            candidate = []
            for le in range(14 - block_size + 1):
                for to in range(14 - block_size + 1):
                    next_index = block_to_index(le, to, block_size, block_size)
                    next_index_not_multiple = [x for x in next_index if x not in context]
                    if len(next_index_not_multiple) == block_size * block_size:
                        candidate.append((le, to))
            if len(candidate) == 0:
                print(f"end @ {t+1}/{T}")
                break

            left, top = random.choice(candidate)
            next_index = block_to_index(left, top, block_size, block_size)
            context = context + next_index

        selected_index_list.append(context)

    os.makedirs("selected_patch_index", exist_ok=True)

    with open(f"selected_patch_index/block_{block_size}_tmax_{T}_test_random_center.json", "w") as f:
        json.dump(selected_index_list, f)
