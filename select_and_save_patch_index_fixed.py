# Fixed (image-independent) block selection orders.
# Inspired by select_and_save_patch_index_random.py (the original is kept intact).
#
# The selection rule does not look at the image at all: every image receives the
# same observation sequence, and the rule itself does not depend on the block
# size or the number of steps (both are simply taken from the CLI).
#
#   --mode center_out : blocks sorted by the distance of their center from the
#                       image center (ascending); ties broken in raster order.
#                       Rings of blocks around the center. This is the
#                       "spatial prior only" baseline for the sigma selection:
#                       if sigma-selection cannot beat this image-free rule,
#                       its value is limited for object-centric data.
#   --mode raster     : plain reading order, top-left -> bottom-right. The
#                       "no prior knowledge at all" lower end.
#
# Protocol (identical to the other variants): the INITIAL block is the exact
# center block (left = top = (14 - b + 1) // 2), then the first T blocks of the
# fixed order that do not overlap with anything chosen so far. Index list per
# image = initial block + chosen blocks in order (same slicing convention as
# the random/sigma JSONs: test_mae takes the first t*b*b indices).
#
# No model is needed for this script (pure deterministic rule).
#
# Output: selected_patch_index/block_{b}_tmax_{T}_test_fixed_{mode}.json
#
# Usage:
#   python select_and_save_patch_index_fixed.py \
#       --fname configs/configs_local.yaml --T 10 --b 3 --mode center_out

import argparse
import json
import os
import pprint

import yaml
import torch
from einops.layers.torch import Rearrange

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
    '--mode', type=str, default='center_out',
    choices=['center_out', 'raster'],
    help='fixed selection rule (center_out: rings around the image center; '
         'raster: top-left -> bottom-right reading order)')
parser.add_argument(
    '--init', type=str, default='center',
    choices=['center', 'order'],
    help='initial block: "center" = the exact center block first (identical '
         'initial condition across variants; default), "order" = no dedicated '
         'initial block, the sequence starts from the first block of the '
         'fixed order (e.g. the top-left corner for raster)')
parser.add_argument(
    '--outdir', type=str, default='selected_patch_index',
    help='output directory for the selection JSONs (default: '
         'selected_patch_index; the per-seed runs of revision/run.sh write '
         'to selected_patch_index/seed_<SEED>)')

if __name__ == "__main__":

    args = parser.parse_args()

    # -- load script args
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # -- DATA
    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']

    # dataset (only used for N; the transform content is irrelevant here)
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

    # all possible (left, top) positions of a b x b sliding window
    n_pos = 14 - block_size + 1
    all_blocks = [(le, to) for le in range(n_pos) for to in range(n_pos)]

    if args.mode == 'center_out':
        # image center in patch coordinates: (6.5, 6.5);
        # center of block (le, to) is (le + (b-1)/2, to + (b-1)/2)
        bc = (block_size - 1) / 2

        def dist(pos):
            cx = pos[0] + bc - 6.5
            cy = pos[1] + bc - 6.5
            return (cx ** 2 + cy ** 2) ** 0.5

        order = sorted(all_blocks, key=lambda pos: (dist(pos), pos[0], pos[1]))
    elif args.mode == 'raster':
        # reading order: row (to) first, then column (le)
        order = sorted(all_blocks, key=lambda pos: (pos[1], pos[0]))
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    # initial block: exact center (same rule as the random_center variant)
    # for --init center, or nothing for --init order (sequence comes purely
    # from the fixed order)
    center = (14 - block_size + 1) // 2

    if args.init == 'center':
        context = block_to_index(center, center, block_size, block_size)
    else:
        context = []

    chosen = 0
    for pos in order:
        if chosen >= T:
            break
        next_index = block_to_index(pos[0], pos[1], block_size, block_size)
        # sliding windows can overlap; keep only fully disjoint blocks
        if any(x in context for x in next_index):
            continue
        context = context + next_index
        chosen += 1

    if chosen < T:
        print(f"end @ {chosen}/{T}")

    # identical sequence for every image
    selected_index_list = [list(context) for _ in range(N)]

    os.makedirs(args.outdir, exist_ok=True)

    if args.init == 'center':
        suffix = f"test_fixed_{args.mode}"
    else:
        suffix = f"test_fixed_{args.mode}_no_center"

    with open(os.path.join(args.outdir,
              f"block_{block_size}_tmax_{T}_{suffix}.json"), "w") as f:
        json.dump(selected_index_list, f)

    print(f"mode={args.mode}, init={args.init}, "
          f"{len(context)} patch indices per image")
