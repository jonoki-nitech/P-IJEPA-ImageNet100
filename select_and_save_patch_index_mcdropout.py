# MC-Dropout based patch selection for the DETERMINISTIC I-JEPA baseline.
#
# This is the DET counterpart of select_and_save_patch_index.py (the sigma
# selection): every step, each active image observes the fully-unobserved
# b x b window whose predicted-block-average uncertainty is currently the
# LARGEST, where the uncertainty comes from MC-Dropout sampling of the
# DET mu-predictor:
#     encoder forward once (deterministic, eval mode)
#     K stochastic predictor forwards with Dropout kept in train mode
#     var(patch) = mean over dropout masks / feat dims of (z_k - mean_k z_k)^2
#     block score = avg-pool of var over the b x b window -> argmax
#
# Everything else (CLI, initial-block seeding so that strategies with the
# same --seed share identical initial blocks, saturation handling, batching,
# JSON/stats file formats) follows the sigma script exactly, so its outputs
# are interchangeable with the sigma/random/fixed/ame/lookahead selection
# JSONs in the downstream evaluation (test_mae_vs_t.py with --index_dir).
#
# The DET checkpoint's sigma modules are random init and are NEVER loaded;
# the stats file keeps the legacy "sigma_sum" key name (for pipeline
# compatibility) but it now holds the total MC-Dropout variance.
#
# Usage:
#   python select_and_save_patch_index_mcdropout.py --T <t> --b <b> \
#       --seed <s> --preload --batch_size <n>
#   (smoke: --fname configs/configs_local.yaml --T 2 --b 2 --mc_k 2)

import torch
from torchvision import transforms as T
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn.functional as F
import argparse
import yaml
from src.datasets.imagenet1k import make_imagenet1k, ImageNet
import pprint
from det_uncertainty import (
    load_det_models,
    mc_predictor_forward)
import json
import random
import os


def block_to_index(left, top, w, h):
    """same block-to-index convention as the sigma selection script: the
    b x b window whose top-left corner is (top, left) on the 14x14 grid."""
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
    default='configs/configs_det.yaml')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='which device to use on local machine')
parser.add_argument(
    '--model_path', type=str,
    help='path to the DET-trained checkpoint',
    default='results/det-pretrained/jepa-det-latest.pth.tar')
parser.add_argument(
    '--T', type=int, default=10,
    help='time step')
parser.add_argument(
    '--b', type=int, default=3,
    help='block size')
parser.add_argument(
    '--start', type=str, default='random',
    choices=['random', 'center'],
    help='initial block: "random" (default, same as the sigma strategy) or '
         '"center" (exact center block first, then MC-Dropout selection)')
parser.add_argument(
    '--mc_k', type=int, default=16,
    help='number of stochastic (dropout-on) predictor forwards per selection '
         'step; the variance is estimated from these K samples (default 16; '
         'use 2-4 only for smoke tests)')
parser.add_argument(
    '--dropout', type=float, default=None,
    help='override the predictor dropout rate (default: meta.'
         'predictor_dropout of the config, i.e. the training value 0.1; '
         'DO NOT change this unless you know why)')
parser.add_argument(
    '--seed', type=int, default=None,
    help='manual seed for the random initial block (and the dropout masks, '
         'so the whole run is reproducible); if omitted, behavior is '
         'unchanged (non-deterministic)')
parser.add_argument(
    '--batch_size', type=int, default=64,
    help='images processed together per selection step (selection is per-'
         'image independent, so batching does not change the selection)')
parser.add_argument(
    '--preload', action='store_true',
    help='preload the normalized images into CPU RAM (~0.6 MB/image) once '
         'so the selection loop does not re-decode image files at every step')
parser.add_argument(
    '--outdir', type=str, default='selected_patch_index_mcdropout',
    help='output directory for the selection/stats JSONs (default: '
         'selected_patch_index_mcdropout, kept apart from the sigma '
         'variants in selected_patch_index; feed it to test_mae_vs_t.py '
         'via --index_dir)')

if __name__ == "__main__":

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"seed set: {args.seed}")

    # -- load script args
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device
        # torch.cuda.set_device(device)

    # -- dataset (test transform identical to the sigma selection)
    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']
    crop_size = params['data']['crop_size']
    copy_data = params['meta']['copy_data']

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
        copy_data=copy_data,
        index_targets=False)

    # -- model: DET triplet (encoder + dropout predictor; the returned
    #    target encoder is not needed for selection and is dropped)
    encoder, predictor, target_encoder = load_det_models(
        checkpoint_path=args.model_path,
        params=params,
        device=device,
        predictor_dropout=args.dropout)
    del target_encoder
    print(f'mc_k = {args.mc_k}')

    N = len(test_dataset)
    T = args.T
    block_size = args.b

    selected_index_list = []

    # ---- initial blocks ----------------------------------------------------
    # identical to the original per-image computation: the RNG calls are
    # made in the same per-image order, so results equal the unbatched run
    anchors = []
    for n in range(N):
        if args.start == 'center':
            # exact center block (rows/cols covered: center..center+b-1)
            left = top = (14 - block_size + 1) // 2
        else:
            left = random.choice(range(14-block_size+1))
            top = random.choice(range(14-block_size+1))
        anchors.append(block_to_index(left, top, block_size, block_size))

    # optional preload: normalized images kept in CPU RAM once, so the
    # selection loop does not re-decode the image files at every step
    imgs = None
    if args.preload and N > 0:
        imgs = [test_dataset[n][0] for n in range(N)]

    # ---- batched MC-Dropout selection ---------------------------------------
    # every image's selection is independent and each step of the loop gives
    # every active image exactly one more observed block (same context size),
    # so all active images can be forwarded together per chunk. An image is
    # frozen (stops observing) when no fully-unobserved b x b window remains.
    contexts = [list(a) for a in anchors]
    active = list(range(N))
    full_idx = torch.arange(196, dtype=torch.long, device=device)
    block_offsets = (torch.arange(block_size, device=device)[:, None] * 14 +
                     torch.arange(block_size, device=device)[None, :]).reshape(-1)

    # -- per-image stats: trajectory of the total uncertainty over the
    #    still-unobserved patches; sigma_sum_traj[i][j] = the sum given the
    #    first j+1 observed blocks (the value the strategy conditions on
    #    before each choice, plus a terminal value after the last block).
    #    "sigma_sum" keeps the sigma-pipeline key name, but with the DET
    #    model it is the summed MC-Dropout variance.
    sigma_sum_traj = [[] for _ in range(N)]

    for t in range(T):
        if not active:
            break

        # -- saturation check: does a fully-unobserved b x b window remain?
        n_before = len(active)
        obs = torch.zeros(n_before, 196, dtype=torch.bool, device=device)
        obs[torch.arange(n_before, device=device)[:, None],
            torch.tensor([contexts[i] for i in active], device=device)] = True
        overlap = F.max_pool2d(obs.float().view(n_before, 1, 14, 14),
                               kernel_size=block_size, stride=1)
        has_candidate = (overlap == 0).view(n_before, -1).any(dim=1)
        obs = obs[has_candidate]
        active = [i for p, i in enumerate(active) if has_candidate[p]]
        if n_before - len(active) > 0:
            print(f"end @ {t+1}/{T}: {n_before - len(active)} image(s) saturated")
        if not active:
            break

        print(f"\r    step {t+1}/{T} (active {len(active)}/{N})",
              end="", flush=True)

        # -- one step: forward the active images in chunks and let each of
        #    them observe its own largest-MC-variance block
        for lo in range(0, len(active), args.batch_size):
            chunk = active[lo:lo + args.batch_size]
            if imgs is not None:
                x = torch.stack([imgs[i] for i in chunk]).to(device)
            else:
                x = torch.stack([test_dataset[i][0] for i in chunk]).to(device)

            mask_ctx = torch.tensor([contexts[i] for i in chunk],
                                    dtype=torch.long, device=device)
            mask_pred = full_idx.unsqueeze(0).repeat(len(chunk), 1)

            with torch.no_grad():
                feature = encoder(x, [mask_ctx])
                # K stochastic predictor forwards (dropout ON);
                # z_mean = point prediction (unused here), var_pooled =
                # per-patch MC-Dropout variance = the selection signal
                _, var_pooled, _ = mc_predictor_forward(
                    predictor, feature, [mask_ctx], [mask_pred], args.mc_k)
            sigma = var_pooled

            # observed patches never win the argmax (same as the -inf write)
            sigma = sigma.masked_fill(obs[lo:lo + len(chunk)], -torch.inf)

            # total variance over the still-unobserved patches (trajectory
            # point for the context the strategy is about to extend)
            sig_sum = sigma.masked_fill(obs[lo:lo + len(chunk)],
                                        0.0).sum(dim=1)
            for row, i in enumerate(chunk):
                sigma_sum_traj[i].append(float(sig_sum[row]))

            # pooled variance map, argmax per image (the same computation as
            # select_next_block, vectorized over the batch)
            pooled = F.avg_pool2d(sigma.reshape(len(chunk), 1, 14, 14),
                                  kernel_size=block_size, stride=1)
            flat = pooled.reshape(len(chunk), -1)
            best = flat.argmax(dim=1)
            Hp, Wp = pooled.shape[-2], pooled.shape[-1]
            tops, lefts = np.unravel_index(best.cpu().numpy(), (Hp, Wp))
            base = (torch.tensor(tops, dtype=torch.long, device=device) * 14 +
                    torch.tensor(lefts, dtype=torch.long, device=device))
            blocks_all = base[:, None] + block_offsets[None, :]

            for row, i in enumerate(chunk):
                contexts[i].extend(blocks_all[row].tolist())

    print()

    # -- terminal uncertainty: one MC forward per image with its FINAL
    #    context (last point of the uncertainty trajectory). Images saturate
    #    at different block counts, so batch same-length contexts together.
    order = sorted(range(N), key=lambda i: len(contexts[i]))
    lo = 0
    while lo < N:
        j = lo
        while j < N and len(contexts[order[j]]) == len(contexts[order[lo]]):
            j += 1
        same = order[lo:j]
        lo = j
        for lo2 in range(0, len(same), args.batch_size):
            chunk = same[lo2:lo2 + args.batch_size]
            if imgs is not None:
                x = torch.stack([imgs[i] for i in chunk]).to(device)
            else:
                x = torch.stack(
                    [test_dataset[i][0] for i in chunk]).to(device)
            mask_ctx = torch.tensor([contexts[i] for i in chunk],
                                    dtype=torch.long, device=device)
            obs_c = torch.zeros(len(chunk), 196, dtype=torch.bool,
                                device=device)
            obs_c[torch.arange(len(chunk), device=device)[:, None],
                  torch.tensor([contexts[i] for i in chunk],
                               device=device)] = True
            with torch.no_grad():
                feature = encoder(x, [mask_ctx])
                _, var_pooled, _ = mc_predictor_forward(
                    predictor, feature, [mask_ctx],
                    [full_idx.unsqueeze(0).repeat(len(chunk), 1)], args.mc_k)
            sig = var_pooled
            sig_sum = sig.masked_fill(obs_c, 0.0).sum(dim=1)
            for row, i in enumerate(chunk):
                sigma_sum_traj[i].append(float(sig_sum[row]))

    selected_index_list = contexts

    os.makedirs(args.outdir, exist_ok=True)

    suffix = "_mcdropout" if args.start == "random" else "_mcdropout_start_center"

    with open(os.path.join(args.outdir,
              f"block_{block_size}_tmax_{T}_test{suffix}.json"), "w") as f:
        json.dump(selected_index_list, f)

    # -- stats file ("stats_" prefix: never matched by the eval glob
    #    block_{b}_tmax_{T}_test*.json, but picked up by the run.sh zip)
    stats = {
        "meta": {"strategy": "selected_mcdropout", "b": block_size, "T": T,
                 "seed": args.seed, "start": args.start, "N": N,
                 "mc_k": args.mc_k,
                 "uncertainty": "mc_dropout_var",
                 "model_path": args.model_path},
        # sigma_sum[i]: len = #observed blocks; entry j = total MC-Dropout
        #   variance over the unobserved patches given the first j+1 blocks
        "sigma_sum": sigma_sum_traj,
    }
    with open(os.path.join(args.outdir,
              f"stats_block_{block_size}_tmax_{T}_test{suffix}.json",
              ), "w") as f:
        json.dump(stats, f)
