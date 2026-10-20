# 1-step-lookahead selection ("lookahead" strategy, mu-predicted rollout).
#
# At every step, instead of directly choosing the block whose CURRENT sigma
# is the largest (the "selected" strategy), each candidate block c is
# SIMULATED before the actual choice:
#
#   1. its features are predicted from the observed context by the mu
#      predictor (the JEPA predictor trained to predict target features):
#            feature_c = predictor(observed_features, ctx_mask, c_mask)
#   2. the synthetic next context (observed features + feature_c) is fed to
#      the sigma predictor, which produces a hypothetical 1-step-ahead
#      uncertainty map:
#            sigma_next = sigma_head(sigma_predictor(
#                [obs_context, feature_c], [ctx + c], all_196))
#   3. the score of c is the total remaining uncertainty
#            score(c) = sum over patches NOT in (ctx U c) of sigma_next
#
# and the candidate with the SMALLEST score is the one actually observed.
# Only information available at the current step is used (observed encoder
# features + the model's own predictions); the true encoder features of the
# candidate blocks are never read, so nothing unobserved leaks into the
# choice (the alternative "use the true features of c" fallback would
# effectively peek at the image content).
#
# Speed: all candidates of the active images are stacked into one batch of
# rows (they share the context length and the target set), so each step
# needs one encoder forward over the active images plus one mu-predictor
# and one sigma-predictor forward per chunk of rows (batch_size rows).
# By default (--topk 16) the candidates are first pre-screened with the
# CURRENT sigma map and only the 16 best anchors per image are simulated
# (full enumeration is available with --topk 0, but costs ~#candidates
# forward rows per image-step and is prohibitive for b=1 on large
# datasets). Images saturate individually: as soon as no fully-unobserved
# b x b sliding window remains, an image stops being observed (terminal
# condition identical to the other strategies).
#
# Initial block: "--start random" (default, seeded through --seed, RNG call
# order identical to the other strategies) or "--start center".
#
# Output:
#   random start : selected_patch_index/block_{b}_tmax_{T}_test_lookahead.json
#                  + stats_block_..._test_lookahead.json: per-image trajectory
#                  of the total sigma over the unobserved patches (one entry
#                  per observed block, initial block included) + per decision
#                  step the rank of the chosen anchor in the CURRENT
#                  pooled-sigma ranking over all valid anchors (1 = the
#                  sigma-greedy choice) and the number of valid anchors
#   center start : ..._test_lookahead_start_center.json (+ its stats file,
#                  same content)
#   (the "stats_" prefix keeps the stats files invisible to the eval glob
#    block_{b}_tmax_{T}_test*.json)
#
# Usage:
#   python select_and_save_patch_index_lookahead.py \
#       --fname configs/configs.yaml --T 196 --b 1 [--seed 0] \
#       [--start center] [--batch_size 64]

import argparse
import json
import os
import pprint
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms as T
from tqdm import tqdm

from src.datasets.imagenet1k import ImageNet
from src.helper import init_model

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/configs.yaml')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='which device to use on local machine')
parser.add_argument(
    '--model_path', type=str,
    help='path to pretrained model',
    default='results/pretrained/jepa-latest.pth.tar')
parser.add_argument(
    '--T', type=int, default=10,
    help='time step')
parser.add_argument(
    '--b', type=int, default=3,
    help='block size')
parser.add_argument(
    '--start', type=str, default='random',
    choices=['random', 'center'],
    help='initial block: "random" (default, same protocol as the other '
         'strategies) or "center" (exact center block first)')
parser.add_argument(
    '--seed', type=int, default=None,
    help='manual seed for the random initial block; if omitted, behavior is '
         'unchanged (non-deterministic)')
parser.add_argument(
    '--batch_size', type=int, default=64,
    help='rows (image x candidate) forwarded together; selection is '
         'per-image independent, so this does not change the selection')
parser.add_argument(
    '--topk', type=int, default=16,
    help='pre-screen candidates with the CURRENT sigma map and simulate '
         'only the best K per image-step (default 16; 0 = simulate every '
         'valid candidate; K >= #candidates is equivalent to 0). Full '
         'enumeration costs ~#candidates forward rows per image-step, '
         'which is prohibitive for b=1 on large datasets; NOTE: with K=16 '
         'the choice differs from the full enumeration in a non-trivial '
         'fraction of steps, so --topk is a cheaper variant of the '
         'strategy, not an approximation of it')
parser.add_argument(
    '--preload', action='store_true',
    help='preload the normalized images into CPU RAM (~0.6 MB/image) once '
         'so the selection loop does not re-decode image files at every step')
parser.add_argument(
    '--outdir', type=str, default='selected_patch_index',
    help='output directory for the selection/stats JSONs (default: '
         'selected_patch_index; the per-seed runs of revision/run.sh write '
         'to selected_patch_index/seed_<SEED>)')

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

    # -- META
    model_name = params['meta']['model_name']
    copy_data = params['meta']['copy_data']
    pred_depth = params['meta']['pred_depth']
    sigma_depth = params['meta']['sigma_depth']
    pred_emb_dim = params['meta']['pred_emb_dim']
    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device

    # -- DATA
    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']
    # image pixels are only used by the encoder; the transform matches the
    # other selection scripts
    crop_size = params['data']['crop_size']

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

    # model (encoder features -> mu predictor -> sigma predictor/head)
    patch_size = params['mask']['patch_size']
    encoder, predictor, sigma_predictor, sigma_head = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        sigma_depth=sigma_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    for target, key in ((encoder, 'encoder'), (predictor, 'predictor'),
                        (sigma_predictor, 'sigma_predictor'),
                        (sigma_head, 'sigma_head')):
        pretrained_dict = checkpoint[key]
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        msg = target.load_state_dict(new_state_dict)
    del checkpoint

    encoder.eval()
    predictor.eval()
    sigma_predictor.eval()
    sigma_head.eval()

    N = len(test_dataset)
    T = args.T
    block_size = args.b

    # optional preload: normalized images kept in CPU RAM once
    imgs = None
    if args.preload and N > 0:
        imgs = [test_dataset[n][0]
                for n in tqdm(range(N), desc="preloading", leave=False)]

    # ---- precomputed sliding windows --------------------------------------
    # anchors in le-major / to-minor order (same convention as the other
    # strategies); window_matrix[a] = the b*b patch indices of anchor a
    n_pos = 14 - block_size + 1
    patch_grid = np.arange(196).reshape(14, 14)
    anchor_pos = [(le, to) for le in range(n_pos) for to in range(n_pos)]
    anchor_index = {pos: i for i, pos in enumerate(anchor_pos)}
    window_matrix = np.stack([
        patch_grid[to:to + block_size, le:le + block_size].reshape(-1)
        for (le, to) in anchor_pos])
    anchor_windows = [window_matrix[i].tolist()
                      for i in range(len(anchor_pos))]

    # ---- initial blocks -----------------------------------------------------
    contexts = []
    observed = np.zeros((N, 196), dtype=bool)
    for n in range(N):
        if args.start == 'center':
            left = top = (14 - block_size + 1) // 2
        else:
            left = random.choice(range(n_pos))
            top = random.choice(range(n_pos))
        win = anchor_windows[anchor_index[(left, top)]]
        contexts.append(list(win))
        observed[n, win] = True

    active = list(range(N))
    full_idx = torch.arange(196, dtype=torch.long, device=device)
    rows_ar = lambda R: torch.arange(R, device=device)[:, None]
    n_saturated = 0

    # -- per-image stats (saved next to the selection JSON) -------------------
    # sigma_sum_traj[i][j]: total sigma over the still-unobserved patches
    #   given the first j+1 observed blocks (trajectory of the total
    #   uncertainty; the value the strategy conditions on before each choice)
    # chosen_rank[i][s]: at decision step s, the rank of the anchor actually
    #   chosen in the current pooled-sigma ranking over ALL valid anchors
    #   (1 = the sigma-greedy choice; the quantity --topk screens on)
    # n_valid_traj[i][s]: number of valid anchors at decision step s
    sigma_sum_traj = [[] for _ in range(N)]
    chosen_rank = [[] for _ in range(N)]
    n_valid_traj = [[] for _ in range(N)]

    for t in range(T):
        if not active:
            break

        # -- saturation check: does a fully-unobserved b x b window remain?
        n_before = len(active)
        obs_active = observed[active]                      # (A, 196)
        overlap = np.take(obs_active, window_matrix,
                          axis=1).any(axis=2)              # (A, nA): anchor overlaps observed?
        has_candidate = (~overlap).any(axis=1)
        frozen = n_before - int(has_candidate.sum())
        if frozen > 0:
            n_saturated += frozen
            print(f"\nend @ {t+1}/{T}: {frozen} image(s) saturated")
        keep_idx = [k for k, ok in enumerate(has_candidate) if ok]
        active = [active[k] for k in keep_idx]
        overlap = overlap[keep_idx]           # re-align to the filtered active
        if not active:
            break

        print(f"\r    step {t+1}/{T} (active {len(active)}/{N})",
              end="", flush=True)

        # -- candidates per active image (order: image-major, le-major)
        k_ctx = len(contexts[active[0]])          # same for all active
        ctx_mat = np.array([contexts[i] for i in active])     # (A, k)
        valid_all = [np.flatnonzero(~overlap[ai]).tolist()
                     for ai in range(len(active))]

        # -- one encoder forward over all active images (chunked)
        feats = [None] * len(active)
        for lo in range(0, len(active), args.batch_size):
            imgs_ids = active[lo:lo + args.batch_size]
            if imgs is not None:
                x = torch.stack([imgs[i] for i in imgs_ids]).to(device)
            else:
                x = torch.stack(
                    [test_dataset[i][0] for i in imgs_ids]).to(device)
            mask_ctx = torch.tensor(ctx_mat[lo:lo + args.batch_size],
                                    dtype=torch.long, device=device)
            with torch.no_grad():
                feature = encoder(x, [mask_ctx])          # (Bc, k, D)
            for row, i in enumerate(imgs_ids):
                feats[lo + row] = feature[row]

        # -- current sigma map: used for the --topk pre-screening AND for the
        #    stats (uncertainty trajectory + chosen-block rank); one
        #    sigma-predictor forward per image
        sig_rows = []
        for lo in range(0, len(active), args.batch_size):
            n_chunk = len(active[lo:lo + args.batch_size])
            fc = torch.stack(feats[lo:lo + n_chunk])          # (Bc, k, D)
            mask_ctx = torch.tensor(ctx_mat[lo:lo + n_chunk],
                                    dtype=torch.long, device=device)
            with torch.no_grad():
                sig = sigma_head(sigma_predictor(
                    fc, [mask_ctx],
                    [full_idx.unsqueeze(0).repeat(n_chunk, 1)]))
            if sig.dim() == 3:
                sig = sig.squeeze(-1)
            sig_rows.append(sig)
        sigma_cur = torch.cat(sig_rows)                       # (A, 196)
        # total uncertainty over the still-unobserved patches (trajectory
        # point for the context the strategy is about to extend)
        obs_t = torch.tensor(observed[active], dtype=torch.bool,
                             device=device)
        sig_sum = sigma_cur.masked_fill(obs_t, 0.0).sum(dim=1)
        for ai, i in enumerate(active):
            sigma_sum_traj[i].append(float(sig_sum[ai]))
        pooled = F.avg_pool2d(sigma_cur.reshape(-1, 1, 14, 14),
                              kernel_size=block_size,
                              stride=1)                       # (A,1,Hp,Wp)
        pooled = (pooled.reshape(len(active), n_pos, n_pos)
                        .transpose(1, 2)
                        .reshape(len(active), -1))            # le-major order
        pool_np = pooled.detach().cpu().numpy()
        valid_used = valid_all
        if args.topk > 0:
            valid_used = [
                valid if len(valid) <= args.topk else
                [valid[j] for j in np.argsort(
                    -pool_np[ai][valid], kind='stable')[:args.topk]]
                for ai, valid in enumerate(valid_all)]

        # -- simulate every candidate: mu-predict its features, then score
        #    the synthetic next context with the sigma predictor
        rep, cand, img_seg = [], [], []
        for ai, valid in enumerate(valid_used):
            img_seg.append((len(rep), len(valid)))
            rep.extend([ai] * len(valid))
            cand.extend(valid)
        reps = rep                            # flattened image index per row
        cands = cand
        seg_bounds = img_seg
        scores = []
        for r0 in range(0, len(reps), args.batch_size):
            rsl = slice(r0, r0 + args.batch_size)
            rr = reps[rsl]
            cc = cands[rsl]
            R = len(rr)
            x_ctx = torch.stack([feats[ai] for ai in rr])  # (R, k, D)
            with torch.no_grad():
                # 1) predicted features of the (virtual) candidate block
                pred = predictor(
                    x_ctx,
                    [torch.tensor(ctx_mat[rr], dtype=torch.long,
                                  device=device)],
                    [torch.tensor(window_matrix[cc], dtype=torch.long,
                                  device=device)])      # (R, b^2, D)
                x_next = torch.cat([x_ctx, pred], dim=1)  # (R, k+b^2, D)
                ctxC = torch.cat(
                    [torch.tensor(ctx_mat[rr], dtype=torch.long,
                                  device=device),
                     torch.tensor(window_matrix[cc], dtype=torch.long,
                                  device=device)], dim=1)
                # 2) 1-step-ahead uncertainty map of the synthetic context
                sigma = sigma_head(sigma_predictor(
                    x_next, [ctxC],
                    [full_idx.unsqueeze(0).repeat(R, 1)]))
            if sigma.dim() == 3:
                sigma = sigma.squeeze(-1)                 # (R, 196)
            # 3) score = total sigma over the still-unobserved patches
            rem = torch.ones(R, 196, dtype=torch.bool, device=device)
            rem[rows_ar(R), ctxC] = False
            scores.append(sigma.masked_fill(~rem, 0.0).sum(dim=1))

        scores = torch.cat(scores)                        # (rows,)

        # -- pick, per image, the candidate with the smallest remaining sigma
        for ai, i in enumerate(active):
            start, cnt = seg_bounds[ai]
            if cnt == 0:
                continue
            best = cands[start + int(scores[start:start + cnt].argmin().item())]
            win = anchor_windows[best]
            contexts[i].extend(win)
            observed[i, win] = True
            # rank of the chosen anchor in the current pooled-sigma ranking
            # over ALL valid anchors (1 = the sigma-greedy choice); same
            # ordering rule as the --topk pre-screening (stable argsort)
            full_valid = valid_all[ai]
            pos = full_valid.index(best)
            order = np.argsort(-pool_np[ai][full_valid], kind='stable')
            chosen_rank[i].append(int(np.flatnonzero(order == pos)[0]) + 1)
            n_valid_traj[i].append(len(full_valid))

    print()

    if n_saturated > 0:
        print(f"end: {n_saturated}/{N} image(s) saturated "
              f"(no fully-unobserved b x b block left before t={T})")

    # -- terminal uncertainty: one sigma forward per image with its FINAL
    #    context (last point of the sigma-sum trajectory). Images saturate
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
                sig = sigma_head(sigma_predictor(
                    feature, [mask_ctx],
                    [full_idx.unsqueeze(0).repeat(len(chunk), 1)]))
            if sig.dim() == 3:
                sig = sig.squeeze(-1)
            sig_sum = sig.masked_fill(obs_c, 0.0).sum(dim=1)
            for row, i in enumerate(chunk):
                sigma_sum_traj[i].append(float(sig_sum[row]))

    os.makedirs(args.outdir, exist_ok=True)

    suffix = "" if args.start == "random" else "_start_center"
    with open(os.path.join(args.outdir,
              f"block_{block_size}_tmax_{T}_test_lookahead{suffix}.json",
              ), "w") as f:
        json.dump(contexts, f)

    # -- stats file ("stats_" prefix: never matched by the eval glob
    #    block_{b}_tmax_{T}_test*.json, but picked up by the run.sh zip)
    stats = {
        "meta": {"strategy": "lookahead", "b": block_size, "T": T,
                 "seed": args.seed, "start": args.start, "topk": args.topk,
                 "N": N},
        # sigma_sum[i]: len = #observed blocks; entry j = total sigma over
        #   the unobserved patches given the first j+1 blocks
        # chosen_sigma_rank[i]: len = #decision steps; rank (1-based) of the
        #   chosen anchor among the valid anchors in the current pooled-sigma
        #   ranking (stable argsort, same rule as --topk)
        # n_valid_anchors[i]: len = #decision steps; #valid anchors at that
        #   step (denominator for a relative rank)
        "sigma_sum": sigma_sum_traj,
        "chosen_sigma_rank": chosen_rank,
        "n_valid_anchors": n_valid_traj,
    }
    with open(os.path.join(args.outdir,
              f"stats_block_{block_size}_tmax_{T}_test_lookahead{suffix}.json",
              ), "w") as f:
        json.dump(stats, f)
