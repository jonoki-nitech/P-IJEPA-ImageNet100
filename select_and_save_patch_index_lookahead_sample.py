# 1-step / H-step lookahead selection with a SAMPLED rollout
# ("lookahead_sample" strategy).
#
# Identical to the "lookahead" strategy (select_and_save_patch_index_
# lookahead.py) except for the simulation of the candidate block:
#
#   * lookahead: the candidate features are the mu-predictor MEAN
#         feature_c = predictor(observed, ctx, c)                     (deterministic)
#     and the synthetic next context is [observed, feature_c].
#   * lookahead_sample: the candidate features are SAMPLED from the
#     model's predictive distribution for the candidate patches
#         feature_c ~ N(mu, sigma^2)
#         mu      = predictor(observed, ctx, c)          (R, b^2, D)
#         sigma^2 = current sigma map at the candidate patches (R, b^2)
#                       (the sigma head outputs the VARIANCE: the train
#                        loss e/v + log v matches a Gaussian NLL with
#                        variance = output, so the sampling std is the
#                        square root of the map value)
#         eps     ~ N(0, I) of shape (R, b^2, D)
#         feature_c = mu + sigma * eps
#     and the synthetic next context [observed, feature_c] is scored with
#     the sigma predictor exactly as in lookahead:
#         sigma_next = sigma_head(sigma_predictor([x_ctx, feat_c], ctx U c,
#                                                 all_196))
#         score(c) = sum of sigma_next over the patches NOT in (ctx U c)
#
# With --horizon H > 1 the rollout is a H-step CASCADE: after the
# candidate's features enter the synthetic context, every FURTHER
# simulated level greedily picks the next block from the model's own
# simulated sigma map (largest pooled sigma over the still-valid anchors),
# samples its features the same way (mu + sqrt(sigma) * eps), appends them
# and re-computes the simulated map. The score is the total remaining
# sigma after the whole chain, so the choice of the REAL block is the
# heuristic on "where does a greedy H-step rollout starting at c end up".
# Each of the --samples K rollouts is an independent chain (the inner
# block choices may differ between chains); K=1 is the pure variant.
# If a row runs out of valid anchors mid-chain (near-saturation image),
# its chain simply stops early and is scored with the map built so far.
# Anti-peeking still holds: only predicted/sampled features and the
# model's own maps enter the simulation, true features are never read.
# Naming: horizon=1 keeps the plain names (lookahead_sample /
# lookahead_sample<K>); horizon>1 appends h<H>
# (e.g. lookahead_sampleh2 for K=1, H=2).
#
# --branch full (H=2 only) upgrades the INNER level to a full enumeration:
# instead of picking one greedy block inside the simulation, EVERY topk
# level-2 block is itself simulated (leaf = image x level-1 x level-2) and
# the score of a level-1 candidate is the minimum residual sigma over its
# leaves — the exact two-step lookahead objective — averaged over the K
# chains. AND: the rollout result is OBSERVED — the selection advances by
# TWO blocks per real step (the argmin level-1 block plus, for real, the
# argmin level-2 block it computed; exact for the default K=1, the first
# chain's argmin for K>1; skipped when the row has no valid inner block).
# Cost: ~8.5x the greedy-inner H=2 rollout (~topk^2 rows per image-step);
# topk^H explodes for H>=3, so full branching is restricted to horizon=2
# (parser error otherwise). Name: the "fb" tag is replaced by "pair"
# (lookahead_sampleh2pair for K=1, H=2, full) — the name reflects the
# two-blocks-per-step observation.
#
# The score is therefore an estimate built only from information available
# at the current step (observed encoder features + the model's own
# predictions); the true encoder features of the candidates are never
# read, so nothing unobserved leaks into the choice (same anti-peeking
# rule as the mu lookahead).
#
# Sampling uses a dedicated torch.Generator seeded from --seed, so the
# Python-side RNG stream that draws the INITIAL BLOCKS is untouched: with
# the same --seed the initial blocks are IDENTICAL to the other random-
# start strategies (greedy / random / lookahead / AME, verified by the
# shared RNG convention), per image and per seed.
#
# The resulting sigma-based next-step map can drift off the manifold the
# sigma predictor saw in training (it was trained with the predicted mean
# as context), which is exactly the hypothesis this experiment tests.
#
# Speed: per candidate row-chunk 1 mu + K*(H sigma + (H-1) mu) forwards
# (the level-1 mu prediction is computed once and reused across the K
# chains). --topk 16 pre-screening as in lookahead. Images saturate
# individually (no fully-unobserved b x b window left).
#
# Initial block: random only, seeded through --seed (same protocol and
# RNG call order as the other random-start strategies). There is NO
# center-start variant of this strategy.
#
# Output:
#   selected_patch_index[<outdir>]/block_{b}_tmax_{T}_test_lookahead_sample.json
#   + stats_block_..._test_lookahead_sample.json (per-image sigma-sum
#   trajectory / chosen-anchor rank / n-valid, same contents as lookahead)
#
# Usage:
#   python select_and_save_patch_index_lookahead_sample.py \
#       --fname configs/configs.yaml --T 49 --b 2 --seed 0 \
#       [--samples 1] [--horizon 1] [--topk 16] [--batch_size 64] [--preload]

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
    '--seed', type=int, default=None,
    help='manual seed: seeds the random initial block (same convention as '
         'the other strategies) AND the sampling generator')
parser.add_argument(
    '--samples', type=int, default=1,
    help='sampled rollouts averaged per candidate (default 1 = the pure '
         'sampling variant; K > 1 averages the score over K chains for a '
         'less noisy choice)')
parser.add_argument(
    '--horizon', type=int, default=1,
    help='number of simulated lookahead steps per candidate (default 1 = '
         'the one-step rollout). H > 1 continues the chain inside the '
         'simulation: at every further simulated level the next block is '
         'chosen greedily from the model\'s own simulated sigma map, its '
         'features are sampled the same way, and the score is the total '
         'remaining sigma after the whole chain. Names are h<H>-tagged for '
         'H>1. Cost per row-chunk: 1 mu + K*(H sigma + (H-1) mu) forwards')
parser.add_argument(
    '--branch', type=str, default='greedy', choices=['greedy', 'full', 'fullseq'],
    help='inner-rollout rule for horizon>1: "greedy" (default) picks ONE '
         'block per simulated level from the simulated sigma map; "full" '
         '(H=2 only) enumerates the top-M2 level-2 blocks of each chain\'s '
         'own SIMULATED map (c2 set depends on the sample) and scores a '
         'level-1 candidate by the minimum residual sigma over its leaves '
         '(sample-then-optimize); "fullseq" (H=2 only, PAPER-EXACT) fixes '
         'the candidate sequence set BEFORE sampling - c2 = top-M2 of the '
         'sigma map of the deterministic MEAN context [obs, mu1(c1)] - and '
         'scores each (c1, c2) sequence by the residual of one sampled '
         'rollout (the paper\'s J-hat_K with K=1), then executes the argmin '
         'sequence (c1 + its best c2) as the pair')
parser.add_argument(
    '--batch_size', type=int, default=64,
    help='rows (image x candidate) forwarded together; selection is '
         'per-image independent, so this does not change the selection')
parser.add_argument(
    '--topk', type=int, default=16,
    help='pre-screen candidates with the CURRENT sigma map and simulate '
         'only the best K per image-step (default 16; 0 = simulate every '
         'valid candidate). Same caveat as the mu lookahead: with K=16 the '
         'choice differs from the full enumeration, so --topk is a cheaper '
         'variant of the strategy, not an approximation of it')
parser.add_argument(
    '--preload', action='store_true',
    help='preload the normalized images into CPU RAM (~0.6 MB/image) once '
         'so the selection loop does not re-decode image files at every step')
parser.add_argument(
    '--outdir', type=str, default='selected_patch_index',
    help='output directory for the selection/stats JSONs')

if __name__ == "__main__":

    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.horizon < 1:
        parser.error("--horizon must be >= 1")
    if args.branch in ("full", "fullseq") and args.horizon != 2:
        parser.error("--branch full/fullseq requires --horizon 2 (topk^H "
                     "explodes for H>=3; H=1 has no inner level to branch)")
    if args.branch == "fullseq" and args.samples != 1:
        parser.error("--branch fullseq is implemented for samples=1 "
                     "(K=1) only; use --branch full for the K>1 variant")

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

    # dedicated generator for the sampled rollouts: never touches the
    # Python-side RNG stream that draws the initial blocks
    sample_gen = torch.Generator(device=device)
    if args.seed is not None:
        sample_gen.manual_seed(int(args.seed))
    else:
        _sd = random.randrange(2 ** 31 - 1)
        sample_gen.manual_seed(_sd)
        print(f"(sampling generator seed: {_sd})")

    # optional preload
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
    # random start ONLY (same protocol and RNG call order as the other
    # random-start strategies: random.choice(n_pos) twice per image, drawn
    # before anything else touches the Python RNG)
    contexts = []
    observed = np.zeros((N, 196), dtype=bool)
    for n in range(N):
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

        # -- current sigma map: used for the --topk pre-screening, the
        #    candidate-feature sampling std AND the stats; one sigma
        #    forward per image
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

        # -- simulate every candidate with a --horizon-step cascade rollout:
        #    level 1 = the candidate itself (features SAMPLED from the
        #    predictive distribution: mu + sqrt(sigma_map) * eps); at every
        #    FURTHER level the next block is chosen GREEDILY from the
        #    model's own simulated sigma map, sampled the same way, and the
        #    score is the total remaining sigma after the whole chain.
        #    Anti-peeking holds: only predicted/sampled features and the
        #    model's own maps enter the simulation. horizon=1 reduces to
        #    the previous one-step rollout.
        rep, cand, img_seg = [], [], []
        for ai, valid in enumerate(valid_used):
            img_seg.append((len(rep), len(valid)))
            rep.extend([ai] * len(valid))
            cand.extend(valid)
        reps = rep                            # flattened image index per row
        cands = cand
        seg_bounds = img_seg
        # pair mode: the level-2 block that achieved the residual argmin in
        # the FIRST rollout chain, per level-1 row (observed for real when
        # --branch full / horizon > 1); global row alignment
        pair_c2 = [None] * len(reps)
        windows_t = torch.tensor(window_matrix, dtype=torch.long,
                                 device=device)        # (n_pos^2, b^2)
        scores = []
        for r0 in range(0, len(reps), args.batch_size):
            rsl = slice(r0, r0 + args.batch_size)
            rr = reps[rsl]
            cc = cands[rsl]
            R = len(rr)
            rr_t = torch.tensor(rr, device=device)
            x_ctx = torch.stack([feats[ai] for ai in rr])  # (R, k, D)
            with torch.no_grad():
                # level 1: predicted FEATURES of the (virtual) candidate
                # block, mean reused across the K chains
                cand_idx = torch.tensor(window_matrix[cc],
                                        dtype=torch.long, device=device)
                mu1 = predictor(
                    x_ctx,
                    [torch.tensor(ctx_mat[rr], dtype=torch.long,
                                  device=device)],
                    [cand_idx])                           # (R, b^2, D)
                # sampling std at the candidate patches: sqrt of the CURRENT
                # sigma-map values. The sigma head outputs the VARIANCE
                # (the train loss e/v + log v matches a Gaussian NLL with
                # variance = output), so the sampling amplitude is the
                # square root, not the raw value
                std1 = (sigma_cur[rr_t]
                        .gather(1, cand_idx)
                        .clamp_min(0.0).sqrt().unsqueeze(-1))   # (R, b^2, 1)
                ctx_t = torch.tensor(ctx_mat[rr], dtype=torch.long,
                                     device=device)
                # baseline occupancy = the observed context only
                occ0 = torch.zeros(R, 196, dtype=torch.bool, device=device)
                occ0[rows_ar(R), ctx_t] = True

                acc = torch.zeros(R, device=device)
                if args.horizon > 1 and args.branch == "full":
                    # ---------- FULL BRANCHING (level 2 enumerated) --------
                    # leaf = (image, level-1 block c1, level-2 block c2):
                    # the score of c1 is the MINIMUM residual sigma over its
                    # level-2 options (the exact 2-step objective; ties with
                    # an equal number of simulated blocks per leaf). The
                    # inner candidate set is re-derived from each chain's
                    # own simulated map, so chains are independent.
                    pos1 = torch.cat([ctx_t, cand_idx], dim=1)
                    # second observed block per level-1 row (filled by the
                    # first chain's residual-argmin leaf)
                    pair_c2_local = [None] * R
                    for _ in range(args.samples):
                        f1 = mu1 + std1 * torch.randn(
                            mu1.shape, generator=sample_gen, device=device)
                        sim1 = torch.cat([x_ctx, f1], dim=1)
                        sig1 = sigma_head(sigma_predictor(
                            sim1, [pos1],
                            [full_idx.unsqueeze(0).repeat(R, 1)]))
                        if sig1.dim() == 3:
                            sig1 = sig1.squeeze(-1)       # (R, 196)
                        occ1 = occ0.clone()
                        occ1[rows_ar(R), cand_idx] = True
                        overlap = occ1[:, windows_t].any(dim=2)
                        have = ~overlap                   # (R, n_pos^2)
                        pooled = (F.avg_pool2d(
                            sig1.reshape(-1, 1, 14, 14),
                            kernel_size=block_size, stride=1)
                            .reshape(R, n_pos, n_pos)
                            .transpose(1, 2).reshape(R, -1))
                        pooled = pooled.masked_fill(~have, float("-inf"))
                        # leaf list: per level-1 row, up to --topk inner
                        # candidates (finite pooled sigma = valid anchor)
                        rows2, c2list = [], []
                        for r1 in range(R):
                            kk = int(min(args.topk, have[r1].sum().item()))
                            if kk == 0:
                                continue   # no valid inner block for this leaf
                            sel = torch.topk(pooled[r1], kk).indices.tolist()
                            rows2.extend([r1] * kk)
                            c2list.extend(sel)
                        # per-row fallback: no inner block (image nearly
                        # saturated) -> score c1 on the level-1 residual
                        chain_min = sig1.masked_fill(occ1, 0.0).sum(dim=1)
                        if rows2:
                            r1_idx = torch.tensor(rows2, dtype=torch.long,
                                                  device=device)
                            w2 = windows_t[torch.tensor(
                                c2list, dtype=torch.long, device=device)]
                            sim_base = sim1[r1_idx]       # (L, k+b^2, D)
                            pos1_exp = pos1[r1_idx]       # (L, k+b^2)
                            sig1_exp = sig1[r1_idx]       # (L, 196)
                            mu2 = predictor(sim_base, [pos1_exp], [w2])
                            std2 = (sig1_exp.gather(1, w2)
                                    .clamp_min(0.0).sqrt().unsqueeze(-1))
                            leaf = torch.cat(
                                [sim_base,
                                 mu2 + std2 * torch.randn(
                                     mu2.shape, generator=sample_gen,
                                     device=device)], dim=1)
                            pos2 = torch.cat([pos1_exp, w2], dim=1)
                            occ2 = occ1[r1_idx]
                            sig2 = sigma_head(sigma_predictor(
                                leaf, [pos2],
                                [full_idx.unsqueeze(0).repeat(occ2.shape[0],
                                                              1)]))
                            if sig2.dim() == 3:
                                sig2 = sig2.squeeze(-1)
                            leaf_occ = occ2.clone()
                            leaf_occ[rows_ar(occ2.shape[0]), w2] = True
                            leaf_score = sig2.masked_fill(leaf_occ, 0).sum(1)
                            # residual-argmin c2 per level-1 row from the
                            # FIRST chain (= the second block of the
                            # actually observed pair; exact for the default
                            # K=1). rows2 is grouped contiguously per r1.
                            if _ == 0:
                                i = 0
                                while i < len(rows2):
                                    r1 = rows2[i]
                                    j = i
                                    while (j < len(rows2)
                                           and rows2[j] == r1):
                                        j += 1
                                    m = min(range(i, j),
                                            key=lambda t: float(
                                                leaf_score[t]))
                                    pair_c2_local[r1] = c2list[m]
                                    i = j
                            chain_min.scatter_reduce_(
                                0, r1_idx, leaf_score, reduce="amin",
                                include_self=True)
                        acc = acc + chain_min
                    pair_c2[r0:r0 + R] = pair_c2_local
                elif args.horizon > 1 and args.branch == "fullseq":
                    # ---------- PAPER-EXACT FIXED SEQUENCE SET (J-hat_K) -------
                    # Candidate SEQUENCES (c1, c2) are FIXED before sampling:
                    # c2 = top-M2 of the sigma map of the DETERMINISTIC mean
                    # context [obs, mu1(c1)]. Each sequence is scored by the
                    # residual of K sampled rollouts (K=1 here; K>1 would average
                    # chains), and the argmin sequence (c1 + its c2) is executed
                    # as the pair. This keeps the c2 candidate set independent of
                    # the sample and K-independent per sequence, matching the
                    # paper's J-hat_K(B) definition (K=1; paper-exact).
                    pos1 = torch.cat([ctx_t, cand_idx], dim=1)
                    pair_c2_local = [None] * R
                    # deterministic mean-context sigma map (chain-independent)
                    sim_mean = torch.cat([x_ctx, mu1], dim=1)
                    sig_mean = sigma_head(sigma_predictor(
                        sim_mean, [pos1],
                        [full_idx.unsqueeze(0).repeat(R, 1)]))
                    if sig_mean.dim() == 3:
                        sig_mean = sig_mean.squeeze(-1)       # (R, 196)
                    occ1 = occ0.clone()
                    occ1[rows_ar(R), cand_idx] = True
                    overlap = occ1[:, windows_t].any(dim=2)
                    have = ~overlap                   # (R, n_pos^2)
                    pooled_mean = (F.avg_pool2d(
                        sig_mean.reshape(-1, 1, 14, 14),
                        kernel_size=block_size, stride=1)
                        .reshape(R, n_pos, n_pos)
                        .transpose(1, 2).reshape(R, -1))
                    pooled_mean = pooled_mean.masked_fill(~have, float("-inf"))
                    rows2, c2list = [], []
                    for r1 in range(R):
                        kk = int(min(args.topk, have[r1].sum().item()))
                        if kk == 0:
                            continue   # no valid inner block for this leaf
                        sel = torch.topk(pooled_mean[r1], kk).indices.tolist()
                        rows2.extend([r1] * kk)
                        c2list.extend(sel)
                    # fallback residual on the level-1 map (no valid inner block)
                    chain_min = sig_mean.masked_fill(occ1, 0.0).sum(dim=1)
                    if rows2:
                        r1_idx = torch.tensor(rows2, dtype=torch.long,
                                              device=device)
                        w2 = windows_t[torch.tensor(
                            c2list, dtype=torch.long, device=device)]
                        # one sampled rollout (K=1; chain rule): f1 sampled from
                        # the CURRENT sigma map, then f2 sampled using the
                        # simulated map's sigma at the candidate (variance)
                        eps1 = torch.randn(mu1.shape, generator=sample_gen,
                                           device=device)
                        f1 = mu1 + std1 * eps1
                        sim1 = torch.cat([x_ctx, f1], dim=1)
                        sig1 = sigma_head(sigma_predictor(
                            sim1, [pos1],
                            [full_idx.unsqueeze(0).repeat(R, 1)]))
                        if sig1.dim() == 3:
                            sig1 = sig1.squeeze(-1)       # (R, 196)
                        sim_base = sim1[r1_idx]           # (L, k+b^2, D)
                        pos1_exp = pos1[r1_idx]            # (L, k+b^2)
                        sig1_exp = sig1[r1_idx]            # (L, 196)
                        mu2 = predictor(sim_base, [pos1_exp], [w2])
                        std2 = (sig1_exp.gather(1, w2)
                                .clamp_min(0.0).sqrt().unsqueeze(-1))
                        leaf = torch.cat(
                            [sim_base,
                             mu2 + std2 * torch.randn(
                                 mu2.shape, generator=sample_gen,
                                 device=device)], dim=1)
                        pos2 = torch.cat([pos1_exp, w2], dim=1)
                        occ2 = occ1[r1_idx]
                        sig2 = sigma_head(sigma_predictor(
                            leaf, [pos2],
                            [full_idx.unsqueeze(0).repeat(occ2.shape[0], 1)]))
                        if sig2.dim() == 3:
                            sig2 = sig2.squeeze(-1)
                        leaf_occ = occ2.clone()
                        leaf_occ[rows_ar(occ2.shape[0]), w2] = True
                        leaf_score = sig2.masked_fill(leaf_occ, 0).sum(1)
                        # residual-argmin c2 per level-1 row -> the second block
                        # of the actually observed pair
                        i = 0
                        while i < len(rows2):
                            r1 = rows2[i]
                            j = i
                            while j < len(rows2) and rows2[j] == r1:
                                j += 1
                            m = min(range(i, j),
                                    key=lambda t: float(leaf_score[t]))
                            pair_c2_local[r1] = c2list[m]
                            i = j
                        chain_min.scatter_reduce_(
                            0, r1_idx, leaf_score, reduce="amin",
                            include_self=True)
                    acc = acc + chain_min
                    pair_c2[r0:r0 + R] = pair_c2_local
                else:
                    # ---------- greedy-inner cascade (default) ------------
                    for _ in range(args.samples):
                        # --- one independent rollout chain ------------------
                        occ = occ0.clone()
                        sim = torch.cat(
                            [x_ctx,
                             mu1 + std1 * torch.randn(mu1.shape,
                                                      generator=sample_gen,
                                                      device=device)], dim=1)
                        pos = torch.cat([ctx_t, cand_idx], dim=1)  # (R, k+b^2)
                        occ[rows_ar(R), cand_idx] = True
                        sig = sigma_head(sigma_predictor(
                            sim, [pos],
                            [full_idx.unsqueeze(0).repeat(R, 1)]))
                        if sig.dim() == 3:
                            sig = sig.squeeze(-1)             # (R, 196)
                        # --- inner rollout: levels 2..horizon ---------------
                        for _lvl in range(2, args.horizon + 1):
                            overlap = (occ[:, windows_t]
                                       .any(dim=2))            # (R, n_pos^2)
                            have = ~overlap
                            if not bool(have.any()):
                                # no valid anchor left for ANY row: stop the
                                # chain and score with the map built so far
                                break
                            # greedy inner pick from the SIMULATED sigma map
                            pooled = F.avg_pool2d(
                                sig.reshape(-1, 1, 14, 14),
                                kernel_size=block_size, stride=1)
                            pooled = (pooled.reshape(R, n_pos, n_pos)
                                            .transpose(1, 2)
                                            .reshape(R, -1))   # le-major order
                            choose = pooled.masked_fill(
                                ~have, float("-inf")).argmax(dim=1)
                            w_i = windows_t[choose]            # (R, b^2)
                            mu_i = predictor(sim, [pos], [w_i])  # (R, b^2, D)
                            std_i = (sig.gather(1, w_i)
                                        .clamp_min(0.0).sqrt()
                                        .unsqueeze(-1))        # (R, b^2, 1)
                            sim = torch.cat(
                                [sim,
                                 mu_i + std_i * torch.randn(
                                     mu_i.shape, generator=sample_gen,
                                     device=device)], dim=1)
                            pos = torch.cat([pos, w_i], dim=1)
                            occ[rows_ar(R), w_i] = True
                            sig = sigma_head(sigma_predictor(
                                sim, [pos],
                                [full_idx.unsqueeze(0).repeat(R, 1)]))
                            if sig.dim() == 3:
                                sig = sig.squeeze(-1)
                        # score = total sigma over the still-unobserved patches
                        # (occupancy includes the chain's simulated blocks)
                        acc = acc + sig.masked_fill(occ, 0.0).sum(dim=1)
                scores.append(acc / args.samples)

        scores = torch.cat(scores)                        # (rows,)

        # -- pick, per image, the candidate with the smallest remaining sigma
        for ai, i in enumerate(active):
            start, cnt = seg_bounds[ai]
            if cnt == 0:
                continue
            arg = int(scores[start:start + cnt].argmin().item())
            row1 = start + arg
            best = cands[row1]
            win = anchor_windows[best]
            contexts[i].extend(win)
            observed[i, win] = True
            # pair mode (--branch full, horizon > 1): the argmin level-2
            # block computed for this rollout is observed as well, so the
            # selection advances 2 blocks per real step
            if args.horizon > 1 and args.branch in ("full", "fullseq"):
                c2a = pair_c2[row1]
                if c2a is not None:
                    win2 = anchor_windows[c2a]
                    contexts[i].extend(win2)
                    observed[i, win2] = True
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

    # K/H/mode-tagged variant name: K=1, H=1 -> "lookahead_sample" (the
    # pure one-step sampling variant), K>1 -> "lookahead_sample<K>",
    # H>1 appends h<H>, and full branching (pair observation) appends
    # "pair" instead of a branch tag
    # (e.g. K=1, H=2, greedy-inner -> "lookahead_sampleh2";
    #       K=1, H=2, full / pair -> "lookahead_sampleh2pair") so every
    # (K, H, branch) artefact coexists
    if args.samples == 1:
        base = "lookahead_sample"
    else:
        base = f"lookahead_sample{args.samples}"
    if args.horizon > 1 and args.branch in ("full", "fullseq"):
        base = f"{base}h{args.horizon}"
    if args.branch == "full" and args.horizon > 1:
        base = f"{base}pair"
    elif args.branch == "fullseq" and args.horizon > 1:
        base = f"{base}seq"
    suffix = base
    with open(os.path.join(args.outdir,
              f"block_{block_size}_tmax_{T}_test_{suffix}.json",
              ), "w") as f:
        json.dump(contexts, f)

    # -- stats file ("stats_" prefix: never matched by the eval glob
    #    block_{b}_tmax_{T}_test*.json)
    stats = {
        "meta": {"strategy": suffix, "b": block_size, "T": T,
                 "seed": args.seed, "start": "random", "topk": args.topk,
                 "samples": args.samples, "horizon": args.horizon,
                 "branch": args.branch, "N": N},
        "sigma_sum": sigma_sum_traj,
        "chosen_sigma_rank": chosen_rank,
        "n_valid_anchors": n_valid_traj,
    }
    with open(os.path.join(args.outdir,
              f"stats_block_{block_size}_tmax_{T}_test_{suffix}.json",
              ), "w") as f:
        json.dump(stats, f)

    print(f"saved selection+stats -> {args.outdir} "
          f"({suffix}, samples={args.samples}, horizon={args.horizon}, "
          f"branch={args.branch})")
