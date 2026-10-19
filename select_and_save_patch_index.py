import torch
from torchvision import transforms as T
from einops.layers.torch import Rearrange
from torchvision.utils import save_image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import argparse
import yaml
from src.datasets.imagenet1k import make_imagenet1k, ImageNet
import pprint
from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
import json
import random
from tqdm import tqdm
import os

inv_normalize = T.Normalize(
    mean=[-m/s for m, s in zip((0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225))],
    std=[1/s for s in (0.229, 0.224, 0.225)]
)


def reconstruct_img_from_patch(ori_img, patch_index, patch_size=16):

    h = 224//patch_size
    w = 224//patch_size

    patchfy = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                        p1=patch_size,
                        p2=patch_size)

    unpatchfy = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                        p1=patch_size,
                        p2=patch_size,
                        h=h, w=w)

    patches = patchfy(inv_normalize(ori_img))

    for i in range(patches.shape[1]):
        if not i in patch_index:
            patches[:, i] = torch.zeros(patch_size**2*3)

    img = unpatchfy(patches)

    return img

def sigma_heatmap(sigma, patch_index=None, patch_size=16):

    h = 224//patch_size
    w = 224//patch_size

    if patch_index is not None:
        sigma[0, patch_index] = 0

    unpatchfy = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                        p1=1,
                        p2=1,
                        h=h, w=w)

    sigma = unpatchfy(sigma)

    sigma = F.interpolate(sigma, size=(224, 224))

    sigma = to_color_tensor(sigma.squeeze())

    return sigma.unsqueeze(0)

def to_color_tensor(x, cmap='viridis'):
    # x: (H, W)
    # x_min, x_max = x.min(), x.max()
    # print(x.max())
    x_min, x_max=0, 0.7
    x_norm = (x - x_min) / (x_max - x_min)
    colormap = cm.get_cmap(cmap)
    x_color = torch.from_numpy(colormap(x_norm.numpy())[:, :, :3])  # (H, W, 3)
    x_color = x_color.permute(2, 0, 1)  # (3, H, W)
    return x_color

def block_to_index(left, top, w, h):

    unpatchfy = Rearrange("(h w)->h w",
                        h=14, w=14)

    blocks = torch.arange(0, 196)
    blocks = unpatchfy(blocks)

    blocks = blocks[top:top+h, left:left+w]

    index = blocks.flatten().tolist()

    return index

def select_next_block(sigma, patch_size=16, k=4, stride=1):

    h = 224//patch_size
    w = 224//patch_size

    index = torch.arange(sigma.shape[1])

    sigma = sigma.reshape(1, 1, h, w)
    index = index.reshape(1, 1, h, w)

    sigma_avg = F.avg_pool2d(sigma, kernel_size=k, stride=1)
    max_idx_flat = torch.argmax(sigma_avg)
    max_idx_pooled = torch.unravel_index(max_idx_flat, sigma_avg.shape[-2:])

    top = max_idx_pooled[0] * stride
    left = max_idx_pooled[1] * stride
    # end_h = start_h + k
    # end_w = start_w + k

    return block_to_index(left, top, k, k)


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
    help='initial block: "random" (default, original behavior) or "center" '
         '(exact center block first, then sigma-based selection)')
parser.add_argument(
    '--seed', type=int, default=None,
    help='manual seed for the random initial block (and any other stochastic '
         'operation); if omitted, behavior is unchanged (non-deterministic)')
parser.add_argument(
    '--batch_size', type=int, default=64,
    help='images processed together per selection step (selection is per-'
         'image independent, so batching does not change the selection)')
parser.add_argument(
    '--preload', action='store_true',
    help='preload the normalized images into CPU RAM (~0.6 MB/image) once '
         'so the selection loop does not re-decode image files at every step')

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
    use_bfloat16 = params['meta']['use_bfloat16']
    model_name = params['meta']['model_name']
    load_model = params['meta']['load_checkpoint']
    r_file = params['meta']['read_checkpoint']
    copy_data = params['meta']['copy_data']
    pred_depth = params['meta']['pred_depth']
    sigma_depth = params['meta']['sigma_depth']
    pred_emb_dim = params['meta']['pred_emb_dim']
    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device
        # torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = params['data']['use_gaussian_blur']
    use_horizontal_flip = params['data']['use_horizontal_flip']
    use_color_distortion = params['data']['use_color_distortion']
    color_jitter = params['data']['color_jitter_strength']
    # --
    batch_size = params['data']['batch_size']
    pin_mem = params['data']['pin_mem']
    num_workers = params['data']['num_workers']
    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']
    crop_size = params['data']['crop_size']
    crop_scale = params['data']['crop_scale']
    # --

    # # -- MASK
    patch_size = params['mask']['patch_size']

    # dataset
    test_transform = T.Compose([
        # T.Resize(320),
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

    # model
    encoder, _, sigma_predictor, sigma_head = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        sigma_depth=sigma_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    del _

    # load pretrained model
    r_path = args.model_path
    checkpoint = torch.load(r_path, map_location="cpu")

    # -- loading encoder
    pretrained_dict = checkpoint['encoder']
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    msg = encoder.load_state_dict(new_state_dict)

    pretrained_dict = checkpoint['sigma_predictor']
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    msg = sigma_predictor.load_state_dict(new_state_dict)

    pretrained_dict = checkpoint['sigma_head']
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    msg = sigma_head.load_state_dict(new_state_dict)


    # inference
    encoder.eval()
    sigma_predictor.eval()
    sigma_head.eval()

    N = len(test_dataset)
    T = args.T
    block_size = args.b

    selected_index_list=[]

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
        imgs = [test_dataset[n][0] for n in tqdm(range(N), desc="preloading", leave=False)]

    # ---- batched sigma-based selection --------------------------------------
    # every image's selection is independent and each step of the loop gives
    # every active image exactly one more observed block (same context size),
    # so all active images can be forwarded together per chunk. An image is
    # frozen (stops observing) when no fully-unobserved b x b window remains.
    contexts = [list(a) for a in anchors]
    active = list(range(N))
    full_idx = torch.arange(196, dtype=torch.long, device=device)
    block_offsets = (torch.arange(block_size, device=device)[:, None] * 14 +
                     torch.arange(block_size, device=device)[None, :]).reshape(-1)

    # -- per-image stats: trajectory of the total sigma over the still-
    #    unobserved patches; sigma_sum_traj[i][j] = the sum given the first
    #    j+1 observed blocks (the value the strategy conditions on before
    #    each choice, plus a terminal value after the last block)
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
        #    them observe its own largest-sigma block
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
                sigma = sigma_head(sigma_predictor(feature,
                                                   [mask_ctx], [mask_pred]))
            if sigma.dim() == 3:     # (B, 196, 1) -> (B, 196)
                sigma = sigma.squeeze(-1)
            # observed patches never win the argmax (same as the -inf write)
            sigma = sigma.masked_fill(obs[lo:lo + len(chunk)], -torch.inf)

            # total sigma over the still-unobserved patches (trajectory point
            # for the context the strategy is about to extend)
            sig_sum = sigma.masked_fill(obs[lo:lo + len(chunk)],
                                        0.0).sum(dim=1)
            for row, i in enumerate(chunk):
                sigma_sum_traj[i].append(float(sig_sum[row]))

            # pooled sigma map, argmax per image (the same computation as
            # select_next_block, vectorized over the batch
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

    selected_index_list = contexts

    os.makedirs("selected_patch_index", exist_ok=True)

    suffix = "" if args.start == "random" else "_start_center"
    with open(f"selected_patch_index/block_{block_size}_tmax_{T}_test{suffix}.json", "w") as f:
        json.dump(selected_index_list, f)

    # -- stats file ("stats_" prefix: never matched by the eval glob
    #    block_{b}_tmax_{T}_test*.json, but picked up by the run.sh zip)
    stats = {
        "meta": {"strategy": "selected", "b": block_size, "T": T,
                 "seed": args.seed, "start": args.start, "N": N},
        # sigma_sum[i]: len = #observed blocks; entry j = total sigma over
        #   the unobserved patches given the first j+1 blocks
        "sigma_sum": sigma_sum_traj,
    }
    with open(f"selected_patch_index/"
              f"stats_block_{block_size}_tmax_{T}_test{suffix}.json",
              "w") as f:
        json.dump(stats, f)
