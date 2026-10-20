# Attention-Map Entropy (AME; Pardyl et al., IJCAI 2023,
# https://github.com/apardyl/AME) patch selection, adapted to this repo's
# conventions as an 8th comparison strategy.
#
# Same structure as the other select_and_save_patch_index_*.py scripts:
# generate the observed-index JSON here, then evaluate reconstruction error
# with the existing test_mae_vs_t.py engine (which auto-detects the JSON by
# name and plots the strategy as one more line).
#
# Method (official AttentionGlimpseSelector, adapted):
#   * first block: uniformly random per image (default, seeded with --seed;
#     drawn with the same python-random stream and call order (left, then
#     top) as the sigma-based / lookahead / random strategies, so for a given
#     --seed the initial blocks are IDENTICAL across all random-initial
#     strategies). The official zero step of the reference code — the decoder
#     runs once on mask tokens only (empty context) to produce an
#     image-independent attention map, giving the same first block for every
#     image — is available with --init zero;
#   * each step: run the MAE forward with the current context and take the
#     softmax attention of one decoder block (default: last decoder layer);
#     the attention rows (queries) of already-observed patches are zeroed and
#     entropy = sum over (heads, keys) of -a * log2(a) (nan -> 0) is computed
#     per unobserved patch; pooled with avg_pool2d(kernel=b, stride=1), the
#     b x b window with the largest pooled entropy is observed next;
#   * truncation: when no fully-unobserved b x b window remains — i.e. the
#     selection runs to its natural saturation at most 196/b^2 blocks
#     (no T cap: the maximum observable block count). The official AME
#     allows re-observing patches; here the selection must stay
#     duplicate-free to match the other strategies and the t-sweep
#     protocol (the evaluation aligns all variants at their smallest
#     budget, capped by T).
#
# The attention maps are obtained through the officially supported
# output_attentions=True path of the HF decoder (no model code is modified).
# The attention-based selection itself is deterministic; --seed only drives
# the random first block.
#
# Output: selected_patch_index/block_{b}_tmax_{T}_test_ame.json
# (same naming rule as the other strategies; run.sh passes T = 196/b^2).
#
# Usage:
#   python select_and_save_patch_index_ame.py --fname configs/configs.yaml --b 3
#   python select_and_save_patch_index_ame.py --N 2 --b 2     # smoke test

import argparse
import json
import os
import pprint
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
from transformers import ViTMAEForPreTraining
import yaml

from src.datasets.imagenet1k import ImageNet

N_PATCHES = 196
GRID = 14
VARIANT = "ame"


# --------------------------------------------------------------------------- #
#  forward with attention (same math as test_mae_vs_t.forward_with_mask_idx,
#  plus output_attentions=True)
# --------------------------------------------------------------------------- #
def select_tokens(x, idx):
    """Return tokens at the given indices, preserving the index order.
    x: (B, tokens, c), idx: (B, k)"""
    idx = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return x.gather(dim=1, index=idx)


def forward_with_attn(model, img, mask_idx, target_idx, attention_layer):
    """MAE forward with per-image context indices; returns (logits,
    rec_patches_for_target, decoder attention of the chosen layer).

    mask_idx may be empty ((B, 0)) for the zero step: the decoder then sees
    only the cls token + mask tokens (no context)."""
    with torch.no_grad():
        idx = torch.cat([mask_idx, target_idx], dim=1)
        k = mask_idx.shape[1]

        pos_embs = model.vit.embeddings.position_embeddings
        embs = model.vit.embeddings.patch_embeddings(img)
        embs = embs + pos_embs[:, 1:, :]
        embs = select_tokens(embs, mask_idx)
        ids_restore = idx.argsort(1)

        cls_token = model.vit.embeddings.cls_token + pos_embs[:, :1, :]
        cls_tokens = cls_token.expand(embs.shape[0], -1, -1)
        embs = torch.cat((cls_tokens, embs), dim=1)

        encoder_output = model.vit.encoder(embs)
        sequence_output = encoder_output[0]
        sequence_output = model.vit.layernorm(sequence_output)

        dec = model.decoder(sequence_output, ids_restore,
                            output_attentions=True)
        # (B, heads, 1+196, 1+196) -> patch tokens only
        attn = dec.attentions[attention_layer][:, :, 1:, 1:]

        logits = dec.logits
        rec_patches = select_tokens(logits, idx[:, k:])

    return logits, rec_patches, attn


# --------------------------------------------------------------------------- #
#  AME selection
# --------------------------------------------------------------------------- #
def pooled_entropy(attn, observed, b):
    """Pooled attention-map entropy, following the official
    AttentionGlimpseSelector.

    attn: (B, heads, 196, 196) softmax attention (patch tokens only)
    observed: (B, 196) bool, True = already-observed patch
    Returns the pooled entropy map (B, 1, h', w') for anchoring a stride-1
    b x b window sliding over the patch grid (-inf where the window would
    overlap an already-observed patch)."""
    B = attn.shape[0]
    if observed is not None:
        # zero the attention rows (queries) of already-observed patches
        attn = attn * (~observed).view(B, 1, N_PATCHES, 1)
    ent = torch.nan_to_num(-attn * torch.log2(attn), nan=0.0)
    ent = ent.sum(dim=(1, 3)).reshape(B, 1, GRID, GRID)
    pooled = F.avg_pool2d(ent, kernel_size=b, stride=1)
    if observed is not None:
        # a b x b window is valid iff it contains no observed patch
        valid = F.max_pool2d(observed.float().view(B, 1, GRID, GRID),
                             kernel_size=b, stride=1)
        pooled = pooled.masked_fill(valid > 0, -torch.inf)
    return pooled


def block_to_index(left, top, block_size, grid=GRID):
    """Indices of the b x b block whose top-left corner is (left, top)."""
    rows = torch.arange(top, top + block_size)
    cols = torch.arange(left, left + block_size)
    return (rows[:, None] * grid + cols[None, :]).flatten().tolist()


def select_ame(model, xs_norm, b, device, attention_layer,
               batch_size=64, init="random"):
    """AME selection for block size b. Returns the flat observed patch lists
    per image (same JSON payload as the other selection strategies).

    init: 'random' = a uniformly random first block per image (default,
    seeded --seed, like the other strategies); 'zero' = the official zero
    step (decoder on mask tokens only; image-independent first block).

    No T cap: selection runs to natural saturation (at most 196 // b^2
    blocks; the evaluation aligns variants at their smallest budget)."""
    N = len(xs_norm)
    max_steps = N_PATCHES // (b * b)

    # ---- first block ------------------------------------------------------- #
    if init == "zero":
        # official zero step: decoder mask tokens only (image-independent)
        img0 = xs_norm[0].unsqueeze(0).to(device)
        empty_ctx = torch.zeros((1, 0), dtype=torch.long, device=device)
        all_idx = torch.arange(N_PATCHES, device=device).unsqueeze(0)
        _, _, attn = forward_with_attn(model, img0, empty_ctx, all_idx,
                                       attention_layer)
        pooled = pooled_entropy(attn, None, b)
        best = pooled.flatten(1).argmax(dim=1).item()
        Hp, Wp = pooled.shape[2], pooled.shape[3]
        top, left = np.unravel_index(best, (Hp, Wp))
        first = block_to_index(int(left), int(top), b)
        contexts = [list(first) for _ in range(N)]
    else:
        # random initial block per image, drawn with the same python-random
        # stream and call order (left, then top) as the sigma-based and
        # lookahead strategies: for a given --seed the initial blocks are
        # IDENTICAL across all random-initial strategies
        anchors = []
        for i in range(N):
            left = random.choice(range(GRID - b + 1))
            top = random.choice(range(GRID - b + 1))
            anchors.append((left, top))
        # NB: anchors rows are (left, top); block_to_index takes (left, top)
        contexts = [block_to_index(int(l_), int(t_), b)
                    for l_, t_ in anchors]
    observed_sets = [set(c) for c in contexts]
    active = list(range(N))
    steps = [1] * N

    # ---- iterative steps --------------------------------------------------- #
    step = 1
    while active and step < max_steps:
        next_blocks = {}
        for lo in range(0, len(active), batch_size):
            chunk = active[lo:lo + batch_size]
            x = torch.stack([xs_norm[i] for i in chunk]).to(device)
            ctx = [contexts[i] for i in chunk]
            mask_idx = torch.tensor(ctx, dtype=torch.long, device=device)
            target_rows = [[p for p in range(N_PATCHES)
                            if p not in observed_sets[i]] for i in chunk]
            target_idx = torch.tensor(target_rows, dtype=torch.long,
                                      device=device)
            _, _, attn = forward_with_attn(model, x, mask_idx, target_idx,
                                           attention_layer)
            observed = torch.zeros(len(chunk), N_PATCHES, dtype=torch.bool,
                                   device=device)
            observed[torch.arange(len(chunk), device=device)[:, None],
                     mask_idx] = True
            pooled = pooled_entropy(attn, observed, b)
            A, _, Hp, Wp = pooled.shape
            pooled_flat = pooled.reshape(A, -1)
            best = pooled_flat.argmax(dim=1)
            best_val = pooled_flat.max(dim=1).values
            for row, i in enumerate(chunk):
                if not torch.isfinite(best_val[row]):
                    continue      # saturation: no fully-unobserved window
                t_, l_ = np.unravel_index(best[row].item(), (Hp, Wp))
                next_blocks[i] = block_to_index(int(l_), int(t_), b)
        still = []
        for i in active:
            window = next_blocks.get(i)
            if window is None:
                continue          # this image is done (saturated)
            contexts[i].extend(window)
            observed_sets[i].update(window)
            steps[i] += 1
            still.append(i)
        active = still
        step += 1
        print(f"\r    {VARIANT}: step {step}/{max_steps} "
              f"(active {len(active)}/{N})", end="", flush=True)
    print()
    return contexts, steps


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str, default='configs/configs.yaml',
    help='name of config file to load')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='which device to use (falls back to cpu when cuda is unavailable; '
         "an explicit 'cpu' is honoured even on cuda machines)")
parser.add_argument(
    '--N', type=int, default=0,
    help='number of images to process (0 = all)')
parser.add_argument(
    '--b', type=int, default=3, help='block size')
parser.add_argument(
    '--attention-layer', type=int, default=-1,
    help='decoder block supplying the attention (0-based, -1 = last decoder '
         'block; the official AME uses its last of 8 blocks)')
parser.add_argument(
    '--T', type=int, default=32,
    help='tmax label in the output filename (should match the T of the '
         'evaluation run; selection itself always runs to saturation — '
         'no step cap)')
parser.add_argument(
    '--batch_size', type=int, default=64,
    help='images forwarded per step (selection is per-image independent; '
         'only a speed knob)')
parser.add_argument(
    '--init', type=str, default='random', choices=['random', 'zero'],
    help="first block: 'random' = uniformly random per image (default; "
         "same convention as the other strategies); 'zero' = official "
         "AttentionGlimpseSelector zero step (decoder on mask tokens "
         "only, image-independent)")
parser.add_argument(
    '--seed', type=int, default=0, help='seed for the random first block')
parser.add_argument(
    '--outdir', type=str, default='selected_patch_index',
    help='output directory for the selection JSONs (default: '
         'selected_patch_index; the per-seed runs of revision/run.sh write '
         'to selected_patch_index/seed_<SEED>)')

if __name__ == "__main__":

    args = parser.parse_args()

    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device

    # the random first block uses the same python-random stream and call
    # order (left, then top per image) as the other random-initial
    # strategies, so the initial blocks coincide for a given --seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']
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

    xs_norm = []
    for i in range(N):
        img, _ = test_dataset[i]
        xs_norm.append(img)
    print(f"images: {N}, block size: {args.b}, device: {device}")

    # the MAE used by the reconstruction evaluation (eager attention so that
    # output_attentions is available)
    model = ViTMAEForPreTraining.from_pretrained(
        "facebook/vit-mae-base").to(device)
    model.eval()
    dec_depth = len(model.decoder.decoder_layers)
    if args.attention_layer == -1:
        attention_layer = dec_depth - 1
    else:
        attention_layer = args.attention_layer
        assert 0 <= attention_layer < dec_depth
    print(f"decoder depth: {dec_depth}, attention layer: {attention_layer}")

    b = args.b
    max_steps = N_PATCHES // (b * b)

    contexts, steps = select_ame(
        model, xs_norm, b, device, attention_layer,
        batch_size=args.batch_size, init=args.init)
    print(f"{VARIANT}: steps per image min={min(steps)}, max={max(steps)} "
          f"(cap={max_steps})")

    # uniqueness check (same assertion policy as the other strategies)
    for i, ctx in enumerate(contexts):
        assert len(set(ctx)) == len(ctx), f"duplicate patch in image {i}"

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"block_{b}_tmax_{args.T}_test_{VARIANT}.json")
    with open(out_path, "w") as f:
        json.dump(contexts, f)
    print(f"saved: {out_path}")
