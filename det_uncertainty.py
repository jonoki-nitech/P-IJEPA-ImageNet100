# det_uncertainty.py — shared loader + MC-Dropout forward for the
# DETERMINISTIC I-JEPA baseline checkpoint (results/det-pretrained/).
#
# The DET checkpoint (trained by main_det.py / src/train_det.py) contains:
#   encoder        : trained I-JEPA context encoder          (used, eval mode)
#   predictor      : trained mu-predictor WITH Dropout(0.1)  -> MC Dropout
#   target_encoder : trained EMA target encoder              (used, eval mode)
#   sigma_predictor / sigma_head : NEVER TRAINED (random init) -> NEVER
#                   loaded by this module (using them would produce
#                   garbage uncertainty; the DET uncertainty comes solely
#                   from MC-Dropout sampling of the mu-predictor).
#
# MC-Dropout (Gal & Ghahramani, 2016): at inference the predictor's Dropout
# modules are kept in train mode and the predictor is forwarded K times
# ("mc_k"), each with a fresh dropout mask. The K predicted feature maps are
# samples of an approximate posterior over predictor weights, so
#     var_dim  = mean_k (z_k - mean_k z_k)^2        (per patch, per dim)
#     var_pooled = mean over dims of var_dim         (per patch)
# is the epistemic uncertainty of the prediction; mean_k z_k is used as the
# prediction itself. (Numerically the loop accumulates sum / sum-of-squares
# instead of stacking all K maps, so memory stays 2 prediction maps.)
#
# Used by calibration_eval_mcdropout.py and
# select_and_save_patch_index_mcdropout.py.

import copy
import os

import torch
import torch.nn as nn

from src.models import vision_transformer as vit
from src.utils.tensors import trunc_normal_


def _strip_module_prefix(state_dict):
    return {k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


def _init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)


def load_det_models(checkpoint_path, params, device,
                    predictor_dropout=None):
    """Build the DET model triplet from `params` (a loaded config dict) and
    load the trained weights from `checkpoint_path`.

    predictor_dropout: if None, taken from params['meta']['predictor_dropout']
    (default 0.1). MUST match the value used at DET training time, otherwise
    the dropout structure of the loaded weights is meaningless (dropping out
    entirely at p=0 would make MC-Dropout degenerate: zero variance).

    Returns (encoder, predictor, target_encoder) in eval mode on `device`.
    """
    model_name = params['meta']['model_name']
    patch_size = params['mask']['patch_size']
    crop_size = params['data']['crop_size']
    pred_depth = params['meta']['pred_depth']
    pred_emb_dim = params['meta']['pred_emb_dim']
    if predictor_dropout is None:
        predictor_dropout = float(params['meta'].get('predictor_dropout', 0.1))

    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        drop_rate=predictor_dropout)
    target_encoder = copy.deepcopy(encoder)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)
    for m in predictor.modules():
        init_weights(m)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'DET checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    for key, module in [('encoder', encoder),
                        ('predictor', predictor),
                        ('target_encoder', target_encoder)]:
        msg = module.load_state_dict(_strip_module_prefix(checkpoint[key]))
        print(f'loaded {key}: {msg}')

    # -- sanity: the predictor really carries active Dropouts (the DET
    #    training structure), else MC-Dropout would silently degenerate
    n_drop = sum(1 for m in predictor.modules()
                 if isinstance(m, nn.Dropout) and m.p > 0)
    if predictor_dropout > 0 and n_drop < 2 * pred_depth:
        raise RuntimeError(
            f'predictor has only {n_drop} active Dropout modules '
            f'(expected >= {2 * pred_depth}); is this really a DET '
            f'(dropout-trained) checkpoint?')
    print(f'MC-Dropout: predictor_dropout={predictor_dropout}, '
          f'active Dropout modules={n_drop}')

    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)
    encoder.eval()
    predictor.eval()
    target_encoder.eval()
    return encoder, predictor, target_encoder


def set_predictor_dropout_mode(predictor, mode):
    """put the predictor's Dropout modules in `mode` ('train'/'eval'),
    leaving every other module untouched (LayerNorms etc. are
    mode-insensitive anyway)."""
    assert mode in ('train', 'eval')
    want = (mode == 'train')
    for m in predictor.modules():
        if isinstance(m, nn.Dropout):
            m.train(want)


def mc_predictor_forward(predictor, feature, mask_enc, mask_pred, mc_k):
    """MC-Dropout sampling of the DET predictor.

    encoder forward must be done OUTSIDE this function (once, deterministic).
    Here the predictor's Dropout modules are switched to train mode and the
    predictor is forwarded `mc_k` times per dropout mask, under no_grad.
    Memory is bounded: only running sums accumulate (no (K, B, P, D) stack).

    Returns:
      z_mean    (P, B, D)-shaped tensor -> actually (B_out, P, D): mean over
                the mc_k sampled predictions  (= the DET point prediction),
      var_pooled (B_out, P): average over the penultimate feat dim of the
                per-dim sample variance                        [uncertainty],
      var_dim (B_out, P, D): per-dim sample variance (coverage use).
    """
    set_predictor_dropout_mode(predictor, 'train')
    with torch.no_grad():
        z_sum = None
        z_sq = None
        for k in range(mc_k):
            z = predictor(feature, mask_enc, mask_pred)
            if z_sum is None:
                z_sum = torch.zeros_like(z)
                z_sq = torch.zeros_like(z)
            z_sum += z
            z_sq += z * z
    z_mean = z_sum / mc_k
    # E[z^2] - (E[z])^2 ; clamp tiny negatives from float rounding
    var_dim = (z_sq / mc_k - z_mean * z_mean).clamp_min_(0.0)
    var_pooled = var_dim.mean(dim=-1)
    set_predictor_dropout_mode(predictor, 'eval')
    return z_mean, var_pooled, var_dim
