# Calibration evaluation for the MC-Dropout uncertainty of the DETERMINISTIC
# I-JEPA baseline (DET model in results/det-pretrained/).
#
# Structure-mirror of calibration_eval.py (the sigma-head evaluation of
# Probabilistic I-JEPA) and it REUSES its analysis code directly (import):
# same reliability diagram (equal-count bins), Pearson / Spearman,
# mean(e)/mean(v), mean(e/v), log-log fit, the NLL proper-scoring-rule
# comparison (ours / homoscedastic / shuffled) and the 95% per-dim coverage.
# The difference is only in where the two measured quantities come from:
#
#   sigma version :  v   = trained sigma head, single deterministic forward
#                    err = || z - h ||^2 per patch (z: mu-predictor)
#   DET version   :  v   = K-sample MC-Dropout variance of the predictor
#                          (var over dropout masks, pooled over the feat dim)
#                    err = || mean_k z_k - h ||^2 per patch (z_mean used)
#   (h = target-encoder features, layer-normed; identical in both versions)
#
# The sigma modules of the DET checkpoint are random init and are NEVER
# loaded (see det_uncertainty.py). The uncertainty is the MC-Dropout
# epistemic variance; do not expect it to behave exactly like the trained
# heteroscedastic sigma head (it covers only model, not data, uncertainty),
# which is precisely what this evaluation quantifies.
#
# Usage examples:
#   python calibration_eval_mcdropout.py --fname configs/configs_local.yaml \
#       --N 2 --mc_k 4
#   python calibration_eval_mcdropout.py --fname configs/configs.yaml \
#       --N 500 --mc_k 16 --b 4 --positions 5:5
#   (analysis-only, pattern identical to calibration_eval.py)
#   python calibration_eval_mcdropout.py --points_pt \
#       images/calibration_det/calibration_points.pt
#
# All CLI flags, outputs (PNG diagram, CSV metrics, calibration_points.pt)
# and score semantics follow calibration_eval.py. Currently the calibration
# points carry the pooled MC variance under the same 'sigma' key, so the
# analysis-only mode of the original calibration_eval.py also works with it.

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms as T
import pprint

from src.datasets.imagenet1k import ImageNet
from src.masks.utils import apply_masks

from calibration_eval import block_to_index, analyze
from det_uncertainty import load_det_models, mc_predictor_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fname', type=str,
        help='name of config file to load (inference mode only; only the '
             'data/model-shape fields are read, the DET dropout rate comes '
             'from meta.predictor_dropout)',
        default='configs/configs_det.yaml')
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='which device to use on local machine (inference mode only)')
    parser.add_argument(
        '--model_path', type=str,
        help='path to the DET-trained checkpoint (inference mode only)',
        default='results/det-pretrained/jepa-det-latest.pth.tar')
    parser.add_argument(
        '--start_index', type=int, default=0,
        help='the first index of test data (inference mode only)')
    parser.add_argument(
        '--N', type=int, default=0,
        help='number of test images to evaluate (0 = all) (inference mode only)')
    parser.add_argument(
        '--b', type=int, default=4,
        help='block size (inference mode only)')
    parser.add_argument(
        '--positions', type=str, nargs='+', default=['5:5'],
        help='context block positions as "top:left" (patch grid coordinates); '
             'rows/cols covered are top..top+b-1 and left..left+b-1 '
             '(inference mode only)')
    parser.add_argument(
        '--mc_k', type=int, default=16,
        help='number of stochastic (dropout-on) predictor forwards per '
             'context; the MC-Dropout variance is estimated from these K '
             'samples (default 16)')
    parser.add_argument(
        '--dropout', type=float, default=None,
        help='override the predictor dropout rate (default: meta.'
             'predictor_dropout of the config, i.e. the training value 0.1; '
             'DO NOT change this unless you know why)')
    parser.add_argument(
        '--bins', type=int, default=10,
        help='number of equal-count bins for the reliability diagram')
    parser.add_argument(
        '--include_context', action='store_true',
        help='keep patches inside the observed block (default: excluded, '
             'matching the training target condition)')
    parser.add_argument(
        '--out_dir', type=str, default='images/calibration_det',
        help='output directory (the sigma version uses images/calibration; '
             'DET results go here to keep them apart)')
    parser.add_argument(
        '--points_pt', type=str, default='',
        help='path to an existing calibration_points.pt; if set, the script '
             'runs analysis only (no model loading, no inference) and never '
             'overwrites the input file')

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  analysis-only mode: load previously collected points
    # ------------------------------------------------------------------ #
    if args.points_pt:
        data = torch.load(args.points_pt, map_location="cpu")
        sigma_arr = data['sigma'].cpu().numpy()        # pooled MC variance
        err_arr = data['pred_error'].cpu().numpy()

        sample_ids = None
        if 'sample_ids' in data:
            sample_ids = np.asarray(data['sample_ids'])

        meta = {
            'n_images': len(set(sample_ids.tolist()))
            if sample_ids is not None else data.get('n_images', -1),
            'positions_str': data.get('positions_str', 'unknown'),
            'block_size': data.get('block_size', 'unknown'),
            'context_included': bool(data.get('include_context', False)),
            'coverage_count': data.get('coverage_count', None),
            'coverage_total': data.get('coverage_total', None),
        }
        print(f'[analysis-only mode] loading points from: {args.points_pt}')
        analyze(sigma_arr, err_arr, args.bins, args.out_dir, meta,
                save_points=False,
                source_note=f'points_pt={args.points_pt}')
        return

    # ------------------------------------------------------------------ #
    #  inference mode: collect (MC variance, residual) from the DET model
    # ------------------------------------------------------------------ #
    # -- load script args
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # -- device
    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device

    # -- DET model (encoder / dropout predictor / target encoder; sigma
    #    modules intentionally never built)
    encoder, predictor, target_encoder = load_det_models(
        checkpoint_path=args.model_path,
        params=params,
        device=device,
        predictor_dropout=args.dropout)
    print(f'mc_k = {args.mc_k}')

    # -- dataset (same test transform as calibration_eval.py)
    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']
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

    positions = []
    for pos in args.positions:
        top, left = pos.split(':')
        positions.append((int(top), int(left)))

    N = args.N if args.N > 0 else len(test_dataset)
    N = min(N, len(test_dataset) - args.start_index)

    os.makedirs(args.out_dir, exist_ok=True)

    sigmas_list = []      # pooled MC variance per patch   (matches 'sigma')
    errors_list = []      # ||z_mean - h||^2 per patch
    sample_ids = []
    pos_ids = []
    cover_count = 0
    cover_total = 0

    for n in range(N):
        i = args.start_index + n
        img, label = test_dataset[i]
        img = img.unsqueeze(0).to(device)

        for p_idx, (top, left) in enumerate(positions):
            context_indices_list = [block_to_index(left, top, args.b, args.b)]
            context_set = set(context_indices_list[0])

            target_indices_list = [list(range(196))]

            with torch.no_grad():
                mask_enc = [torch.tensor(context_indices_list).to(device)]
                mask_pred = [torch.tensor(target_indices_list).to(device)]

                feature = encoder(img, mask_enc)

                # MC-Dropout: K stochastic predictor forwards
                z_mean, var_pooled, var_dim = mc_predictor_forward(
                    predictor, feature, mask_enc, mask_pred, args.mc_k)

                h = target_encoder(img)
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                h = apply_masks(h, mask_pred)

                resid = z_mean - h
                err = (resid ** 2).mean(dim=-1)     # per-patch mean sq error

                # per-dim 95% coverage with the POOLED variance broadcast
                # over dims: |resid_d| <= 1.96 * sqrt(v_pooled)  — the same
                # semantics as the sigma-head version for comparability
                cover_count += (resid.abs()
                                <= 1.96 * torch.sqrt(var_pooled).unsqueeze(-1)) \
                    .sum().item()
                cover_total += resid.numel()

            sigma_np = var_pooled.flatten().float().cpu().numpy()
            err_np = err.flatten().float().cpu().numpy()
            keep = np.ones(196, dtype=bool)
            if not args.include_context:
                keep = np.array([i2 not in context_set for i2 in range(196)])

            sigmas_list.append(sigma_np[keep])
            errors_list.append(err_np[keep])
            sample_ids.extend([i] * int(keep.sum()))
            pos_ids.extend([p_idx] * int(keep.sum()))

        if (n + 1) % 25 == 0:
            print(f'\r  {n + 1}/{N} images', end='', flush=True)
    print()

    sigma_arr = np.concatenate(sigmas_list)
    err_arr = np.concatenate(errors_list)

    meta = {
        'n_images': N,
        'positions_str': ' '.join(args.positions),
        'block_size': args.b,
        'context_included': bool(args.include_context),
        'coverage_count': cover_count,
        'coverage_total': cover_total,
    }

    analyze(sigma_arr, err_arr, args.bins, args.out_dir, meta,
            save_points=True,
            source_note=(f'DET mc-dropout (model={args.model_path}, '
                         f'mc_k={args.mc_k})'))

    # -- save the collected points (same schema as calibration_eval.py; the
    #    pooled MC variance is stored under the 'sigma' key so that the
    #    analysis-only mode of BOTH scripts works)
    torch.save({
        'sigma': torch.tensor(sigma_arr),
        'pred_error': torch.tensor(err_arr),
        'sample_ids': sample_ids,
        'pos_ids': pos_ids,
        'n_images': N,
        'positions_str': ' '.join(args.positions),
        'block_size': args.b,
        'include_context': bool(args.include_context),
        'mc_k': args.mc_k,
        'coverage_count': meta['coverage_count'],
        'coverage_total': meta['coverage_total'],
    }, os.path.join(args.out_dir, 'calibration_points_det.pt'))
    print(f"saved: {os.path.join(args.out_dir, 'calibration_points_det.pt')}")


if __name__ == "__main__":
    main()
