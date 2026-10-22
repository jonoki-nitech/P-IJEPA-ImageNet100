# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# =============================================================================
# src/train_det.py — DETERMINISTIC I-JEPA BASELINE
#
#   A copy of src/train.py (kept byte-identical wherever possible; original
#   file untouched) that trains the original, deterministic I-JEPA: the
#   mu-predictor is the predictor, the sigma head is never trained, and the
#   loss is plain MSE (== the probabilistic loss with sigma == 1, so the mu
#   paths of both models are directly comparable):
#
#       loss = mean( (z - h)^2 )
#
#   The mu-predictor additionally gets Dropout (configs 'meta.pred_depth'-
#   style key: meta.predictor_dropout, default 0.1) placed exactly like the
#   vanilla ViT predictor's drop_rate: two Dropouts per block (after the
#   attention projection and after both MLP linear layers). This enables
#   MC-Dropout inference with this checkpoint later: multiple noised
#   predictor forwards at test time give an empirical variance as the
#   uncertainty for the DET variant.
#
#   The sigma modules (sigma_predictor, sigma_head) are STILL constructed
#   and CHECKPOINTED so loaders written for the probabilistic format do not
#   crash, but they are never forwarded or trained (their weights stay at
#   random init). Consequently:
#     * they are NOT wrapped in DistributedDataParallel (avoids unused-
#       parameter handling in DDP; they are plain modules in save/load);
#     * the sigma branches of the stock probabilistic evaluation pipeline
#       (sigma-based selection, calibration of the sigma head) MUST NOT be
#       used with this checkpoint — evaluate the DET model through
#       MC-Dropout variance instead.
#
#   Other copy-only supportive adjustments (all no-ops on the GPU machine):
#     * STL10 monitoring: STL10Eval gained an optional root= argument
#       (stl10_eval.py); here the root comes from the STL10_ROOT env var
#       (falls back to the original GPU-machine path) and, when the data is
#       missing at all, the monitoring is skipped with a logged note instead
#       of killing the training;
#     * when nccl is unavailable (local CPU machines), the process group
#       falls back to gloo (world_size 1) so smoke tests can run.
#
#   Entry point: main_det.py (mirror of main.py; CLI identical).
#   Config: configs/configs_det.yaml (production; logging folder/tag and
#   meta.predictor_dropout changed so the probabilistic checkpoint and the
#   DET checkpoint never collide).
# =============================================================================

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k

from src.helper import (
    load_checkpoint,
    init_opt)

from src.models import vision_transformer as vit
from src.utils.tensors import trunc_normal_
from src.transforms import make_transforms
from stl10_eval import STL10Eval
import pandas as pd

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_model_det(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    sigma_depth=2,
    pred_emb_dim=384,
    predictor_dropout=0.1
):
    """Same architecture as src.helper.init_model, except the mu-predictor
    gets drop_rate = predictor_dropout (MC-Dropout baseline). The sigma
    modules are built identically but are never trained by train_det."""
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
    sigma_predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=sigma_depth,
        num_heads=encoder.num_heads)
    sigma_head = torch.nn.Sequential(torch.nn.Linear(encoder.embed_dim, 1), torch.nn.Softplus())

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

    for m in sigma_predictor.modules():
        init_weights(m)

    for m in sigma_head.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    sigma_predictor.to(device)
    sigma_head.to(device)
    logger.info(encoder)
    logger.info('predictor dropout rate: %.3f' % predictor_dropout)
    return encoder, predictor, sigma_predictor, sigma_head


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    sigma_depth = args['meta']['sigma_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    # -- DETERMINISTIC BASELINE: dropout rate for the mu-predictor
    #    (MC-Dropout at test time); the vanilla ViT predictor's drop_rate is
    #    reused, so no model code is touched
    predictor_dropout = float(args['meta'].get('predictor_dropout', 0.1))
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    backward_per_step = args['data']['backward_per_step']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    os.makedirs(folder, exist_ok=True)

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        # -- DETERMINISTIC BASELINE: gloo fallback for machines without nccl
        #    (local CPU smoke tests); on the GPU machine the process group is
        #    already initialized (nccl) inside main_det.py
        try:
            torch.distributed.init_process_group(
                backend='gloo',
                world_size=world_size,
                rank=rank)
            logger.info('initialized gloo process group (fallback)')
        except Exception as e:
            logger.info(f'gloo fallback failed ({e}); continuing single-process')
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor, sigma_predictor, sigma_head = init_model_det(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        sigma_depth=sigma_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        predictor_dropout=predictor_dropout)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(unsupervised_loader) // backward_per_step

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        sigma_predictor=sigma_predictor,
        sigma_head=sigma_head,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)

    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    # -- DETERMINISTIC BASELINE: the sigma modules are NOT wrapped in DDP
    #    (never forwarded -> avoids the unused-parameter question entirely;
    #    plain modules, checkpointed at random init for format compatibility)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, sigma_predictor, sigma_head, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            sigma_predictor=sigma_predictor,
            sigma_head=sigma_head,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(int(start_epoch*ipe)):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'sigma_predictor': sigma_predictor.state_dict(),
            'sigma_head': sigma_head.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    def stl_evaluation(target_encoder, results, folder):
        if rank == 0 and stl10_eval is not None:
            with torch.no_grad():
                model_for_eval = getattr(target_encoder, "module", target_encoder)
                try:
                    acc = stl10_eval.evaluate(model_for_eval)
                except:
                    print("error!!!")
            results['epoch'].append(epoch+1)
            results['stl10_acc'].append(acc)
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(folder, "stl10_acc_monitoring.csv"), index=False)


    # -- TRAINING LOOP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # -- DETERMINISTIC BASELINE: root overridable; monitoring skipped
    #    gracefully when the STL10 data is not available on this machine
    try:
        stl10_eval = STL10Eval(device=f"cuda:{local_rank}")
        stl10_note = 'STL10 monitoring enabled'
    except Exception as e:
        stl10_eval = None
        stl10_note = f'STL10 monitoring disabled ({e})'
    logger.info(stl10_note)
    # in100_eval = IN100Eval(device=f"cuda:{local_rank}")
    results = {'epoch':[], 'stl10_acc':[]}


    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():

                if (itr+1) %  backward_per_step == 0:
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()
                else:
                    _new_lr = 0
                    _new_wd = 0
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    feature = encoder(imgs, masks_enc)
                    # -- DETERMINISTIC BASELINE: mu-predictor only; the
                    #    sigma predictor/head are never forwarded
                    z = predictor(feature, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    # -- DETERMINISTIC BASELINE: plain MSE; identical to the
                    #    probabilistic loss with sigma == 1 (comparable mu)
                    loss = F.mse_loss(z, h, reduction='none').mean()
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h) / backward_per_step


                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    if (itr+1) % backward_per_step == 0:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss.backward()
                    if (itr+1) % backward_per_step == 0:
                        optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())

                if (itr+1) % backward_per_step == 0:
                    optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                if (itr+1) % backward_per_step == 0:
                    with torch.no_grad():
                        m = next(momentum_scheduler)
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   (torch.cuda.max_memory_allocated() / 1024.**2
                                    if torch.cuda.is_available() else 0.0),
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)

        # -- Evaluation using STL10
        if (epoch + 1) % 10 == 0:
            stl_evaluation(target_encoder, results, folder)
            # in100_evaluation(target_encoder, results, folder)

if __name__ == "__main__":
    main()
