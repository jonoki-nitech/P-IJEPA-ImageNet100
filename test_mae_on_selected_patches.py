
import json
# from simple_ijepa.transformer import  VisionTransformer
import torch
from torch import nn
# from simple_ijepa.utils import trunc_normal_
from einops.layers.torch import Rearrange
from lars import LARS
import torch.nn.functional as F
import csv
from torchvision.utils import save_image
from tqdm import tqdm
# from mae import MAE
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.datasets.imagenet1k import make_imagenet1k, ImageNet
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
import pprint
from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
import json
import random
from tqdm import tqdm
import timm
from transformers import ViTMAEForPreTraining
import os

normalize = T.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
inv = T.Normalize(
    mean=[-m/s for m, s in zip((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))],
    std=[1/s for s in (0.229, 0.224, 0.225)]
)


class ImageNetWithMasks(ImageNet):
    def __init__(self, root, image_folder, train, copy_data, index_targets, input_patch_num=10, mask_list=None, blocksize=2, transform=None):
        super().__init__(root=root_path,
        image_folder=image_folder,
        transform=test_transform,
        train=train,
        copy_data=copy_data,
        index_targets=index_targets,)

        if mask_list is None:
            raise ValueError("mask_list must be provided.")
        if len(mask_list) != len(self.samples):
            raise ValueError("Length of mask_list must match ImageNet dataset length.")

        self.mask_list = mask_list   # 例: 各サンプルに対応するマスクのリスト（list of lists）

        self.input_patch_num = input_patch_num
        self.block_size = block_size

    def __getitem__(self, idx):
        # ImageNet の標準サンプル取得
        img, target = super().__getitem__(idx)

        # ここでマスクを取り出す
        masks = self.mask_list[idx][:self.input_patch_num*self.block_size*self.block_size]   # 例: [mask1, mask2, ...] のリスト
        target_masks = [x for x in range(196) if x not in masks ]

        return img, target, torch.tensor(masks), torch.tensor(target_masks)

def visualize_rec_image(img, rec_patches, target_mask):
    patchify = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=16,
                p2=16)
    unpatchify = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=16,
            p2=16,
            h=14, w=14)
    img_patches = patchify(img)
    img_patches[:, target_mask[0], :] = rec_patches
    img_rec = unpatchify(img_patches)

    return img_rec

def select_tokens(x: torch.Tensor, idx: torch.Tensor):
    idx = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return x.gather(dim=1, index=idx)

def image_as_tokens(x):
    b, c, h, w = x.shape
    p = 16
    x = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(b, (h * w) // p**2, c * p * p)

    return x

def tokens_as_image(x: torch.Tensor):
    '''Reshape a set of token vectors into an image. This is the reverse operation
    of `image_as_tokens`.
    '''
    b = x.shape[0]
    im, p = [224,224], 16
    hh, ww = im[0] // p, im[1] // p
    x = x.reshape(b, hh, ww, 3, p, p).permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(b, 3, p * hh, p * ww)
    return x

def forward_with_mask_idx(model, img, mask_idx=None, target_idx=None):

    with torch.no_grad():

        idx = torch.cat([mask_idx, target_idx], dim=1)
        n = idx.shape[1]
        k = mask_idx.shape[1]
        # y = image_as_tokens(img)
        # target_patches = select_tokens(y, idx[:, k:])
        # context_patches = select_tokens(y, idx[:, :k])

        # embedding
        pos_embs = model.vit.embeddings.position_embeddings
        embs = model.vit.embeddings.patch_embeddings(img)
        embs = embs + pos_embs[:, 1:, :]

        # mask
        embs = select_tokens(embs, mask_idx)
        # embs, mask, _= fixed_masking_per_sample(embs, mask_idx)
        ids_restore = idx.argsort(1)

        # add cls token
        cls_token = model.vit.embeddings.cls_token + pos_embs[:, :1, :]
        cls_tokens = cls_token.expand(embs.shape[0], -1, -1)
        embs = torch.cat((cls_tokens, embs), dim=1)

        # encode
        encoder_output = model.vit.encoder(embs)
        sequence_output = encoder_output[0]
        sequence_output = model.vit.layernorm(sequence_output)

        # decode
        decoder_outputs = model.decoder(sequence_output, ids_restore)
        logits = decoder_outputs.logits

        # img = model.unpatchify(logits)
        # save_image(img, "reconstruction.png", normalize=True)

        rec_patches = select_tokens(logits, idx[:, k:])

    return logits, rec_patches # , target_patches, context_patches

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/configs.yaml')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='which devices to use on local machine')
parser.add_argument(
    '--T', type=int, default=10,
    help='time step')
parser.add_argument(
    '--t', type=int, default=8,
    help='time step for reconstruction')
parser.add_argument(
    '--b', type=int, default=3,
    help='block size')
parser.add_argument(
    '--variants', type=str, nargs='+', default=['selected', 'random'],
    help='selection variants to evaluate. "selected" reads '
         'block_{b}_tmax_{T}_test.json; any other name V reads '
         'block_{b}_tmax_{T}_test_V.json (e.g. random, random_center, '
         'fixed_center_out, fixed_raster from the new baseline scripts).')

if __name__ == "__main__":

    args = parser.parse_args()

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
    if not torch.cuda.is_available():
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

    block_size = args.b

    input_patch_num = args.t

    t_max = args.T

    variants = args.variants

    variant_data = {}   # variant -> (mask_list, output folder, mse file name)
    for variant in variants:
        if variant == "selected":
            mask_path = f"selected_patch_index/block_{block_size}_tmax_{t_max}_test.json"
            folder = "selected"
            mse_name = "mse_select_patches.pt"
        else:
            mask_path = f"selected_patch_index/block_{block_size}_tmax_{t_max}_test_{variant}.json"
            folder = variant
            mse_name = f"mse_{variant}_patches.pt"
        with open(mask_path, "r") as f:
            mask_list = json.load(f)
        variant_data[variant] = (mask_list, folder, mse_name)

    # dataset (one per variant; ImageNetWithMasks picks up block_size from
    # the module-level variable as before)
    test_datasets = {}
    for variant in variants:
        mask_list, folder, mse_name = variant_data[variant]
        test_datasets[variant] = ImageNetWithMasks(
            root=root_path,
            image_folder=image_folder,
            transform=test_transform,
            train=False,
            copy_data=copy_data,
            index_targets=False,
            mask_list=mask_list,
            input_patch_num=input_patch_num)


    # model
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", attn_implementation="sdpa").to(device)

    # test (visualize reconstructed image) for each variant
    mse_all = {}
    for variant in variants:
        ds_model = test_datasets[variant]
        mask_list, folder, mse_name = variant_data[variant]
        N = len(ds_model)

        os.makedirs(f"images/rec_images/n={input_patch_num}/{folder}", exist_ok=True)

        mse_patches = torch.zeros(N)
        for n in tqdm(range(N), desc=variant):
            x, y, input_mask, target_mask = ds_model[n]
            x, input_mask, target_mask = x.unsqueeze(0).to(device), input_mask.unsqueeze(0).to(device), target_mask.unsqueeze(0).to(device)
            pred, rec_patches = forward_with_mask_idx(model, x, input_mask, target_mask)

            x_input = visualize_rec_image(x, torch.zeros_like(rec_patches), target_mask)
            mask = visualize_rec_image(torch.ones_like(x, dtype=torch.bool), torch.zeros_like(rec_patches, dtype=bool), target_mask)
            pred_image = model.unpatchify(pred.detach())
            pred_image[mask] = x_input[mask]

            x_input = inv(x_input)
            x = inv(x)
            pred_image = inv(pred_image)
            img_result = torch.cat([x_input, pred_image, x], dim=0)
            mse = F.mse_loss(x[~mask], pred_image[~mask])
            mse_patches[n] = mse.item()
            if (n+1) % 500 == 0 or n < 10:
                vutils.save_image(img_result, "tmp.png")
                image = mpimg.imread("tmp.png")
                plt.figure(figsize=(6, 6))
                plt.imshow(image)
                plt.axis("off")
                plt.title(f"MSE={mse:.6f}", fontsize=10)
                plt.savefig(f"images/rec_images/n={input_patch_num}/{folder}/{n+1}.png", bbox_inches="tight", pad_inches=0.2)
                plt.close()

        torch.save(mse_patches, f"images/rec_images/n={input_patch_num}/{folder}/{mse_name}")
        mse_all[variant] = mse_patches
        print(f"{variant}: mean={mse_patches.mean()} std={mse_patches.std()}")
