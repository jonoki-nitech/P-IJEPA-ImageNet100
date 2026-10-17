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

    for n in tqdm(range(N)):
        img, label = test_dataset[n]
        img = img.unsqueeze(0)

        all_index = range(196)

        if args.start == 'center':
            # exact center block (rows/cols covered: center..center+b-1)
            left = top = (14 - block_size + 1) // 2
        else:
            left = random.choice(range(14-block_size+1))
            top = random.choice(range(14-block_size+1))

        context_indices_list = [block_to_index(left, top, block_size, block_size)]
        target_indices_list = [all_index]

        context_images_list = []
        sigma_images_list = []

        for t in range(T):

            # saturation check (same terminal condition as the random variant):
            # stop when no fully-unobserved b x b sliding window remains.
            # Without this, argmax picks an all-(-inf) window after saturation
            # and keeps appending blocks that overlap the observed region.
            candidate_exists = False
            for le in range(14-block_size+1):
                for to in range(14-block_size+1):
                    next_index = block_to_index(le, to, block_size, block_size)
                    if all(x not in context_indices_list[0] for x in next_index):
                        candidate_exists = True
                        break
                if candidate_exists:
                    break
            if not candidate_exists:
                print(f"end @ {t+1}/{T}")
                break

            with torch.no_grad():
                img = img.to(device)

                mask_enc = [torch.tensor(context_indices_list).to(device)]
                mask_pred = [torch.tensor(target_indices_list).to(device)]
                feature = encoder(img, mask_enc)
                sigma = sigma_head(sigma_predictor(feature, mask_enc, mask_pred))

                # visualize
                img = img.to("cpu")
                sigma = sigma.to("cpu")
                context_img = reconstruct_img_from_patch(img, context_indices_list[0])
                # target_img = reconstruct_img_from_patch(img, target_indices_list[0])
                sigma_img = sigma_heatmap(sigma, patch_index=torch.tensor(context_indices_list))

                context_images_list.append(context_img.squeeze().permute(1,2,0).cpu().numpy())
                sigma_images_list.append(sigma_img.squeeze().permute(1,2,0).cpu().numpy())


                # select patches which has largest sigma(4*4block)
                sigma[:, context_indices_list[0], :] = -torch.inf
                next_index = select_next_block(sigma, k=block_size)
                # context_indices_list = context_indices_list.tolist()
                context_indices_list[0] = context_indices_list[0] + next_index

        # # visualize saccade simulation
        # if (n+1) % 100 == 0 or n < 10:
        #     row_labels = ['context', 'sigma']      # 左に表示
        #     col_labels = [f't = {i}' for i in range(0, T)]  # 上に表示

        #     fig, axes = plt.subplots(2, T, figsize=(T*2, 4))
        #     for i in range(T):
        #         axes[0, i].imshow(context_images_list[i])
        #         axes[0, i].axis('off')
        #         axes[1, i].imshow(sigma_images_list[i])
        #         axes[1, i].axis('off')
        #         axes[0, i].set_title(col_labels[i], fontsize=10)

        #     for ax, row_label in zip(axes[:,0], row_labels):
        #         ax.set_ylabel(row_label, fontsize=12, rotation=0, labelpad=40, va='center')

        #     plt.tight_layout()
        #     plt.savefig(f"images/saccade/image_{n}.png")
        #     plt.close()

        if torch.is_tensor(context_indices_list):
            context_indices_list = context_indices_list.tolist()
        # print(context_indices_list)
        selected_index_list.append(context_indices_list[0])

    os.makedirs("selected_patch_index", exist_ok=True)

    suffix = "" if args.start == "random" else "_start_center"
    with open(f"selected_patch_index/block_{block_size}_tmax_{T}_test{suffix}.json", "w") as f:
        json.dump(selected_index_list, f)
