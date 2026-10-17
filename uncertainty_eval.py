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
from src.utils.tensors import repeat_interleave_batch
from src.masks.utils import apply_masks
import copy
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
        sigma[0, patch_index] = 0.4

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
    # print(x_min, x_max)
    x_min, x_max=0.4, 0.7
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
    help='which devices to use on local machine')
parser.add_argument(
    '--model_path', type=str,
    help='path to pretrained model',
    default='results/pretrained/jepa-latest.pth.tar')
parser.add_argument(
    '--start_index', type=int, default=0,
    help='the first index of test data')
parser.add_argument(
    '--top', type=int, default=4,
    help='top coordinate of context')
parser.add_argument(
    '--left', type=int, default=4,
    help='left coordinate of context')
parser.add_argument(
    '--N', type=int, default=10,
    help='number of sumples to evaluate')
parser.add_argument(
    '--T', type=int, default=10,
    help='time step')
parser.add_argument(
    '--b', type=int, default=4,
    help='block size')

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
    test_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=test_transform,
        train=False,
        copy_data=copy_data,
        index_targets=False)

    # model
    encoder, predictor, sigma_predictor, sigma_head = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        sigma_depth=sigma_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)

    target_encoder = copy.deepcopy(encoder)

    # load pretrained model
    r_path = args.model_path
    checkpoint = torch.load(r_path, map_location="cpu")

    # -- loading encoder
    pretrained_dict = checkpoint['encoder']
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    msg = encoder.load_state_dict(new_state_dict)


    pretrained_dict = checkpoint['predictor']
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    msg = predictor.load_state_dict(new_state_dict)

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

    pretrained_dict = checkpoint['target_encoder']
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    msg = target_encoder.load_state_dict(new_state_dict)

    # inference
    encoder.eval()
    sigma_predictor.eval()
    sigma_head.eval()

    top = args.top
    left = args.left
    start_index = args.start_index
    N = args.N
    block_size = args.b

    sigmas_list = torch.zeros([N, 196])
    pred_error_list = torch.zeros_like(sigmas_list)

    os.makedirs("images/uncertainty", exist_ok = True)

    for n in range(N):
        i = n
        n = start_index + n
        img, label = test_dataset[n]
        img = img.unsqueeze(0)

        all_index = range(196)

        context_indices_list = [block_to_index(left, top, block_size, block_size)]
        target_indices_list = [all_index]

        with torch.no_grad():
            img = img.to(device)

            mask_enc = [torch.tensor(context_indices_list).to(device)]
            mask_pred = [torch.tensor(target_indices_list).to(device)]
            feature = encoder(img, mask_enc)
            z = predictor(feature, mask_enc, mask_pred)
            sigma = sigma_head(sigma_predictor(feature, mask_enc, mask_pred))

            h = target_encoder(img)
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)
            # -- create targets (masked regions of h)
            h = apply_masks(h, mask_pred)
            h = repeat_interleave_batch(h, B, repeat=len(mask_enc))


            # plot sigmas and pred error
            pred_error = ((z - h)**2).mean(dim=-1)
            sigmas_list[i] = sigma.squeeze()
            pred_error_list[i] = pred_error.squeeze()

            # visualize
            img = img.to("cpu")
            sigma = sigma.to("cpu")
            context_img = reconstruct_img_from_patch(img, context_indices_list[0])
            target_index = [x for x in target_indices_list[0] if x not in context_indices_list[0]]
            target_img = reconstruct_img_from_patch(img, target_index)
            # sigma_img = sigma_heatmap(sigma)
            img = inv_normalize(img)
            sigma_img = sigma_heatmap(sigma, patch_index=torch.tensor(context_indices_list))

            imgs = torch.cat([img, context_img, target_img, sigma_img], dim=0)
            save_image(imgs, f'images/uncertainty/image_{n}.png', nrow=4)


    sigmas_list = sigmas_list.flatten().cpu().numpy()
    pred_error_list = pred_error_list.flatten().cpu().numpy()

    corr = np.corrcoef(sigmas_list, pred_error_list)[0, 1]
    plt.scatter(sigmas_list, pred_error_list, alpha=0.1)
    plt.xlabel("uncertainty")
    plt.ylabel("prediction error")
    plt.title(f"corr = {corr}")
    plt.savefig("images/sigma_pred_error.png")
    plt.close()
