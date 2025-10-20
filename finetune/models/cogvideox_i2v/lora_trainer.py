from typing import Any, Dict, List, Tuple
import os
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    # CogVideoXImageToVideoPipeline,
    # CogVideoXTransformer3DModel,
)
from models.cogvideox_transformer_MD import CogVideoXTransformer3DModel
from models.pipeline import CogVideoXImageToVideoPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image, ImageDraw
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register, sample_by_mask_multi

import cv2
import numpy as np
import torch.nn.functional as F
# from ultralytics import YOLO
from ..train_utils.unimatch.unimatch.unimatch import UniMatch
from ..train_utils.unimatch.utils.flow_viz import flow_to_image
from torchvision.transforms.functional import to_pil_image
import time
import random
from ..depth_anything_v2.dpt import DepthAnythingV2
    
palette = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)  # For tracking IDs
MOTION_FLAG = 6
FLOW_SCALE = 1.0
TRAIN_MODE = ["Motion", "RGB"]
DENSE_POINTS = 512

def draw_arrow_img(arrow_tensor, img_size, scale=50):
    img = Image.new("RGB", img_size, "white")
    draw = ImageDraw.Draw(img)

    dx, dy = arrow_tensor[0] * scale, arrow_tensor[1] * scale

    center_x, center_y = img_size[0] // 2, img_size[1] // 2
    end_x = center_x + dx
    end_y = center_y - dy  

    # 绘制箭头（用线段模拟）
    draw.line([(center_x, center_y), (end_x, end_y)], fill="red", width=3)
    draw.polygon([(end_x, end_y), (end_x - 10, end_y - 5), (end_x - 10, end_y + 5)], fill="red")

    return img


def preprocess_size(image1, image2, padding_factor=32):
    transpose_img = False
    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    # inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
    #                 int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        
    inference_size = [384, 512]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)
    
    return image1, image2, inference_size, ori_size, transpose_img

def postprocess_size(flow_pr, inference_size, ori_size, transpose_img):
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)
    
    return flow_pr

def depthflow_to_pil(delta, value=10):  # [-v, v] -> [0, 255]
    d = np.asarray(delta, np.float32)
    out = np.clip((d + value) * (255.0 / (2 * value)), 0, 255)
    rgb = np.repeat(np.rint(out)[..., None], 3, axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, 'RGB')

def depth_flow_from_masks(masks, depth):
    assert masks.ndim == 4, "masks should be (K,T,H,W)"
    K, T, H, W = masks.shape
    depth1, depthN = depth[0], depth[MOTION_FLAG]
    assert depth1.shape == (H, W) and depthN.shape == (H, W)

    device = depth1.device
    # 保证都在同一 device、类型合适
    m1 = masks[:, 0].to(device=device, dtype=torch.bool)        # (K,H,W)
    mN = masks[:, MOTION_FLAG].to(device=device, dtype=torch.bool)        # (K,H,W)
    d1 = depth1.to(device=device, dtype=torch.float32)           # (H,W)
    dN = depthN.to(device=device, dtype=torch.float32)           # (H,W)

    # 每实例像素数
    cnt1 = m1.sum(dim=(1, 2))                                    # (K,)
    cntN = mN.sum(dim=(1, 2))                                    # (K,)

    # 每实例深度和/均值（广播 (K,H,W)*(H,W) -> (K,H,W)）
    sum1 = (m1.float() * d1).sum(dim=(1, 2))                     # (K,)
    sumN = (mN.float() * dN).sum(dim=(1, 2))                     # (K,)
    mean1 = sum1 / cnt1.clamp_min(1)
    meanN = sumN / cntN.clamp_min(1)

    # 基础有效性：两帧都至少有像素
    valid = (cnt1 > 0) & (cntN > 0)

    delta_per_inst = torch.zeros(K, device=device, dtype=torch.float32)
    delta_per_inst[valid] = (meanN - mean1)[valid]

    # 回填到第1帧实例区域；若实例重叠，按求和叠加
    delta_map = (m1.float() * delta_per_inst.view(K, 1, 1)).sum(dim=0)  # (H,W)
    return delta_map, delta_per_inst

@torch.no_grad()
def get_depth_maps(depth_anything, video_frames):
    depth_anything = depth_anything.float()
    assert video_frames.shape[0] == 1, "batchsize must be 1"
    frame_height, frame_width = video_frames.shape[3:]  # Get frame dimensions (H, W)

    frame_nps = video_frames[0].permute(0, 2, 3, 1).byte().cpu().numpy()  # [T, H, W, 3]

    depth_maps = []
    for i in range(frame_nps.shape[0]):
        image = cv2.cvtColor(frame_nps[i], cv2.COLOR_RGB2BGR)
        depth = depth_anything.infer_image(image)
        if depth.shape != (frame_height, frame_width):
            depth = cv2.resize(depth, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
        depth_maps.append(torch.from_numpy(depth))

    return torch.stack(depth_maps, dim=0)   # [T, H, W]

@torch.no_grad()
def get_optical_flow(unimatch, video_frame):
    '''
        video_frame: [b, t, c, w, h]
    '''
    image1, image2 = video_frame[:, 0], video_frame[:, MOTION_FLAG]
    image1_r, image2_r, inference_size, ori_size, transpose_img = preprocess_size(image1, image2)
    results_r = unimatch(image1_r, image2_r,
        attn_type='swin',
        attn_splits_list=[2, 8],
        corr_radius_list=[-1, 4],
        prop_radius_list=[-1, 1],
        num_reg_refine=6,
        task='flow',
        pred_bidir_flow=False,
        )['flow_preds'][-1]
    direct_flow = postprocess_size(results_r, inference_size, ori_size, transpose_img)
    return direct_flow


def _get_instance_flow(instance_mask, direct_flow, perterb=True):
    assert len(direct_flow.shape) == 4
    if instance_mask.shape != direct_flow.shape[2:]:
        instance_mask = instance_mask.unsqueeze(0).unsqueeze(0)
        instance_mask = F.interpolate(instance_mask, direct_flow.shape[2:])
    
    instance_mask = (instance_mask > 0).float()
    average_flow = (direct_flow * instance_mask).flatten(2, 3).mean(dim=2)
    
    if perterb:
        average_flow += torch.rand_like(average_flow)

    instance_motion_flow = average_flow.unsqueeze(2).unsqueeze(3) * instance_mask
    return instance_motion_flow

def separate_instances_video(video_seg_map, count_threshold=20):
    """
    Separates instances in a video segmentation map.

    Parameters:
    - video_seg_map (torch.Tensor): The video segmentation map tensor of shape [frame, 3, h, w].
    - count_threshold (int): Minimum number of pixels for an instance to be considered valid.

    Returns:
    - List[torch.Tensor or None]: A list of tensors, each of shape [num_instances, frames, h, w] with boolean masks, or None if no valid instances.
    """
    frames, _, h, w = video_seg_map.shape
    seg_map_quant_list = []

    for i in range(frames):
        seg_map = video_seg_map[i]  # Shape: [3, h, w]

        # Ensure the segmentation map is of type uint8
        if seg_map.dtype != torch.uint8:
            seg_map = seg_map.to(torch.uint8)

        seg_map_quant = (seg_map // 4).to(torch.uint8)
        seg_map_quant = seg_map_quant.permute(1, 2, 0)  # Shape: [h, w, 3]
        seg_map_quant_list.append(seg_map_quant)

    # Stack segmentation maps along a new dimension for frames
    seg_map_quant_stack = torch.stack(seg_map_quant_list, dim=0)  # Shape: [frames, h, w, 3]
    pixels = seg_map_quant_stack.reshape(-1, 3)  # flatten across all frames

    unique_colors, counts = torch.unique(pixels, dim=0, return_counts=True)

    # Remove the zero color (background) if present
    is_non_zero = ~(unique_colors == 0).all(dim=1)
    unique_colors = unique_colors[is_non_zero]
    counts = counts[is_non_zero]

    # Filter out colors with counts less than the threshold
    valid_colors_mask = counts >= count_threshold * frames
    valid_colors = unique_colors[valid_colors_mask]
    counts = counts[valid_colors_mask]

    # If no valid colors remain, return None
    if valid_colors.size(0) == 0:
        return None

    num_instances = valid_colors.size(0)
    result = torch.zeros((num_instances, frames, h, w), dtype=torch.bool)

    for idx, color in enumerate(valid_colors):
        color = color.view(1, 1, 1, 3)  # Shape: [1, 1, 1, 3]
        mask = (seg_map_quant_stack == color).all(dim=-1)  # Shape: [frames, h, w]
        result[idx] = mask

    return result

@torch.no_grad()
def get_seg_flow(unimatch, video_frames, seg_frames, 
                 perterb=True, random_mask=True, depth_model=None):
    '''
        Input:
            video_frames: [b, t, c, h, w]
        Output (fp16, cuda):
            optical_flow: [b, t-1, 2, h, w], 
            seg_map: [b, t-1, 3, h, w].
    '''
    unimatch.to(torch.float32)

    video_frames = (video_frames + 1) * 127.5
    direct_flow = get_optical_flow(unimatch, video_frames.to(torch.float32))  # flow, direct_flow: [b, 2, h, w]

    depth = get_depth_maps(depth_model, video_frames) if depth_model is not None else None
    # flow = torch.cat([flow, torch.zeros_like(flow[:, 0:1])], dim=1)

    camera_motion = torch.zeros_like(-direct_flow.flatten(2, 3).mean(dim=2))
    return direct_flow, camera_motion, depth

@torch.no_grad()
def get_instance_flow(direct_flow, seg_masks, depth_map=None, perterb=False):
    seg_masks = seg_masks.add(1).mul(127.5).type(torch.uint8).permute(0,2,1,3,4)
    if len(direct_flow.shape) == 4:
        direct_flow = direct_flow.unsqueeze(1)

    instances_flow = torch.zeros_like(direct_flow)
    depth_flow = None
    for b, seg_mask_batch in enumerate(seg_masks): 
        assert b == 0, "only support batchsize == 1"
        # Image.fromarray(seg_mask_batch.permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save('m.png')
        masks = separate_instances_video(seg_mask_batch)
        if masks is not None:
            if depth_map is not None:
                depth_flow, _ = depth_flow_from_masks(masks, depth_map)

            first_mask = masks[:, 0].to(direct_flow.device)
            for idx, mask in enumerate(first_mask):
                instance_flow = _get_instance_flow(mask, direct_flow[b], perterb=perterb)
                instances_flow[b] += instance_flow

    # assert depth_map is not None
    if depth_map is not None:
        if depth_flow is None:
            depth_flow = torch.zeros_like(instances_flow[:, :, 0:1])
        else:
            depth_flow = depth_flow[None, None, None, ...]
            depth_flow = depth_flow.to(instances_flow.device, instances_flow.dtype)
        instances_flow = torch.cat([instances_flow, depth_flow], dim=2)

    return instances_flow.squeeze(1)

def depth_to_rgb(depth: torch.Tensor, vmin=0.0, vmax=10.0):
    """
    depth: [T,H,W] 或 [H,W] 的 float 张量
    return: uint8 RGB, 形状 [T,3,H,W] 或 [3,H,W]
    """
    d = depth.clone().float()
    d = (d - vmin) / (vmax - vmin + 1e-8)
    d = d.clamp(0, 1)
    d = (d * 255.0).round().to(torch.uint8)

    if d.ndim == 2:
        return d.unsqueeze(0).repeat(3, 1, 1)         # [3,H,W]
    else:
        return d.unsqueeze(1).repeat(1, 3, 1, 1)      # [T,3,H,W]
    
class CogVideoXI2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        _model_path = '/data/user/zmai090/.cache/huggingface/hub/models--THUDM--CogVideoX1.5-5B-I2V/snapshots/46c90528707aebbe69066390b4fe7e7d24c9c2a4/'
        components.transformer = CogVideoXTransformer3DModel.from_pretrained(_model_path, subfolder="transformer")
        
        cc = 2
        if self.args.use_depth:
            cc += 1
        if self.args.use_mass:
            cc += 2

        components.transformer.init_all_weights(ins_flow_channels=cc*64, N=1792)  # 0.5N

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        components.flow_model = UniMatch(feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task='flow')

        checkpoint = torch.load('/data/user/zmai090/physion_forcing_dense_rope/finetune/models/train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
        components.flow_model.load_state_dict(checkpoint['model'])

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        d_encoder = "vits"
        components.depth_model = DepthAnythingV2(**model_configs[d_encoder])
        components.depth_model.load_state_dict(torch.load(
            f'/data/user/zmai090/.cache/depth_anything_v2_{d_encoder}.pth', 
            map_location='cpu'))
        # components.seg_model = YOLO(path_hpc+'code/segment-anything-2/yolo11l-seg.pt')  # Replace with your model variant if needed

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video, seg_video, mass_map=None) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], [-1, 1]
        vae = self.components.vae.to(self.accelerator.device)
        # seg_model = self.components.seg_model.to(self.accelerator.device)
        flow_model = self.components.flow_model.to(self.accelerator.device)
        if self.args.use_depth:
            depth_model = self.components.depth_model.to(self.accelerator.device)
        else:
            depth_model = None

        video = video.to(vae.device)
        
        direct_flow, camera_motion, depth = get_seg_flow(
            flow_model, video.permute(0, 2, 1, 3, 4), 
            seg_video, depth_model=depth_model)

        dtype = torch.float32
        direct_flow = direct_flow.to(dtype).contiguous()
        if self.args.use_mass:
            assert mass_map is not None
            # 可视化验证无误，tianshuo
            # img_arr = mass_map[0, 1].float().detach().cpu().numpy()  
            # img_arr = (img_arr / (img_arr.max()+1e-3) * 255.0).astype("uint8")
            # Image.fromarray(img_arr, mode="L").save('mass.png')
            mass_map = mass_map.to(direct_flow.device, direct_flow.dtype) 
            direct_flow = torch.cat([direct_flow, mass_map], dim=1)


        direct_flow = get_instance_flow(direct_flow, seg_video, depth)

        ins_flow_img = flow_to_image(direct_flow[0, :2].permute(1, 2, 0).float())
        ins_flow_img = Image.fromarray(ins_flow_img)

        if self.args.use_depth:
            depth_flow_img = depthflow_to_pil(direct_flow[0, -1].cpu().numpy())
        else:
            depth_flow_img = None
        
        video = video.to(vae.dtype)
        latent = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor

        depth_video = depth_to_rgb(depth).permute(1, 0, 2, 3).unsqueeze(0)
        depth_video = depth_video.float() / 127.5 - 1.0
        depth_video = depth_video.to(vae.device, vae.dtype)
        depth_latent = vae.encode(depth_video).latent_dist.sample() * vae.config.scaling_factor

        return (latent, depth_latent, direct_flow.to(vae.device), 
            ins_flow_img, camera_motion, depth_flow_img, depth_video)

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        self.components.text_encoder.to(self.accelerator.device)
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids.to(self.accelerator.device)
        prompt_embedding = self.components.text_encoder(prompt_token_ids)[0]

        return prompt_embedding
    
    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"prompt": [], "images": [], "video": [],
                "seg_video": [], "mass_map": []}

        for sample in samples:
            ret["video"].append(sample["frames"])
            ret["prompt"].append(sample["prompt"])
            ret["images"].append(sample["image"])
            ret["seg_video"].append(sample["seg_frames"])
            ret["mass_map"].append(sample["mass_map"])

        ret["video"] = torch.stack(ret["video"])
        ret["images"] = torch.stack(ret["images"])
        ret["seg_video"] = torch.stack(ret["seg_video"])
        ret["mass_map"] = torch.stack(ret["mass_map"])
        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        vae = self.components.vae

        prompt = batch["prompt"]
        prompt_embedding = self.encode_text(prompt)
        video = batch["video"].to(vae.dtype)
        video_seg = batch["seg_video"].to(vae.dtype)
        images = batch["images"].unsqueeze(2).to(vae.dtype)
        mass = batch["mass_map"].to(vae.dtype) if self.args.use_mass else None
        latent, latent_depth, direct_flow, _, camera_motion, _, depth_video = self.encode_video(
            video, video_seg, mass_map=mass)
        
        mask = video_seg[:, :, 0]
        dense_mask = F.interpolate(mask, scale_factor=1/16, mode='nearest-exact')
        direct_flow = F.interpolate(direct_flow, scale_factor=1/2, mode='nearest-exact')
        dense_mask = (dense_mask.max(dim=1).values > -0.9).to(vae.dtype)
        images = torch.cat((depth_video[:, :, 0:1], images), dim=2)

        # video_cond = video_flow.to(vae.dtype)
        direct_flow = direct_flow.to(vae.dtype)
        camera_motion = camera_motion.to(vae.device, vae.dtype)  # [B, 1, 2]

        if len(camera_motion.shape) == 4:
            camera_motion = camera_motion.squeeze(1)
        
        if self.args.rgb_only:
            latent_cond = latent.clone()   # ablation, only rgb, tianshuo
        else:
            latent_cond = latent_depth

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, 2, H, W]
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

            ncopy = latent_cond.shape[2] % patch_size_t
            first_frame_cond = latent_cond[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent_cond = torch.cat([first_frame_cond.repeat(1, 1, ncopy, 1, 1), latent_cond], dim=2)
            assert latent_cond.shape[2] % patch_size_t == 0
        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        # Add noise to images
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        
        image_latent_dist = torch.cat((
            vae.encode(noisy_images[:, :, 0:1].to(dtype=vae.dtype)).latent_dist.sample(),
            vae.encode(noisy_images[:, :, 1:2].to(dtype=vae.dtype)).latent_dist.sample(),
            ), dim=2
        )
        image_latents = image_latent_dist * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        train_mode = random.choice(TRAIN_MODE)
        # train_mode = "RGB"  # tianshuo

        timesteps_rgb = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps_rgb = timesteps_rgb.long()
        timesteps_motion = timesteps_rgb

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        latent_cond = latent_cond.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (
            latent.shape[0], 
            latent.shape[1] + latent_cond.shape[1] - image_latents.shape[1], 
            *latent.shape[2:]
        )
        latent_padding = image_latents.new_zeros(padding_shape)
        s = latent_padding.shape[1] // 2

        image_latents = torch.cat([
            image_latents[:, 0:1], latent_padding[:, :s],
            image_latents[:, 1:2], latent_padding[:, s:],
            ], dim=1)

        # Add noise to latent
        noise_motion = torch.randn_like(latent_cond)
        latent_cond_noisy = self.components.scheduler.add_noise(latent_cond, noise_motion, timesteps_motion)

        noise_rgb = torch.randn_like(latent)
        latent_rgb_noisy = self.components.scheduler.add_noise(latent, noise_rgb, timesteps_rgb)
        latent_noisy = torch.cat((latent_cond_noisy, latent_rgb_noisy), dim=1)
        # Concatenate latent and image_latents in the channel dimension
        # print(image_latents.shape, latent_noisy.shape)
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames * 2 + 2,  # tianshuo, + ins_flow
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )

        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps_motion,
            timestep_2=timesteps_rgb,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            camera_motion=camera_motion, # [b, 1, 2]
            instance_flow=direct_flow,  # [b, 1, 2, h, w]
            return_dict=False,
            dense_mask=dense_mask,
            dense_points=DENSE_POINTS,
        )[0]

        # Denoise
        motion_noise = predicted_noise[:, :num_frames]
        rgb_noise = predicted_noise[:, num_frames:]

        if train_mode == "Motion":
            latent_pred = self.components.scheduler.get_velocity(motion_noise, latent_cond_noisy, timesteps_motion)
            timesteps = timesteps_motion
        else:
            latent_pred = self.components.scheduler.get_velocity(rgb_noise, latent_rgb_noisy, timesteps_rgb)
            timesteps = timesteps_rgb

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        if train_mode == "Motion":
            latent_error = (latent_pred - latent_cond) ** 2
        else:
            latent_error = (latent_pred - latent) ** 2
        # Compute loss
        loss = torch.mean((weights * latent_error).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)

        prompt = eval_data["prompt"]
        video = eval_data["video"].to(vae.device, vae.dtype)
        video_seg = eval_data["seg_video"].to(vae.device, vae.dtype)
        images = eval_data["images"].unsqueeze(2).to(vae.dtype)
        mass = eval_data["mass_map"].to(vae.dtype) if self.args.use_mass else None
        _, _, direct_flow, ins_flow_pil, camera_motion, depth_flow_pil, depth_video = self.encode_video(
            video, video_seg, mass_map=mass)
        
        mask = video_seg[:, :, 0]
        dense_mask = F.interpolate(mask, scale_factor=1/16, mode='nearest-exact')
        direct_flow = F.interpolate(direct_flow, scale_factor=1/2, mode='nearest-exact')
        dense_mask = (dense_mask.max(dim=1).values > -0.9).to(vae.dtype)
        
        images = torch.cat((depth_video[:, :, 0:1].cpu(), images), dim=2)
        video_cond = depth_video.to(vae.dtype)
        # if not self.args.average_ins_flow:
        #     mask = (mask.max(dim=1).values > -0.9).to(vae.dtype)[None, None, ...]
        #     ins_flow_img = ins_flow_img * mask.to(vae.dtype)
        #     ins_flow_img = ins_flow_img.to(vae.dtype)
        
        if len(camera_motion.shape) == 4:
            camera_motion = camera_motion.squeeze(1)

        # ins_flow_pil = to_pil_image(ins_flow_img[0, :, 0].add(1).mul(127.5).byte().cpu())              
        video_generate = pipe(
            num_frames=self.state.train_frames * 2,  # 25*2, seg+flow
            height=self.state.train_height,      # 480
            width=self.state.train_width,        # 720
            prompt=prompt,  # 'a realistic driving scenario with high visual quality, the overall scene is moving forward.'
            image=images, 
            camera_motion=camera_motion.to(vae.device, dtype=vae.dtype),  # [b, 1, 2]
            instance_flow=direct_flow.to(vae.device, dtype=vae.dtype),  # [b, 1, 2, h, w]
            generator=self.state.generator,
            num_inference_steps=20,  # tianshuo
            dense_mask=dense_mask,
            dense_points=DENSE_POINTS,
        ).frames[0]
        
        video = video.add(1).mul(127.5).byte().cpu()
        video_cond = video_cond.add(1).mul(127.5).byte().cpu()
        gt_video = torch.cat([video_cond, video], dim=3).squeeze().permute(1, 0, 2, 3) 

        assert len(video_generate) == gt_video.shape[0]

        width = video_cond.shape[-1]
        height = video_cond.shape[-2]
        total_width = width * 2
        total_height = height * 3

        if depth_flow_pil is not None:
            camera_arrow_img = depth_flow_pil
        else:
            camera_arrow = camera_motion[0].float().cpu().numpy()
            camera_arrow_img = draw_arrow_img(camera_arrow, ins_flow_pil.size, scale=5)

        generate_frames = []
        for frame in range(gt_video.shape[0]):
            new_frame = Image.new("RGB", (total_width, total_height))
            gt_frame = to_pil_image(gt_video[frame])
            pred_frame = video_generate[frame]
            
            new_frame.paste(camera_arrow_img, (0, 0))
            new_frame.paste(ins_flow_pil, (width, 0))
            new_frame.paste(gt_frame, (0, height))
            new_frame.paste(pred_frame, (width, height))
            generate_frames.append(new_frame)
            
        # new_frame.save('/hpc2hdd/home/zmai090/code/C/frame.png')
        return [("video", generate_frames)]


# Implements the 3D rotary positional embeddings for CogVideoX. 
# The Dense rope is detail in transformer_MD file.
    def prepare_rotary_positional_embeddings( 
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        assert (base_num_frames - 1) % 2 == 0, "motion forcing requires an odd number of frames"
        

        # 8:8:2 / 2 -> 4:4:1
        freqs_cos_0, freqs_sin_0 = get_3d_rotary_pos_embed( # rope for RGB
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=(base_num_frames - 1) // 2, # 4
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        freqs_cos_1, freqs_sin_1 = freqs_cos_0.clone(), freqs_sin_0.clone() # duplicate for auxiliary modality

        freqs_cos_2, freqs_sin_2 = get_3d_rotary_pos_embed( # rope for instance Cue
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=1,  # anchor to first frame
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        freqs_cos = torch.cat([freqs_cos_0, freqs_cos_1, freqs_cos_2], dim=0)
        freqs_sin = torch.cat([freqs_sin_0, freqs_sin_1, freqs_sin_2], dim=0)

        return freqs_cos, freqs_sin


register("cogvideox-i2v", "lora", CogVideoXI2VLoraTrainer)
