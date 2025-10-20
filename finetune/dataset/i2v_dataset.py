import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import random
from copy import deepcopy

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
import pandas as pd
from decord import VideoReader, cpu
import numpy as np 
from PIL import Image
# from ..models.train_utils.degradation import degrade_image
# from ..models.train_utils.unimatch.utils.flow_viz import flow_to_image
import torch.nn.functional as F
import cv2
# import albumentations as A

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .utils import (   # .utils
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)

PROMPT = "a scene with high visual quality."

if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip
decord.bridge.set_bridge("torch")

# from ..constants import LOG_LEVEL, LOG_NAME
# logger = get_logger(LOG_NAME, LOG_LEVEL)


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


class BaseI2VDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        video_column: str,
        device: torch.device,
        trainer: "Trainer" = None,
        perterb: bool = False,
        random_mask: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.videos = load_videos(video_column)
        self.trainer = trainer

        self.device = device
        # self.encode_video = trainer.encode_video
        # self.encode_text = trainer.encode_text
        self.perterb = perterb
        self.random_mask = random_mask

        if trainer is not None:  # for debug mode
            self.is_validation = trainer.args.is_validation
            self.validation_perturb = trainer.args.validation_perturb

            if self.validation_perturb:
                assert self.is_validation is True

            print(f"validation_perturb: {self.validation_perturb}")

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                prompt = PROMPT
                video = self.videos[index]

                seg_path = str(video).replace('rgb_video', 'seg_video')
                mass_path = str(video).replace('rgb_video', 'mass_density').replace('.mp4', '.npy')
                mass = np.load(mass_path, allow_pickle=True).item()['mass']
                density = np.load(mass_path, allow_pickle=True).item()['density']

                frames, _ = self.preprocess(video, None)  # video is a path here
                _, _, h, w = frames.shape
                mass = cv2.resize(mass, (w, h), interpolation=cv2.INTER_LINEAR)
                density = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
                mass = torch.stack([torch.from_numpy(mass), torch.from_numpy(density)])
                # flow = torch.from_numpy(np.load(flow_path)).permute(2, 0, 1)
                # flow = F.interpolate(flow.unsqueeze(0), (h, w), mode='nearest-exact')
                seg_frames, _ = self.preprocess(seg_path, None)
                
                frames = frames.to('cpu')
                seg_frames = seg_frames.to('cpu')
                
                # Current shape of frames: [F, C, H, W]
                frames = self.video_transform(frames)  # [0, 255] -> [-1, 1]
                seg_frames = self.video_transform(seg_frames)  # [0, 255] -> [-1, 1]
                image = frames[0].clone()

                frames = frames.permute(1, 0, 2, 3).contiguous()
                seg_frames = seg_frames.permute(1, 0, 2, 3).contiguous()
                break

            except Exception as e:
                print(f"Error processing index {index}: {e}")
                index = random.randint(0, len(self.videos) - 1)
                continue

        return {
            "image": image,
            "prompt": prompt,
            "frames": frames,
            "seg_frames": seg_frames,
            "mass_map": mass
        }

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class I2VDatasetWithResize(BaseI2VDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, stride=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.stride = stride
        print(f"stride: {self.stride}, max_num_frames: {self.max_num_frames}")
        
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width, self.stride)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


class I2VDatasetWithBuckets(BaseI2VDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_buckets(video_path, self.video_resolution_buckets)
        image = preprocess_image_with_resize(image_path, video.shape[2], video.shape[3])
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


if __name__ == "__main__":
    # Example usage
    from torchvision.io import write_video
    from utils import flow_to_image
    from torchvision.transforms.functional import to_pil_image

    dataset = I2VDatasetWithResize(
        data_root="not important",
        video_column="/data/user/zmai090/physion_forcing_dense_rope/finetune/videos_2-5objs_train.txt",
        device='cpu',
        max_num_frames=25,
        height=256,
        width=256,
    )
    print(len(dataset))
    sample = dataset[1]
    path = '/data/user/txu647/code/C/'

    mass_map = sample["mass_map"]

    breakpoint()
    
