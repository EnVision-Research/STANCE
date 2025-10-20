import os, io, csv, math, random
import numpy as np
from einops import rearrange
import pandas as pd

import torch
from decord import VideoReader, cpu
import torch.distributed as dist

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import h5py
import cv2


def get_depth_map(depth_np):
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    if depth_max - depth_min > 0:
        depth_normalized = (255 * (depth_max - depth_np) / (depth_max - depth_min)).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)

    # Apply a colormap to the normalized depth map
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
    return depth_colormap


def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


class WebVid10M(Dataset):
    def __init__(
            self,
            meta_path='/apdcephfs/share_1290939/0_public_datasets/WebVid/metadata/results_2M_train.csv',
            data_dir='/apdcephfs/share_1290939/0_public_datasets/WebVid',
            sample_size=[256, 256], 
            sample_stride=1, 
            sample_n_frames=14,
        ):
        zero_rank_print(f"loading annotations from {meta_path} ...")

        metadata = pd.read_csv(meta_path)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        self.data_dir = data_dir

        self.length = len(self.metadata)
        print(f"data scale: {self.length}")

        self.sample_stride   = sample_stride
        print(f"sample stride: {self.sample_stride}")
        self.sample_n_frames = sample_n_frames
        
        # sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size", sample_size)
        self.sample_size = sample_size

        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(sample_size),
            # transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp, rel_video_fp
    
    def get_batch(self, index):

        while True:

            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, rel_path = self._get_video_path(sample)

            required_frame_num = self.sample_stride * self.sample_n_frames

            try:
                video_reader = VideoReader(video_path, ctx=cpu(0))
                if len(video_reader) < required_frame_num:
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            frame_num = len(video_reader)

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + self.sample_stride*i for i in range(self.sample_n_frames)]

            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue

        assert(frames.shape[0] == self.sample_n_frames),f'{len(frames)}, self.video_length={self.sample_n_frames}'

        frames = frames.asnumpy()

        resized_frames = []
        for i in range(frames.shape[0]):
            frame = np.array(Image.fromarray(frames[i]).convert('RGB').resize([self.sample_size[1], self.sample_size[0]]))
            resized_frames.append(frame)
        resized_frames = np.array(resized_frames)

        resized_frames = torch.tensor(resized_frames).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
    
        return resized_frames, rel_path
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        pixel_values, video_name = self.get_batch(idx)

        # pixel_values = self.pixel_transforms(pixel_values)
        pixel_values = pixel_values / 255.
        
        sample = dict(pixel_values=pixel_values, video_name=video_name)
        return sample


class YoutubeVideoData(Dataset):
    def __init__(
            self,
            meta_path='/hpc2hdd/home/zmai090/code/video_data/metas_40v_driving/meta_clips_caption_cleaned.csv',
            data_dir='/hpc2hdd/home/zmai090/code/video_data/clips',
            sample_size=[256, 256], 
            sample_stride=1, 
            sample_n_frames=14,
            random_flip=True,
            random_crop=True,
        ):
        zero_rank_print(f"loading annotations from {meta_path} ...")

        metadata = pd.read_csv(meta_path)
        metadata['caption'] = metadata['text']
        del metadata['text']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        self.data_dir = data_dir
        self.random_flip = random_flip
        self.random_crop = random_crop

        print(f"random flip: {random_flip}, random crop: {random_crop}")
        self.length = len(self.metadata)
        print(f"data scale: {self.length}")

        self.sample_stride   = sample_stride
        print(f"sample stride: {self.sample_stride}")
        self.sample_n_frames = sample_n_frames
        
        # sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size", sample_size)
        self.sample_size = sample_size
        self.temp_sample_size = deepcopy(self.sample_size)
        if self.random_crop:
            self.temp_sample_size[0] = self.sample_size[0] + self.sample_size[0] // 8
            self.temp_sample_size[1] = self.sample_size[1] + self.sample_size[1] // 8
    
    def _get_video_path(self, sample):
        rel_video_fp = sample['path'].split('/')[-1]
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp
    
    def get_batch(self, index):
        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, rel_path = self._get_video_path(sample)

            required_frame_num = self.sample_stride * self.sample_n_frames

            try:
                video_reader = VideoReader(video_path, ctx=cpu(0))
                if len(video_reader) < required_frame_num:
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            frame_num = len(video_reader)

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + self.sample_stride*i for i in range(self.sample_n_frames)]

            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue

        assert(frames.shape[0] == self.sample_n_frames),f'{len(frames)}, self.video_length={self.sample_n_frames}'

        frames = frames.asnumpy()

        resized_frames = []
        for i in range(frames.shape[0]):
            frame = np.array(Image.fromarray(frames[i]).convert('RGB').resize(
                [self.temp_sample_size[1], self.temp_sample_size[0]]))
            resized_frames.append(frame)
        resized_frames = np.array(resized_frames)

        resized_frames = torch.tensor(resized_frames).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
    
        return resized_frames, rel_path
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pixel_values, video_name = self.get_batch(idx)
        if self.random_crop:
            crop_y = random.randrange(pixel_values.shape[2] - self.sample_size[0] + 1)
            crop_x = random.randrange(pixel_values.shape[3] - self.sample_size[1] + 1)
            pixel_values = pixel_values[:, :, crop_y : crop_y + self.sample_size[0], crop_x : crop_x + self.sample_size[1]]
            video_name += '_crop'

        if self.random_flip and random.random() <= 0.5:
            pixel_values = torch.flip(pixel_values, dims=[3])
            video_name += '_flip'

        pixel_values = pixel_values / 255.  # [0, 1]
        
        sample = dict(pixel_values=pixel_values, video_name=video_name)
        return sample


class Physion(Dataset):
    def __init__(
            self,
            data_dir='/hpc2hdd/home/zmai090/code/video_data/physics/physion',
            sample_size=[256, 256], 
            sample_stride=1, 
            sample_n_frames=14,
            random_flip=True,
        ):
        zero_rank_print(f"loading data from {data_dir} ...")

        self.data = self._get_video_paths(data_dir)

        self.length = len(self.data)
        self.sample_stride = sample_stride
        self.sample_size = sample_size
        self.sample_n_frames = sample_n_frames
        
        self.random_flip = random_flip
        print(f"random flip: {random_flip}")
        print(f"data scale: {self.length}")
        print(f"sample stride: {self.sample_stride}")
        print("sample size", sample_size)
    
    def _get_video_paths(self, root_dir):
        """
        Recursively finds all .mp4 files in root_dir and its subdirectories.

        """
        video_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".hdf5"):
                    video_paths.append(os.path.join(dirpath, filename))
        return video_paths
    
    def get_batch(self, index):
        while True:
            try:
                index = index % self.length
                video_path = self.data[index]
                rgb_frames, depth_frames, seg_frames = [], [], []
                with h5py.File(video_path, 'r') as f:
                    frames = f['frames']
                    frame_num = len(sorted(frames.keys()))
                                        
                    ## select a random clip
                    required_frame_num = self.sample_stride * self.sample_n_frames
                    random_range = frame_num - required_frame_num
                    start_idx = random.randint(0, random_range) if random_range > 0 else 0
                    frame_indices = [f"{(start_idx + self.sample_stride*i):04d}" for i in range(self.sample_n_frames)]

                    for frame_idx in frame_indices:
                        frame = frames[frame_idx]['images']
                        
                        try:
                            rgb_frame = Image.open(io.BytesIO(frame['_img_cam0'][:].tobytes())).convert('RGB')
                            depth_frame = Image.fromarray(get_depth_map(frame['_depth_cam0'][:])).convert('RGB')
                            seg_frame = Image.open(io.BytesIO(frame['_id_cam0'][:].tobytes())).convert('RGB')
                        except KeyError:
                            rgb_frame = Image.open(io.BytesIO(frame['_img'][:].tobytes())).convert('RGB')
                            depth_frame = Image.fromarray(get_depth_map(frame['_depth'][:])).convert('RGB')
                            seg_frame = Image.open(io.BytesIO(frame['_id'][:].tobytes())).convert('RGB')

                        rgb_frame = np.array(rgb_frame.resize([self.sample_size[1], self.sample_size[0]]))
                        depth_frame = np.array(depth_frame.resize([self.sample_size[1], self.sample_size[0]]))
                        seg_frame = np.array(seg_frame.resize([self.sample_size[1], self.sample_size[0]]))

                        rgb_frames.append(rgb_frame)
                        depth_frames.append(depth_frame)
                        seg_frames.append(seg_frame)
                    
                rgb_frames = np.stack(rgb_frames)
                depth_frames = np.stack(depth_frames)
                seg_frames = np.stack(seg_frames)
                break
            except:
                index += 1
                continue

        assert(rgb_frames.shape[0] == self.sample_n_frames), f'{len(frames)}, self.video_length={self.sample_n_frames}'

        rgb_frames = torch.tensor(rgb_frames).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
        depth_frames = torch.tensor(depth_frames).permute(0, 3, 1, 2).float()
        seg_frames = torch.tensor(seg_frames).permute(0, 3, 1, 2).float()

        return rgb_frames, seg_frames, depth_frames, video_path
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixel_values, seg_frames, depth_frames, video_path = self.get_batch(idx)
        video_name = video_path.split('/')[-1].split('.')[0]

        if self.random_flip and random.random() <= 0.5:
            pixel_values = torch.flip(pixel_values, dims=[3])
            seg_frames = torch.flip(seg_frames, dims=[3])
            depth_frames = torch.flip(depth_frames, dims=[3])

        sample = dict(pixel_values=pixel_values / 255., 
                      seg_frames=seg_frames / 255., 
                      depth_frames=depth_frames / 255.,
                      video_name=video_name)
        return sample


from copy import deepcopy
import torchvision.transforms.functional as tvF


def get_color_jitter_params(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1):
    # Generate random factors within the specified ranges
    brightness_factor = None
    contrast_factor = None
    saturation_factor = None
    hue_factor = None

    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    if hue > 0:
        hue_factor = random.uniform(-hue, hue)

    color_jitter_params = (brightness_factor, contrast_factor, saturation_factor, hue_factor)
    return color_jitter_params


def apply_color_jitter(frame, color_jitter_params):
    brightness_factor, contrast_factor, saturation_factor, hue_factor = color_jitter_params
    frame = tvF.adjust_brightness(frame, brightness_factor)
    frame = tvF.adjust_contrast(frame, contrast_factor)
    frame = tvF.adjust_saturation(frame, saturation_factor)
    frame = tvF.adjust_hue(frame, hue_factor)
    return frame


class Physion2(Dataset):
    def __init__(
            self,
            data_dir='/hpc2hdd/home/zmai090/code/video_data/physics/physion',
            sample_size=[256, 256], 
            sample_stride=1, 
            sample_n_frames=14,
            test_only=False,
            random_flip=True,
            random_crop=True,
            color_jitter=True,
        ):
        zero_rank_print(f"loading data from {data_dir} ...")

        self.data = self._get_video_paths(data_dir)

        self.length = len(self.data)
        self.sample_stride = sample_stride
        self.sample_size = sample_size
        self.sample_n_frames = sample_n_frames
        self.test_only = test_only

        if test_only:
            random_flip, random_crop, color_jitter = False, False, False
            print("Careful! Physion test only!!")

        self.random_flip = random_flip
        self.random_crop = random_crop
        self.color_jitter = color_jitter
        print(f"random flip: {random_flip}, random crop: {random_crop}, color jitter: {color_jitter}")
        print(f"data scale: {self.length}")
        print(f"sample stride: {self.sample_stride}")
        print("sample size", sample_size)

        self.temp_sample_size = deepcopy(self.sample_size)
        if self.random_crop:
            self.temp_sample_size[0] = self.sample_size[0] + self.sample_size[0] // 8
            self.temp_sample_size[1] = self.sample_size[1] + self.sample_size[1] // 8        
    
    def _get_video_paths(self, root_dir):
        """
        Recursively finds all .mp4 files in root_dir and its subdirectories.

        """
        video_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".hdf5"):
                    video_paths.append(os.path.join(dirpath, filename))
        return video_paths
    
    def get_batch(self, index):
        if self.color_jitter:
            color_jitter_params_rgb = get_color_jitter_params()
            color_jitter_params_seg = get_color_jitter_params()

        while True:
            try:
                index = index % self.length
                video_path = self.data[index]
                rgb_frames, depth_frames, seg_frames = [], [], []
                with h5py.File(video_path, 'r') as f:
                    frames = f['frames']
                    frame_num = len(sorted(frames.keys()))
                                        
                    ## select a random clip
                    required_frame_num = self.sample_stride * self.sample_n_frames
                    random_range = frame_num - required_frame_num
                    if self.test_only:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, random_range) if random_range > 0 else 0
                    frame_indices = [f"{(start_idx + self.sample_stride*i):04d}" for i in range(self.sample_n_frames)]

                    for frame_idx in frame_indices:
                        frame = frames[frame_idx]['images']
                        
                        rgb_frame = Image.open(io.BytesIO(frame['_img_cam0'][:].tobytes())).convert('RGB')
                        depth_frame = Image.fromarray(get_depth_map(frame['_depth_cam0'][:])).convert('RGB')
                        seg_frame = Image.open(io.BytesIO(frame['_id_cam0'][:].tobytes())).convert('RGB')
                        if self.color_jitter:
                            rgb_frame = apply_color_jitter(rgb_frame, color_jitter_params_rgb)
                            seg_frame = apply_color_jitter(seg_frame, color_jitter_params_seg)

                        rgb_frame = np.array(rgb_frame.resize([self.temp_sample_size[1], self.temp_sample_size[0]]))
                        depth_frame = np.array(depth_frame.resize([self.temp_sample_size[1], self.temp_sample_size[0]]))
                        seg_frame = np.array(seg_frame.resize([self.temp_sample_size[1], self.temp_sample_size[0]]))

                        rgb_frames.append(rgb_frame)
                        depth_frames.append(depth_frame)
                        seg_frames.append(seg_frame)
                    
                rgb_frames = np.stack(rgb_frames)
                depth_frames = np.stack(depth_frames)
                seg_frames = np.stack(seg_frames)
                break
            except:
                index += 1
                continue

        assert(rgb_frames.shape[0] == self.sample_n_frames), f'{len(frames)}, self.video_length={self.sample_n_frames}'

        rgb_frames = torch.tensor(rgb_frames).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
        depth_frames = torch.tensor(depth_frames).permute(0, 3, 1, 2).float()
        seg_frames = torch.tensor(seg_frames).permute(0, 3, 1, 2).float()

        return rgb_frames, seg_frames, depth_frames, video_path
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixel_values, seg_frames, depth_frames, video_path = self.get_batch(idx)
        video_name = video_path.split('/')[-1].split('.')[0]

        if self.random_crop:
            crop_y = random.randrange(pixel_values.shape[2] - self.sample_size[0] + 1)
            crop_x = random.randrange(pixel_values.shape[3] - self.sample_size[1] + 1)

            pixel_values = pixel_values[:, :, crop_y : crop_y + self.sample_size[0], crop_x : crop_x + self.sample_size[1]]
            seg_frames = seg_frames[:, :, crop_y : crop_y + self.sample_size[0], crop_x : crop_x + self.sample_size[1]]
            depth_frames = depth_frames[:, :, crop_y : crop_y + self.sample_size[0], crop_x : crop_x + self.sample_size[1]]
            video_name += '_crop'

        if self.random_flip and random.random() <= 0.5:
            pixel_values = torch.flip(pixel_values, dims=[3])
            seg_frames = torch.flip(seg_frames, dims=[3])
            depth_frames = torch.flip(depth_frames, dims=[3])
            video_name += '_flip'

        sample = dict(pixel_values=pixel_values / 255., 
                      seg_frames=seg_frames / 255., 
                      depth_frames=depth_frames / 255.,
                      video_name=video_name)
        return sample


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torchvision
    dataset = Physion(
        data_dir="/hpc2hdd/home/zmai090/code/video_data/physics/physion/dominoes_new/dataset_210",
        sample_size=[256, 256],
        sample_stride=4,
        sample_n_frames=25,
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Enumerate through the DataLoader
    for index, data in enumerate(dataloader):
        video = data['pixel_values'][0].permute(0, 2, 3, 1)
        seg = data['seg_frames'][0].permute(0, 2, 3, 1)
        depth = data['depth_frames'][0].permute(0, 2, 3, 1)

        video = (video * 255).type(torch.uint8)
        seg = (seg * 255).type(torch.uint8)
        depth = (depth * 255).type(torch.uint8)

        total = torch.cat((video, seg, depth), dim=1)

        video_name = 'debug/' + data['video_name'][0] + '.mp4'
        torchvision.io.write_video(
            filename=video_name, 
            video_array=total,
            fps=7
        )
        breakpoint()

        # print(f"Index: {index}, Data: {data}")
