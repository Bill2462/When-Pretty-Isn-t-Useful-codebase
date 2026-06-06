import os
import io
import torch
import msgpack
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from torchvision.transforms import (
    CenterCrop,
    RandomCrop,
    ColorJitter,
    Compose,
    GaussianBlur,
    Grayscale,
    InterpolationMode,
    Normalize,
    RandomChoice,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    RandomSolarize,
    Resize,
    Lambda,
    ToTensor
)

from file_io import BinaryReader, BinaryWriter

def texture_transform(is_test=False, resize_size=256, crop_size=128):
    transforms = [
        ToTensor(),
        Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        RandomCrop(crop_size),
    ]

    if is_test:
        pass
    else:
        transforms.append(RandomHorizontalFlip(p=0.5))
        transforms.append(RandomVerticalFlip(p=0.5))

    transforms.append(
        Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
        )
    )
    
    return Compose(transforms)

def depth_space_transform(is_test=False, imsize=224):
    transforms = [
        ToTensor(),
    ]
    
    if is_test:
        transforms.append(Resize(imsize, interpolation=InterpolationMode.BICUBIC))
        transforms.append(CenterCrop(imsize))
    else:
        transforms.append(RandomResizedCrop(imsize, interpolation=InterpolationMode.BICUBIC))
        transforms.append(RandomHorizontalFlip(p=0.5))

    transforms.append(
        Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
    )

    return Compose(transforms)

def rescale_bp_filtered_image(img):
    # Rescale each channel to [0, 1]
    img_min = img.amin(dim=(-2, -1), keepdim=True)
    img_max = img.amax(dim=(-2, -1), keepdim=True)
    return (img - img_min) / (img_max - img_min + 1e-8)  # Add small epsilon to avoid division by zero

def bp_filtered_rgb_space_transform(is_test=False, imsize=224):
    transforms = [
        Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),  # Convert PIL Image to float tensor
        Lambda(rescale_bp_filtered_image),
    ]
    
    if is_test:
        transforms.append(Resize(imsize, interpolation=InterpolationMode.BICUBIC))
        transforms.append(CenterCrop(imsize))
    else:
        transforms.append(RandomResizedCrop(imsize, interpolation=InterpolationMode.BICUBIC))
        transforms.append(RandomHorizontalFlip(p=0.5))

    transforms.append(
        Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
    )

    return Compose(transforms)

def rgb_space_transform(is_test=False, imsize=224):
    transforms = [
        ToTensor(),
    ]
    
    if is_test:
        transforms.append(Resize(imsize, interpolation=InterpolationMode.BICUBIC))
        transforms.append(CenterCrop(imsize))
    else:
        transforms.append(RandomResizedCrop(imsize, interpolation=InterpolationMode.BICUBIC))
        
        augs_choice = [
            Grayscale(num_output_channels=3),
            RandomSolarize(threshold=0.5, p=1.0),
            GaussianBlur(kernel_size=7, sigma=(0.2, 2.0)),
        ]

        transforms.append(RandomChoice(augs_choice))
        transforms.append(ColorJitter(0.3, 0.3, 0.3))
        transforms.append(RandomHorizontalFlip(p=0.5))

    transforms.append(
        Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
    )

    return Compose(transforms)

class DatasetWriter:
    def __init__(self, output_dir: str, max_shard_size: int):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_shard_size = max_shard_size
        
        self.current_shard_index = 0
        self.current_shard_count = 0
        self.current_sample_count = 0
        
        self.writer = None
        self._open_new_shard()

    def _open_new_shard(self):
        if self.writer is not None:
            self.writer.close()
        
        shard_path = os.path.join(self.output_dir, f"{self.current_shard_index}")
        self.writer = BinaryWriter(shard_path)
        self.current_shard_count = 0
        self.current_shard_index += 1

    def write(self, element: dict):
        if self.current_shard_count == self.max_shard_size:
            self._open_new_shard()
        
        self.writer.write(msgpack.packb(element, use_bin_type=True))
        
        self.current_shard_count += 1
        self.current_sample_count += 1
    
    def close(self):
        if self.writer is not None:
            self.writer.close()

        self.writer = None

class ImageDataset(Dataset):
    def __init__(self, dataset_dir: str, transform=None, npy_mode: bool = False):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.npy_mode = npy_mode

        # Discover all shards in the dataset directory
        self.shard_paths = []
        self.shard_readers = {}
        self.shard_lengths = []
        self.total_length = 0
        
        self._discover_shards()
        self._calculate_offsets()
    
    def _discover_shards(self):
        """Discover all shard directories and initialize readers."""
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        # Find all numeric shard directories
        shard_indices = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                shard_indices.append(int(item))
        
        # Sort shard indices to ensure consistent ordering
        shard_indices.sort()
        
        for shard_idx in shard_indices:
            shard_path = os.path.join(self.dataset_dir, str(shard_idx))
            self.shard_paths.append(shard_path)
            
            # Initialize reader for this shard
            reader = BinaryReader(shard_path, close_file_after_reading=True)
            self.shard_readers[shard_idx] = reader
            self.shard_lengths.append(len(reader))
            self.total_length += len(reader)
    
    def _calculate_offsets(self):
        """Calculate cumulative offsets for each shard."""
        self.shard_offsets = [0]
        for length in self.shard_lengths:
            self.shard_offsets.append(self.shard_offsets[-1] + length)
    
    def _find_shard_and_index(self, global_index):
        """Find which shard contains the global index and return local index."""
        for i, offset in enumerate(self.shard_offsets[1:]):
            if global_index < offset:
                shard_idx = i
                local_idx = global_index - self.shard_offsets[i]
                return shard_idx, local_idx
        raise IndexError(f"Index {global_index} out of bounds")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        if index < 0 or index >= self.total_length:
            raise IndexError(f"Index {index} out of bounds")
        
        # Find which shard contains this index
        shard_idx, local_idx = self._find_shard_and_index(index)
        
        # Get the reader for this shard
        reader = self.shard_readers[shard_idx]
        
        # Read the raw bytes
        raw_bytes = reader[local_idx]
        
        # Decode the msgpack data
        data_dict = msgpack.unpackb(raw_bytes, raw=False)
        
        processed_data = self._process_sample(data_dict)
        
        return processed_data

    def _process_sample(self, data_dict):
        label_bytes = data_dict['label']
        label = int(np.load(io.BytesIO(label_bytes))[0])
        label = torch.tensor(label, dtype=torch.long)

        img_bytes = data_dict['image']

        # Convert bytes to PIL Image
        if self.npy_mode:
            img = np.load(io.BytesIO(img_bytes))
        else:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'label': label
        }
