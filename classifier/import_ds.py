import os
import io
import torch
import argparse
import numpy as np
import random

from PIL import Image

from torchvision.transforms import ToTensor, ToPILImage

from dataset import DatasetWriter

def bandpass_filter(bx, cutoff_freq: float, lowpass=True):
    assert cutoff_freq >= 0 and cutoff_freq <= 1, "cutoff must be in [0, 1]"
    fft = torch.fft.fftshift(torch.fft.fft2(bx))

    if not lowpass:
        cutoff_freq = 1 - cutoff_freq
    
    h, w = fft.shape[-2:]  # height and width
    cy, cx = h // 2, w // 2  # center y, center x
    ry, rx = int(cutoff_freq * cy), int(cutoff_freq * cx)
    
    if lowpass:
        mask = torch.zeros_like(fft)
        mask[:, cy-ry:cy+ry, cx-rx:cx+rx] = 1
    else:
        mask = torch.ones_like(fft)
        mask[:, cy-ry:cy+ry, cx-rx:cx+rx] = 0

    fft = torch.fft.ifft2(torch.fft.ifftshift(fft * mask)).real.clip(0, 1)
    return fft

def get_args():
    parser = argparse.ArgumentParser(description="Dataset Writer")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory to read input data from')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the dataset shards')
    parser.add_argument('--max_shard_size', type=int, default=10000, help='Maximum number of samples per shard')
    parser.add_argument('--apply_bps_filter', type=str, choices=['lowpass', 'highpass'], default=None,
                        help='Apply bandpass filter to images before saving')
    parser.add_argument('--cutoff_freq', type=float, default=0.1, help='Cutoff frequency for bandpass filter (0 to 1)')
    parser.add_argument('--save_as_npy', action='store_true', help='Save images as .npy files instead of jpg')
    parser.add_argument('--resize_size', type=int, help='Resize images to this size (square)')
    parser.add_argument('--ipc', type=int, default=None, help='Images per class (randomly sample if set)')
    return parser.parse_args()

def find_subdirectories(input_dir):
    return [os.path.join(input_dir, d) for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))]

def find_imgs(input_dir):   
    img_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                if file.lower() == "mean.png":
                    continue
                img_files.append(os.path.join(root, file))
    return img_files

def main():
    args = get_args()

    writer = DatasetWriter(args.output_dir, args.max_shard_size)
    subdirs = find_subdirectories(args.input_dir)
    subdirs = sorted(subdirs)  # Ensure consistent ordering

    print(f"Found {len(subdirs)} subdirectories.", flush=True)
    n_written = 0
    for i, subdir in enumerate(subdirs):
        imgs = find_imgs(subdir)
        if not imgs:
            print(f"No images found in {subdir}, skipping...", flush=True)
            continue
        
        # Randomly sample ipc images if specified
        if args.ipc is not None and len(imgs) > args.ipc:
            imgs = random.sample(imgs, args.ipc)
            print(f"Randomly sampled {args.ipc} images from {subdir}...", flush=True)
        else:
            print(f"Processing {len(imgs)} images from {subdir}...", flush=True)
        
        for img_path in imgs:
            if args.apply_bps_filter is not None or args.save_as_npy or args.resize_size:
                image = Image.open(img_path).convert("RGB")
                if args.resize_size:
                    image = image.resize((args.resize_size, args.resize_size))
                
                if args.apply_bps_filter is not None:
                    tensor = ToTensor()(image)  # shape (1, 3, H, W)
                    lowpass = args.apply_bps_filter == 'lowpass'
                    filtered_tensor = bandpass_filter(tensor, args.cutoff_freq, lowpass=lowpass)
                    image = filtered_tensor.numpy()
                
                if args.save_as_npy:
                    np_array = np.array(image)
                    io_bytes = io.BytesIO()
                    np.save(io_bytes, np_array, allow_pickle=False)
                    img_bytes = io_bytes.getvalue()
                else:
                    image = ToPILImage()(image)

                    io_bytes = io.BytesIO()
                    image.save(io_bytes, format='jpeg')
                    img_bytes = io_bytes.getvalue()
            else:
                with open(img_path, 'rb') as img_file:
                    img_bytes = img_file.read()
            
            label = np.array([i], dtype=np.int64)  # Using the index as the label
            io_bytes = io.BytesIO()
            np.save(io_bytes, label, allow_pickle=False)
            label_bytes = io_bytes.getvalue()
            
            sample = {
                'image': img_bytes,
                'label': label_bytes
            }

            writer.write(sample)
            n_written += 1

    print(f"Total samples written: {n_written}", flush=True)
    
    writer.close()

if __name__ == "__main__":
    main()
