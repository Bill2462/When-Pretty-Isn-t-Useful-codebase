import os
import json
import random
import argparse
import numpy as np

import torch
import torch.utils.data
import clip
import vendi

from tqdm import tqdm
from PIL import Image
from prdc import compute_prdc

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, preprocess):
        self.image_files = image_files
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        return self.preprocess(image)

class ClipImageEmbedder:
    def __init__(self, model_filepath, device):
        self.device = device
        self.model, self.preprocess = clip.load(model_filepath, device=device)

    @torch.no_grad()
    def embed_images(self, image_files, batch_size=256, num_workers=8):
        dataset = ImageDataset(image_files, self.preprocess)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )
        
        all_embeddings = []
        for images in tqdm(dataloader):
            images = images.to(self.device)
            embeddings = self.model.encode_image(images)
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--clip_model_filepath", type=str, required=True,
                        help="Path to the CLIP model file.")
    
    parser.add_argument('--fake_image_path', type=str, required=True,
                        help='Path to directory containing images.')
    
    parser.add_argument('--real_image_path', type=str, required=True,
                        help='Path to directory containing real images for PRDC comparison.. ')
    
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output metrics.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for processing images.')
    
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers for image loading.')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation (e.g., "cpu" or "cuda").')
    
    parser.add_argument("--class_slice_size", type=int, default=200,
                        help="Number of classes to process at a time.")
    
    return parser.parse_args()

def find_classes_in_dir(dir_path):
    classes = [d for d in os.listdir(dir_path)
               if os.path.isdir(os.path.join(dir_path, d))]
    classes.sort()
    return classes

def find_image_files(base_path):
    image_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(os.path.join(root, file))
    return image_files

def list_to_batches(input_list, batch_size):
    n = len(input_list)
    for i in range(0, n, batch_size):
        yield input_list[i:min(i + batch_size, n)]

def find_image_list_in_class_list(base_path, class_list):
    image_files = []
    for cls in class_list:
        class_path = os.path.join(base_path, cls)
        class_image_files = find_image_files(class_path)
        image_files.extend(class_image_files)
    return image_files

def normalize_file_lists(fake_files, real_files):
    min_len = min(len(fake_files), len(real_files))
    fake_files = fake_files[:min_len]
    real_files = real_files[:min_len]
    return fake_files, real_files

def main():
    args = get_args()

    classes_real = find_classes_in_dir(args.real_image_path)
    classes_fake = find_classes_in_dir(args.fake_image_path)
    assert classes_real == classes_fake, "Class directories"
    # Now shuffle the classes
    random.shuffle(classes_real)
    classes = classes_real

    embedder = ClipImageEmbedder(args.clip_model_filepath, args.device)
    metrics = []

    for i, class_slice in enumerate(list_to_batches(classes, args.class_slice_size)):
        print(f"Processing class slice {i+1}/{(len(classes) + args.class_slice_size - 1) // args.class_slice_size}", flush=True)
        fake_files_slice = find_image_list_in_class_list(args.fake_image_path, class_slice)
        real_files_slice = find_image_list_in_class_list(args.real_image_path, class_slice)
        random.shuffle(fake_files_slice)
        random.shuffle(real_files_slice)

        fake_files_slice, real_files_slice = normalize_file_lists(fake_files_slice, real_files_slice)
        print(f"Number of images in this slice: {len(fake_files_slice)}", flush=True)
        
        fake_embeddings = embedder.embed_images(fake_files_slice, batch_size=args.batch_size, num_workers=args.num_workers)
        real_embeddings = embedder.embed_images(real_files_slice, batch_size=args.batch_size, num_workers=args.num_workers)

        batch_metrics = compute_prdc(
            real_features=real_embeddings,
            fake_features=fake_embeddings,
            nearest_k=5,
        )

        print(f"Batch PRDC: {batch_metrics}", flush=True)

        #batch_metrics["vendi_score"] = vendi.score_X(fake_embeddings)
        #print(f"Batch VENDI Score: {batch_metrics['vendi_score']}", flush=True)
        metrics.append(batch_metrics)

    # Aggregate metrics by averaging
    final_metrics = {}
    for key in metrics[0].keys():
        final_metrics[key] = np.mean([m[key] for m in metrics])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

if __name__ == '__main__':
    main()
