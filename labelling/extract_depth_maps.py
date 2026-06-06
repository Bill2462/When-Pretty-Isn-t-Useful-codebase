import os
import argparse

from math import ceil
from tqdm import tqdm
from PIL import Image

from depth_models import MidasDepthModel, DepthAnythingV2

class DataBatcher:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        else:
            batch = self.data[self.index:min(self.index + self.batch_size, len(self.data))]
            self.index += self.batch_size
            return batch
    
    def reset(self):
        self.index = 0
    
    def __len__(self):
        return ceil(len(self.data) / self.batch_size)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output image")

    parser.add_argument("--model", type=str, choices=["midasv3", "depth_anything_v2"], default="midasv3",
                        help="Method to generate control net input")
    
    parser.add_argument("--model_path", type=str, help="Path to the depth model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for depth extraction")
    args = parser.parse_args()
    return args

def crop_to_square(image):
    """
    Take the smaller dimension as the output size and crop out the center square.
    """
    width, height = image.size

    # Get the smaller dimension as the square size
    square_size = min(width, height)
    
    # Calculate the crop coordinates to center the square
    left = (width - square_size) // 2
    top = (height - square_size) // 2
    right = left + square_size
    bottom = top + square_size
    
    # Crop the image to a square
    return image.crop((left, top, right, bottom))
    
def find_images(input_path):
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    image_filepaths = []

    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_filepaths.append(os.path.join(root, file))
    else:
        raise ValueError(f"Input path {input_path} is not a valid directory.")
    
    return image_filepaths

def main():
    args = get_args()

    if args.model == "midasv3":
        depth_model = MidasDepthModel(model_path=args.model_path, device=args.device)
    elif args.model == "depth_anything_v2":
        depth_model = DepthAnythingV2(model_path=args.model_path, device=args.device)

    image_filepath_list = find_images(args.input_path)
    os.makedirs(args.output_path, exist_ok=True)

    data_batcher = DataBatcher(image_filepath_list, batch_size=args.batch_size)
    for image_filepaths in tqdm(data_batcher):
        
        images = []
        for image_filepath in image_filepaths:
            image = Image.open(image_filepath).convert("RGB")
            if image.width != image.height:
                image = crop_to_square(image)
            images.append(image)
        
        depth_maps = depth_model(images)

        for img_fp, depth_map in zip(image_filepaths, depth_maps):
            relative_path = os.path.relpath(img_fp, args.input_path)
            output_filepath = os.path.join(args.output_path, relative_path.replace(os.path.splitext(relative_path)[1], ".png"))
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            depth_map.save(output_filepath)

if __name__ == "__main__":
    main()
