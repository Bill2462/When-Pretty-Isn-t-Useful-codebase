import os
import json
import argparse

from math import ceil
from t2i_diffusers import get_t2i_model

def get_args():
    parser = argparse.ArgumentParser(description="Generate images using a text-to-image model.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Name of the T2I model to use.")

    parser.add_argument("--model_type", type=str, required=True,
                        choices=["stable-diffusion-v1", "stable-diffusion-v2.0", "stable-diffusion-v2.1",
                                 "stable-diffusion-xl", "stable-diffusion-xl-refiner", "stable-diffusion-turbo",
                                 "stable-diffusion-3", "stable-diffusion-3.5", "stable-diffusion-3.5-medium",
                                 "stable-diffusion-3.5-turbo", "qwen-image", "sana", "pixart-alpha", "lumina2",
                                 "flux-schnell", "flux-dev"],
                        help="Type of T2I model to use.")
    
    parser.add_argument("--class_idx_start", type=int, required=True,
                        help="Starting index of the class to generate images for (0-199).")
    
    parser.add_argument("--class_idx_end", type=int, required=True,
                        help="Ending index of the class to generate images for (0-199).")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
    
    parser.add_argument("--classes_file", type=str, default="class_labels/tiny_imagenet.json",
                        help="Path to the file containing class names for image generation.")
    
    parser.add_argument("--ipc", type=int, default=500,
                        help="Number of images per class to generate.")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of images to generate in each batch.")
    
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target size for the generated images (width and height).")
    
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of inference steps for the T2I model.")
    
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="Guidance scale for the T2I model.")

    return parser.parse_args()

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

def load_json_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)

def process_class(class_id, class_label, args, model):
    output_dir = os.path.join(args.output_dir, class_id)
    os.makedirs(output_dir, exist_ok=True)

    if model.guidance_scale == 0.0 and args.model_type not in ["flux-schnell", "stable-diffusion-3.5-turbo", "stable-diffusion-turbo"]:
        all_prompts = ["" for i in range(args.ipc)]
    else:
        all_prompts = [f"{class_label}" for i in range(args.ipc)]
    
    # Count how many jpg images already exist
    existing_images = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    existing_count = len(existing_images)
    print(f"Found {existing_count} existing images in {output_dir}.", flush=True)
    all_prompts = all_prompts[existing_count:]
    if not all_prompts:
        print("No new prompts to process. Exiting.", flush=True)
        return

    print(f"Generating images for class: {class_label}", flush=True)

    batcher = DataBatcher(all_prompts, args.batch_size)

    sample_idx = existing_count
    for batch_idx, prompts in enumerate(batcher):
        print(f"Processing batch {batch_idx + 1}/{len(batcher)}", flush=True)
        images = model(prompts, width=args.target_size, height=args.target_size)

        for image in images:
            image_path = os.path.join(output_dir, f"{sample_idx}.jpg")
            image.save(image_path, format="JPEG", quality=95)

            sample_idx += 1

def main():
    args = get_args()

    classes = load_json_file(args.classes_file)

    model = get_t2i_model(args.model_type, args.model_path, device="cuda")
    model.num_inference_steps = args.num_inference_steps
    model.guidance_scale = args.guidance_scale
    
    print(f"Using model: {args.model_type} from {args.model_path}", flush=True)

    for class_idx in range(args.class_idx_start, args.class_idx_end + 1):
        cls = classes[class_idx]
        class_id = cls['id']
        class_label = cls['label']
        process_class(class_id, class_label, args, model)

if __name__ == "__main__":
    main()
