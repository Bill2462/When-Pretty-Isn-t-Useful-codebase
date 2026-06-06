import os
import json
import argparse
import numpy as np

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
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated images.")
    
    parser.add_argument("--captions_filepath", type=str, required=True,
                        help="Path to the file containing class names for image generation.")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of images to generate in each batch.")
    
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of inference steps for the T2I model.")
    
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="Guidance scale for the T2I model.")

    parser.add_argument("--classes_filepath", type=str, default="class_labels/tiny_imagenet.json",
                        help="Path to the file containing class names for image generation.")
    
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target size (width and height) for generated images.")
    
    parser.add_argument("--limit-to-first-n-classes", type=int, default=None,
                        help="If set, only process the first N classes.")

    parser.add_argument("--limit-to-first-n-captions-per-class", type=int, default=None,
                        help="If set, only process the first N captions per class.")
    
    parser.add_argument("--n-images-per-prompt", type=int, default=1,
                        help="Number of images to generate per prompt.")
    
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

def load_prompts(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        obj = json.load(f)
    
    prompts = {}
    for entry in obj:
        class_id = entry['class_id']
        if class_id not in prompts:
            prompts[class_id] = []
        
        del entry ['class_id']
        prompts[class_id].append(entry)

    return prompts

def process_class(class_id, class_name, prompts, args, model):
    #Determine output directory
    output_dir = os.path.join(args.output_dir, class_id)
    os.makedirs(output_dir, exist_ok=True)

    # Apply limit to first N captions if specified
    if args.limit_to_first_n_captions_per_class is not None:
        prompts = prompts[:args.limit_to_first_n_captions_per_class]

    # Count how many jpg images already exist
    existing_images = [f for f in os.listdir(output_dir)]
    existing_count = len(existing_images)
    print(f"Found {existing_count} existing images in {output_dir}.", flush=True)

    # Calculate how many prompts to skip based on existing images and n-images-per-prompt
    prompts_to_skip = existing_count // args.n_images_per_prompt
    prompts = prompts[prompts_to_skip:]
    
    if not prompts:
        print("No new prompts to process. Exiting.", flush=True)
        return

    print(f"Generating images for class: {class_id}", flush=True)

    batcher = DataBatcher(prompts, args.batch_size)

    for batch_idx, inputs in enumerate(batcher):
        prompt_batch = [f"{class_name}, {entry['caption']}" for entry in inputs]
        filenames = [entry['filename'] for entry in inputs]
        print(f"Processing batch {batch_idx + 1}/{len(batcher)}", flush=True)

        # Generate n images per prompt
        for img_idx in range(args.n_images_per_prompt):
            images = model(prompt_batch, width=args.target_size, height=args.target_size)

            for filename, image in zip(filenames, images):
                # Adjust filename if generating multiple images per prompt
                if args.n_images_per_prompt > 1:
                    base_name, ext = os.path.splitext(filename)
                    adjusted_filename = f"{base_name}_{img_idx}{ext}"
                else:
                    adjusted_filename = filename
                
                image_path = os.path.join(output_dir, adjusted_filename)
                image.save(image_path, format="JPEG", quality=95)

def main():
    args = get_args()

    all_captions = load_prompts(args.captions_filepath)
    classes = load_json_file(args.classes_filepath)
    classes = {e['id']: e['label'] for e in classes}

    model = get_t2i_model(args.model_type, args.model_path, device="cuda")
    model.num_inference_steps = args.num_inference_steps
    model.guidance_scale = args.guidance_scale

    print(f"Using model: {args.model_type} from {args.model_path}", flush=True)

    # Get the class IDs to process
    class_ids = list(all_captions.keys())
    
    # Apply limit to first N classes if specified
    if args.limit_to_first_n_classes is not None:
        class_ids = class_ids[:args.limit_to_first_n_classes]
    
    # Filter by class index range
    class_ids = [class_ids[i] for i in range(args.class_idx_start, min(args.class_idx_end + 1, len(class_ids)))]

    for class_id in class_ids:
        prompts = all_captions[class_id]
        class_name = classes[class_id]
        process_class(class_id, class_name, prompts, args, model)

if __name__ == "__main__":
    main()
