import os
import argparse

from PIL import Image
from t2i_vae_pipelines import get_t2i_vae_pipeline

def list_image_paths_in_directory(directory: str):
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                paths.append(os.path.join(root, file))
    
    return paths

def get_args():
    parser = argparse.ArgumentParser(description="Generate images using a text-to-image model.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Name of the T2I model to use.")

    parser.add_argument("--model_type", type=str, required=True,
                        choices=["stable-diffusion-v1",
                                 "stable-diffusion-v2.0",
                                 "stable-diffusion-v2.1",
                                 "stable-diffusion-xl",
                                 "stable-diffusion-turbo",
                                 "stable-diffusion-3",
                                 "stable-diffusion-3.5",
                                 "stable-diffusion-3.5-medium",
                                 "stable-diffusion-3.5-turbo",
                                 "qwen-image",
                                 "sana",
                                 "pixart-alpha",
                                 "lumina2",
                                 "flux-dev"],
                        help="Type of T2I model to use.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to re-encode.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")

    return parser.parse_args()
        
def main():
    args = get_args()

    vae_pipeline = get_t2i_vae_pipeline(model_name=args.model_type, path=args.model_path, device='cuda')

    image_paths = list_image_paths_in_directory(args.input_dir)

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        reencoded_image = vae_pipeline(image)[0]

        base_name = os.path.basename(img_path)
        class_name = os.path.basename(os.path.dirname(img_path))
        output_path = os.path.join(args.output_dir, class_name, base_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        reencoded_image.save(output_path)

if __name__ == "__main__":
    main()
