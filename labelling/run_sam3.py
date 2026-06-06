import os
import json
import torch
import argparse

from PIL import Image

from transformers import Sam3Processor, Sam3Model

def list_image_paths_in_directory(directory: str):
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                paths.append(os.path.join(root, file))
    
    return paths

def load_class_list(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        class_list = json.load(f)

    out = {}
    for item in class_list:
        out[item['id']] = item['label']
    return out

def get_args():
    parser = argparse.ArgumentParser(description="Object Detection with SAM3 model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained SAM3 model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the directory containing images for object detection.")
    parser.add_argument("--output_path_mask", type=str, required=True, help="Path to save the output masks.")
    parser.add_argument("--output_filepath_object_boxes", type=str, required=True, help="Path to save the output object bounding boxes.")
    parser.add_argument("--class_list_path", type=str, required=True, help="Path to the JSON file containing the list of class names.")
    return parser.parse_args()

class ObjectDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model = Sam3Model.from_pretrained(model_path).to(self.device)

    @torch.no_grad()
    def detect_objects(self, image, class_name, threshold=0.3, mask_threshold=0.3):
        inputs = self.processor(images=image, text=class_name, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        return results
    
def main():
    args = get_args()

    detector = ObjectDetector(args.model_path)

    image_paths = list_image_paths_in_directory(args.image_path)
    class_list = load_class_list(args.class_list_path)

    os.makedirs(args.output_path_mask, exist_ok=True)

    object_boxes_output = {}
    for idx, image_path in enumerate(image_paths):
        if idx % 1000 == 0:
            print(f"Processing image {idx+1}/{len(image_paths)}: {image_path}", flush=True)
        
        image_name = os.path.basename(image_path).rsplit('.', 1)[0]
        
        image = Image.open(image_path).convert("RGB")

        # get class name from the image path
        class_id = os.path.basename(os.path.dirname(image_path))
        class_name = class_list[class_id]
        
        results = detector.detect_objects(image, class_name=class_name)
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()

        try:
            final_mask = torch.zeros_like(results["masks"][0], dtype=torch.uint8)
            for mask in results["masks"]:
                final_mask[mask > 0.3] = 255  # Assign unique value for each mask
        except Exception as e:
            final_mask = torch.zeros((image.height, image.width), dtype=torch.uint8)

        final_mask_image = Image.fromarray(final_mask.cpu().numpy())
        
        mask_save_path = os.path.join(args.output_path_mask, class_id)
        os.makedirs(mask_save_path, exist_ok=True)
        final_mask_image.save(os.path.join(mask_save_path, f"{image_name}_mask.png"))
        
        object_boxes_output[class_id+"/"+image_name] = {
            "boxes": boxes,
            "scores": scores
        }
    
    # Save object boxes output as JSON
    with open(args.output_filepath_object_boxes, "w") as f:
        json.dump(object_boxes_output, f)

if __name__ == "__main__":
    main()
