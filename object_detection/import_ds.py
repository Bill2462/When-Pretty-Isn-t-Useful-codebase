import os
import argparse

import json
import shutil

from PIL import Image
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(
        description="Import trash detection dataset in COCO format"
    )

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output COCO dataset directory")
    
    parser.add_argument("--input_images", type=str, required=True,
                        help="Path to input images directory")
    
    parser.add_argument("--input_sam3_annotations", type=str, required=True,
                        help="Path to input SAM3 annotations directory")
    
    return parser.parse_args()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dummy_polygon_from_bbox(bbox):
    x, y, w, h = bbox
    return [[
        x, y,
        x + w, y,
        x + w, y + h,
        x, y + h
    ]]

class COCODetectionWriter:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root

        self.train_images = []
        self.val_images = []
        self.train_annotations = []
        self.val_annotations = []
        self.categories = []

        self.image_id = 1
        self.annotation_id = 1

        # Prepare folders
        ensure_dir(os.path.join(dataset_root, "train"))
        ensure_dir(os.path.join(dataset_root, "val"))
        ensure_dir(os.path.join(dataset_root, "annotations"))

    def add_category(self, category_id, name, supercategory="none"):
        self.categories.append({
            "id": category_id,
            "name": name,
            "supercategory": supercategory
        })

    def _add_image(self, image_path, split):
        with Image.open(image_path) as img:
            width, height = img.size

        file_name = os.path.basename(image_path)
        target_path = os.path.join(
            self.dataset_root, split, file_name + f"_{self.image_id}"
        )

        shutil.copy(image_path, target_path)

        image_info = {
            "id": self.image_id,
            "file_name": file_name + f"_{self.image_id}",
            "width": width,
            "height": height
        }

        if split == "train":
            self.train_images.append(image_info)
        else:
            self.val_images.append(image_info)

        self.image_id += 1
        return image_info["id"]

    def add_image_with_bboxes(
        self,
        image_path,
        bboxes,
        category_ids,
        split="train"
    ):
        """
        image_path: path to image
        bboxes: list of [x, y, w, h]
        category_ids: list of category_id (same length as bboxes)
        split: 'train' or 'val'
        """
        assert split in ("train", "val")
        assert len(bboxes) == len(category_ids)

        image_id = self._add_image(image_path, split)

        for bbox, cat_id in zip(bboxes, category_ids):
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(v) for v in bbox],
                "area": float(bbox[2] * bbox[3]),
                "segmentation": dummy_polygon_from_bbox(bbox),
                "iscrowd": 0
            }

            if split == "train":
                self.train_annotations.append(annotation)
            else:
                self.val_annotations.append(annotation)

            self.annotation_id += 1


    def write(self):
        train_coco = {
            "images": self.train_images,
            "annotations": self.train_annotations,
            "categories": self.categories
        }

        val_coco = {
            "images": self.val_images,
            "annotations": self.val_annotations,
            "categories": self.categories
        }

        with open(
            os.path.join(self.dataset_root, "annotations", "instances_train.json"),
            "w"
        ) as f:
            json.dump(train_coco, f, indent=2)

        with open(
            os.path.join(self.dataset_root, "annotations", "instances_val.json"),
            "w"
        ) as f:
            json.dump(val_coco, f, indent=2)

def load_json_annotations(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def process_set(writer: COCODetectionWriter, class_mapping, ann_data, images_dir, split):
    for key, value in tqdm(ann_data.items(), desc=f"Processing {split} set"):
        class_name, image_filename = key.split("/")
        bboxes = value["boxes"]

        # Round bbox values to integers and cap at 0
        bboxes = [
            [max(0, coord) for coord in bbox
            ] for bbox in bboxes]

        if "n" in image_filename or "ILSVRC" in image_filename:
            extension = ".JPEG"
        else:
            extension = ".jpg"

        try:
            category_id = class_mapping[class_name]
        except KeyError:
            # Skip classes not in the mapping
            continue

        image_filepath = os.path.join(images_dir, split, class_name, image_filename+extension)

        writer.add_image_with_bboxes(
            image_filepath,
            bboxes,
            category_ids=[category_id] * len(bboxes),
            split=split
        )

def main():
    args = get_args()
    writer = COCODetectionWriter(args.output_dir)

    # Load annotations for train and val
    ann_train = load_json_annotations(os.path.join(args.input_sam3_annotations, "train", "bounding_boxes.json"))
    ann_val = load_json_annotations(os.path.join(args.input_sam3_annotations, "val", "bounding_boxes.json"))

    # Define categories
    all_classes = []
    for ann in [ann_train, ann_val]:
        for key in ann.keys():
            class_name = key.split("/")[0]
            all_classes.append(class_name)

    all_classes = sorted(list(set(all_classes)))[:50]
    print(f"Found {len(all_classes)} unique classes.")

    class_id_map = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
    for cls_id, mapped_id in class_id_map.items():
        writer.add_category(mapped_id, str(cls_id))

    # Process train and val sets
    process_set(writer, class_id_map, ann_val, args.input_images, split="val")
    process_set(writer, class_id_map, ann_train, args.input_images, split="train")

    writer.write()
    print("COCO dataset has been created.")

if __name__ == "__main__":
    main()
