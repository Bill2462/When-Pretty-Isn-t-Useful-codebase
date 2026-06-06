# Training Object Detection Model

## Import Dataset

Converts SAM3 bounding-box annotations (see [../labelling](../labelling)) into a
COCO-format detection dataset. The script reads `bounding_boxes.json` from the
`train/` and `val/` subdirectories of the SAM3 annotations, copies the source
images into `train/` and `val/` splits, and writes
`annotations/instances_train.json` and `annotations/instances_val.json`. Classes
are derived from the annotation keys (`class_name/image_name`), sorted, and
capped at the first 50.

```
python3 import_ds.py \
--input_images path/to/rgb/images \
--input_sam3_annotations path/to/sam3/annotations \
--output_dir path/to/coco/dataset
```

The resulting `--output_dir` is the COCO dataset directory passed as
`--data-path` to training.

## Train Faster-RCNN

`train.py` trains a torchvision detection model (e.g. `fasterrcnn_resnet50_fpn`)
on the COCO dataset at `--data-path`, saving a checkpoint per epoch to
`--output-dir`.

```
python3 train.py \
--data-path path/to/data \
--output-dir path/to/save/checkpoint \
-b 16 \
--dataset coco \
--model fasterrcnn_resnet50_fpn \
--epochs 26 \
--lr-steps 16 22 \
--aspect-ratio-group-factor 3 \
--weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

## Evaluate Faster-RCNN

Run with `--test-only` and point `--resume` at a trained checkpoint to evaluate
on the validation split (reports COCO mAP metrics).

```
python3 train.py \
--data-path path/to/data \
--dataset coco \
--model fasterrcnn_resnet50_fpn \
--resume path/to/save/checkpoint/model_25.pth \
--test-only
```
