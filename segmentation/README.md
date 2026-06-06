# Training Segmentation Model

## Import Dataset

Converts the per-class binary masks produced by SAM3 (see
[../labelling](../labelling)) into a VOC-style multi-class segmentation dataset.
The script walks the class subdirectories under `--images`, matches each image
to its `<name>_mask.png` under `--masks`, binarizes the mask, and writes a
VOC-palette PNG where foreground pixels carry the class id (background = 0).
Output images and masks are flattened into `--out-images` and `--out-masks` with
`<class>_<name>` filenames. Classes are sorted and capped at the first 50.

```
python3 import_ds.py \
--images path/to/rgb/images \
--masks path/to/sam3/masks \
--out-images path/to/data/images \
--out-masks path/to/data/masks
```

## Train DeeplabV3

`train.py` trains a torchvision segmentation model (e.g. `deeplabv3_resnet50`)
on the VOC-style dataset at `--data-path`, saving a checkpoint per epoch to
`--output-dir`.

```
python3 train.py \
--data-path path/to/data \
--output-dir path/to/save/checkpoint \
--lr 0.02 \
--dataset custom \
-b 20 \
--model deeplabv3_resnet50 \
--aux-loss \
--weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

## Evaluate DeeplabV3

Run with `--test-only` and point `--resume` at a trained checkpoint to evaluate
on the validation split (reports per-class and mean IoU plus pixel accuracy).

```
python3 train.py \
--data-path path/to/data \
--dataset custom \
--model deeplabv3_resnet50 \
--aux-loss \
--resume path/to/save/checkpoint/model_29.pth \
--test-only
```
