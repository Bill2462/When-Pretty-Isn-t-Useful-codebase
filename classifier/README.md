# Classifier Training

## Import Dataset

`import_ds.py` packs a class-folder image directory into sharded training data,
optionally applying a low/high-pass bandpass filter and saving as `.npy`.

For RGB / depth maps

```
python3 import_ds.py \
--input_dir path/to/images \
--output_dir path/to/storage/for/training
```

For bandpass filtering 

```
python3 import_ds.py \
--input_dir path/to/images \
--output_dir path/to/storage/for/training \
--apply_bps_filter lowpass|highpass \
--save_as_npy
```

Please import both train and test data.

## Train Classifier

`train.py` trains one of the supported architectures (`--model_arch`) on the
imported data and saves checkpoints to `--output_dir`.

```
python3 train.py \
--train_data_path /data/train \
--val_data_path /data/val \
--output_dir path/to/save/checkpoint \
--num_workers 24 \
--model_arch resnet50|resnet18|vit_tiny_patch16_224|convnext_tiny|swin_v2_tiny|bagnet9|bagnet17|bagnet33
```

The `--data_type` flag (`rgb`, `rgb_bp`, `depth`, `texture`) must match how the
dataset was imported above; both `train.py` and `eval.py` accept it.

## Evaluate Classifier

`eval.py` loads a trained checkpoint, evaluates it on `--data_path`, and writes
the metrics to a gzipped JSON log.

```
python3 eval.py \
--data_path /data/val \
--output_filepath path/to/save/eval/log.json.gz \
--ckpt_filepath path/to/model/best.ckpt \
--num_workers 8 \
--model_arch resnet50|resnet18|vit_tiny_patch16_224|convnext_tiny|swin_v2_tiny|bagnet9|bagnet17|bagnet33
```
