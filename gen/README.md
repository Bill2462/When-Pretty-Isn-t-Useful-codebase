# Image generation

This folder contains code for making images from class name prompts, captions and reencoding via VAE.

In all commands below you will need to adjust your model and input/output paths.

## Generating from class names

`gen_images.py` synthesises images from class-name prompts for the class index
range `[--class_idx_start, --class_idx_end]`, writing them under `--output_dir`.

Example command:

```
python3 gen_images.py \
--model_path /storage/t2i/stable-diffusion-v1-5 \
--model_type stable-diffusion-v1 \
--class_idx_start 0 \
--class_idx_end 199 \
--num_inference_steps 50 \
--guidance_scale 2.0 \
--classes_file class_labels/classes_sweep.json \
--target_size 512 \
--output_dir out \
--batch_size 16
```

Please change model type (see `t2i_diffusers.py`), path and batch size to fit your resources. 

## Generating from captions

`gen_images_from_cap.py` synthesises images from the per-image captions in
`--captions_filepath` (produced by `labelling/openai_captioner.py`) instead of
bare class names.

Example commands:

```
python3 gen_images_from_cap.py \
--model_path /storage/t2i/stable-diffusion-v1-5 \
--model_type stable-diffusion-v1 \
--class_idx_start 0 \
--class_idx_end 199 \
--num_inference_steps 50 \
--guidance_scale 2.0 \
--classes_filepath class_labels/classes_sweep.json \
--captions_filepath captions.json \
--target_size 512 \
--output_dir out \
--batch_size 16
```

## VAE reencoding

`vae_reencode.py` passes existing images through a model's VAE (encode then
decode) and saves the reconstructions, isolating the VAE's contribution to image
quality.

```
python3 vae_reencode.py \
--model_path /storage/t2i/stable-diffusion-v1-5 \
--model_type stable-diffusion-v1 \
--input_dir input_dir \
--output_dir out
```
