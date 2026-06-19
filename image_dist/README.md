# Image Distribution Metrics

This folder contains code for measuring how well a set of generated (fake)
images covers the distribution of a set of real images. It embeds both sets with
CLIP and reports Precision, Recall, Density and Coverage (PRDC), with optional
Vendi diversity scoring.

## Compute PRDC

`calc.py` expects the real and fake images to be organised into matching
class subdirectories (the same class names under both `--real_image_path` and
`--fake_image_path`). It embeds every image with the CLIP model at
`--clip_model_filepath`, processes the classes in slices of `--class_slice_size`,
balances the number of fake and real images per slice, computes PRDC per slice,
and writes the averaged metrics to `--output_path` as JSON.

```
python3 calc.py \
--clip_model_filepath path/to/clip/ViT-B-32.pt \
--real_image_path path/to/real/images \
--fake_image_path path/to/generated/images \
--output_path path/to/metrics.json \
--batch_size 256 \
--num_workers 16 \
--class_slice_size 200 \
--device cuda
```

The output JSON contains the averaged `precision`, `recall`, `density` and
`coverage` values.

## Notes

 - `prdc.py` implements the Precision/Recall/Density/Coverage metrics
   ([Naeem et al., 2020](https://github.com/clovaai/generative-evaluation-prdc)).
 - `vendi.py` provides Vendi diversity scores; `image_utils.py` adds Inception
   embeddings and pixel/embedding Vendi helpers. Vendi scoring is wired into
   `calc.py` but disabled by default (commented out) — uncomment it to also log
   the Vendi score per slice.
