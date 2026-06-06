# Labelling

## Depth maps

`extract_depth_maps.py` runs a Depth-Anything-V2 model over a folder of RGB
images and writes a depth map for each one.

Link to model: [https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)

```
python3 extract_depth_maps.py \
--input_path rgb/image/path \
--output_path output/path \
--model_path path/to/models/Depth-Anything-V2-Large-hf \
--batch_size 250 \
--model depth_anything_v2
```

## Segmentation Masks and Bounding Boxes (SAM3)

`run_sam3.py` prompts a SAM3 model with the class names in `--class_list_path`
and saves a segmentation mask per image plus a JSON of detected bounding boxes.
These outputs feed the `import_ds.py` scripts in `object_detection` and
`segmentation`.

Link to model: [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)

```
python3 run_sam3.py \
--model_path path/to/models/sam3 \
--image_path rgb/image/path \
--output_path_mask output/masks \
--output_filepath_object_boxes output/bounding_boxes.json \
--class_list_path class_labels/tiny_imagenet.json
```

## Captioning

`openai_captioner.py` generates a text caption for every image in `--data_path`
via the OpenAI API and collects them into a single JSON file (consumed by
`gen/gen_images_from_cap.py`).

```
export OPENAI_API_KEY="........"
python3 openai_captioner.py \
--data_path path/to/images
--output_filepath path/to/captions.json
--model gpt-4.1-nano
```
