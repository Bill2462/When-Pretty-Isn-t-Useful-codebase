import os
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# -------------------------------------------------------
# VOC COLOR MAP
# -------------------------------------------------------
def voc_colormap(N=256):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= bitget(cid, 0) << (7 - j)
            g |= bitget(cid, 1) << (7 - j)
            b |= bitget(cid, 2) << (7 - j)
            cid >>= 3
        cmap[i] = [r, g, b]
    return cmap

def save_voc_mask(binary_mask: np.ndarray, class_id: int, out_path: str):
    """
    binary_mask: HxW, values {0,1}
    class_id: VOC class index (>=1)
    """
    voc_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    voc_mask[binary_mask > 0] = class_id

    mask_img = Image.fromarray(voc_mask, mode="P")
    mask_img.putpalette(voc_colormap().flatten())
    mask_img.save(out_path)


# -------------------------------------------------------
# MAIN PROCESSING
# -------------------------------------------------------
def process_dataset(
    images_root: str,
    masks_root: str,
    out_images: str,
    out_masks: str,
    min_area_ratio: float,
):
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    # Assign class IDs (background = 0)
    class_names = sorted(
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    )
    class_names = class_names[:50]
    class_to_id = {cls: i + 1 for i, cls in enumerate(class_names)}

    print("Class mapping:")
    for cls, cid in class_to_id.items():
        print(f"  {cid}: {cls}")

    kept, skipped = 0, 0

    for cls in tqdm(class_names):
        img_dir = os.path.join(images_root, cls)
        mask_dir = os.path.join(masks_root, cls)

        if not os.path.isdir(mask_dir):
            print(f"⚠️ Missing masks for class {cls}, skipping")
            continue

        class_id = class_to_id[cls]

        for fname in os.listdir(img_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue

            img_path = os.path.join(img_dir, fname)

            # Get mask fname without image extension
            mask_fname = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, mask_fname + "_mask.png")

            if not os.path.exists(mask_path):
                skipped += 1
                continue

            # Load and binarize mask
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)
            binary_mask = (mask_np > 0).astype(np.uint8)

            base = f"{cls}_{os.path.splitext(fname)[0]}"

            out_img = os.path.join(out_images, base + ext)
            out_mask = os.path.join(out_masks, base + ".png")

            shutil.copy2(img_path, out_img)
            save_voc_mask(binary_mask, class_id, out_mask)

            kept += 1

    print("\n✅ Finished")
    print(f"Kept samples:   {kept}")
    print(f"Skipped samples: {skipped}")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ImageNet-style binary masks to VOC-style multi-class masks"
    )
    parser.add_argument("--images", required=True, help="Root directory of images")
    parser.add_argument("--masks", required=True, help="Root directory of masks")
    parser.add_argument("--out-images", required=True, help="Output image directory")
    parser.add_argument("--out-masks", required=True, help="Output mask directory")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.01,
        help="Minimum foreground ratio to keep a mask",
    )

    args = parser.parse_args()

    process_dataset(
        args.images,
        args.masks,
        args.out_images,
        args.out_masks,
        args.min_area_ratio,
    )
