import os
import io
import json
import base64
import argparse
import asyncio
from tqdm import tqdm
from PIL import Image
from openai import AsyncOpenAI
from tempfile import NamedTemporaryFile

client = AsyncOpenAI()

class GPT4oCaptioner:
    def __init__(self, model_name="gpt-4.1-nano", prompt=None, concurrency=5, min_side=256):
        self.model_name = model_name
        self.prompt = prompt or (
            "You are an expert image captioning model. You will write a detailed 5-6 sentence caption that describes background, foreground, objects and camera angle of the image."
        )
        self.semaphore = asyncio.Semaphore(concurrency)
        self.min_side = min_side

    def _resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if min(width, height) == self.min_side:
            return image
        if width < height:
            new_width = self.min_side
            new_height = int(height * (self.min_side / width))
        else:
            new_height = self.min_side
            new_width = int(width * (self.min_side / height))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    async def _caption_single(self, image_path: str) -> str:
        try:
            with Image.open(image_path).convert("RGB") as image:
                image = self._resize_image(image)
                image_b64 = self._image_to_base64(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load/process {image_path}: {e}")

        max_retries, delay = 5, 5
        for attempt in range(1, max_retries + 1):
            try:
                async with self.semaphore:
                    response = await asyncio.wait_for(
                        client.responses.create(
                            model=self.model_name,
                            input=[{
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": self.prompt},
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                                ],
                            }],
                        ),
                        timeout=90,
                    )
                return response.output_text.strip()
            except Exception as e:
                print(f"Attempt {attempt} failed for {image_path}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise

class ImageDataset:
    def __init__(self, path: str):
        self.samples = []
        for class_id in sorted(os.listdir(path)):
            class_path = os.path.join(path, class_id)
            if not os.path.isdir(class_path):
                continue
            for img in sorted(os.listdir(class_path)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.samples.append({
                        "class_id": class_id,
                        "filename": img,
                        "image_path": os.path.join(class_path, img),
                    })
    def __len__(self): return len(self.samples)
    def __iter__(self): yield from self.samples

def safe_load_json(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Try to recover partial file (truncate last line)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for i in range(len(lines) - 1, -1, -1):
                try:
                    return json.loads("".join(lines[:i]))
                except Exception:
                    continue
        except Exception:
            pass
        print("⚠ Checkpoint corrupted, starting fresh.")
        return []

def safe_write_json(filepath, data):
    tmp = NamedTemporaryFile("w", delete=False, dir=os.path.dirname(filepath), encoding="utf-8")
    json.dump(data, tmp, indent=4, ensure_ascii=False)
    tmp.flush()
    os.fsync(tmp.fileno())
    tmp.close()
    os.replace(tmp.name, filepath)

async def async_main(args):
    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    dataset = ImageDataset(args.data_path)
    captioner = GPT4oCaptioner(args.model, args.prompt, args.concurrency, args.min_side)

    results = safe_load_json(args.output_filepath)
    completed = {(r["class_id"], r["filename"]) for r in results}
    remaining = [s for s in dataset if (s["class_id"], s["filename"]) not in completed]

    print(f"{len(remaining)} images to process ({len(completed)} already done).")
    pbar = tqdm(total=len(remaining), desc="Captioning")

    lock = asyncio.Lock()
    pending_write_count = 0  # track captions since last save

    async def process(sample):
        nonlocal pending_write_count
        try:
            caption = await captioner._caption_single(sample["image_path"])
            item = {"class_id": sample["class_id"], "filename": sample["filename"], "caption": caption}
            async with lock:
                results.append(item)
                pending_write_count += 1
                # Save only once every 100 captions
                if pending_write_count >= 100:
                    safe_write_json(args.output_filepath, results)
                    pending_write_count = 0
        except Exception as e:
            print(f"❌ {sample['filename']} failed: {e}")
        finally:
            pbar.update(1)

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded_process(sample):
        async with sem:
            await process(sample)

    await asyncio.gather(*(bounded_process(s) for s in remaining))
    pbar.close()

    # Final save for any remaining unsaved captions
    if pending_write_count > 0:
        safe_write_json(args.output_filepath, results)

    print(f"✅ Completed {len(results)} total captions.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_filepath", required=True)
    parser.add_argument("--model", default="gpt-4.1-nano")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--min_side", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()
#
