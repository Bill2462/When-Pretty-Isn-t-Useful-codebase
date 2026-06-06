import torch
import numpy as np

from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class MidasDepthModel:
    def __init__(self, model_path: str, device="cuda"):
        model = DPTForDepthEstimation.from_pretrained(model_path).to(device)
        image_processor = DPTImageProcessor.from_pretrained(model_path)
        self.model = model
        self.image_processor = image_processor
        self.device = device
    
    @torch.no_grad()
    def __call__(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        
        if self.device == "cuda":
            with torch.autocast("cuda"):
                depth_maps = self.model(**inputs).predicted_depth
        else:
            depth_maps = self.model(**inputs).predicted_depth
        
        depth_min = torch.amin(depth_maps, dim=[1, 2], keepdim=True)
        depth_max = torch.amax(depth_maps, dim=[1, 2], keepdim=True)
        depth_maps = (depth_maps - depth_min) / (depth_max - depth_min)
        
        imgs = []
        for depth_map in depth_maps:
            depth_map = depth_map.cpu().numpy()
            depth_map = (depth_map * 255.0).clip(0, 255).astype(np.uint8)
            depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
            img = Image.fromarray(depth_map).convert("RGB")
            imgs.append(img)
        return imgs

class DepthAnythingV2:
    def __init__(self, model_path: str, device="cuda"):
        model = AutoModelForDepthEstimation.from_pretrained(model_path).to(device)
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = model
        self.image_processor = image_processor
        self.device = device
    
    @torch.no_grad()
    def __call__(self, images) -> Image.Image:
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        depth_maps = outputs.predicted_depth

        depth_min = torch.amin(depth_maps, dim=[1, 2], keepdim=True)
        depth_max = torch.amax(depth_maps, dim=[1, 2], keepdim=True)
        depth_maps = (depth_maps - depth_min) / (depth_max - depth_min)
        imgs = []
        for depth_map in depth_maps:
            depth_map = depth_map.cpu().numpy()
            depth_map = (depth_map * 255.0).clip(0, 255).astype(np.uint8)
            depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
            img = Image.fromarray(depth_map).convert("RGB")
            imgs.append(img)
        return imgs
