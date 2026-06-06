import os
import torch

import diffusers
from diffusers import (StableDiffusionPipeline, EulerDiscreteScheduler,
                       DPMSolverMultistepScheduler,
                       DiffusionPipeline, AutoPipelineForText2Image,
                       StableDiffusion3Pipeline, FluxPipeline,
                       QwenImagePipeline, SanaPipeline,
                       PixArtAlphaPipeline, Lumina2Pipeline)

class QwenImage:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = QwenImagePipeline.from_pretrained(model_path,
                                                     torch_dtype=torch.float16)
        self.pipe.vae.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, image) -> list:
        image = self.pipe.image_processor.preprocess(image).to(self.device, dtype=torch.float16)
        latents = self.pipe.vae(image).latent_dist.sample()
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image)
        return image

class Sana:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = SanaPipeline.from_pretrained(model_path,
                                                torch_dtype=torch.float16)
        self.pipe.vae.to(device)
        self.device = device
    
    @torch.no_grad()
    def __call__(self, image):
        image = self.pipe.image_processor.preprocess(image).to(self.device, dtype=torch.float16)
        latents = self.pipe.vae.encode(image).latent_dist.sample()
        recon_image = self.pipe.vae.decode(latents, return_dict=False)[0]
        #recon_image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(recon_image)
        return image

class PixArtAlpha:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = PixArtAlphaPipeline.from_pretrained(model_path,
                                                       torch_dtype=torch.float16)
        self.pipe.vae.to(device)
        self.device = device
    
    @torch.no_grad()
    def __call__(self, image):
        image = self.pipe.image_processor.preprocess(image).to(self.device, dtype=torch.float16)
        latents = self.pipe.vae.encode(image).latent_dist.sample()
        recon_image = self.pipe.vae.decode(latents, return_dict=False)[0]
        #recon_image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(recon_image)
        return image

class Lumina2:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = Lumina2Pipeline.from_pretrained(model_path,
                                                   torch_dtype=torch.float16)
        self.pipe.vae.to(device)
        self.device = device
    
    @torch.no_grad()
    def __call__(self, image):
        image = self.pipe.image_processor.preprocess(image).to(self.device, dtype=torch.float16)
        latents = self.pipe.vae.encode(image).latent_dist.sample()
        #latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        recon_image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(recon_image)
        return image

class StableDiffusionV1:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                            safety_checker=None,
                                                            torch_dtype=torch.float16)
                                                            
        self.pipe.vae.to(device)
        self.device = device
    
    @torch.no_grad()
    def __call__(self, image):
        image = self.pipe.image_processor.preprocess(image).to(self.device, dtype=torch.float16)
        latents = self.pipe.vae.encode(image).latent_dist.sample()
        #latents = latents / self.pipe.vae.config.scaling_factor
        recon_image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(recon_image)
        return image

class StableDiffusionV2:
    def __init__(self, model_path: str, device: str = 'cuda', variant="2.0"):
        if variant not in ["2.0", "2.1"]:
            raise ValueError(f"Variant {variant} is not supported. Use '2.0' or '2.1'.")
        
        if variant == "2.0":
            scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
            self.pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                    scheduler=scheduler,
                                                    safety_checker=None,
                                                    torch_dtype=torch.float16)
        elif variant == "2.1":
            self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.vae.to(device)
        self.device = device
    
    @torch.no_grad()
    def __call__(self, image):
        image = self.pipe.image_processor.preprocess(image).to(self.device, dtype=torch.float16)
        latents = self.pipe.vae.encode(image).latent_dist.sample()
        #latents = latents / self.pipe.vae.config.scaling_factor
        recon_image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(recon_image)
        return image

class StableDiffusionXL:
    def __init__(self, model_path: str, device: str = 'cuda'):
        base_model_path = os.path.join(model_path, "stable-diffusion-xl-base-1.0")

        self.pipe = DiffusionPipeline.from_pretrained(base_model_path,
                                                      use_safetensors=True)
        self.pipe.vae.to(device)
        self.device = device
    
    @torch.no_grad()
    def __call__(self, image):
        image = self.pipe.image_processor.preprocess(image).to(self.device)
        latents = self.pipe.vae.encode(image).latent_dist.sample()
        #has_latents_mean = hasattr(self.pipe.vae.config, "latents_mean") and self.pipe.vae.config.latents_mean is not None
        #has_latents_std = hasattr(self.pipe.vae.config, "latents_std") and self.pipe.vae.config.latents_std is not None
        #if has_latents_mean and has_latents_std:
        #    latents_mean = (
        #        torch.tensor(self.pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        #    )
        #    latents_std = (
        #        torch.tensor(self.pipe.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        #    )
        #    latents = latents * latents_std / self.pipe.vae.config.scaling_factor + latents_mean
        #else:
        #    latents = latents / self.pipe.vae.config.scaling_factor
        
        recon_image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(recon_image)
        return image

class StableDiffusion3:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_path,
                                                             torch_dtype=torch.float16)
        self.pipe.vae.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, image):
        image = self.pipe.image_processor.preprocess(image).to(self.device, dtype=torch.float16)
        latents = self.pipe.vae.encode(image).latent_dist.sample()
        #latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        recon_image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(recon_image)
        return image

class FLUX:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        self.pipe.vae.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, image):
        image = self.pipe.image_processor.preprocess(image).to(self.device, dtype=torch.float16)
        latents = self.pipe.vae.encode(image).latent_dist.sample()
        #latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        recon_image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(recon_image)
        return image

def get_t2i_vae_pipeline(model_name: str, path: str, device: str = 'cuda'):
    diffusers.utils.logging.disable_progress_bar()

    if model_name == "stable-diffusion-v1":
        return StableDiffusionV1(model_path=path, device=device)
    elif model_name == "stable-diffusion-v2.0":
        return StableDiffusionV2(model_path=path, device=device)
    elif model_name == "stable-diffusion-v2.1":
        return StableDiffusionV2(model_path=path, device=device)
    elif model_name == "stable-diffusion-xl":
        return StableDiffusionXL(model_path=path, device=device)
    elif model_name == "stable-diffusion-3":
        return StableDiffusion3(model_path=path, device=device)
    elif model_name == "stable-diffusion-3.5":
        return StableDiffusion3(model_path=path, device=device)
    elif model_name == "stable-diffusion-3.5-medium":
        return StableDiffusion3(model_path=path, device=device)
    elif model_name == "stable-diffusion-3.5-turbo":
        return StableDiffusion3(model_path=path, device=device)
    elif model_name == "flux-schnell":
        return FLUX(model_path=path, device=device)
    elif model_name == "flux-dev":
        return FLUX(model_path=path, device=device)
    elif model_name == "qwen-image":
        return QwenImage(model_path=path, device=device)
    elif model_name == "sana":
        return Sana(model_path=path, device=device)
    elif model_name == "pixart-alpha":
        return PixArtAlpha(model_path=path, device=device)
    elif model_name == "lumina2":
        return Lumina2(model_path=path, device=device)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
