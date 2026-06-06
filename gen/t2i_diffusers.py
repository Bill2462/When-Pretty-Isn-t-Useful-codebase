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
                                                     torch_dtype=torch.bfloat16)
        self.pipe.to(device)
        self.guidance_scale = 7.5
        self.num_inference_steps = 50

    def __call__(self, prompts: list[str], width: int = 512, height: int = 512) -> list:
        out = self.pipe(prompt=prompts,
                        height=height,
                        width=width,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps)
        
        images = out.images
        return images

class Sana:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = SanaPipeline.from_pretrained(model_path,
                                                torch_dtype=torch.bfloat16)
        self.pipe.to(device)
        self.guidance_scale = 7.5
        self.num_inference_steps = 50
    
    def __call__(self, prompts: list[str], width: int = 512, height: int = 512) -> list:
        out = self.pipe(prompt=prompts,
                        height=height,
                        width=width,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps,
                        complex_human_instruction=None)
        
        images = out.images
        return images

class PixArtAlpha:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = PixArtAlphaPipeline.from_pretrained(model_path,
                                                       torch_dtype=torch.bfloat16)
        self.pipe.to(device)
        self.guidance_scale = 7.5
        self.num_inference_steps = 50
    
    def __call__(self, prompts: list[str], width: int = 512, height: int = 512) -> list:
        out = self.pipe(prompt=prompts,
                        height=height,
                        width=width,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps)
        
        images = out.images
        return images
    
class Lumina2:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = Lumina2Pipeline.from_pretrained(model_path,
                                                   torch_dtype=torch.bfloat16)
        self.pipe.to(device)
        self.guidance_scale = 7.5
        self.num_inference_steps = 50
    
    def __call__(self, prompts: list[str], width: int = 512, height: int = 512) -> list:
        out = self.pipe(prompt=prompts,
                        height=height,
                        width=width,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps)
        
        images = out.images
        return images

class StableDiffusionV1:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                            safety_checker=None,
                                                            torch_dtype=torch.float16)
                                                            
        self.pipe.to(device)
        self.guidance_scale = 7.5
        self.num_inference_steps = 50

    def __call__(self, prompts: list[str], width: int = 512, height: int = 512,
                 output_latent: bool = False) -> list:
        out = self.pipe(prompt=prompts,
                        height=height,
                        width=width,
                        guidance_scale=self.guidance_scale,
                        output_type="latent" if output_latent else "pil",
                        num_inference_steps=self.num_inference_steps)
        
        images = out.images
        return images
    
    @torch.no_grad()
    def decode_latents(self, latents):
        imgs = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        imgs = self.pipe.image_processor.postprocess(imgs, output_type="pil", do_denormalize=[True] * imgs.shape[0])
        return imgs

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

        self.pipe.to(device)

        self.guidance_scale = 7.5
        self.num_inference_steps = 50

    def __call__(self, prompts: list[str], width: int = 512, height: int = 512) -> list:
        out = self.pipe(prompt=prompts,
                        height=height,
                        width=width,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps)
        
        images = out.images
        return images
    
class StableDiffusionXL:
    def __init__(self, model_path: str, device: str = 'cuda', use_refiner: bool = False):
        refiner_path = os.path.join(model_path, "stable-diffusion-xl-refiner-1.0")
        base_model_path = os.path.join(model_path, "stable-diffusion-xl-base-1.0")

        self.use_refiner = use_refiner

        self.pipe = DiffusionPipeline.from_pretrained(base_model_path,
                                                      torch_dtype=torch.float16,
                                                      use_safetensors=True,
                                                      variant="fp16")
        self.pipe.to(device)

        if use_refiner:
            self.refiner = DiffusionPipeline.from_pretrained(
                refiner_path,
                text_encoder_2=self.pipe.base.text_encoder_2,
                vae=self.pipe.base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.refiner.to(device)

        self.num_inference_steps = 50
        self.guidance_scale = 5.0
        self.high_noise_frac = 0.8

    def _pipe_no_refiner(self, prompts, width, height, output_latent):
        out = self.pipe(prompt=prompts,
                        height=height,
                        width=width,
                        output_type="latent" if output_latent else "pil",
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale)
        images = out.images
        return images
    
    def _pipe_with_refiner(self, prompts, width, height, output_latent):
        out = self.pipe(
            prompt=prompts,
            height=height,
            width=width,
            num_inference_steps=self.num_inference_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
        ).images

        out = self.refiner(
            prompt=prompts,
            num_inference_steps=self.num_inference_steps,
            denoising_start=self.high_noise_frac,
            output_type="latent" if output_latent else "pil",
            image=out,
        )

        images = out.images
        return images
    
    def __call__(self, prompts: list[str],
                 width: int = 1024,
                 height: int = 1024,
                 output_latent: bool = False) -> list:
        if self.use_refiner:
            return self._pipe_with_refiner(prompts, width, height, output_latent)
        else:
            return self._pipe_no_refiner(prompts, width, height, output_latent)

class StableDiffusionTurbo:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_path,
                                                              torch_dtype=torch.float16,
                                                              variant="fp16")
        self.pipe.to(device)

        self.guidance_scale = 0.0
        self.num_inference_steps = 4

    def __call__(self, prompts: list[str], width: int, height: int) -> list:
        out = self.pipe(prompt=prompts,
                        num_inference_steps=4,
                        guidance_scale=0.0)
        
        images = out.images
        return images

class StableDiffusion3:
    def __init__(self, model_path: str, device: str = 'cuda', variant: str = "3.0"):
        self.variant = variant
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_path,
                                                             torch_dtype=torch.bfloat16)
        
        if variant == "3.0":
            self.guidance_scale = 7.0
            self.num_inference_steps = 28
        elif variant == "3.5":
            self.guidance_scale = 3.5
            self.num_inference_steps = 28
        elif variant == "3.5-medium":
            self.guidance_scale = 4.5
            self.num_inference_steps = 40
        elif variant == "3.5-turbo":
            self.guidance_scale = 0.0
            self.num_inference_steps = 4
        else:
            raise ValueError(f"Variant {variant} is not supported.")
        
        self.pipe.to(device)

    def __call__(self, prompts: list[str], width: int = 512, height: int = 512,
                 output_latent: bool = False) -> list:
        if self.variant == "3.0":
            out = self.pipe(prompt=prompts,
                            height=height,
                            width=width,
                            guidance_scale=self.guidance_scale,
                            output_type="latent" if output_latent else "pil",
                            num_inference_steps=self.num_inference_steps)
        else:
            out = self.pipe(prompt=prompts,
                            height=height,
                            width=width,
                            num_inference_steps=self.num_inference_steps,
                            output_type="latent" if output_latent else "pil",
                            guidance_scale=self.guidance_scale)
        
        images = out.images
        return images

class FLUX:
    def __init__(self, model_path: str, device: str = 'cuda', variant="dev"):
        self.pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.pipe.to(device)
        self.variant = variant

        if variant == "dev":
            self.guidance_scale = 3.5
            self.num_inference_steps = 50
        elif variant == "shnell":
            self.guidance_scale = 0.0
            self.num_inference_steps = 4
        else:
            raise ValueError(f"Variant {variant} is not supported.")
    

    def __call__(self, prompts: list[str], width: int = 1024, height: int = 1024) -> list:
        if self.variant == "dev":
            output = self.pipe(prompts,
                               height=height,
                               width=width,
                               guidance_scale=self.guidance_scale,
                               num_inference_steps=self.num_inference_steps,
                               max_sequence_length=512)
        elif self.variant == "shnell":
            output = self.pipe(prompts,
                               guidance_scale=self.guidance_scale,
                               num_inference_steps=self.num_inference_steps,
                               height=height,
                               width=width,
                               max_sequence_length=256)
        else:
            raise ValueError(f"Variant {self.variant} is not supported.")

        images = output.images
        return images

def get_t2i_model(model_name: str, path: str, device: str = 'cuda'):
    diffusers.utils.logging.disable_progress_bar()

    if model_name == "stable-diffusion-v1":
        return StableDiffusionV1(model_path=path, device=device)
    elif model_name == "stable-diffusion-v2.0":
        return StableDiffusionV2(model_path=path, device=device, variant="2.0")
    elif model_name == "stable-diffusion-v2.1":
        return StableDiffusionV2(model_path=path, device=device, variant="2.1")
    elif model_name == "stable-diffusion-xl":
        return StableDiffusionXL(model_path=path, device=device)
    elif model_name == "stable-diffusion-xl-refiner":
        return StableDiffusionXL(model_path=path, device=device, use_refiner=True)
    elif model_name == "stable-diffusion-turbo":
        return StableDiffusionTurbo(model_path=path, device=device)
    elif model_name == "stable-diffusion-3":
        return StableDiffusion3(model_path=path, device=device, variant="3.0")
    elif model_name == "stable-diffusion-3.5":
        return StableDiffusion3(model_path=path, device=device, variant="3.5")
    elif model_name == "stable-diffusion-3.5-medium":
        return StableDiffusion3(model_path=path, device=device, variant="3.5-medium")
    elif model_name == "stable-diffusion-3.5-turbo":
        return StableDiffusion3(model_path=path, device=device, variant="3.5-turbo")
    elif model_name == "flux-schnell":
        return FLUX(model_path=path, device=device, variant="shnell")
    elif model_name == "flux-dev":
        return FLUX(model_path=path, device=device, variant="dev")
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
