#!/usr/bin/env python3
#author:rangapv@yahoo.com
#20-01-26

from io import BytesIO
import PIL
import torch
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline

import gc

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


clear_memory()
torch.no_grad()
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))

def download_image(url):
    response = url
    return PIL.Image.open(response).convert("RGB")

init_image=download_image("./salmon.png").resize((512, 512))
mask_image=download_image("./square.png").resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=torch.float16, use_safetensors=False
)
pipe = pipe.to("cuda")

prompt = """
Blue color salmon fillet, high resolution
"""
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
image.save("result57.png")
