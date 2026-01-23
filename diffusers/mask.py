#!/usr/bin/env python3
#author:rangapv@yahoo.com
#20-01-26

import torch
from diffusers import DiffusionPipeline

import gc

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


clear_memory()
torch.no_grad()
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="balanced"
)

prompt = """
Draw a medium size solid white rectangle at the center with black background
I want to use this image as a mask image
"""
image=pipeline(prompt).images[0]
image.save("square.png")
