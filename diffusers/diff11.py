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
A breakfast plate with items Salmon fillet, brocolli on the side and mushroom on the side
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
image=pipeline(prompt).images[0]
image.save("result.png")
