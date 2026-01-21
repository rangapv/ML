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
cinematic film still of a cat talking on a iPhone on a beach during sunrise 
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
image4=pipeline(prompt).images[0]
image4.save("result4.png")
