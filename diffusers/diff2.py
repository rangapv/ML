#!/usr/bin/env python3

import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="balanced"
)
pipeline.load_lora_weights(
  "flymy-ai/qwen-image-realism-lora",
)

prompt = """
super Realism cinematic film still of a cat sipping a margarita in a pool in Palm Springs in the style of umempart, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
torch.no_grad()
image1=pipeline(prompt).images[0]
image1.save("result2.png")
