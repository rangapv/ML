#!/usr/bin/env python3

import torch
from llava.model import LlavaLlamaForCausalLM
from transformers import AutoTokenizer

model_path = "../../checkpoints/Meta-Llama-3-8B-Instruct"
mm_projector_path = "./mm_projector.bin"
output_path = "./merged_model"

# Load model
model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)

# Load mm_projector weights
mm_weights = torch.load(mm_projector_path, map_location="cpu")

# Check model has these keys
model_keys = [k for k in model.state_dict().keys() if "mm_projector" in k]
print("Model projector keys:", model_keys)

if len(model_keys) == 0:
    # Model not initialized with vision — manually add the weights
    model.model.mm_projector = torch.nn.Sequential(
        torch.nn.Linear(mm_weights["model.mm_projector.0.weight"].shape[1],
                        mm_weights["model.mm_projector.0.weight"].shape[0]),
        torch.nn.GELU(),
        torch.nn.Linear(mm_weights["model.mm_projector.2.weight"].shape[1],
                        mm_weights["model.mm_projector.2.weight"].shape[0]),
    )
    model.load_state_dict(mm_weights, strict=False)
else:
    # Keys exist, just load
    model.load_state_dict(mm_weights, strict=False)

# Save
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print("Saved merged model to:", output_path)
