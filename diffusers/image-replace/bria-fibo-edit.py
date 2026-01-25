#!/usr/bin/env python3
#author:rangapv@yahoo.com
#25-01-26

import torch
from diffusers import BriaFiboEditPipeline
from diffusers.modular_pipelines import ModularPipelineBlocks
import json
from PIL import Image

torch.set_grad_enabled(False)
vlm_pipe = ModularPipelineBlocks.from_pretrained("briaai/FIBO-VLM-prompt-to-JSON", trust_remote_code=True)
vlm_pipe = vlm_pipe.init_pipeline()

pipe = BriaFiboEditPipeline.from_pretrained(
    "briaai/fibo-edit",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

output = vlm_pipe(
    prompt="A breakfast plate with cooked salmon fillet, with brocolli and mushrooms on the side, high resolution, attention to detail"
)
json_prompt_generate = json.loads(output.values["json_prompt"])

image = Image.open("./salmon.png")

edit_prompt = "Make the salmon fillet to be a steamed carrots" 

json_prompt_generate["edit_instruction"] = edit_prompt

results_generate = pipe(
    prompt=json_prompt_generate, num_inference_steps=50, guidance_scale=3.5, image=image, output_type="pil"
)

results_generate.images[0].save("image_generate.png")

