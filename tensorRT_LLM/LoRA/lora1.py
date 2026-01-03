#!/usr/bin/env python3
#author:rangapv@yahoo.com
#03-01-26

from tensorrt_llm import LLM
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.sampling_params import SamplingParams

# Configure LoRA
lora_config = LoraConfig(
    lora_dir=["./new"],
    max_lora_rank=8,
    max_loras=1,
    max_cpu_loras=1
)

base_model="meta-llama/Llama-2-7b-hf"

# Initialize LLM with LoRA support
llm = LLM(
    model=base_model,
    lora_config=lora_config
)

# Create LoRA request
lora_request = LoRARequest("my-lora-task", 0, "./new")

# Generate with LoRA
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(max_tokens=50)

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=[lora_request]
)

print(outputs)
