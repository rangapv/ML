#!/usr/bin/env python3
#author:rangapv@yahoo.com
#30-12-25


from tensorrt_llm import LLM
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.sampling_params import SamplingParams

#base_model="meta-llama/llama-7b-hf/"
base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#base_model="meta-llama/Llama-2-7b-hf/"
lora_weights="kunishou/Japanese-Alpaca-LoRA-7b-v0"
#kunishou/Japanese-Alpaca-LoRA-7b-v0


# Configure LoRA
lora_config = LoraConfig(
    lora_dir=["kunishou/Japanese-Alpaca-LoRA-7b-v0"],
    max_lora_rank=8,
    max_loras=1,
    max_cpu_loras=1
)

# Initialize LLM with LoRA support
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_config=lora_config
)

# Create LoRA request
lora_request = LoRARequest("my-lora-task", 0, "kunishou/Japanese-Alpaca-LoRA-7b-v0")

# Generate with LoRA
prompts = ["<s> アメリカ合衆国の首都はどこですか? \n答え:"]
sampling_params = SamplingParams(max_tokens=50)

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=[lora_request]
)
