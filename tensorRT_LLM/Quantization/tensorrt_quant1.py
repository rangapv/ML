#!/usr/bin/env python3
#author:rangapv@yahoo.com
#12-12-25

from tensorrt_llm import SamplingParams
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import KvCacheConfig, QuantAlgo, QuantConfig

def main():
    # Model could accept HF model name, a path to local HF model,
    # or TensorRT Model Optimizer's quantized checkpoints like nvidia/Llama-3.1-8B-Instruct-FP8 on HF.
#   config_quant=QuantConfig(quant_algo=algo_quant)
    quant_algo1=QuantAlgo('W4A16_AWQ')
    kv_cache_algo1=quant_algo1
    quant_config1=QuantConfig(quant_algo=quant_algo1)

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",quant_config=quant_config1)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
        "What day is celebrated today ?",
         ]
    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    for output in llm.generate(prompts, sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )

    # Got output like
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The president of the United States is', Generated text: 'likely to nominate a new Supreme Court justice to fill the seat vacated by the death of Antonin Scalia. The Senate should vote to confirm the'
    # Prompt: 'The capital of France is', Generated text: 'Paris.'
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'

if __name__ == '__main__':
    main()
