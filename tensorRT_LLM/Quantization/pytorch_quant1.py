#!/usr/bin/env python3
#author:rangapv@yahoo.com
#09-12-25

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.quantization import QuantAlgo

def main():
    # Model could accept HF model name, a path to local HF model,
    # or TensorRT Model Optimizer's quantized checkpoints like nvidia/Llama-3.1-8B-Instruct-FP8 on HF.
    algo_quant=QuantAlgo('W4A16_AWQ')
#   config_quant=QuantConfig(quant_algo=algo_quant)
    quantization=algo_quant
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
        "What day is celebrated today ?",

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    for output in llm.generate(prompts, sampling_params, quantization):
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
