#!/usr/bin/env python3
#author:rangapv@yahoo.com
#03-01-26

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

#base_model_name = "gpt2"
base_model="meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

lora_config = LoraConfig(
    r=8,                         # Low-rank dimension
    #lora_dir=["./"],
    #lora_alpha=32,
    #target_modules=["c_attn"],  # Target GPT2's attention layers
    lora_dropout=0.1,
    bias="none",
    #max_loras=1,
    #max_cpu_loras=1,
    task_type=TaskType.CAUSAL_LM # Causal Language Modeling task
)

model = get_peft_model(model, lora_config)

model.save_pretrained("./new")

model.print_trainable_parameters()

