#!/usr/bin/env python3
#author:rangapv@yahoo.com
#04-01-26

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training

base_model = "gpt2"
#base_model="meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    r=8,                         # Low-rank dimension
    lora_alpha=32,
    target_modules=["c_attn"],  # Target GPT2's attention layers
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM # Causal Language Modeling task
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#model = get_peft_model(model, lora_config)

# Load small dataset
dataset = load_dataset("imdb", split="train[:1%]")

# Preprocess the data
def tokenize(example):
    encoding = tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
#    encoding["label"] = [label for label in example["label"]]
    return encoding 
    
  
tokenized_dataset = dataset.map(tokenize, batched=True)
#tokenized_dataset.set_format(type="torch", columns=["input_ids", "label"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

small_train_dataset = tokenized_dataset

#print("before decode")
#print(small_train_dataset)
#for i in small_train_dataset['input_ids']: 
#    print(f"After decoding {tokenizer.decode(i)}")
#    print(f"")
print("after decode")


training_args = TrainingArguments(
    output_dir="./lora_imdb",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    #label_names=["label"],
#    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# Save the LoRA adapter (not full model)
model.save_pretrained("./lora_adapter_only")
tokenizer.save_pretrained("./lora_adapter_only")
base_model1 = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

peft_model = PeftModel.from_pretrained(base_model1, "./lora_adapter_only")
peft_model.eval()

# Inference
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = peft_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

