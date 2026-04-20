import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 配置
base_model = "meta-llama/Llama-3.1-8B-Instruct"   # 第一次运行会自动下载，建议后续改为本地路径
output_dir = "./checkpoints/llama8b-gomoku-maxlora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",            # 自动分配，如果显存不足可改为 {"":0}
    trust_remote_code=True,
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 使用中文训练数据
dataset = load_dataset("json", data_files="datasets/real_games_v2/train.json")

def tokenize_function(examples):
    texts = []
    for inst, out in zip(examples["instruction"], examples["output"]):
        # Llama-3 对话模板
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|>"
        texts.append(text)
    tokenized = tokenizer(texts, truncation=True, padding=False, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    report_to="none",
    gradient_checkpointing=True,
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Starting training...")
trainer.train()
model.save_pretrained(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")
print("Training completed.")