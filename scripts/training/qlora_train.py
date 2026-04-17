import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# ===============================
# 1. 模型选择（0.5B，小、稳、CPU 可跑）
# ===============================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ===============================
# 2. Tokenizer
# ===============================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False,   # Windows + sentencepiece 更稳
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===============================
# 3. 模型加载（CPU / fp32）
# ===============================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=None,          # 强制 CPU
    trust_remote_code=True,
)

# ===============================
# 4. LoRA 配置（QLoRA 的“L”部分）
# ===============================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Qwen 架构通用
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

print("\n===== Trainable Parameters =====")
model.print_trainable_parameters()

# ===============================
# 5. 极小数据集（只为跑通）
# ===============================
data = [
    {
        "text": "用户：什么是 LoRA？\n助手："
    },
    {
        "text": "用户：QLoRA 的核心思想是什么？\n助手："
    },
]

dataset = Dataset.from_list(data)

def tokenize_fn(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    result["labels"] = result["input_ids"].copy()
    return result

dataset = dataset.map(tokenize_fn, remove_columns=["text"])

# ===============================
# 6. 训练参数（只跑 1 step）
# ===============================
training_args = TrainingArguments(
    output_dir="./lora-qwen05b-debug",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=1,              # ⭐ 关键：只跑 1 step
    logging_steps=1,
    save_steps=1,
    fp16=False,
    bf16=False,
    report_to="none",
)

# ===============================
# 7. Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# ===============================
# 8. 开始训练
# ===============================
trainer.train()

print("\n✅ CPU LoRA 调试完成")
