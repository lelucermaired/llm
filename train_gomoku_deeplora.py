"""
train_gomoku_deeplora.py

深层LoRA训练：冻结前10层，只训练后18层
理论依据：浅层重置实验（reset_10_qv = +0.020）
  前10层LoRA是OOD损失来源 → 从训练阶段就不更新前10层
"""

import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0+PTX;8.6+PTX;8.9+PTX'
import sys, json, re
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

_orig = torch.load
def _patched(*a, **kw): kw['weights_only'] = False; return _orig(*a, **kw)
torch.load = _patched

import warnings; warnings.filterwarnings("ignore")

CONFIG = {
    "model_name":        "Qwen/Qwen2.5-7B-Instruct",
    "local_model_path":  None,
    "dataset_path":      "./datasets/real_games_v2/train.json",
    "output_dir":        "./checkpoints/qwen-gomoku-deeplora",
    "cache_dir":         "./cache",
    "freeze_layers":     10,

    "quantization": {
        "load_in_4bit":              True,
        "bnb_4bit_quant_type":       "nf4",
        "bnb_4bit_compute_dtype":    torch.float16,
        "bnb_4bit_use_double_quant": True,
    },
    "lora": {
        "r":              8,
        "lora_alpha":     16,
        "lora_dropout":   0.05,
        "bias":           "none",
        "target_modules": ["q_proj", "v_proj"],
    },
    "training": {
        "num_train_epochs":            1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate":               3e-6,
        "warmup_steps":                100,
        "max_grad_norm":               2.0,
        "optim":                       "adamw_torch",
        "lr_scheduler_type":           "linear",
        "weight_decay":                0.1,
        "logging_steps":               5,
        "save_strategy":               "epoch",
        "save_total_limit":            1,
        "seed":                        42,
        "gradient_checkpointing":      True,
        "fp16":                        False,
        "bf16":                        True,
    },
    "generation": {"max_length": 1024}
}


def main():
    print("=" * 65)
    print("五子棋深层LoRA训练（冻结前10层）")
    print(f"理论依据：浅层重置实验 reset_10_qv = +0.020")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    set_seed(CONFIG["training"]["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    print("\n加载数据集...")
    ds = load_dataset("json", data_files=CONFIG["dataset_path"],
                      cache_dir=CONFIG["cache_dir"])
    print(f"✅ 样本数: {len(ds['train'])}")

    # 加载模型
    print("\n加载模型...")
    model_name = CONFIG["local_model_path"] or CONFIG["model_name"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=CONFIG["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=CONFIG["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=CONFIG["quantization"]["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=CONFIG["quantization"]["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, cache_dir=CONFIG["cache_dir"], use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=CONFIG["cache_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora"]["r"],
        lora_alpha=CONFIG["lora"]["lora_alpha"],
        lora_dropout=CONFIG["lora"]["lora_dropout"],
        bias=CONFIG["lora"]["bias"],
        target_modules=CONFIG["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)

    # 冻结前10层LoRA
    freeze_n = CONFIG["freeze_layers"]
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            m = re.search(r'\.(\d+)\.', name)
            if m:
                if int(m.group(1)) < freeze_n:
                    param.requires_grad = False
                    frozen += 1
                else:
                    trainable += 1

    print(f"✅ 冻结前{freeze_n}层LoRA：{frozen}个参数张量")
    print(f"   可训练后{28-freeze_n}层LoRA：{trainable}个参数张量")
    model.print_trainable_parameters()

    # 预处理
    def tokenize(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            text = (f"<|im_start|>user\n{examples['instruction'][i]}<|im_end|>\n"
                    f"<|im_start|>assistant\n{examples['output'][i]}<|im_end|>")
            texts.append(text)
        tok = tokenizer(texts, truncation=True, padding=False,
                        max_length=CONFIG["generation"]["max_length"])
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = ds.map(tokenize, batched=True,
                       remove_columns=ds["train"].column_names,
                       desc="Tokenizing")

    # 训练
    print("\n开始训练...")
    gpu_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
    use_bf16 = CONFIG["training"]["bf16"] and gpu_cap[0] >= 8

    args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["training"]["num_train_epochs"],
        per_device_train_batch_size=CONFIG["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["training"]["gradient_accumulation_steps"],
        learning_rate=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
        warmup_steps=CONFIG["training"]["warmup_steps"],
        max_grad_norm=CONFIG["training"]["max_grad_norm"],
        optim=CONFIG["training"]["optim"],
        lr_scheduler_type=CONFIG["training"]["lr_scheduler_type"],
        logging_steps=CONFIG["training"]["logging_steps"],
        save_strategy=CONFIG["training"]["save_strategy"],
        save_total_limit=CONFIG["training"]["save_total_limit"],
        bf16=use_bf16, fp16=False,
        report_to="none",
        seed=CONFIG["training"]["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=CONFIG["training"]["gradient_checkpointing"],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
    )

    result = trainer.train()
    print(f"✅ 训练完成！Loss: {result.training_loss:.4f}")

    final_dir = os.path.join(CONFIG["output_dir"], "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ 模型保存至: {final_dir}")

    print("\n" + "=" * 65)
    print("🎉 完成！下一步：eval_all_50.py加入deeplora评测")
    print(f"  v2（全28层）         → 零迁移")
    print(f"  reset_10_qv（后处理）→ +0.020")
    print(f"  deeplora（后18层）   → 待评测")
    print("=" * 65)


if __name__ == "__main__":
    main()