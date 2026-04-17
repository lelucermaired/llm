import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0+PTX;8.6+PTX;8.9+PTX'
import sys, json
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

_orig = torch.load
def _patched(*a, **kw): kw['weights_only'] = False; return _orig(*a, **kw)
torch.load = _patched

import warnings; warnings.filterwarnings("ignore")

CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "local_model_path": None,
    "dataset_path": "./datasets/gsm8k_sft/train.json",
    "output_dir": "./checkpoints/qwen-gsm8k-sft",
    "cache_dir": "./cache",

    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
    },
    "lora": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj"],  # 与v2保持一致，便于对比
    },
    "training": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 3e-6,
        "warmup_steps": 50,
        "max_grad_norm": 2.0,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": 0.1,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "seed": 42,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
    },
    "generation": {
        "max_length": 1024,
    }
}


def setup():
    print("=" * 65)
    print("GSM8K SFT 训练")
    print("源任务：数学推理（1000条CoT数据）")
    print("目标：测试数学SFT能否向逻辑/空间推理迁移")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    set_seed(CONFIG["training"]["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} | "
              f"显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print(f"\n实验设计说明：")
    print(f"  - 与v2模型配置完全一致（r=8，q+v，1 epoch）")
    print(f"  - 只有源任务不同：五子棋→数学推理")
    print(f"  - 评测时对比：base / v2（五子棋）/ gsm8k（数学）")
    print(f"  - 若数学SFT→逻辑正向迁移，说明源任务结构相似性是关键")


def load_data():
    print("\n加载GSM8K数据集...")
    ds = load_dataset("json", data_files=CONFIG["dataset_path"],
                      cache_dir=CONFIG["cache_dir"])
    print(f"✅ 样本数: {len(ds['train'])}")
    return ds


def build_model():
    print("\n加载模型，应用QLoRA...")
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
    model.print_trainable_parameters()
    return model, tokenizer


def preprocess(dataset, tokenizer):
    print("\n预处理数据集...")

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

    tokenized = dataset.map(tokenize, batched=True,
                            remove_columns=dataset["train"].column_names,
                            desc="Tokenizing")
    print(f"✅ 预处理完成，样本数: {len(tokenized['train'])}")
    return tokenized


def save_model(model, tokenizer, output_dir):
    print(f"\n保存模型至: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    weight_files = ["adapter_model.safetensors", "adapter_model.bin"]
    has_cfg = os.path.exists(os.path.join(output_dir, "adapter_config.json"))
    has_w = any(os.path.exists(os.path.join(output_dir, f)) and
                os.path.getsize(os.path.join(output_dir, f)) > 1024*1024
                for f in weight_files)
    if not has_cfg or not has_w:
        raise RuntimeError("模型保存验证失败")
    print("✅ 保存验证通过")


def train(model, tokenizer, dataset):
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
        logging_strategy="steps",
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
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model, args=args,
        train_dataset=dataset["train"],
        data_collator=collator,
    )
    result = trainer.train()
    print(f"✅ 训练完成！Loss: {result.training_loss:.4f}")

    final_dir = os.path.join(CONFIG["output_dir"], "final_model")
    save_model(model, tokenizer, final_dir)

    with open(os.path.join(CONFIG["output_dir"], "training_metrics.json"), "w") as f:
        json.dump(result.metrics, f, indent=2)
    return result


def main():
    try:
        setup()
        dataset = load_data()
        model, tokenizer = build_model()
        tokenized = preprocess(dataset, tokenizer)
        result = train(model, tokenizer, tokenized)

        print("\n" + "=" * 65)
        print("🎉 GSM8K SFT 训练完成！")
        final_dir = os.path.join(CONFIG['output_dir'], 'final_model')
        print(f"模型保存在: {final_dir}")
        print(f"\n下一步：运行 gsm8k_transfer_eval.py 评测迁移效果")
        print(f"对比：base / v2（五子棋SFT）/ gsm8k（数学SFT）")
        print(f"评测任务：数学推理 / 空间推理 / 序列规划 / 逻辑推理")
        print("=" * 65)
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback; traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())