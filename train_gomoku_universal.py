"""
train_gomoku_universal.py

通用五子棋LoRA-SFT训练脚本，支持多种基座模型。
训练配置与原Qwen v2/cot_short完全一致（r=8, q+v, 1 epoch, lr=3e-6）。

用法:
  # Qwen2.5-7B (原始配置)
  python train_gomoku_universal.py --model qwen7b

  # Llama-3.1-8B
  python train_gomoku_universal.py --model llama8b

  # Qwen2.5-3B
  python train_gomoku_universal.py --model qwen3b

  # Mistral-7B
  python train_gomoku_universal.py --model mistral7b

  # 自定义数据集
  python train_gomoku_universal.py --model llama8b --dataset ./datasets/real_games_cot/train.json
"""

import os, sys, json, argparse
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
import warnings
warnings.filterwarnings("ignore")

# ==================== 模型配置 ====================
MODEL_CONFIGS = {
    "qwen7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "short": "qwen7b",
        "template": "qwen",  # <|im_start|> 格式
        "target_modules": ["q_proj", "v_proj"],
    },
    "qwen3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "short": "qwen3b",
        "template": "qwen",
        "target_modules": ["q_proj", "v_proj"],
    },
    "llama8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "short": "llama8b",
        "template": "llama3",
        "target_modules": ["q_proj", "v_proj"],
    },
    "mistral7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "short": "mistral7b",
        "template": "mistral",
        "target_modules": ["q_proj", "v_proj"],
    },
}

# ==================== 训练配置（与原v2/cot_short完全一致）====================
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
}

TRAIN_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 3e-6,
    "warmup_steps": 100,
    "max_grad_norm": 2.0,
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "weight_decay": 0.1,
    "logging_steps": 5,
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "seed": 42,
    "gradient_checkpointing": True,
    "bf16": True,
    "max_length": 1024,
}

# AutoDL路径适配
if os.path.exists("/root/autodl-tmp"):
    CACHE_DIR = "/root/autodl-tmp/hf_cache"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", CACHE_DIR)
else:
    CACHE_DIR = "./cache"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="基座模型: qwen7b / qwen3b / llama8b / mistral7b")
    parser.add_argument("--dataset", default="./datasets/real_games_v2/train.json",
                        help="训练数据路径")
    parser.add_argument("--output", default=None,
                        help="输出目录(默认自动生成)")
    return parser.parse_args()


def format_chat(instruction, output, template):
    """根据模型类型格式化chat template"""
    if template == "qwen":
        return (f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>")
    elif template == "llama3":
        return (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{instruction}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{output}<|eot_id|>")
    elif template == "mistral":
        return f"[INST] {instruction} [/INST]{output}</s>"
    else:
        raise ValueError(f"未知template: {template}")


def setup(args, model_cfg):
    output_dir = args.output or f"./checkpoints/{model_cfg['short']}-gomoku-sft"
    print("=" * 65)
    print(f"五子棋 LoRA-SFT 训练")
    print(f"基座模型: {model_cfg['name']}")
    print(f"数据集:   {args.dataset}")
    print(f"输出目录: {output_dir}")
    print(f"配置:     r=8, q+v, 1 epoch, lr=3e-6")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    set_seed(TRAIN_CONFIG["seed"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump({
            "base_model": model_cfg["name"],
            "model_key": args.model,
            "dataset": args.dataset,
            "lora": LORA_CONFIG,
            "train": TRAIN_CONFIG,
            "start_time": datetime.now().isoformat(),
        }, f, indent=2)

    return output_dir


def load_data(dataset_path):
    print("\n加载数据集...")
    ds = load_dataset("json", data_files=dataset_path, cache_dir=CACHE_DIR)
    n = len(ds["train"])
    print(f"  样本数: {n}")
    sample = ds["train"][0]
    if "instruction" not in sample or "output" not in sample:
        raise ValueError(f"数据集缺少instruction/output字段")
    return ds


def build_model(model_cfg):
    print(f"\n加载 {model_cfg['name']} ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        target_modules=model_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def preprocess(dataset, tokenizer, template):
    print("\n预处理数据集...")

    def tokenize(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            text = format_chat(examples["instruction"][i], examples["output"][i], template)
            texts.append(text)
        tok = tokenizer(texts, truncation=True, padding=False,
                        max_length=TRAIN_CONFIG["max_length"])
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = dataset.map(
        tokenize, batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    print(f"  预处理完成, 样本数: {len(tokenized['train'])}")
    return tokenized


def train(model, tokenizer, dataset, output_dir):
    print("\n开始训练（1 epoch）...")
    gpu_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    use_bf16 = TRAIN_CONFIG["bf16"] and gpu_cap[0] >= 8

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        max_grad_norm=TRAIN_CONFIG["max_grad_norm"],
        optim=TRAIN_CONFIG["optim"],
        lr_scheduler_type=TRAIN_CONFIG["lr_scheduler_type"],
        logging_strategy="steps",
        logging_steps=TRAIN_CONFIG["logging_steps"],
        save_strategy=TRAIN_CONFIG["save_strategy"],
        save_total_limit=TRAIN_CONFIG["save_total_limit"],
        bf16=use_bf16,
        fp16=False,
        report_to="none",
        seed=TRAIN_CONFIG["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=TRAIN_CONFIG["gradient_checkpointing"],
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
    print(f"  训练完成! Loss: {result.training_loss:.4f}")

    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(result.metrics, f, indent=2)

    return result


def save_model(model, tokenizer, output_dir):
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    has_cfg = os.path.exists(os.path.join(final_dir, "adapter_config.json"))
    weight_files = ["adapter_model.safetensors", "adapter_model.bin"]
    has_w = any(os.path.exists(os.path.join(final_dir, f)) and
                os.path.getsize(os.path.join(final_dir, f)) > 1024 * 1024
                for f in weight_files)
    if not has_cfg or not has_w:
        raise RuntimeError("模型保存验证失败!")
    print(f"  模型保存至: {final_dir}")
    return final_dir


def main():
    args = parse_args()
    model_cfg = MODEL_CONFIGS[args.model]

    try:
        output_dir = setup(args, model_cfg)
        dataset = load_data(args.dataset)
        model, tokenizer = build_model(model_cfg)
        tokenized = preprocess(dataset, tokenizer, model_cfg["template"])
        result = train(model, tokenizer, tokenized, output_dir)
        final_dir = save_model(model, tokenizer, output_dir)

        print("\n" + "=" * 65)
        print(f"训练完成!")
        print(f"模型: {model_cfg['name']}")
        print(f"Adapter: {final_dir}")
        print(f"Loss: {result.training_loss:.4f}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 65)

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()