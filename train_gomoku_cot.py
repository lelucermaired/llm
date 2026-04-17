"""
train_gomoku_cot.py

训练五子棋CoT模型，支持不同质量的推理链数据
配置与v2完全一致（r=8, q+v, 1 epoch），唯一变量是数据集

用法：
  # 结构化短推理链
  python train_gomoku_cot.py \
    --dataset ./datasets/real_games_cot/train.json \
    --output  ./checkpoints/qwen-gomoku-cot-short

  # 详细长推理链
  python train_gomoku_cot.py \
    --dataset ./datasets/real_games_detailed_cot/train.json \
    --output  ./checkpoints/qwen-gomoku-cot-detailed
"""

import os, sys, json, argparse
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0+PTX;8.6+PTX;8.9+PTX'

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

# ==================== 固定配置（与v2完全一致）====================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR  = "./cache"

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
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
    "fp16": False,
    "max_length": 1024,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="训练数据路径，如 ./datasets/real_games_cot/train.json")
    parser.add_argument("--output",  required=True,
                        help="模型输出目录，如 ./checkpoints/qwen-gomoku-cot-short")
    return parser.parse_args()


def setup(args):
    dataset_name = os.path.basename(os.path.dirname(args.dataset))
    print("=" * 65)
    print("五子棋CoT SFT训练")
    print(f"数据集：{args.dataset}")
    print(f"输出目录：{args.output}")
    print(f"配置：r=8, q+v, 1 epoch, lr=3e-6（与v2完全一致）")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    set_seed(TRAIN_CONFIG["seed"])
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(CACHE_DIR,   exist_ok=True)

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} | "
              f"显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # 保存实验配置
    with open(os.path.join(args.output, "experiment_config.json"), "w") as f:
        json.dump({
            "dataset": args.dataset,
            "base_model": BASE_MODEL,
            "lora": LORA_CONFIG,
            "train": TRAIN_CONFIG,
            "start_time": datetime.now().isoformat(),
        }, f, indent=2)


def load_data(dataset_path):
    print("\n加载数据集...")
    ds = load_dataset("json", data_files=dataset_path, cache_dir=CACHE_DIR)
    n = len(ds["train"])
    print(f"✅ 样本数: {n}")

    # 检查字段
    sample = ds["train"][0]
    if "instruction" not in sample or "output" not in sample:
        raise ValueError(f"数据集缺少instruction/output字段，实际字段：{list(sample.keys())}")

    # 打印第一条样本的output长度分布信息
    lengths = [len(ds["train"][i]["output"]) for i in range(min(100, n))]
    avg_len = sum(lengths) / len(lengths)
    print(f"output平均长度（前100条）：{avg_len:.0f} 字符")

    return ds


def build_model():
    print("\n加载模型，应用QLoRA...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, cache_dir=CACHE_DIR, use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, cache_dir=CACHE_DIR)
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
        target_modules=LORA_CONFIG["target_modules"],
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
                        max_length=TRAIN_CONFIG["max_length"])
        tok["labels"] = tok["input_ids"].copy()
        return tok

    # 只保留instruction和output列
    keep_cols = ["instruction", "output"]
    remove_cols = [c for c in dataset["train"].column_names if c not in keep_cols]

    tokenized = dataset.map(
        tokenize, batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    print(f"✅ 预处理完成，样本数: {len(tokenized['train'])}")
    return tokenized


def save_model(model, tokenizer, output_dir):
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # 验证
    has_cfg = os.path.exists(os.path.join(final_dir, "adapter_config.json"))
    weight_files = ["adapter_model.safetensors", "adapter_model.bin"]
    has_w = any(os.path.exists(os.path.join(final_dir, f)) and
                os.path.getsize(os.path.join(final_dir, f)) > 1024*1024
                for f in weight_files)
    if not has_cfg or not has_w:
        raise RuntimeError("模型保存验证失败")
    print(f"✅ 模型保存至: {final_dir}")
    return final_dir


def train(model, tokenizer, dataset, output_dir):
    print("\n开始训练（1 epoch）...")
    gpu_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
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
        bf16=use_bf16, fp16=False,
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
    print(f"✅ 训练完成！Loss: {result.training_loss:.4f}")

    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(result.metrics, f, indent=2)

    return result


def main():
    args = parse_args()
    try:
        setup(args)
        dataset  = load_data(args.dataset)
        model, tokenizer = build_model()
        tokenized = preprocess(dataset, tokenizer)
        result = train(model, tokenizer, tokenized, args.output)
        final_dir = save_model(model, tokenizer, args.output)

        print("\n" + "=" * 65)
        print("🎉 训练完成！")
        print(f"模型：{final_dir}")
        print(f"Loss：{result.training_loss:.4f}")
        print(f"\n下一步：运行评测脚本，对比三组模型的OOD迁移效果")
        print(f"  base          → 基准")
        print(f"  v2            → 伪推理链（原始数据）")
        print(f"  cot-short     → 结构化短推理链")
        print(f"  cot-detailed  → Qwen生成详细推理链（193条）")
        print("=" * 65)

    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()