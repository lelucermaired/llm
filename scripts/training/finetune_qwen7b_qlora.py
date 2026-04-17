import os

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0+PTX;8.6+PTX;8.9+PTX'
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

import torch

# 保存原始 torch.load 函数
_original_torch_load = torch.load


# 覆盖 torch.load，默认强制 weights_only=False
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


# 应用 monkey patch
torch.load = patched_torch_load

import warnings

warnings.filterwarnings("ignore")

# ==================== 实验配置（根据你的情况修改） ====================
CONFIG = {
    # 模型配置
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "local_model_path": None,

    # 数据配置
    "dataset_path": "./datasets/real_games_v2/train.json",
    "output_dir": "./checkpoints/qwen-gomoku-real",
    "cache_dir": "./cache",

    # QLoRA 量化配置
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
        "target_modules": ["q_proj", "v_proj"],
    },

    # 训练超参数
    "training": {
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
        "save_steps": 100,
        "save_total_limit": 2,
        "seed": 42,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
    },

    # 生成配置
    "generation": {
        "max_length": 1024,
        "temperature": 0.1,
        "do_sample": False,
    }
}


# =====================================================================

def setup_environment():
    """设置实验环境"""
    print("=" * 70)
    print("Qwen2.5-7B-Instruct QLoRA 微调脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    set_seed(CONFIG["training"]["seed"])

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_capability = torch.cuda.get_device_capability(0)
        print(f"✅ GPU可用: {gpu_name}")
        print(f"✅ 显存大小: {gpu_memory:.2f} GB")
        print(f"✅ CUDA版本: {torch.version.cuda}")
        print(f"✅ GPU计算能力: {gpu_capability[0]}.{gpu_capability[1]}")
    else:
        print("❌ 警告: 未检测到GPU，将在CPU上运行（极慢！）")

    config_file = os.path.join(CONFIG["output_dir"], "training_config.json")
    serializable_config = {}
    for key, value in CONFIG.items():
        if key == "quantization":
            quant_config = value.copy()
            quant_config["bnb_4bit_compute_dtype"] = str(quant_config["bnb_4bit_compute_dtype"])
            serializable_config[key] = quant_config
        else:
            serializable_config[key] = value.copy() if isinstance(value, dict) else value

    serializable_config["environment"] = {
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(serializable_config, f, indent=2, ensure_ascii=False)
    print(f"✅ 配置已保存至: {config_file}")


def load_and_prepare_data():
    """加载并准备五子棋数据集"""
    print("\n" + "=" * 70)
    print("步骤1: 加载和预处理数据集")
    print("=" * 70)

    dataset_path = CONFIG["dataset_path"]

    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 数据集文件不存在: {dataset_path}")
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    print(f"加载数据集: {dataset_path}")
    try:
        dataset = load_dataset("json", data_files=dataset_path, cache_dir=CONFIG["cache_dir"])
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        raise

    print(f"数据集加载成功，样本数: {len(dataset['train'])}")

    print("\n数据集前3个样本预览:")
    for i in range(min(3, len(dataset["train"]))):
        example = dataset["train"][i]
        print(f"\n--- 样本 {i + 1} ---")
        print(f"指令: {example['instruction'][:100]}...")
        if 'input' in example and example['input']:
            print(f"输入: {example['input'][:50]}...")
        print(f"输出: {example['output'][:100]}...")

    return dataset


def create_model_and_tokenizer():
    """加载模型和分词器，应用QLoRA配置"""
    print("\n" + "=" * 70)
    print("步骤2: 加载模型和分词器 (应用4-bit QLoRA量化)")
    print("=" * 70)

    model_name = CONFIG["local_model_path"] or CONFIG["model_name"]

    print("配置4-bit量化参数...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=CONFIG["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=CONFIG["quantization"]["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=CONFIG["quantization"]["bnb_4bit_use_double_quant"],
    )

    print(f"加载模型: {model_name}")
    print("注意: 首次下载需要较长时间和约15GB磁盘空间...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CONFIG["cache_dir"],
            use_cache=False,
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=CONFIG["cache_dir"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("✅ 分词器加载成功！")

    print("准备模型用于QLoRA训练...")
    model = prepare_model_for_kbit_training(model)

    print("配置LoRA适配器...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora"]["r"],
        lora_alpha=CONFIG["lora"]["lora_alpha"],
        lora_dropout=CONFIG["lora"]["lora_dropout"],
        bias=CONFIG["lora"]["bias"],
        target_modules=CONFIG["lora"]["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def preprocess_dataset(dataset, tokenizer):
    """预处理数据集"""
    print("\n" + "=" * 70)
    print("步骤3: 预处理数据集")
    print("=" * 70)

    def tokenize_function(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            output = examples["output"][i]
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=CONFIG["generation"]["max_length"],
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("正在tokenize数据集...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )

    print(f"✅ 数据集预处理完成！样本数: {len(tokenized_dataset['train'])}")
    return tokenized_dataset


def save_lora_model(model, tokenizer, output_dir):
    print(f"\n保存 LoRA adapter 到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 只验证关键文件，忽略 training_args.bin 等内部文件
    print("\n验证保存文件:")
    critical_files = ["adapter_config.json"]
    weight_files = ["adapter_model.safetensors", "adapter_model.bin"]

    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  {'✅' if size > 0 else '⚠️ '} {fname}: {size / 1024:.1f} KB")

    # 只检查关键文件
    has_config = os.path.exists(os.path.join(output_dir, "adapter_config.json")) and \
                 os.path.getsize(os.path.join(output_dir, "adapter_config.json")) > 0
    has_weights = any(
        os.path.exists(os.path.join(output_dir, f)) and
        os.path.getsize(os.path.join(output_dir, f)) > 1024 * 1024  # 至少1MB
        for f in weight_files
    )

    if not has_config:
        raise RuntimeError("❌ adapter_config.json 缺失或为空！")
    if not has_weights:
        raise RuntimeError("❌ LoRA 权重文件缺失或过小！")

    print(f"\n✅ 模型保存验证通过！")
def train_model(model, tokenizer, dataset):
    """训练模型"""
    print("\n" + "=" * 70)
    print("步骤4: 开始训练")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu_capability = torch.cuda.get_device_capability(0)
        supports_bf16 = gpu_capability[0] >= 8
    else:
        supports_bf16 = False

    use_bf16 = CONFIG["training"]["bf16"] and supports_bf16
    use_fp16 = CONFIG["training"]["fp16"] and not use_bf16

    print(f"GPU支持bf16: {supports_bf16}")
    print(f"使用bf16: {use_bf16} | 使用fp16: {use_fp16}")

    training_args = TrainingArguments(
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
        # ✅ 训练过程只保存 checkpoint，不在这里保存最终模型
        save_strategy=CONFIG["training"]["save_strategy"],
        save_steps=CONFIG["training"].get("save_steps", 500),
        save_total_limit=CONFIG["training"]["save_total_limit"],
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=CONFIG["training"]["seed"],
        data_seed=CONFIG["training"]["seed"],
        disable_tqdm=False,
        remove_unused_columns=False,
        push_to_hub=False,
        gradient_checkpointing=CONFIG["training"]["gradient_checkpointing"],
        dataloader_num_workers=2,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    print("开始训练...")
    print(f"预计耗时: 约{CONFIG['training']['num_train_epochs'] * 0.5:.1f}小时 (RTX 5070 Ti)")
    print("-" * 50)

    try:
        train_result = trainer.train()

        print(f"\n✅ 训练完成！训练损失: {train_result.training_loss:.4f}")

        # ✅ 修复：使用专用函数保存，而不是 trainer.save_model
        final_model_dir = os.path.join(CONFIG["output_dir"], "final_model")
        save_lora_model(model, tokenizer, final_model_dir)

        # 保存训练指标
        metrics_file = os.path.join(CONFIG["output_dir"], "training_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(train_result.metrics, f, indent=2)
        print(f"✅ 训练指标已保存至: {metrics_file}")

        return trainer

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU显存不足！请尝试:")
            print(f"  1. 减小 max_length（当前: {CONFIG['generation']['max_length']}）")
            print(f"  2. 减小 gradient_accumulation_steps（当前: {CONFIG['training']['gradient_accumulation_steps']}）")
            print(f"  3. 减小 per_device_train_batch_size（当前: {CONFIG['training']['per_device_train_batch_size']}）")
        raise e


def test_inference(model, tokenizer):
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    """推理测试，验证模型是否正常工作"""
    print("\n" + "=" * 70)
    print("步骤5: 推理测试")
    print("=" * 70)

    test_prompt = """你是一个五子棋大师。规则：黑白交替落子，先在横、竖、斜方向连成五子者胜。
请分析棋盘：
棋盘状态（●黑子，○白子，·空位）：
  A B C D E
1 · · · · ·
2 · ● ● · ·
3 · ○ ○ · ·
4 · · · · ·
5 · · · · ·
轮到黑棋（●）走。最优落子位置是什么？请简要说明理由。"""

    print("测试提示词（前200字符）:")
    print(test_prompt[:200] + "...")
    print("-" * 50)

    input_text = f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(input_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=CONFIG["generation"]["temperature"],
            do_sample=CONFIG["generation"]["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()

    print("模型回复:")
    print(response)
    print("-" * 50)
    print("✅ 推理测试完成！")


def main():
    """主函数"""
    try:
        setup_environment()
        dataset = load_and_prepare_data()
        model, tokenizer = create_model_and_tokenizer()
        tokenized_dataset = preprocess_dataset(dataset, tokenizer)
        trainer = train_model(model, tokenizer, tokenized_dataset)
        test_inference(model, tokenizer)

        print("\n" + "=" * 70)
        print("🎉 微调完成！")
        print("=" * 70)
        final_dir = os.path.join(CONFIG['output_dir'], 'final_model')
        print(f"模型保存在: {final_dir}")
        print("\n加载方式:")
        print(f"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "{CONFIG['model_name']}",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "{final_dir}")
tokenizer = AutoTokenizer.from_pretrained("{final_dir}")
        """)
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())