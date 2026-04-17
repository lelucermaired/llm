import os

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0+PTX;8.6+PTX;8.9+PTX'
import sys
import json
from datetime import datetime
from datasets import load_dataset
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

_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

import warnings
warnings.filterwarnings("ignore")

# ==================== 关键修改：浅层LoRA配置 ====================
# 理论依据：浅层（0-9层）处理通用语义，更新落在此处更易产生跨域迁移
# 对比：v2模型更新均匀分散在全部28层，无法建立跨域桥梁
# 目标模块：只更新前10层的q_proj和v_proj
SHALLOW_LAYERS = list(range(10))  # 层0-9
TARGET_MODULES = [f"model.layers.{i}.self_attn.{proj}"
                  for i in SHALLOW_LAYERS
                  for proj in ["q_proj", "v_proj"]]

CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "local_model_path": None,

    "dataset_path": "./datasets/real_games_v2/train.json",
    "output_dir": "./checkpoints/qwen-gomoku-shallow",  # 新模型目录
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
        # 核心修改：只更新前10层，而非全部28层
        "target_modules": TARGET_MODULES,
    },

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

    "generation": {
        "max_length": 1024,
        "temperature": 0.1,
        "do_sample": False,
    }
}


def setup_environment():
    print("=" * 70)
    print("浅层LoRA微调：只更新前10层（通用语义层）")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标模块数量: {len(TARGET_MODULES)}个（每层q+v，共10层）")
    print("=" * 70)

    set_seed(CONFIG["training"]["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} | 显存: {gpu_memory:.2f} GB")
    else:
        print("❌ 未检测到GPU")

    # 打印目标模块列表
    print("\n将更新的模块（前10层）：")
    for m in TARGET_MODULES[:6]:
        print(f"  {m}")
    print(f"  ... 共{len(TARGET_MODULES)}个模块")

    # 对比说明
    print("\n对比配置：")
    print(f"  v2模型：全部28层 q+v，56个模块")
    print(f"  本模型：前10层 q+v，{len(TARGET_MODULES)}个模块（更新量更集中）")


def load_and_prepare_data():
    print("\n" + "=" * 70)
    print("步骤1: 加载数据集")
    print("=" * 70)

    dataset_path = CONFIG["dataset_path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    print(f"加载: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path,
                          cache_dir=CONFIG["cache_dir"])
    print(f"✅ 样本数: {len(dataset['train'])}")
    return dataset


def create_model_and_tokenizer():
    print("\n" + "=" * 70)
    print("步骤2: 加载模型，应用浅层LoRA")
    print("=" * 70)

    model_name = CONFIG["local_model_path"] or CONFIG["model_name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=CONFIG["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=CONFIG["quantization"]["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=CONFIG["quantization"]["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CONFIG["cache_dir"],
        use_cache=False,
    )
    print("✅ 模型加载成功")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=CONFIG["cache_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

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
    print(f"\n✅ 浅层LoRA配置完成（前10层，r=8）")

    return model, tokenizer


def preprocess_dataset(dataset, tokenizer):
    print("\n" + "=" * 70)
    print("步骤3: 预处理数据集")
    print("=" * 70)

    def tokenize_function(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            output = examples["output"][i]
            text = (f"<|im_start|>user\n{instruction}<|im_end|>\n"
                   f"<|im_start|>assistant\n{output}<|im_end|>")
            texts.append(text)

        tokenized = tokenizer(
            texts, truncation=True, padding=False,
            max_length=CONFIG["generation"]["max_length"],
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    print(f"✅ 预处理完成，样本数: {len(tokenized_dataset['train'])}")
    return tokenized_dataset


def save_lora_model(model, tokenizer, output_dir):
    print(f"\n保存模型至: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    weight_files = ["adapter_model.safetensors", "adapter_model.bin"]
    has_config = os.path.exists(os.path.join(output_dir, "adapter_config.json"))
    has_weights = any(os.path.exists(os.path.join(output_dir, f)) and
                     os.path.getsize(os.path.join(output_dir, f)) > 1024 * 1024
                     for f in weight_files)

    if not has_config or not has_weights:
        raise RuntimeError("模型保存验证失败")
    print("✅ 模型保存验证通过")


def train_model(model, tokenizer, dataset):
    print("\n" + "=" * 70)
    print("步骤4: 开始训练")
    print("=" * 70)

    gpu_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    use_bf16 = CONFIG["training"]["bf16"] and gpu_capability[0] >= 8
    use_fp16 = CONFIG["training"]["fp16"] and not use_bf16

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
        save_strategy=CONFIG["training"]["save_strategy"],
        save_total_limit=CONFIG["training"]["save_total_limit"],
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=CONFIG["training"]["seed"],
        remove_unused_columns=False,
        push_to_hub=False,
        gradient_checkpointing=CONFIG["training"]["gradient_checkpointing"],
        dataloader_num_workers=2,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    print("开始训练...")
    train_result = trainer.train()
    print(f"✅ 训练完成！Loss: {train_result.training_loss:.4f}")

    final_model_dir = os.path.join(CONFIG["output_dir"], "final_model")
    save_lora_model(model, tokenizer, final_model_dir)

    metrics_file = os.path.join(CONFIG["output_dir"], "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    return trainer


def main():
    try:
        setup_environment()
        dataset = load_and_prepare_data()
        model, tokenizer = create_model_and_tokenizer()
        tokenized_dataset = preprocess_dataset(dataset, tokenizer)
        train_model(model, tokenizer, tokenized_dataset)

        print("\n" + "=" * 70)
        print("🎉 浅层LoRA微调完成！")
        final_dir = os.path.join(CONFIG['output_dir'], 'final_model')
        print(f"模型保存在: {final_dir}")
        print("\n下一步：运行 fulllora_eval.py 加入 shallow 模型对比")
        print("=" * 70)

    except Exception as e:
        print(f"❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())