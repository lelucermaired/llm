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
    "model_name": "Qwen/Qwen2.5-7B-Instruct",  # HuggingFace模型ID
    "local_model_path": None,  # 如果已下载到本地，可指定路径如"./models/Qwen2.5-7B-Instruct"

    # 数据配置
    "dataset_path": "./datasets/complete_gomoku_dataset.json",  # 你的五子棋数据集路径
    "output_dir": "./checkpoints/qwen7b-gomoku-lora",  # 保存路径
    "cache_dir": "./cache",  # 缓存目录

    # QLoRA 量化配置（关键！针对16GB显存优化）
    "quantization": {
        "load_in_4bit": True,  # 4-bit量化是必须的
        "bnb_4bit_quant_type": "nf4",  # 使用NF4量化（精度最好）
        "bnb_4bit_compute_dtype": torch.float16,  # 计算时用float16
        "bnb_4bit_use_double_quant": True,  # 双重量化，进一步节省显存
    },

    # LoRA 适配器配置
    "lora": {
        "r": 10,  # 秩大小。越大能力越强但参数越多（8-64之间）
        "lora_alpha": 20,  # 缩放因子，通常设为2*r
        "lora_dropout": 0.1,  # Dropout防止过拟合
        "bias": "none",  # 偏置训练模式
        "target_modules": [  # 针对Qwen2.5架构
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    },

    # 训练超参数（针对RTX 5070 Ti优化）
    "training": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "warmup_steps": 50,  # 注意：这里是warmup_steps，不是warmup_ratio
        "max_grad_norm": 2.0,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": 0.05,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_steps": 100,  # 添加save_steps
        "save_total_limit": 2,
        "seed": 42,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
    },

    # 生成配置
    "generation": {
        "max_length": 1024,  # 生成最大长度
        "temperature": 0.1,  # 温度参数
        "do_sample": False,  # 评估时不采样
    }
}


# =====================================================================

def setup_environment():
    """设置实验环境"""
    print("=" * 70)
    print("Qwen2.5-7B-Instruct QLoRA 微调脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 设置随机种子
    set_seed(CONFIG["training"]["seed"])

    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    # 检查GPU
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
        print("强烈建议修复GPU问题后再继续！7B模型在CPU上训练极慢。")

    # 保存配置（先清理不可序列化的对象）
    config_file = os.path.join(CONFIG["output_dir"], "training_config.json")

    # 创建可序列化的配置副本
    serializable_config = {}
    for key, value in CONFIG.items():
        if key == "quantization":
            # 处理量化配置中的torch数据类型
            quant_config = value.copy()
            quant_config["bnb_4bit_compute_dtype"] = str(quant_config["bnb_4bit_compute_dtype"])
            serializable_config[key] = quant_config
        elif key == "training":
            # 直接复制训练配置
            serializable_config[key] = value.copy()
        elif key == "lora":
            # 直接复制lora配置
            serializable_config[key] = value.copy()
        elif key == "generation":
            # 直接复制生成配置
            serializable_config[key] = value.copy()
        else:
            serializable_config[key] = value

    # 添加环境信息
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
    """
    加载并准备五子棋数据集
    """
    print("\n" + "=" * 70)
    print("步骤1: 加载和预处理数据集")
    print("=" * 70)

    dataset_path = CONFIG["dataset_path"]

    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 数据集文件不存在: {dataset_path}")
        print("请创建数据集文件，示例格式如下：")
        print('''
        [
            {
                "instruction": "你是一个五子棋大师。规则：黑白交替落子，先在横、竖、斜方向连成五子者胜。请分析棋盘：\\n棋盘：●在B3,C3；○在C4,D4\\n轮到黑棋走。",
                "input": "",
                "output": "<think>黑棋有两个连子，白棋也有两个连子。根据规则，我需要防守白棋的潜在活三，同时发展自己的进攻。最佳位置可能是D3，既阻挡白棋又连接自己的棋子。</think>\\n最佳落子：D3"
            },
            ...
        ]
        ''')
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    # 加载数据集
    print(f"加载数据集: {dataset_path}")
    try:
        # 从本地JSON文件加载
        dataset = load_dataset("json", data_files=dataset_path, cache_dir=CONFIG["cache_dir"])
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        raise

    print(f"数据集加载成功，样本数: {len(dataset['train'])}")

    # 显示前3个样本作为检查
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
    """
    加载模型和分词器，应用QLoRA配置
    """
    print("\n" + "=" * 70)
    print("步骤2: 加载模型和分词器 (应用4-bit QLoRA量化)")
    print("=" * 70)

    model_name = CONFIG["local_model_path"] or CONFIG["model_name"]

    # 1. 配置4-bit量化
    print("配置4-bit量化参数...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=CONFIG["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=CONFIG["quantization"]["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=CONFIG["quantization"]["bnb_4bit_use_double_quant"],
    )

    # 2. 加载模型 (自动应用量化)
    print(f"加载模型: {model_name}")
    print("注意: 首次下载需要较长时间和约15GB磁盘空间...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",  # 自动分配模型层到GPU/CPU
            trust_remote_code=True,  # Qwen模型需要这个
            cache_dir=CONFIG["cache_dir"],
            use_cache=False,  # 训练时禁用缓存以节省显存
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("可能的原因:")
        print("1. 网络问题，无法下载模型")
        print("2. 磁盘空间不足")
        print("3. transformers库版本过低 (需要>=4.37.0)")
        print("解决方案: pip install transformers -U")
        raise

    # 3. 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=CONFIG["cache_dir"],
    )

    # 设置必要的分词器参数
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 填充在右侧

    print("✅ 分词器加载成功！")

    # 4. 准备模型用于k-bit训练
    print("准备模型用于QLoRA训练...")
    model = prepare_model_for_kbit_training(model)

    # 5. 配置LoRA适配器
    print("配置LoRA适配器...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora"]["r"],
        lora_alpha=CONFIG["lora"]["lora_alpha"],
        lora_dropout=CONFIG["lora"]["lora_dropout"],
        bias=CONFIG["lora"]["bias"],
        target_modules=CONFIG["lora"]["target_modules"],
    )

    # 应用LoRA配置
    model = get_peft_model(model, lora_config)

    # 打印可训练参数信息
    model.print_trainable_parameters()

    return model, tokenizer


def preprocess_dataset(dataset, tokenizer):
    """
    预处理数据集：将文本转换为模型输入
    """
    print("\n" + "=" * 70)
    print("步骤3: 预处理数据集")
    print("=" * 70)

    def tokenize_function(examples):
        """将文本转换为token"""
        # 构造完整的训练文本: instruction + output
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            output = examples["output"][i]
            # 使用Qwen2.5的聊天模板格式
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)

        # 进行tokenization
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,  # 训练时使用collator动态填充
            max_length=CONFIG["generation"]["max_length"],
        )

        # 将labels设置为与input_ids相同（用于因果语言建模）
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    print("正在tokenize数据集...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,  # 移除原始列
        desc="Tokenizing dataset"
    )

    print(f"✅ 数据集预处理完成！")
    print(f"样本数: {len(tokenized_dataset['train'])}")

    return tokenized_dataset


def train_model(model, tokenizer, dataset):
    """
    训练模型
    """
    print("\n" + "=" * 70)
    print("步骤4: 开始训练")
    print("=" * 70)

    # 检查GPU是否支持bfloat16
    if torch.cuda.is_available():
        gpu_capability = torch.cuda.get_device_capability(0)
        supports_bf16 = gpu_capability[0] >= 8  # Ampere架构及以上支持bf16
    else:
        supports_bf16 = False

    # 根据配置和GPU能力设置精度
    use_bf16 = CONFIG["training"]["bf16"] and supports_bf16
    use_fp16 = CONFIG["training"]["fp16"] and not use_bf16

    print(f"GPU支持bf16: {supports_bf16}")
    print(f"使用bf16: {use_bf16}")
    print(f"使用fp16: {use_fp16}")

    # 1. 配置训练参数
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],

        # 训练循环
        num_train_epochs=CONFIG["training"]["num_train_epochs"],
        per_device_train_batch_size=CONFIG["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["training"]["gradient_accumulation_steps"],

        # 优化器
        learning_rate=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
        warmup_steps=CONFIG["training"]["warmup_steps"],  # 修复：使用warmup_steps而不是warmup_ratio
        max_grad_norm=CONFIG["training"]["max_grad_norm"],
        optim=CONFIG["training"]["optim"],
        lr_scheduler_type=CONFIG["training"]["lr_scheduler_type"],

        # 日志和保存
        logging_strategy="steps",
        logging_steps=CONFIG["training"]["logging_steps"],
        save_strategy=CONFIG["training"]["save_strategy"],
        save_steps=CONFIG["training"].get("save_steps", 500),  # 使用get方法避免KeyError
        save_total_limit=CONFIG["training"]["save_total_limit"],

        # 精度设置
        bf16=use_bf16,
        fp16=use_fp16,

        # 其他
        report_to="none",
        seed=CONFIG["training"]["seed"],
        data_seed=CONFIG["training"]["seed"],
        disable_tqdm=False,
        remove_unused_columns=False,
        push_to_hub=False,

        # 针对小批量训练的优化
        gradient_checkpointing=CONFIG["training"]["gradient_checkpointing"],
        dataloader_num_workers=2,
    )

    # 2. 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # 填充到8的倍数，提高效率
    )

    # 3. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    # 4. 开始训练
    print("开始训练...")
    print(f"总训练步数: {trainer.state.max_steps}")
    print(f"预计耗时: 约{CONFIG['training']['num_train_epochs'] * 0.5:.1f}小时 (RTX 5070 Ti)")
    print("-" * 50)

    try:
        train_result = trainer.train()

        # 5. 保存最终模型
        print("\n✅ 训练完成！")
        print(f"训练损失: {train_result.training_loss:.4f}")

        # 保存模型
        final_model_dir = os.path.join(CONFIG["output_dir"], "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        # 保存训练指标
        metrics_file = os.path.join(CONFIG["output_dir"], "training_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(train_result.metrics, f, indent=2)

        print(f"✅ 模型已保存至: {final_model_dir}")
        print(f"✅ 训练指标已保存至: {metrics_file}")

        return trainer

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU显存不足！")
            print("尝试以下解决方案：")
            print("1. 减小 'max_length' (当前: {})".format(CONFIG["generation"]["max_length"]))
            print("2. 减小 'gradient_accumulation_steps' (当前: {})".format(
                CONFIG["training"]["gradient_accumulation_steps"]))
            print("3. 减小 'per_device_train_batch_size' (当前: {})".format(
                CONFIG["training"]["per_device_train_batch_size"]))
            print("4. 关闭其他占用显存的程序")
        raise e


def test_inference(model, tokenizer):
    """
    简单推理测试，验证模型是否正常工作
    """
    print("\n" + "=" * 70)
    print("步骤5: 推理测试")
    print("=" * 70)

    # 使用一个简单的五子棋问题测试
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

    print("测试提示词:")
    print(test_prompt[:200] + "...")
    print("-" * 50)

    # 格式化输入
    input_text = f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"

    # 编码
    inputs = tokenizer(input_text, return_tensors="pt")

    # 移动到模型所在设备
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=CONFIG["generation"]["temperature"],
            do_sample=CONFIG["generation"]["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取助手回复部分
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()

    print("模型回复:")
    print(response)
    print("-" * 50)
    print("✅ 推理测试完成！")


def main():
    """主函数"""
    try:
        # 1. 设置环境
        setup_environment()

        # 2. 加载数据
        dataset = load_and_prepare_data()

        # 3. 加载模型和分词器
        model, tokenizer = create_model_and_tokenizer()

        # 4. 预处理数据
        tokenized_dataset = preprocess_dataset(dataset, tokenizer)

        # 5. 训练模型
        trainer = train_model(model, tokenizer, tokenized_dataset)

        # 6. 测试推理
        test_inference(model, tokenizer)

        # 7. 最终提示
        print("\n" + "=" * 70)
        print("🎉 微调完成！")
        print("=" * 70)
        print("下一步操作:")
        print(f"1. 模型保存在: {CONFIG['output_dir']}/final_model")
        print("2. 使用以下代码加载微调后的模型:")
        print(f'''
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model = AutoModelForCausalLM.from_pretrained(
            "{CONFIG['model_name']}",
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, "{CONFIG['output_dir']}/final_model")
        ''')
        print("3. 运行五子棋能力验证脚本，检查模型是否学会规则")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())