"""
eval_migration.py
用于评估经棋类任务LoRA微调后的大语言模型在通用推理任务（如GSM8K）上的表现，
并与原始基座模型进行对比，以探究能力迁移效应。
支持断点续评，避免长时间运行意外中断。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import re
import json
import os
from datetime import datetime

# ========== 评估配置 ==========
# 模型配置
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 基座模型名称或路径
LORA_PATH = "lora_gomoku_cpu"  # 你的LoRA适配器路径

# 数据集配置
EVAL_DATASET = "gsm8k"  # HuggingFace数据集名称
EVAL_CONFIG = "main"  # 数据集配置名（gsm8k为'main'）
EVAL_SPLIT = "test"  # 使用的数据集划分

# 评估参数
SAMPLE_SIZE = 500  # 评估样本数量（设为0则评估全部）
MAX_NEW_TOKENS = 256  # 生成的最大token数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 优先使用GPU

# 生成参数（评估时建议使用贪婪解码以保证结果可复现）
GENERATION_CONFIG = {
    "do_sample": False,  # 设置为False进行贪婪解码
    "temperature": 0.0,  # 温度设为0配合贪婪解码
    "top_p": 1.0,  # 核采样参数设为1（即不起作用）
    "top_k": 0,  # top-k设为0（即不起作用）
}

# 断点与结果保存配置
RESULTS_DIR = "results"  # 结果保存目录
SAVE_INTERVAL = 50  # 每评估多少个样本保存一次中间结果


# ==============================

def ensure_dir(directory):
    """确保目录存在"""
    os.makedirs(directory, exist_ok=True)


def get_timestamp():
    """获取当前时间戳，用于文件命名"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_base_model():
    """加载原始基座模型"""
    print("=" * 50)
    print("加载原始基座模型...")
    print(f"模型路径: {BASE_MODEL}")
    print(f"运行设备: {DEVICE}")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 根据设备选择加载方式
    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,  # 半精度以节省显存
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

    return model, tokenizer


def load_lora_model():
    """加载融合了LoRA适配器的模型"""
    print("=" * 50)
    print("加载LoRA微调模型...")
    print(f"基座模型: {BASE_MODEL}")
    print(f"LoRA适配器: {LORA_PATH}")
    print(f"运行设备: {DEVICE}")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 加载基座模型
    if DEVICE == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

    # 加载LoRA适配器并合并到基座模型
    try:
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        model = model.merge_and_unload()  # 合并适配器，便于推理
    except Exception as e:
        print(f"加载LoRA适配器失败: {e}")
        print("尝试直接加载基座模型...")
        model = base_model

    return model, tokenizer


def extract_answer_from_gsm8k(text):
    """
    从GSM8K模型输出中提取最终数字答案。
    GSM8K的标准答案格式通常是"#### 25"或"#### 25.5"
    """
    # 方法1: 匹配"####数字"模式
    matches = re.findall(r'####\s*([-+]?\d*\.?\d+)', text)
    if matches:
        return matches[-1]

    # 方法2: 匹配文本中最后的数字
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    if matches:
        return matches[-1]

    return None


def is_correct_gsm8k(model_output, true_answer):
    """判断GSM8K问题的答案是否正确"""
    pred = extract_answer_from_gsm8k(model_output)
    if pred is None:
        return False

    try:
        # 转换为浮点数比较，处理整数和小数
        pred_num = float(pred)
        true_num = float(true_answer)
        return abs(pred_num - true_num) < 1e-6  # 允许极小误差
    except ValueError:
        # 如果转换失败，尝试字符串匹配
        return str(pred).strip() == str(true_answer).strip()


def construct_prompt_gsm8k(question):
    """为GSM8K数学问题构造提示词"""
    # 根据你的模型调整提示词格式
    prompt = f"请解决以下数学问题，给出最终数字答案。\n\n问题：{question}\n\n答案："
    return prompt


def evaluate_model_on_gsm8k(model, tokenizer, dataset, model_name, checkpoint_file=None):
    """
    在GSM8K数据集上评估模型

    Args:
        model: 要评估的模型
        tokenizer: 对应的分词器
        dataset: 数据集
        model_name: 模型名称（用于结果记录）
        checkpoint_file: 断点文件路径（如果有则从中恢复）

    Returns:
        accuracy: 准确率
        details: 每个样本的详细结果
    """
    print(f"\n开始评估 {model_name} 模型...")

    # 确定实际评估的样本数量
    eval_size = SAMPLE_SIZE if 0 < SAMPLE_SIZE < len(dataset) else len(dataset)
    print(f"计划评估 {eval_size} 个样本（共 {len(dataset)} 个）")

    # 加载之前的检查点（如果存在）
    details = []
    start_idx = 0
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                details = json.load(f)
            start_idx = len(details)
            print(f"从检查点恢复，已评估 {start_idx} 个样本，继续从第 {start_idx + 1} 个开始...")
        except Exception as e:
            print(f"加载检查点失败，将从头开始评估: {e}")
            details = []
            start_idx = 0

    # 评估循环
    for i in range(start_idx, eval_size):
        item = dataset[i]
        question = item["question"]

        # 从GSM8K的答案中提取最终数字（格式通常为"#### 25"）
        true_answer = item["answer"].split("#### ")[-1].strip()

        # 构造提示词
        prompt = construct_prompt_gsm8k(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                **GENERATION_CONFIG
            )

        # 解码并清理答案
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = full_output[len(prompt):].strip() if len(full_output) > len(prompt) else full_output

        # 判断是否正确
        correct = is_correct_gsm8k(model_answer, true_answer)

        # 记录结果
        details.append({
            "index": i,
            "question": question,
            "true_answer": true_answer,
            "model_answer": model_answer,
            "correct": correct
        })

        # 定期打印进度并保存检查点
        if (i + 1) % 10 == 0:
            correct_count = sum(1 for d in details if d["correct"])
            accuracy_so_far = correct_count / (i + 1)
            print(f"  已处理 {i + 1}/{eval_size}，当前准确率: {accuracy_so_far:.2%}")

            # 定期清理GPU缓存（如果在GPU上运行）
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # 定期保存检查点
        if (i + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(RESULTS_DIR, f"checkpoint_{model_name}.json")
            ensure_dir(RESULTS_DIR)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(details, f, ensure_ascii=False, indent=2)
            print(f"  [检查点] 已保存中间结果到 {checkpoint_path}")

    # 计算最终准确率
    correct_count = sum(1 for d in details if d["correct"])
    total_count = len(details)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    print(f"\n{model_name} 模型评估完成!")
    print(f"总样本数: {total_count}, 正确数: {correct_count}, 准确率: {accuracy:.2%}")

    return accuracy, details


def main():
    """主函数"""
    print("=" * 60)
    print("大语言模型推理能力迁移评估")
    print(f"开始时间: {get_timestamp()}")
    print(f"评估数据集: {EVAL_DATASET} ({EVAL_CONFIG})")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # 确保结果目录存在
    ensure_dir(RESULTS_DIR)

    # 加载数据集
    print(f"\n正在加载数据集 {EVAL_DATASET}...")
    try:
        dataset = load_dataset(EVAL_DATASET, EVAL_CONFIG, split=EVAL_SPLIT)
        print(f"数据集加载成功，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return

    # 分别评估两个模型
    results = {}

    # 评估基座模型
    base_model, base_tokenizer = load_base_model()
    base_accuracy, base_details = evaluate_model_on_gsm8k(
        base_model, base_tokenizer, dataset, "Base",
        checkpoint_file=os.path.join(RESULTS_DIR, "checkpoint_Base.json")
    )
    results["Base"] = {
        "accuracy": base_accuracy,
        "sample_size": len(base_details),
        "details": base_details
    }

    # 清理显存
    del base_model, base_tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # 评估LoRA微调模型
    lora_model, lora_tokenizer = load_lora_model()
    lora_accuracy, lora_details = evaluate_model_on_gsm8k(
        lora_model, lora_tokenizer, dataset, "LoRA",
        checkpoint_file=os.path.join(RESULTS_DIR, "checkpoint_LoRA.json")
    )
    results["LoRA"] = {
        "accuracy": lora_accuracy,
        "sample_size": len(lora_details),
        "details": lora_details
    }

    # 清理显存
    del lora_model, lora_tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # 保存最终结果
    timestamp = get_timestamp()
    results_file = os.path.join(RESULTS_DIR, f"migration_eval_{timestamp}.json")

    # 添加实验配置信息
    results["config"] = {
        "base_model": BASE_MODEL,
        "lora_path": LORA_PATH,
        "dataset": EVAL_DATASET,
        "config": EVAL_CONFIG,
        "split": EVAL_SPLIT,
        "sample_size": SAMPLE_SIZE if 0 < SAMPLE_SIZE < len(dataset) else len(dataset),
        "max_new_tokens": MAX_NEW_TOKENS,
        "device": DEVICE,
        "generation_config": GENERATION_CONFIG,
        "timestamp": timestamp
    }

    # 保存结果
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    print(f"基座模型准确率: {base_accuracy:.2%}")
    print(f"LoRA微调模型准确率: {lora_accuracy:.2%}")
    print(f"准确率差异: {(lora_accuracy - base_accuracy):+.2%}")

    # 简单分析
    if lora_accuracy > base_accuracy:
        print("结论: 观察到正向迁移效果!")
    elif lora_accuracy < base_accuracy:
        print("结论: 观察到负向迁移效果。")
    else:
        print("结论: 未观察到明显迁移效果。")

    print(f"\n详细结果已保存至: {results_file}")

    # 打印一些错误案例以供分析
    print("\n" + "=" * 60)
    print("错误案例分析（前3个）:")
    print("=" * 60)

    # 找出两个模型都答错的样本
    base_wrong = [d for d in base_details if not d["correct"]]
    lora_wrong = [d for d in lora_details if not d["correct"]]

    common_wrong_indices = set([d["index"] for d in base_wrong]) & set([d["index"] for d in lora_wrong])

    if common_wrong_indices:
        common_wrong = list(common_wrong_indices)[:3]  # 取前3个
        for idx in common_wrong:
            base_item = base_details[idx]
            lora_item = lora_details[idx]

            print(f"\n样本 {idx}:")
            print(f"问题: {base_item['question'][:100]}...")
            print(f"标准答案: {base_item['true_answer']}")
            print(f"基座模型回答: {base_item['model_answer'][:100]}...")
            print(f"LoRA模型回答: {lora_item['model_answer'][:100]}...")
    else:
        print("未找到两个模型都答错的样本。")

    print("\n评估结束!")


if __name__ == "__main__":
    # 设置环境变量（可选，用于避免某些警告）
    import warnings

    warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

    # 运行主函数
    main()