"""
GSM8K 数据准备脚本
从 HuggingFace 下载 GSM8K 训练集，取前 500 条，
转换为与五子棋数据相同的 instruction/output 格式，
保存至 datasets/gsm8k/gsm8k_500.json
"""

import os
import json
import re
from datasets import load_dataset

OUTPUT_DIR = "./datasets/gsm8k"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gsm8k_500.json")
NUM_SAMPLES = 500
CACHE_DIR = "./cache"


def extract_answer(answer_text: str) -> str:
    """提取 GSM8K 答案中的最终数字"""
    # GSM8K 答案格式：推理过程 + "#### 数字"
    match = re.search(r"####\s*([\d,\-]+)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return ""


def format_gsm8k_sample(question: str, answer: str) -> dict:
    """
    将 GSM8K 样本转换为 instruction/output 格式
    与五子棋数据保持一致的结构
    """
    # 分离推理过程和最终答案
    steps = answer.split("####")[0].strip()
    final_answer = extract_answer(answer)

    instruction = (
        "你是一个数学推理助手。请一步步分析并解答下面的数学题。\n\n"
        f"题目：{question}"
    )

    # 使用 <thinking> 标签包裹推理过程，与五子棋数据格式对齐
    output = f"<thinking>\n{steps}\n</thinking>\n答案：{final_answer}"

    return {
        "instruction": instruction,
        "output": output,
        "task_type": "math_reasoning"  # 标记任务类型，便于后续分析
    }


def main():
    print("=" * 60)
    print("GSM8K 数据准备脚本")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"正在从 HuggingFace 加载 GSM8K 数据集...")
    print("（首次下载约需1-2分钟）")

    try:
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
            cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"加载失败: {e}")
        print("\n尝试使用镜像源...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
            cache_dir=CACHE_DIR
        )

    print(f"GSM8K 训练集总量: {len(dataset)} 条")
    print(f"取前 {NUM_SAMPLES} 条...")

    samples = []
    for i in range(NUM_SAMPLES):
        item = dataset[i]
        formatted = format_gsm8k_sample(item["question"], item["answer"])
        samples.append(formatted)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 已保存 {len(samples)} 条到: {OUTPUT_FILE}")
    print(f"文件大小: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # 预览前3条
    print("\n--- 前3条样本预览 ---")
    for i, s in enumerate(samples[:3]):
        print(f"\n[样本 {i+1}]")
        print(f"instruction: {s['instruction'][:120]}...")
        print(f"output: {s['output'][:200]}...")

    print("\n完成！下一步运行 mix_datasets.py 合并数据集。")


if __name__ == "__main__":
    main()
