"""
prepare_gsm8k.py

从GSM8K训练集取1000条，转换为SFT训练格式
输出：./datasets/gsm8k_sft/train.json

格式与五子棋数据一致：
{"instruction": "...", "output": "..."}
"""

import json, os, re
from datasets import load_dataset

OUTPUT_PATH = "./datasets/gsm8k_sft/train.json"
N_SAMPLES = 1000

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("加载GSM8K训练集...")
ds = load_dataset("gsm8k", "main")
train = ds["train"]
print(f"✅ 训练集样本数: {len(train)}")
print(f"字段: {train.column_names}")
print(f"\n前2条样本原始格式：")
for i in range(2):
    s = train[i]
    print(f"\n--- 样本{i+1} ---")
    print(f"question: {s['question'][:120]}")
    print(f"answer:   {s['answer'][:120]}")

# GSM8K的answer字段格式：
# 包含推理步骤，最后一行是 "#### 数字"
# 例：
# Natalia sold clips to 48 of her friends...
# Natalia sold 48/2 = <<48/2=24>>24 clips in May.
# #### 72

def extract_answer(answer_text):
    """提取####后的最终数字答案"""
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip().split('\n')[-1].strip()

def build_instruction(question):
    return (
        "Please solve the following math problem step by step. "
        "Show your reasoning process and give the final answer.\n\n"
        f"Problem: {question}"
    )

def build_output(answer_text):
    """保留完整推理链，作为CoT输出"""
    # 清理<<计算过程>>标注，保留文字推理
    clean = re.sub(r'<<[^>]+>>', '', answer_text)
    # 将####替换为更清晰的格式
    clean = re.sub(r'####\s*(.+)', r'Therefore, the answer is: \1', clean)
    return clean.strip()

print(f"\n转换前2条样本示例：")
for i in range(2):
    s = train[i]
    inst = build_instruction(s['question'])
    out = build_output(s['answer'])
    print(f"\n--- 转换后样本{i+1} ---")
    print(f"instruction: {inst[:150]}...")
    print(f"output:      {out[:150]}...")

# 取前N_SAMPLES条
print(f"\n取前{N_SAMPLES}条样本...")
samples = []
for i in range(min(N_SAMPLES, len(train))):
    s = train[i]
    samples.append({
        "instruction": build_instruction(s['question']),
        "output": build_output(s['answer']),
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"✅ 已保存 {len(samples)} 条样本至: {OUTPUT_PATH}")
print(f"文件大小: {os.path.getsize(OUTPUT_PATH)/1024:.1f} KB")
print(f"\n示例输出（第1条）：")
print(f"instruction: {samples[0]['instruction'][:200]}")
print(f"output:      {samples[0]['output'][:200]}")