"""
gsm8k_transfer_redesign.py

重新设计的数学推理迁移实验
解决原实验的问题：
1. Logic任务天花板 → 使用有难度梯度的LogiQA
2. Math评估不匹配 → 加入GSM8K测试集评估
3. 样本量不平衡 → 统一控制
4. 无法验证假设 → 设计结构相似的任务链

实验假设：
- 若数学SFT → 物理应用题正向迁移，说明结构相似性是迁移条件
- 若数学SFT → 逻辑推理正向迁移，说明抽象推理能力可迁移
- 若数学SFT → 空间推理零迁移，说明领域特异性限制迁移

用法:
    python scripts/evaluation/gsm8k_transfer_redesign.py
"""

import os
import json
import torch
import gc
import random
import re
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# ==================== 配置 ====================

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "models": {
        # "base": None,  # 已有数据，跳过
        # "v2": "./archive/checkpoints/qwen-gomoku-v2/final_model",  # 已有数据，跳过
        "gsm8k": "./checkpoints/qwen-gsm8k-sft/final_model",  # 数学SFT
    },
    "output_dir": "./results/evaluations/gsm8k_transfer_redigned",
    "max_new_tokens": 256,
    "seed": 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
random.seed(CONFIG["seed"])

# ==================== 任务设计 ====================

# L1: GSM8K风格数学应用题（源任务内评估）
GSM8K_SAMPLES = [
    # 简单（2-3步）
    ("A bakery sells 48 loaves of bread per day. If each loaf costs $3, how much does the bakery make in 5 days?", "720"),
    ("Tom has 156 marbles. He gives 23 to each of his 4 friends. How many marbles does he have left?", "64"),
    ("A train travels 120 miles in 2 hours. At the same speed, how far will it travel in 5 hours?", "300"),
    ("Sarah reads 35 pages per day. Her book has 280 pages. How many days will it take to finish?", "8"),
    # 中等（3-4步）
    ("A store buys shirts for $15 each and sells them for $23. If they sold 145 shirts and had 12 left unsold, what's the profit from the sold shirts?", "1160"),
    ("John earns $18 per hour for the first 40 hours and $27 per hour for overtime. If he worked 52 hours last week, what's his total pay?", "1056"),
    ("A rectangle's length is 3 times its width. If the perimeter is 64 cm, what's the area?", "192"),
    # 复杂（4+步）
    ("A farmer has chickens and rabbits. There are 35 heads and 94 legs total. How many rabbits are there?", "12"),
    ("Two trains leave stations 300 km apart, traveling toward each other at 60 km/h and 40 km/h. A bird flies at 80 km/h between them until they meet. How far does the bird fly?", "240"),
]

# L2: 物理应用题（数学+单位转换，结构相似）
PHYSICS_SAMPLES = [
    # 力学（需要数学+物理概念）
    ("A car accelerates from rest at 3 m/s² for 8 seconds. What distance does it travel?", "96"),
    ("A 5 kg object is pushed with 20 N force. What's its acceleration in m/s²?", "4"),
    ("A ball is thrown upward at 25 m/s. How high does it go? (g=10 m/s²)", "31.25"),
    ("A cyclist travels at 15 km/h for 2 hours, then 20 km/h for 1.5 hours. Total distance in km?", "60"),
    # 电学（需要公式应用）
    ("A circuit has 12V battery and 4Ω resistor. What's the current in amperes?", "3"),
    ("A device uses 500W for 6 hours. How many kWh of energy?", "3"),
    # 单位转换
    ("A car travels 90 km/h. How many meters per second is that?", "25"),
    ("Water flows at 5 liters per minute. How many liters in 3.5 hours?", "1050"),
]

# L3: 逻辑推理（LogiQA风格，抽象结构相似）
LOGIC_SAMPLES = [
    # 三段论（有难度）
    ("All programmers know Python. Some engineers are programmers. Can we conclude that some engineers know Python?", "yes"),
    ("All metals conduct electricity. Not all conductors are metals. Copper is a conductor. Is copper necessarily a metal?", "no"),
    # 条件推理
    ("If the alarm sounds, there is either a fire or a drill. There is no fire and no drill. Did the alarm sound?", "no"),
    ("If it rains, the ground is wet. The ground is wet. Did it necessarily rain?", "no"),
    # 数理逻辑
    ("For all x, if x is even then x² is even. 16 is even squared. Is the original number even?", "yes"),
    ("If A then B. If B then C. Not C. Can we conclude not A?", "yes"),
    # 排除推理
    ("Either Tom or Jerry stole the cake. Tom has an alibi. Who stole the cake?", "jerry"),
    ("Of A, B, C, D, exactly one is guilty. A and B were together. C was out of town. Who is guilty?", "d"),
    # 量化推理
    ("At least 3 of 5 marbles are red. At most 2 are blue. How many are red?", "3"),
    ("All items in box A are in box B. Some items in box B are in box C. Are all items in box A in box C?", "no"),
]

# L4: 空间推理（结构不相似，对照组）
SPATIAL_SAMPLES = [
    ("A is north of B. B is east of C. Where is A relative to C?", "northeast"),
    ("If you face north, turn 90° right, then 180°, which direction?", "south"),
    ("A cube has side 3. What's its volume?", "27"),
    ("A rectangular room is 4m by 5m. What's the area?", "20"),
    ("Point X is at (2,3). Point Y is at (5,7). What's the distance? Round to 1 decimal.", "5.0"),
    ("A triangle has base 6 and height 4. What's the area?", "12"),
    ("A circle has radius 7. What's the circumference? Use π=22/7.", "44"),
    ("A is to the left of B. B is above C. Where is A relative to C?", "upper-left"),
]

# L5: 原始简单算术（保留作为基线）
ARITHMETIC_SAMPLES = [
    ("What is 15 + 27?", "42"),
    ("What is 7 × 8?", "56"),
    ("What is 100 ÷ 4?", "25"),
    ("What is √144?", "12"),
    ("What is 3⁴?", "81"),
]

TASKS = {
    "gsm8k": ("L1:数学应用题", GSM8K_SAMPLES),
    "physics": ("L2:物理应用题", PHYSICS_SAMPLES),
    "logic": ("L3:逻辑推理", LOGIC_SAMPLES),
    "spatial": ("L4:空间推理", SPATIAL_SAMPLES),
    "arithmetic": ("L5:简单算术", ARITHMETIC_SAMPLES),
}

# ==================== 模型加载 ====================

def cleanup_model(model):
    """彻底清理模型和显存"""
    if hasattr(model, 'base_model'):
        # PeftModel
        base = model.base_model.model
        del model
        model = base
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("  [显存已清理]")


def load_model(base_path, adapter_path=None):
    """加载模型"""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,  # 允许CPU卸载
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "12GB", "cpu": "8GB"},  # 限制显存使用
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model

# ==================== 评估 ====================

def extract_number(text):
    """从回答中提取数字"""
    # 尝试提取数字
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    if matches:
        return matches[0]
    return None

def evaluate_sample(model, tokenizer, prompt, expected_answer, device):
    """评估单个样本"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip().lower()
    
    expected = expected_answer.lower().strip()
    
    # 多种匹配方式
    # 1. 精确包含
    if expected in response:
        return True
    # 2. 数字匹配
    expected_num = extract_number(expected)
    if expected_num:
        response_num = extract_number(response)
        if response_num and expected_num == response_num:
            return True
    # 3. yes/no 匹配
    if expected in ["yes", "no"]:
        if expected in response:
            return True
    
    return False

def evaluate_model(model, tokenizer, device, model_name):
    """评估单个模型"""
    results = {}
    
    for task_name, (task_desc, samples) in TASKS.items():
        correct = 0
        for prompt, answer in tqdm(samples, desc=f"{model_name}/{task_name}", leave=False):
            if evaluate_sample(model, tokenizer, prompt, answer, device):
                correct += 1
        results[task_name] = {
            "accuracy": correct / len(samples),
            "correct": correct,
            "total": len(samples),
        }
    
    return results

# ==================== 主函数 ====================

def main():
    print("=" * 75)
    print("重新设计：数学推理迁移实验")
    print("=" * 75)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {list(CONFIG['models'].keys())}")
    print(f"任务: {list(TASKS.keys())}")
    print()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 已有的base和v2结果（来自之前的评估）
    PREVIOUS_RESULTS = {
        "base": {
            "gsm8k": {"accuracy": 0.444, "correct": 4, "total": 9},
            "physics": {"accuracy": 1.0, "correct": 8, "total": 8},
            "logic": {"accuracy": 0.80, "correct": 8, "total": 10},
            "spatial": {"accuracy": 0.75, "correct": 6, "total": 8},
            "arithmetic": {"accuracy": 1.0, "correct": 5, "total": 5},
        },
        "v2": {
            "gsm8k": {"accuracy": 0.444, "correct": 4, "total": 9},
            "physics": {"accuracy": 1.0, "correct": 8, "total": 8},
            "logic": {"accuracy": 0.70, "correct": 7, "total": 10},
            "spatial": {"accuracy": 0.75, "correct": 6, "total": 8},
            "arithmetic": {"accuracy": 1.0, "correct": 5, "total": 5},
        },
    }
    
    all_results = {**PREVIOUS_RESULTS}
    
    # 只评估gsm8k模型
    for model_name, adapter_path in CONFIG["models"].items():
        print(f"\n{'='*60}")
        print(f"评估模型: {model_name}")
        print("=" * 60)
        
        model = load_model(CONFIG["base_model_path"], adapter_path)
        results = evaluate_model(model, tokenizer, device, model_name)
        all_results[model_name] = results
        
        # 打印结果
        print(f"\n{model_name} 结果:")
        for task_name, (task_desc, _) in TASKS.items():
            acc = results[task_name]["accuracy"]
            correct = results[task_name]["correct"]
            total = results[task_name]["total"]
            print(f"  {task_desc}: {acc:.1%} ({correct}/{total})")
        
        # 清理显存
        cleanup_model(model)
    
    # ==================== 迁移分析 ====================
    print("\n" + "=" * 75)
    print("迁移分析")
    print("=" * 75)

    base = PREVIOUS_RESULTS["base"]
    
    # 汇总表格
    print(f"\n{'模型':<10} " + " ".join(f"{t:>10}" for t in TASKS.keys()))
    print("-" * 70)
    
    for model_name, results in all_results.items():
        accs = " ".join(f"{results[t]['accuracy']:>10.1%}" for t in TASKS.keys())
        print(f"{model_name:<10} {accs}")
    
    # Delta分析
    print("\n" + "-" * 70)
    print("相对于 base 的变化 (Δ)")
    print("-" * 70)
    
    for model_name in ["v2", "gsm8k"]:
        if model_name not in all_results:
            continue
        print(f"\n{model_name}:")
        for task_name in TASKS.keys():
            base_acc = base.get(task_name, {}).get("accuracy", 0)
            model_acc = all_results[model_name][task_name]["accuracy"]
            delta = model_acc - base_acc
            sign = "+" if delta >= 0 else ""
            tag = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "=")
            print(f"  {task_name}: {sign}{delta:.1%} {tag}")
    
    # ==================== 核心结论 ====================
    print("\n" + "=" * 75)
    print("核心结论")
    print("=" * 75)
    
    # 检查各层迁移
    base = all_results.get("base", {})
    gsm8k = all_results.get("gsm8k", {})
    v2 = all_results.get("v2", {})
    
    conclusions = []
    
    # L1: 源任务内迁移
    gsm8k_delta_l1 = gsm8k.get("gsm8k", {}).get("accuracy", 0) - base.get("gsm8k", {}).get("accuracy", 0)
    if gsm8k_delta_l1 > 0.05:
        conclusions.append(f"✅ L1源任务内提升: +{gsm8k_delta_l1:.1%}")
    else:
        conclusions.append(f"❌ L1源任务内无提升: {gsm8k_delta_l1:+.1%}")
    
    # L2: 结构相似任务迁移
    gsm8k_delta_l2 = gsm8k.get("physics", {}).get("accuracy", 0) - base.get("physics", {}).get("accuracy", 0)
    v2_delta_l2 = v2.get("physics", {}).get("accuracy", 0) - base.get("physics", {}).get("accuracy", 0)
    if gsm8k_delta_l2 > 0.05 and gsm8k_delta_l2 > v2_delta_l2:
        conclusions.append(f"✅ L2物理应用题正向迁移: +{gsm8k_delta_l2:.1%} (vs v2: {v2_delta_l2:+.1%})")
    
    # L3: 抽象推理迁移
    gsm8k_delta_l3 = gsm8k.get("logic", {}).get("accuracy", 0) - base.get("logic", {}).get("accuracy", 0)
    v2_delta_l3 = v2.get("logic", {}).get("accuracy", 0) - base.get("logic", {}).get("accuracy", 0)
    if gsm8k_delta_l3 > 0.05:
        conclusions.append(f"✅ L3逻辑推理正向迁移: +{gsm8k_delta_l3:.1%}")
    
    # L4: 对照组
    gsm8k_delta_l4 = gsm8k.get("spatial", {}).get("accuracy", 0) - base.get("spatial", {}).get("accuracy", 0)
    if abs(gsm8k_delta_l4) < 0.05:
        conclusions.append(f"➖ L4空间推理无迁移: {gsm8k_delta_l4:+.1%} (预期)")
    
    for c in conclusions:
        print(c)
    
    # ==================== 保存结果 ====================
    output_path = os.path.join(CONFIG["output_dir"], "gsm8k_transfer_redesign_results.json")
    save_data = {
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "tasks": {k: v[0] for k, v in TASKS.items()},
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n[OK] 结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
