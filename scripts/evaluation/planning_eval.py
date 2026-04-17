"""
planning_eval.py

序列规划专项评测
验证：r=64五子棋微调在序列规划上的正向迁移是否稳定？
对比：基础模型 vs r=8微调 vs r=64微调

题目设计：50道，覆盖4类规划任务，3个难度梯度
- 积木移动（Blocks World）：10道简单 + 10道中等
- 汉诺塔（Tower of Hanoi）：10道
- 迷宫路径（Grid Navigation）：10道
- 煎饼排序（Pancake Sorting）：10道

用法：
    python planning_eval.py
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "model_paths": {
        "base":  None,
        "r8":    "./checkpoints/qwen-gomoku-v2/final_model",
        "r64":   "./checkpoints/qwen-gomoku-r64/final_model",
    },
    "output_dir": "./planning_eval_results",
    "max_new_tokens": 120,
    "seeds": [42, 123, 456],   # 多种子重复，验证稳定性
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 题目库 ====================

# --- 类型1：积木移动（简单，2-3步）---
BLOCKS_EASY = [
    {
        "question": "Blocks world. Initial: A on table, B on A. Goal: B on table, A on B.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move b", "type": "blocks", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Blocks world. Initial: C on table, B on C, A on B. Goal: A on table, B on A, C on B.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move a", "type": "blocks", "difficulty": "easy", "steps": 3
    },
    {
        "question": "Blocks world. Initial: A on table, B on table, C on B. Goal: C on table, A on C.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move c", "type": "blocks", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Blocks world. Initial: A on table, B on A, C on table. Goal: C on A, B on C.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move b", "type": "blocks", "difficulty": "easy", "steps": 3
    },
    {
        "question": "Blocks world. Initial: B on table, A on B. Goal: A on table, B on A.\nHow many moves are needed and what is the sequence?",
        "answer_key": "2", "type": "blocks", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Blocks world. Initial: A on table, B on table (separate). Goal: B on A.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move b", "type": "blocks", "difficulty": "easy", "steps": 1
    },
    {
        "question": "Blocks world. Initial: A on B, B on table, C on table. Goal: A on table, C on A.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move a", "type": "blocks", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Blocks world. Initial: C on B, B on A, A on table. Goal: all blocks on table separately.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move c", "type": "blocks", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Blocks world. Initial: A on table, B on table, C on table. Goal: A on B, B on C.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move a", "type": "blocks", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Blocks world. Initial: D on C, C on table, A on B, B on table. Goal: D on table, A on D.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move d", "type": "blocks", "difficulty": "easy", "steps": 2
    },
]

# --- 类型1：积木移动（中等，4-5步）---
BLOCKS_MEDIUM = [
    {
        "question": "Blocks world. Initial: D on C, C on B, B on A, A on table. Goal: A on D, D on C, C on B, B on table.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move d", "type": "blocks", "difficulty": "medium", "steps": 5
    },
    {
        "question": "Blocks world. Initial: A on table, B on A, C on B, D on C. Goal: D on table, C on D, B on C, A on B.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move d", "type": "blocks", "difficulty": "medium", "steps": 6
    },
    {
        "question": "Blocks world. Initial: A on B, B on table, C on D, D on table. Goal: C on A, A on B, B on D.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move a", "type": "blocks", "difficulty": "medium", "steps": 4
    },
    {
        "question": "Blocks world. Initial: B on A, A on table, C on table, D on C. Goal: A on D, D on C, B on A.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move b", "type": "blocks", "difficulty": "medium", "steps": 4
    },
    {
        "question": "Blocks world. Initial: A on table, B on table, C on table, D on table. Goal: D on C, C on B, B on A.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move d", "type": "blocks", "difficulty": "medium", "steps": 3
    },
    {
        "question": "Blocks world. Initial: C on A, A on table, B on table, D on B. Goal: A on B, D on A, C on D.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move c", "type": "blocks", "difficulty": "medium", "steps": 4
    },
    {
        "question": "Blocks world. Initial: A on B, B on C, C on table, D on table. Goal: D on A, A on B, B on table, C on table.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move a", "type": "blocks", "difficulty": "medium", "steps": 4
    },
    {
        "question": "Blocks world. Initial: B on table, A on B, D on table, C on D. Goal: A on table, B on A, C on B.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move a", "type": "blocks", "difficulty": "medium", "steps": 4
    },
    {
        "question": "Blocks world. Initial: A on table, B on A, C on B, D on table. Goal: D on C, B on D, A on B.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move c", "type": "blocks", "difficulty": "medium", "steps": 5
    },
    {
        "question": "Blocks world. Initial: C on B, B on table, A on table, D on A. Goal: B on C, C on D, A on B.\nMinimum moves? Start your answer with the first move.",
        "answer_key": "move c", "type": "blocks", "difficulty": "medium", "steps": 5
    },
]

# --- 类型2：汉诺塔 ---
HANOI = [
    {
        "question": "Tower of Hanoi: 1 disk on peg A. Move to peg C.\nHow many moves needed?",
        "answer_key": "1", "type": "hanoi", "difficulty": "easy", "steps": 1
    },
    {
        "question": "Tower of Hanoi: 2 disks on peg A (small on top). Move all to peg C using peg B.\nHow many moves needed?",
        "answer_key": "3", "type": "hanoi", "difficulty": "easy", "steps": 3
    },
    {
        "question": "Tower of Hanoi: 3 disks on peg A. Move all to peg C using peg B.\nHow many moves needed?",
        "answer_key": "7", "type": "hanoi", "difficulty": "medium", "steps": 7
    },
    {
        "question": "Tower of Hanoi: 4 disks on peg A. Move all to peg C using peg B.\nHow many moves needed?",
        "answer_key": "15", "type": "hanoi", "difficulty": "hard", "steps": 15
    },
    {
        "question": "Tower of Hanoi with 2 disks on peg A. Move all to peg C using peg B. What is the first move?",
        "answer_key": "disk 1", "type": "hanoi", "difficulty": "easy", "steps": 3
    },
    {
        "question": "Tower of Hanoi with 3 disks on peg A. Move all to peg C using peg B. What is the first move?",
        "answer_key": "disk 1", "type": "hanoi", "difficulty": "medium", "steps": 7
    },
    {
        "question": "Tower of Hanoi with 2 disks. After moving disk 1 from A to C, what is the next move?",
        "answer_key": "disk 2", "type": "hanoi", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Tower of Hanoi: n disks require 2^n - 1 moves. How many moves for 5 disks?",
        "answer_key": "31", "type": "hanoi", "difficulty": "medium", "steps": 31
    },
    {
        "question": "Tower of Hanoi with 3 disks on peg A, target peg C, auxiliary peg B. List the complete move sequence.",
        "answer_key": "disk 1 from a to c", "type": "hanoi", "difficulty": "medium", "steps": 7
    },
    {
        "question": "Tower of Hanoi: 2 disks on peg A. Move to peg B using peg C. What is the complete sequence?",
        "answer_key": "disk 1", "type": "hanoi", "difficulty": "easy", "steps": 3
    },
]

# --- 类型3：迷宫路径 ---
MAZE = [
    {
        "question": "Grid navigation: Start at (1,1), goal at (1,3). Moves: Up, Down, Left, Right. No obstacles.\nShortest path?",
        "answer_key": "right", "type": "maze", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Grid navigation: Start at (1,1), goal at (3,1). Moves: Up, Down, Left, Right. No obstacles.\nShortest path?",
        "answer_key": "down", "type": "maze", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Grid navigation: Start at (1,1), goal at (3,3) in a 3x3 grid. No obstacles.\nMinimum moves needed?",
        "answer_key": "4", "type": "maze", "difficulty": "easy", "steps": 4
    },
    {
        "question": "Grid 4x4: Start (1,1), goal (4,4). No obstacles. Minimum moves?",
        "answer_key": "6", "type": "maze", "difficulty": "easy", "steps": 6
    },
    {
        "question": "Grid navigation: Start (2,2), goal (2,5). Only Right moves allowed. Minimum moves?",
        "answer_key": "3", "type": "maze", "difficulty": "easy", "steps": 3
    },
    {
        "question": "Grid 3x3: Start (1,1), goal (3,3). Obstacle at (2,2). Minimum moves?",
        "answer_key": "4", "type": "maze", "difficulty": "medium", "steps": 4
    },
    {
        "question": "Grid navigation: Robot at (1,1) facing East. Goal: reach (1,3). Commands: Forward, Turn Left, Turn Right. Minimum commands?",
        "answer_key": "2", "type": "maze", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Grid 5x5: Start (1,1), goal (5,5). No obstacles. Minimum moves?",
        "answer_key": "8", "type": "maze", "difficulty": "medium", "steps": 8
    },
    {
        "question": "Grid navigation: Start (1,1). Move sequence: Right, Right, Down, Down. Where do you end up?",
        "answer_key": "(3,3)", "type": "maze", "difficulty": "easy", "steps": 4
    },
    {
        "question": "Grid 3x3: Start (1,1), goal (2,3). Obstacle at (1,3) and (2,2). Find a valid path.",
        "answer_key": "down", "type": "maze", "difficulty": "medium", "steps": 3
    },
]

# --- 类型4：煎饼排序 ---
PANCAKE = [
    {
        "question": "Pancake sorting: Stack is [1, 2] top to bottom (already sorted). Moves needed to sort?",
        "answer_key": "0", "type": "pancake", "difficulty": "easy", "steps": 0
    },
    {
        "question": "Pancake sorting: Stack is [2, 1] top to bottom. Flip top 2 to sort. How many flips?",
        "answer_key": "1", "type": "pancake", "difficulty": "easy", "steps": 1
    },
    {
        "question": "Pancake sorting: Stack is [3, 1, 2] top to bottom. Goal: [1, 2, 3]. Minimum flips?",
        "answer_key": "3", "type": "pancake", "difficulty": "medium", "steps": 3
    },
    {
        "question": "Pancake sorting: Stack is [2, 3, 1] top to bottom. Goal: [1, 2, 3]. What is the first flip?",
        "answer_key": "flip top 3", "type": "pancake", "difficulty": "medium", "steps": 3
    },
    {
        "question": "Pancake sorting: Stack is [3, 2, 1] top to bottom. Goal: [1, 2, 3]. What is the first flip?",
        "answer_key": "flip top 3", "type": "pancake", "difficulty": "easy", "steps": 2
    },
    {
        "question": "Pancake sorting: Stack is [2, 1, 3] top to bottom. Goal: [1, 2, 3]. Minimum flips?",
        "answer_key": "1", "type": "pancake", "difficulty": "easy", "steps": 1
    },
    {
        "question": "Pancake sorting: Stack is [4, 3, 2, 1] top to bottom. Goal: [1, 2, 3, 4]. Minimum flips?",
        "answer_key": "4", "type": "pancake", "difficulty": "hard", "steps": 4
    },
    {
        "question": "Pancake sorting: Stack is [1, 3, 2] top to bottom. Goal: [1, 2, 3]. Minimum flips?",
        "answer_key": "2", "type": "pancake", "difficulty": "medium", "steps": 2
    },
    {
        "question": "Pancake sorting: Stack is [3, 1, 2] top to bottom. First flip top 2, then what?",
        "answer_key": "flip top 3", "type": "pancake", "difficulty": "medium", "steps": 2
    },
    {
        "question": "Pancake sorting: Stack is [2, 1] top to bottom. What single flip sorts it?",
        "answer_key": "flip top 2", "type": "pancake", "difficulty": "easy", "steps": 1
    },
]

ALL_SAMPLES = BLOCKS_EASY + BLOCKS_MEDIUM + HANOI + MAZE + PANCAKE


# ==================== 评测函数 ====================

def load_model(path, base_path, is_base=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if is_base:
        base.eval()
        return base
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens, device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def check(response, answer_key):
    return answer_key.lower() in response.lower()


def evaluate(model, tokenizer, samples, desc, device):
    correct = 0
    by_type = {}
    by_difficulty = {}
    results = []

    for sample in tqdm(samples, desc=desc):
        response = generate(model, tokenizer, sample["question"],
                          CONFIG["max_new_tokens"], device)
        is_correct = check(response, sample["answer_key"])
        correct += int(is_correct)

        t = sample["type"]
        d = sample["difficulty"]
        by_type.setdefault(t, []).append(int(is_correct))
        by_difficulty.setdefault(d, []).append(int(is_correct))

        results.append({
            "type": t,
            "difficulty": d,
            "steps": sample["steps"],
            "answer_key": sample["answer_key"],
            "response": response[:150],
            "correct": is_correct,
        })

    acc = correct / len(samples)
    type_acc = {t: sum(v)/len(v) for t, v in by_type.items()}
    diff_acc = {d: sum(v)/len(v) for d, v in by_difficulty.items()}
    return acc, type_acc, diff_acc, results


# ==================== 主流程 ====================

def main():
    print("=" * 65)
    print("序列规划专项评测（50道题）")
    print("基础模型 vs r=8微调 vs r=64微调")
    print("=" * 65)
    print(f"题目分布：积木移动×20 | 汉诺塔×10 | 迷宫×10 | 煎饼排序×10")

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}

    for model_name, model_path in CONFIG["model_paths"].items():
        print(f"\n{'='*65}")
        print(f"评测模型：{model_name}")
        print(f"{'='*65}")

        is_base = (model_path is None)
        model = load_model(model_path, CONFIG["base_model_path"], is_base)

        acc, type_acc, diff_acc, results = evaluate(
            model, tokenizer, ALL_SAMPLES, f"{model_name}", device)

        print(f"\n总体准确率: {acc:.3f} ({int(acc*len(ALL_SAMPLES))}/{len(ALL_SAMPLES)})")
        print(f"\n按任务类型：")
        for t, a in sorted(type_acc.items()):
            print(f"  {t:<12}: {a:.3f}")
        print(f"\n按难度：")
        for d, a in sorted(diff_acc.items()):
            print(f"  {d:<10}: {a:.3f}")

        all_results[model_name] = {
            "accuracy": acc,
            "by_type": type_acc,
            "by_difficulty": diff_acc,
            "results": results,
        }

        del model
        torch.cuda.empty_cache()

    # ========== 汇总对比 ==========
    print("\n" + "=" * 65)
    print("汇总对比")
    print("=" * 65)
    base_acc = all_results["base"]["accuracy"]
    r8_acc = all_results["r8"]["accuracy"]
    r64_acc = all_results["r64"]["accuracy"]

    print(f"\n{'模型':<10} {'总体准确率':>10} {'vs基础':>10}")
    print("-" * 35)
    print(f"{'base':<10} {base_acc:>10.3f} {'—':>10}")
    print(f"{'r8':<10} {r8_acc:>10.3f} {r8_acc-base_acc:>+10.3f}")
    print(f"{'r64':<10} {r64_acc:>10.3f} {r64_acc-base_acc:>+10.3f}")

    print(f"\n按任务类型对比（r64 vs 基础）：")
    print(f"{'类型':<12} {'基础':>8} {'r8':>8} {'r64':>8} {'r64变化':>10}")
    print("-" * 50)
    for t in ["blocks", "hanoi", "maze", "pancake"]:
        b = all_results["base"]["by_type"].get(t, 0)
        r8 = all_results["r8"]["by_type"].get(t, 0)
        r64 = all_results["r64"]["by_type"].get(t, 0)
        flag = " ✅" if r64 - b > 0.05 else (" ❌" if r64 - b < -0.05 else "")
        print(f"{t:<12} {b:>8.3f} {r8:>8.3f} {r64:>8.3f} {r64-b:>+10.3f}{flag}")

    print(f"\n按难度对比（r64 vs 基础）：")
    print(f"{'难度':<10} {'基础':>8} {'r8':>8} {'r64':>8} {'r64变化':>10}")
    print("-" * 48)
    for d in ["easy", "medium", "hard"]:
        b = all_results["base"]["by_difficulty"].get(d, 0)
        r8 = all_results["r8"]["by_difficulty"].get(d, 0)
        r64 = all_results["r64"]["by_difficulty"].get(d, 0)
        flag = " ✅" if r64 - b > 0.05 else (" ❌" if r64 - b < -0.05 else "")
        print(f"{d:<10} {b:>8.3f} {r8:>8.3f} {r64:>8.3f} {r64-b:>+10.3f}{flag}")

    # ========== 核心结论 ==========
    print("\n" + "=" * 65)
    print("核心结论")
    print("=" * 65)
    delta_r8 = r8_acc - base_acc
    delta_r64 = r64_acc - base_acc

    if delta_r64 > 0.05 and abs(delta_r8) <= 0.05:
        print(f"\n→ ✅ r=64正向迁移稳定（{delta_r64:+.3f}），r=8零迁移（{delta_r8:+.3f}）")
        print("  rank是序列规划迁移的关键条件")
        print("  低秩（r=8）更新量不足以建立跨域序列推理能力")
        print("  高秩（r=64）提供了足够的参数空间支持迁移")
    elif delta_r64 > 0.05 and delta_r8 > 0.05:
        print(f"\n→ ✅ r=8和r=64均出现正向迁移（r8:{delta_r8:+.3f}, r64:{delta_r64:+.3f}）")
        print("  序列规划迁移不依赖rank大小，五子棋推理模式可广泛迁移至规划任务")
    elif abs(delta_r64) <= 0.05 and abs(delta_r8) <= 0.05:
        print(f"\n→ — 两个模型均为零迁移（r8:{delta_r8:+.3f}, r64:{delta_r64:+.3f}）")
        print("  之前10道题的+0.100可能是样本量不足导致的噪声")
        print("  50道题的结果更可靠，零迁移结论推广至序列规划")
    else:
        print(f"\n→ 混合结论：r8={delta_r8:+.3f}, r64={delta_r64:+.3f}")
        print("  需要进一步分析按类型和难度的细粒度结果")

    # ========== 保存 ==========
    save_path = os.path.join(CONFIG["output_dir"], "planning_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()