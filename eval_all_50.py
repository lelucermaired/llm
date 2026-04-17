"""
eval_all_50.py

统一评测所有模型（50题版）
对比：base / v2 / cot-short / cot-detailed / dft

每个模型独立加载（子进程方式避免显存冲突）
"""

import os, json, sys
import numpy as np
from scipy import stats

# ==================== 评测配置 ====================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./results/evaluations/all_models_50"
CACHE_DIR  = os.path.join(OUTPUT_DIR, "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 待评测模型（name -> adapter_path，None表示base）
MODELS = {
    # "base":         None,            # 已完成
    # "v2":           "...",           # 已完成
    # "cot_short":    "...",           # 已完成
  #  "cot_detailed": "./checkpoints/qwen-gomoku-cot-detailed/final_model",
    #"dft": "./checkpoints/qwen-gomoku-dft/final_model",
  #  "grpo": "./checkpoints/qwen-gomoku-grpo/final_model",
   # "deeplora": "./checkpoints/qwen-gomoku-deeplora/final_model",
   # "ood_monitor": "./checkpoints/qwen-gomoku-ood-monitor/final_model",  # 新跑
"ood_best": "./checkpoints/qwen-gomoku-ood-monitor/best_ood_checkpoint",
}

# 模型描述
MODEL_DESC = {
    "base":         "基础模型",
    "v2":           "CE loss + 伪推理链（原始）",
    "cot_short":    "CE loss + 结构化短推理链",
    "cot_detailed": "CE loss + Qwen详细推理链（193条）",
    "dft":          "DFT loss + 伪推理链",
}

# ==================== 题库 ====================

MATH_SAMPLES = [
    ("What is 15 + 27?", "42"),
    ("What is 7 multiplied by 8?", "56"),
    ("What is 100 divided by 4?", "25"),
    ("What is the square root of 144?", "12"),
    ("What is 3 to the power of 4?", "81"),
    ("A rectangle has length 8 and width 5. What is its area?", "40"),
    ("If x + 3 = 10, what is x?", "7"),
    ("What is 25% of 80?", "20"),
    ("What is the average of 10, 20, 30, 40, and 50?", "30"),
    ("How many seconds are in 3 hours?", "10800"),
    ("What is 15% of 200?", "30"),
    ("If 2x - 4 = 10, what is x?", "7"),
    ("What is the perimeter of a square with side length 9?", "36"),
    ("Divide 360 by 12.", "30"),
    ("What is the next prime number after 13?", "17"),
    ("What is 45 + 67?", "112"),
    ("What is 9 multiplied by 9?", "81"),
    ("What is 200 divided by 8?", "25"),
    ("If a = 4 and b = 3, what is a squared plus b squared?", "25"),
    ("What is 2 to the power of 8?", "256"),
    ("A car travels 150 miles on 5 gallons. Miles per gallon?", "30"),
    ("If 3 workers finish in 6 days, how long for 9 workers?", "2"),
    ("A circle has radius 7. Circumference? Use pi=3.14.", "43.96"),
    ("A shirt costs $40, 20% off. Sale price?", "32"),
    ("How many minutes in 2.5 hours?", "150"),
    ("What is 17 + 38?", "55"),
    ("What is 13 multiplied by 6?", "78"),
    ("What is 90 divided by 6?", "15"),
    ("What is the square root of 81?", "9"),
    ("What is 5 to the power of 3?", "125"),
    ("If y = 3x and x = 4, what is y?", "12"),
    ("A triangle has base 10 and height 6. Area?", "30"),
    ("What is 8% of 250?", "20"),
    ("What is the median of 2, 4, 6, 8, 10?", "6"),
    ("How many days in 4 weeks?", "28"),
    ("What is 23 + 49?", "72"),
    ("What is 6 multiplied by 7?", "42"),
    ("What is 144 divided by 12?", "12"),
    ("What is the square root of 225?", "15"),
    ("What is 4 to the power of 3?", "64"),
    ("If p - 5 = 12, what is p?", "17"),
    ("A square has perimeter 40. What is its side length?", "10"),
    ("What is 35% of 60?", "21"),
    ("What is the mean of 5, 10, 15, 20?", "12.5"),
    ("How many hours in 3.5 days?", "84"),
    ("What is 66 + 34?", "100"),
    ("What is 11 multiplied by 12?", "132"),
    ("What is 500 divided by 20?", "25"),
    ("What is the square root of 400?", "20"),
    ("What is 6 to the power of 2?", "36"),
]

SPATIAL_SAMPLES = [
    ("A is to the left of B. B is above C. What is the relation of A to C?", "upper-left"),
    ("A is above B. B is to the right of C. What is the relation of A to C?", "upper-right"),
    ("A is to the left of B. B is to the left of C. What is the relation of A to C?", "left"),
    ("A is above B. B is above C. What is the relation of A to C?", "above"),
    ("A is to the right of B. B is below C. What is the relation of A to C?", "lower-right"),
    ("A is below B. B is to the left of C. What is the relation of A to C?", "lower-left"),
    ("A is to the left of B. B is below C. What is the relation of A to C?", "lower-left"),
    ("A is to the right of B. B is above C. What is the relation of A to C?", "upper-right"),
    ("A is below B. B is below C. What is the relation of A to C?", "below"),
    ("A is to the right of B. B is to the right of C. What is the relation of A to C?", "right"),
    ("A is above B. B is to the left of C. C is below D. What is A relative to D?", "upper-left"),
    ("A is to the right of B. B is above C. C is to the left of D. What is A relative to D?", "above"),
    ("A is to the left of B. B is to the left of C. C is above D. What is A relative to D?", "upper-left"),
    ("P above Q. Q left of R. R below S. What is P relative to S?", "upper-left"),
    ("X is north of Y. Y is east of Z. Where is X relative to Z?", "upper-right"),
    ("A above B. B right of C. C above D. What is A relative to D?", "upper-right"),
    ("A left of B. B below C. C left of D. What is A relative to D?", "lower-left"),
    ("A right of B. B above C. C right of D. What is A relative to D?", "upper-right"),
    ("A below B. B left of C. C below D. What is A relative to D?", "lower-left"),
    ("A left of B. B above C. C left of D. What is A relative to D?", "upper-left"),
    ("A left of B. B left of C. C left of D. D left of E. What is A relative to E?", "left"),
    ("A above B. B above C. C above D. What is A relative to D?", "above"),
    ("A right of B. B right of C. C above D. D right of E. What is A relative to E?", "upper-right"),
    ("A above B. B right of C. C above D. D left of E. What is A relative to E?", "above"),
    ("A left of B. B above C. C right of D. D below E. E left of F. What is A relative to F?", "left"),
    ("A above B. B above C. C left of D. D above E. What is A relative to E?", "upper-left"),
    ("A right of B. B below C. C right of D. D below E. What is A relative to E?", "lower-right"),
    ("A left of B. B below C. C left of D. D below E. What is A relative to E?", "lower-left"),
    ("A above B. B left of C. C above D. D left of E. What is A relative to E?", "upper-left"),
    ("A right of B. B above C. C right of D. D above E. What is A relative to E?", "upper-right"),
    ("Start facing East. Turn left. Turn left again. Which direction now?", "west"),
    ("Start facing North. Turn right three times. Which direction?", "west"),
    ("Start facing South. Turn right. Which direction now?", "west"),
    ("Start facing West. Turn left twice. Which direction now?", "west"),
    ("Start facing North. Turn left. Turn around. Which direction?", "east"),
    ("Start facing East. Turn right twice. Which direction now?", "west"),
    ("Start facing South. Turn left twice. Which direction now?", "north"),
    ("Start facing West. Turn right. Which direction now?", "north"),
    ("Start facing North. Turn around. Turn left. Which direction?", "west"),
    ("Start facing East. Turn left three times. Which direction?", "south"),
    ("A is upper-left of B. B is lower-right of C. What is A relative to C?", "overlap"),
    ("A is north of B. B is west of C. A is __ of C?", "upper-left"),
    ("A is south of B. B is east of C. A is __ of C?", "lower-right"),
    ("A is north of B. B is north of C. C is east of D. A is __ of D?", "upper-right"),
    ("A is west of B. B is south of C. C is west of D. A is __ of D?", "lower-left"),
    ("A is east of B. B is north of C. C is east of D. A is __ of D?", "upper-right"),
    ("A is south of B. B is west of C. C is south of D. A is __ of D?", "lower-left"),
    ("A is north of B. B is east of C. C is north of D. A is __ of D?", "upper-right"),
    ("A is west of B. B is north of C. C is west of D. A is __ of D?", "upper-left"),
    ("A is northeast of B. B is southwest of C. A is __ of C?", "overlap"),
]

PLANNING_SAMPLES = [
    ("Blocks: A on B, B on table. Goal: B on A. Minimum moves?", "2"),
    ("Tower of Hanoi: 2 disks. Minimum moves?", "3"),
    ("Tower of Hanoi: 3 disks. Minimum moves?", "7"),
    ("Grid start (1,1) goal (3,3) no obstacles. Minimum moves?", "4"),
    ("Pancake sort [2,1]. Minimum flips?", "1"),
    ("Pancake sort [3,2,1]. Minimum flips?", "2"),
    ("Blocks: C on B, B on A. Goal: all separate. First move?", "move c"),
    ("Blocks: A on table, B on table. Goal: B on A. Minimum moves?", "1"),
    ("Tower of Hanoi 1 disk. Minimum moves?", "1"),
    ("Tower of Hanoi 4 disks. Minimum moves?", "15"),
    ("Grid start (1,1) goal (4,4). Minimum moves?", "6"),
    ("Pancake sort [2,3,1]. First flip?", "flip top 3"),
    ("Blocks: D on C, C on B, B on A. Goal: all on table. First move?", "move d"),
    ("Tower of Hanoi: n disks need 2^n-1 moves. For 5 disks?", "31"),
    ("Grid 2x2: start (1,1) goal (2,2). Minimum moves?", "2"),
    ("Pancake sort [1,2,3]. Already sorted. Flips needed?", "0"),
    ("Blocks A B C D all on table. Goal: D on C, C on B, B on A. Minimum moves?", "3"),
    ("Tower of Hanoi 2 disks. First move?", "disk 1"),
    ("Pancake sort [3,1,2]. Minimum flips?", "3"),
    ("Grid start (1,1) goal (1,5). Only move right. Minimum moves?", "4"),
    ("Tower of Hanoi 5 disks. Minimum moves?", "31"),
    ("Blocks: A on B. Goal: A on table. Minimum moves?", "1"),
    ("Grid 3x3: start top-left, goal bottom-right. Minimum moves?", "4"),
    ("Pancake sort [4,3,2,1]. Minimum flips?", "3"),
    ("Blocks: B on A, C on table. Goal: C on B on A. Minimum moves?", "1"),
    ("Tower of Hanoi 6 disks. Minimum moves?", "63"),
    ("Grid start (1,1) goal (3,1). Only move right. Minimum moves?", "2"),
    ("Pancake sort [2,4,1,3]. First flip?", "flip top 4"),
    ("Blocks: A on B, C on D. Goal: C on A on B. Minimum moves?", "2"),
    ("Tower of Hanoi: 2^n - 1 moves for n disks. For 7 disks?", "127"),
    ("Grid start (2,2) goal (4,4). Minimum moves?", "4"),
    ("Pancake sort [3,2,4,1]. Minimum flips to sort?", "4"),
    ("Blocks: A B C all on table. Goal: A on B on C. Minimum moves?", "2"),
    ("Tower of Hanoi 3 disks. First move?", "disk 1"),
    ("Grid start (1,1) goal (5,5). Minimum moves?", "8"),
    ("Pancake sort [1,3,2]. Minimum flips?", "2"),
    ("Blocks: E on D on C on B on A. Goal: all on table. First move?", "move e"),
    ("Tower of Hanoi: minimum moves for 10 disks?", "1023"),
    ("Grid 4x4: start (1,1) goal (4,4). Minimum moves?", "6"),
    ("Pancake sort [4,1,2,3]. First flip?", "flip top 4"),
    ("Blocks: A on B, B on C, C on table. Goal: C on A. Minimum moves?", "5"),
    ("Tower of Hanoi: ratio of moves for n+1 vs n disks?", "2"),
    ("Grid start (1,1) goal (2,3). Minimum moves?", "3"),
    ("Pancake sort [2,1,4,3]. Minimum flips?", "2"),
    ("Blocks: A on table, B on A. Goal: A on B. Minimum moves?", "2"),
    ("Tower of Hanoi 3 disks. Last move?", "disk 1"),
    ("Grid start (3,3) goal (1,1). Minimum moves?", "4"),
    ("Pancake sort [3,4,1,2]. First flip?", "flip top 2"),
    ("Blocks: C on B on A. Goal: A on C on B. Minimum moves?", "5"),
    ("Tower of Hanoi: peg A to peg C. First disk goes to?", "c"),
]

LOGIC_SAMPLES = [
    ("All cats are mammals. Whiskers is a cat. Is Whiskers a mammal? Answer yes or no.", "yes"),
    ("If it rains the ground is wet. It did not rain. Is the ground wet? Answer yes or no.", "no"),
    ("All A are B. All B are C. Is every A also a C? Answer yes or no.", "yes"),
    ("No fish can walk. Nemo is a fish. Can Nemo walk? Answer yes or no.", "no"),
    ("Either P or Q. Not P. Is Q true? Answer yes or no.", "yes"),
    ("All students who passed studied hard. Tom did not study. Did Tom pass? Answer yes or no.", "no"),
    ("If P then Q. If Q then R. P is true. Is R true? Answer yes or no.", "yes"),
    ("All squares are rectangles. Shape X is not a rectangle. Is X a square? Answer yes or no.", "no"),
    ("Some birds cannot fly. All penguins are birds. Can all penguins fly? Answer yes or no.", "no"),
    ("If alarm rings then fire or drill. No fire and no drill. Did alarm ring? Answer yes or no.", "no"),
    ("All roses are flowers. All flowers need water. Do all roses need water? Answer yes or no.", "yes"),
    ("No dogs are cats. Rex is a dog. Is Rex a cat? Answer yes or no.", "no"),
    ("If A implies B, and B implies C, and A is true, is C true? Answer yes or no.", "yes"),
    ("All integers divisible by 4 are divisible by 2. 8 is divisible by 4. Is 8 divisible by 2? Answer yes or no.", "yes"),
    ("No prime number is divisible by 6. 7 is prime. Is 7 divisible by 6? Answer yes or no.", "no"),
    ("All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded? Answer yes or no.", "yes"),
    ("If it is Monday, the store is closed. The store is open. Is it Monday? Answer yes or no.", "no"),
    ("Either the key is in the drawer or on the table. The key is not in the drawer. Is the key on the table? Answer yes or no.", "yes"),
    ("All even numbers are divisible by 2. 13 is not divisible by 2. Is 13 even? Answer yes or no.", "no"),
    ("If X then Y. If Y then Z. X is false. Is Z necessarily true? Answer yes or no.", "no"),
]

SAMPLES = {
    "math":     MATH_SAMPLES,
    "spatial":  SPATIAL_SAMPLES,
    "planning": PLANNING_SAMPLES,
    "logic":    LOGIC_SAMPLES,
}

N_SAMPLES = {k: len(v) for k, v in SAMPLES.items()}


# ==================== 直接评测（主进程顺序执行）====================

def eval_model(model_name, adapter_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from tqdm import tqdm

    cache_file = os.path.join(CACHE_DIR, f"{model_name}.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            result = json.load(f)
        print(f"  [缓存] {model_name}: "
              f"math={result['math']:.3f} spatial={result['spatial']:.3f} "
              f"planning={result['planning']:.3f} logic={result['logic']:.3f}")
        return result

    print(f"\n{'='*50}")
    print(f"  加载模型：{model_name}")
    print(f"{'='*50}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True, trust_remote_code=True)

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = {}
    for task_name, task_samples in SAMPLES.items():
        correct = 0
        for prompt, answer in tqdm(task_samples, desc=f"  {task_name}", leave=False):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=80,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            resp = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()
            correct += int(answer.lower() in resp.lower())
        result[task_name] = correct / len(task_samples)

    del model, base_model
    torch.cuda.empty_cache()

    with open(cache_file, "w") as f:
        json.dump(result, f)
    print(f"  ✅ {model_name}: "
          f"math={result['math']:.3f} spatial={result['spatial']:.3f} "
          f"planning={result['planning']:.3f} logic={result['logic']:.3f}")
    return result


# ==================== 主流程 ====================

def main():
    print("=" * 75)
    print("全模型统一评测（50题版）")
    print(f"数学{N_SAMPLES['math']}题 / 空间{N_SAMPLES['spatial']}题 / "
          f"规划{N_SAMPLES['planning']}题 / 逻辑{N_SAMPLES['logic']}题")
    print("=" * 75)

    all_results = {}
    for model_name, adapter_path in MODELS.items():
        result = eval_model(model_name, adapter_path)
        if result:
            all_results[model_name] = result

    if not all_results:
        print("没有成功的评测结果")
        return

    base = all_results.get("base", {t: 0 for t in SAMPLES})

    # ==================== 汇总表 ====================
    print("\n" + "=" * 80)
    print("完整对比结果")
    print("=" * 80)
    tasks = list(SAMPLES.keys())
    header = f"{'模型':<16} " + " ".join(f"{t:>10}" for t in tasks) + "  说明"
    print(header)
    print("-" * 80)

    for name, res in all_results.items():
        vals = " ".join(f"{res[t]:>10.3f}" for t in tasks)
        desc = MODEL_DESC.get(name, "")
        if name != "base":
            deltas = [res[t] - base.get(t, 0) for t in tasks]
            delta_str = " Δ(" + " ".join(f"{d:+.2f}" for d in deltas) + ")"
        else:
            delta_str = ""
        print(f"{name:<16} {vals}  {desc}{delta_str}")

    # ==================== 核心问题：推理链质量的影响 ====================
    print("\n" + "=" * 75)
    print("核心问题：推理链质量对OOD迁移的影响")
    print("（控制变量：数据量相同时比较）")
    print("=" * 75)

    comparisons = [
        ("v2",          "cot_short",    "伪推理链 vs 结构化短推理链（数据量相同3168条）"),
        ("cot_short",   "cot_detailed", "结构化短推理链 vs 详细长推理链（数量差异：3168 vs 193）"),
        ("v2",          "dft",          "CE loss vs DFT loss（推理链相同，loss函数不同）"),
    ]

    for name_a, name_b, desc in comparisons:
        if name_a not in all_results or name_b not in all_results:
            continue
        res_a = all_results[name_a]
        res_b = all_results[name_b]
        print(f"\n{desc}")
        for t in tasks:
            delta = res_b[t] - res_a[t]
            tag = "↑正向" if delta > 0.02 else ("↓负向" if delta < -0.02 else "=持平")
            print(f"  {t:<10}: {name_a}={res_a[t]:.3f}  {name_b}={res_b[t]:.3f}  "
                  f"Δ={delta:+.3f}  [{tag}]")

    # ==================== 结论 ====================
    print("\n" + "=" * 75)
    print("综合结论")
    print("=" * 75)

    any_positive = False
    for name, res in all_results.items():
        if name == "base":
            continue
        for t in tasks:
            if res[t] - base.get(t, 0) > 0.05:
                any_positive = True
                print(f"✅ {name} 在 {t} 出现正向迁移 ({res[t]-base.get(t,0):+.3f})")

    if not any_positive:
        print("所有变量（推理链质量/loss函数）均未产生显著OOD正向迁移")
        print("结论：推理链质量不是SFT跨域迁移的瓶颈，根本限制在于SFT训练范式本身")
        print("支持：Chu et al. (2025) 'SFT Memorizes, RL Generalizes'")

    # 保存
    save_path = os.path.join(OUTPUT_DIR, "all_models_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "n_samples": N_SAMPLES,
            "base_deltas": {
                name: {t: res[t] - base.get(t, 0) for t in tasks}
                for name, res in all_results.items() if name != "base"
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()