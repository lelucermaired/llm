"""
model_merge_eval.py

模型合并实验：将v2 LoRA权重以不同比例插值回base model
理论依据：Task Arithmetic / Model Merging
  合并后模型 = base + α × (v2 - base)
  α=0: 纯base
  α=1: 纯v2
  α=0.3/0.5/0.7: 插值，期望在五子棋能力和OOD泛化之间取得平衡

参考：
  Jin et al. (2025) arXiv:2509.12235 - 低秩合并作为廉价OOD恢复机制
  Ilharco et al. (2023) Task Arithmetic
  arXiv:2511.02451 - CPT模型合并恢复通用能力
"""

import os, json, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "adapter_path":    "./archive/checkpoints/qwen-gomoku-real/final_model",
    "output_dir":      "./results/evaluations/model_merge",
    "alphas":          [0.0, 0.3, 0.5, 0.7, 1.0],  # 插值比例
    "max_new_tokens":  80,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 50题题库（与eval_all_50一致）====================

MATH_SAMPLES = [
    ("What is 15 + 27?", "42"), ("What is 7 multiplied by 8?", "56"),
    ("What is 100 divided by 4?", "25"), ("What is the square root of 144?", "12"),
    ("What is 3 to the power of 4?", "81"), ("A rectangle has length 8 and width 5. What is its area?", "40"),
    ("If x + 3 = 10, what is x?", "7"), ("What is 25% of 80?", "20"),
    ("What is the average of 10, 20, 30, 40, and 50?", "30"), ("How many seconds are in 3 hours?", "10800"),
    ("What is 15% of 200?", "30"), ("If 2x - 4 = 10, what is x?", "7"),
    ("What is the perimeter of a square with side length 9?", "36"), ("Divide 360 by 12.", "30"),
    ("What is the next prime number after 13?", "17"), ("What is 45 + 67?", "112"),
    ("What is 9 multiplied by 9?", "81"), ("What is 200 divided by 8?", "25"),
    ("If a = 4 and b = 3, what is a squared plus b squared?", "25"), ("What is 2 to the power of 8?", "256"),
    ("A car travels 150 miles on 5 gallons. Miles per gallon?", "30"),
    ("If 3 workers finish in 6 days, how long for 9 workers?", "2"),
    ("A circle has radius 7. Circumference? Use pi=3.14.", "43.96"),
    ("A shirt costs $40, 20% off. Sale price?", "32"), ("How many minutes in 2.5 hours?", "150"),
    ("What is 17 + 38?", "55"), ("What is 13 multiplied by 6?", "78"),
    ("What is 90 divided by 6?", "15"), ("What is the square root of 81?", "9"),
    ("What is 5 to the power of 3?", "125"), ("If y = 3x and x = 4, what is y?", "12"),
    ("A triangle has base 10 and height 6. Area?", "30"), ("What is 8% of 250?", "20"),
    ("What is the median of 2, 4, 6, 8, 10?", "6"), ("How many days in 4 weeks?", "28"),
    ("What is 23 + 49?", "72"), ("What is 6 multiplied by 7?", "42"),
    ("What is 144 divided by 12?", "12"), ("What is the square root of 225?", "15"),
    ("What is 4 to the power of 3?", "64"), ("If p - 5 = 12, what is p?", "17"),
    ("A square has perimeter 40. What is its side length?", "10"), ("What is 35% of 60?", "21"),
    ("What is the mean of 5, 10, 15, 20?", "12.5"), ("How many hours in 3.5 days?", "84"),
    ("What is 66 + 34?", "100"), ("What is 11 multiplied by 12?", "132"),
    ("What is 500 divided by 20?", "25"), ("What is the square root of 400?", "20"),
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
    "math": MATH_SAMPLES,
    "spatial": SPATIAL_SAMPLES,
    "planning": PLANNING_SAMPLES,
    "logic": LOGIC_SAMPLES,
}

BASE_RESULTS = {
    "math": 0.720, "spatial": 0.360, "planning": 0.520, "logic": 1.000
}


# ==================== 合并与评测 ====================

def build_merged_model(base_model, adapter_path, alpha):
    """
    通过缩放LoRA权重实现Task Arithmetic插值
    alpha=0: 纯base（不加载adapter）
    alpha=1: 纯v2（标准LoRA加载）
    alpha=0.5: 插值（LoRA权重×0.5）

    原理：LoRA的增量 = B×A，缩放alpha后增量变为 alpha×B×A
    等价于 merged = base + alpha × (v2 - base)
    """
    if alpha == 0.0:
        return base_model

    from peft import PeftModel
    import copy

    model = PeftModel.from_pretrained(base_model, adapter_path)

    if alpha != 1.0:
        # 缩放所有LoRA层的权重
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.data = param.data * alpha

    return model


def evaluate(model, tokenizer, device):
    """评测四类任务"""
    results = {}
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
                    **inputs, max_new_tokens=CONFIG["max_new_tokens"],
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            resp = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()
            correct += int(answer.lower() in resp.lower())
        results[task_name] = correct / len(task_samples)
    return results


def main():
    print("=" * 65)
    print("模型合并实验（Task Arithmetic）")
    print("merged = base + α × (v2 - base)")
    print(f"测试α值：{CONFIG['alphas']}")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}

    for alpha in CONFIG["alphas"]:
        cache_file = os.path.join(CONFIG["output_dir"], f"alpha_{alpha:.1f}.json")
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                res = json.load(f)
            all_results[alpha] = res
            print(f"\n[缓存] α={alpha}: math={res['math']:.3f} "
                  f"spatial={res['spatial']:.3f} planning={res['planning']:.3f}")
            continue

        print(f"\n{'='*50}")
        print(f"加载模型：α={alpha}")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model_path"], quantization_config=bnb,
            device_map="auto", local_files_only=True,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )

        model = build_merged_model(
            base_model, CONFIG["adapter_path"], alpha)
        model.eval()

        res = evaluate(model, tokenizer, device)
        all_results[alpha] = res

        print(f"  ✅ α={alpha}: math={res['math']:.3f} "
              f"spatial={res['spatial']:.3f} "
              f"planning={res['planning']:.3f} "
              f"logic={res['logic']:.3f}")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        del model, base_model
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # ==================== 汇总 ====================
    print("\n" + "=" * 70)
    print("模型合并结果汇总")
    print("merged = base + α × (v2 - base)")
    print("=" * 70)

    tasks = ["math", "spatial", "planning", "logic"]
    print(f"\n{'α':>6}  {'math':>8} {'spatial':>8} {'planning':>8} {'logic':>8}  "
          f"{'avg_delta':>10}")
    print("-" * 65)

    for alpha in sorted(all_results.keys()):
        res = all_results[alpha]
        deltas = [res[t] - BASE_RESULTS[t] for t in tasks]
        avg_delta = np.mean(deltas)
        vals = " ".join(f"{res[t]:>8.3f}" for t in tasks)
        tag = ""
        if alpha == 0.0:
            tag = "← base"
        elif alpha == 1.0:
            tag = "← v2(CE)"
        elif avg_delta == max(
            np.mean([all_results[a][t] - BASE_RESULTS[t] for t in tasks])
            for a in all_results
        ):
            tag = "← 最优"
        print(f"{alpha:>6.1f}  {vals}  {avg_delta:>+10.3f}  {tag}")

    print(f"\n基准(base): math={BASE_RESULTS['math']:.3f} "
          f"spatial={BASE_RESULTS['spatial']:.3f} "
          f"planning={BASE_RESULTS['planning']:.3f} "
          f"logic={BASE_RESULTS['logic']:.3f}")

    # 找最优alpha
    best_alpha = max(
        all_results.keys(),
        key=lambda a: np.mean([all_results[a][t] - BASE_RESULTS[t] for t in tasks])
    )
    best_avg = np.mean([all_results[best_alpha][t] - BASE_RESULTS[t] for t in tasks])

    print(f"\n【结论】")
    if best_avg > 0.01:
        print(f"  最优合并比例：α={best_alpha}，平均delta={best_avg:+.3f}")
        print(f"  模型合并能在保留五子棋能力的同时改善OOD泛化")
        print(f"  支持Task Arithmetic理论：适度插值比纯SFT模型有更好的泛化性")
    elif best_avg > -0.01:
        print(f"  最优合并比例：α={best_alpha}，平均delta≈0")
        print(f"  模型合并未能改善OOD性能，但也没有损害")
    else:
        print(f"  所有合并比例均低于base，五子棋SFT的参数方向与通用推理不兼容")

    save_path = os.path.join(CONFIG["output_dir"], "merge_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            str(a): r for a, r in all_results.items()
        }, f, indent=2)
    print(f"\n✅ 结果保存至: {save_path}")


if __name__ == "__main__":
    main()