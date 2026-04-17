"""
dft_eval_50.py
扩充到每类50题的DFT评测脚本

数学推理：50题（与math_transfer_verify一致）
空间推理：50题（扩充，难度递进2-6步）
序列规划：50题（与planning_eval一致）
逻辑推理：20题（base已满分，区分度低，20题足够）

已有结果（20题版本）：
base:      math=0.750, spatial=0.500, planning=0.550, logic=1.000
v2(CE):    math=0.750, spatial=0.500, planning=0.500, logic=1.000
shallow:   math=0.750, spatial=0.500, planning=0.500, logic=1.000
full_lora: math=0.800, spatial=0.450, planning=0.550, logic=1.000
"""

import os, json, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "model_path": "./checkpoints/qwen-gomoku-dft/final_model",
    "output_dir": "./results/evaluations/dft_50",
    "max_new_tokens": 80,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

PREVIOUS = {
    "base":      {"math": 0.750, "spatial": 0.500, "planning": 0.550, "logic": 1.000},
    "v2(CE)":    {"math": 0.750, "spatial": 0.500, "planning": 0.500, "logic": 1.000},
    "shallow":   {"math": 0.750, "spatial": 0.500, "planning": 0.500, "logic": 1.000},
    "full_lora": {"math": 0.800, "spatial": 0.450, "planning": 0.550, "logic": 1.000},
}

# ==================== 数学推理 50题 ====================
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

# ==================== 空间推理 50题（难度递进） ====================
SPATIAL_SAMPLES = [
    # 2步推理（10题）
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
    # 3步推理（10题）
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
    # 4步推理（10题）
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
    # 方向推理（10题）
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
    # 组合推理（10题）
    ("A is upper-left of B. B is lower-right of C. What is A relative to C?", "overlap"),
    ("A is north of B. B is west of C. A is __ of C?", "upper-left"),
    ("A is south of B. B is east of C. A is __ of C?", "lower-right"),
    ("A is northeast of B. B is southwest of C. A is __ of C?", "overlap"),
    ("A is north of B. B is north of C. C is east of D. A is __ of D?", "upper-right"),
    ("A is west of B. B is south of C. C is west of D. A is __ of D?", "lower-left"),
    ("A is east of B. B is north of C. C is east of D. A is __ of D?", "upper-right"),
    ("A is south of B. B is west of C. C is south of D. A is __ of D?", "lower-left"),
    ("A is north of B. B is east of C. C is north of D. A is __ of D?", "upper-right"),
    ("A is west of B. B is north of C. C is west of D. A is __ of D?", "upper-left"),
]

# ==================== 序列规划 50题 ====================
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

# ==================== 逻辑推理 20题 ====================
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


def generate(model, tokenizer, prompt, device):
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
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()


def evaluate_task(model, tokenizer, samples, desc, device):
    correct = 0
    for prompt, answer in tqdm(samples, desc=desc, leave=False):
        correct += int(answer.lower() in
                       generate(model, tokenizer, prompt, device).lower())
    return correct / len(samples)


def main():
    print("=" * 65)
    print("DFT模型评测（50题版）")
    print(f"数学50题 / 空间50题 / 规划50题 / 逻辑20题")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n加载dft模型...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model_path"], quantization_config=bnb,
        device_map="auto", local_files_only=True,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, CONFIG["model_path"])
    model.eval()

    math_acc    = evaluate_task(model, tokenizer, MATH_SAMPLES,    "数学(50)", device)
    spatial_acc = evaluate_task(model, tokenizer, SPATIAL_SAMPLES, "空间(50)", device)
    plan_acc    = evaluate_task(model, tokenizer, PLANNING_SAMPLES, "规划(50)", device)
    logic_acc   = evaluate_task(model, tokenizer, LOGIC_SAMPLES,   "逻辑(20)", device)

    dft = {"math": math_acc, "spatial": spatial_acc,
           "planning": plan_acc, "logic": logic_acc}

    del model; torch.cuda.empty_cache()

    base = PREVIOUS["base"]
    all_models = {**PREVIOUS, "dft(DFT)": dft}

    print("\n" + "=" * 78)
    print("完整对比（50题版，唯一变量：loss函数）")
    print("=" * 78)
    print(f"\n{'模型':<14} {'数学/50':>8} {'空间/50':>8} {'规划/50':>8} {'逻辑/20':>8}  说明")
    print("-" * 75)

    labels = {
        "base":      "基础模型",
        "v2(CE)":    "五子棋 标准CE",
        "shallow":   "五子棋 前10层 CE",
        "full_lora": "五子棋 全模块 CE",
        "dft(DFT)":  "五子棋 DFT loss  ← 本次",
    }
    for name, res in all_models.items():
        if name == "base":
            delta_str = ""
        else:
            d = [res[t] - base[t] for t in ["math","spatial","planning","logic"]]
            delta_str = f"  Δ({d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f} {d[3]:+.2f})"
        print(f"{name:<14} {res['math']:>8.3f} {res['spatial']:>8.3f} "
              f"{res['planning']:>8.3f} {res['logic']:>8.3f}  "
              f"{labels.get(name,'')}{delta_str}")

    print("\n" + "=" * 65)
    print("核心结论：DFT vs CE（v2）")
    print("=" * 65)

    tasks = {"math":"数学推理","spatial":"空间推理",
             "planning":"序列规划","logic":"逻辑推理"}
    positive, negative = [], []

    for task, name in tasks.items():
        dft_d = dft[task] - base[task]
        ce_d  = PREVIOUS["v2(CE)"][task] - base[task]
        better = ("DFT更好" if dft_d > ce_d + 0.02 else
                  "CE更好"  if ce_d  > dft_d + 0.02 else "两者相当")
        print(f"  {name}: dft={dft_d:+.3f}  ce={ce_d:+.3f}  [{better}]")
        if dft_d > 0.05:
            positive.append(f"{name}({dft_d:+.3f})")
        elif dft_d < -0.05:
            negative.append(f"{name}({dft_d:+.3f})")

    print()
    if positive:
        print(f"→ ✅ DFT产生正向迁移：{', '.join(positive)}")
        print(f"   改变loss函数改善了OOD泛化，支持DFT论文结论")
    elif not negative:
        print(f"→ DFT与CE相当，均为零迁移")
        print(f"   loss函数改变不足以突破SFT迁移的根本瓶颈")
        print(f"   结合机制分析：根本原因在于MLP未被修改，而非梯度稳定性")
    else:
        print(f"→ DFT出现负向：{', '.join(negative)}")
        print(f"   与DFT论文局限性一致（factual knowledge域DFT可能有损害）")

    save_path = os.path.join(CONFIG["output_dir"], "dft_50_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "dft": dft,
            "all_models": all_models,
            "n_samples": {"math": 50, "spatial": 50, "planning": 50, "logic": 20}
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()