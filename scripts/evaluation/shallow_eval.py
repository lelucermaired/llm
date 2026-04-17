"""
shallow_eval.py
只评测shallow模型，结果与已有数据对比
base: math=0.750, spatial=0.500, planning=0.550, logic=1.000
v2:   math=0.750, spatial=0.500, planning=0.500, logic=1.000
r64:  math=0.750, spatial=0.500, planning=0.550, logic=1.000
full: math=0.800, spatial=0.450, planning=0.550, logic=1.000
"""

import os, json, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "model_path": "./checkpoints/qwen-gomoku-shallow/final_model",
    "output_dir": "./shallow_eval_results",
    "max_new_tokens": 80,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

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
]

SPATIAL_SAMPLES = [
    ("A is to the left of B. B is above C. What is the relation of A to C?", "upper-left"),
    ("A is above B. B is to the right of C. What is the relation of A to C?", "upper-right"),
    ("A is to the left of B. B is to the left of C. What is the relation of A to C?", "left"),
    ("A is above B. B is above C. What is the relation of A to C?", "above"),
    ("A is to the right of B. B is below C. What is the relation of A to C?", "lower-right"),
    ("A is below B. B is to the left of C. What is the relation of A to C?", "lower-left"),
    ("A is to the left of B. B is below C. What is the relation of A to C?", "lower-left"),
    ("A is above B. B is to the left of C. C is below D. What is A relative to D?", "upper-left"),
    ("A is to the right of B. B is above C. C is to the left of D. What is A relative to D?", "above"),
    ("A left of B. B left of C. C left of D. D left of E. What is A relative to E?", "left"),
    ("A above B. B above C. C above D. What is A relative to D?", "above"),
    ("P above Q. Q left of R. R below S. What is P relative to S?", "upper-left"),
    ("X is north of Y. Y is east of Z. Where is X relative to Z?", "upper-right"),
    ("A right of B. B right of C. C above D. D right of E. What is A relative to E?", "upper-right"),
    ("A above B. B right of C. C above D. D left of E. What is A relative to E?", "above"),
    ("A left of B. B above C. C right of D. D below E. E left of F. What is A relative to F?", "left"),
    ("Start facing East. Turn left. Turn left again. Which direction now?", "west"),
    ("Start facing North. Turn right three times. Which direction?", "west"),
    ("A is to the left of B. B is to the left of C. C is above D. What is A relative to D?", "upper-left"),
    ("A is upper-left of B. B is lower-right of C. What is A relative to C?", "overlap"),
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
]

PREVIOUS_RESULTS = {
    "base":      {"math": 0.750, "spatial": 0.500, "planning": 0.550, "logic": 1.000},
    "v2":        {"math": 0.750, "spatial": 0.500, "planning": 0.500, "logic": 1.000},
    "r64":       {"math": 0.750, "spatial": 0.500, "planning": 0.550, "logic": 1.000},
    "full_lora": {"math": 0.800, "spatial": 0.450, "planning": 0.550, "logic": 1.000},
}


def load_model(path, base_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path, quantization_config=bnb_config,
        device_map="auto", local_files_only=True,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    return model


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
        response = generate(model, tokenizer, prompt, device)
        correct += int(answer.lower() in response.lower())
    return correct / len(samples)


def main():
    print("=" * 65)
    print("Shallow LoRA 评测（前10层 q+v，r=8）")
    print("loss=2.089（vs v2 loss=1.446）")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True,
        trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n加载shallow模型...")
    model = load_model(CONFIG["model_path"], CONFIG["base_model_path"])

    math_acc    = evaluate_task(model, tokenizer, MATH_SAMPLES,     "数学推理", device)
    spatial_acc = evaluate_task(model, tokenizer, SPATIAL_SAMPLES,  "空间推理", device)
    plan_acc    = evaluate_task(model, tokenizer, PLANNING_SAMPLES,  "序列规划", device)
    logic_acc   = evaluate_task(model, tokenizer, LOGIC_SAMPLES,    "逻辑推理", device)

    shallow_results = {
        "math": math_acc, "spatial": spatial_acc,
        "planning": plan_acc, "logic": logic_acc,
    }

    del model
    torch.cuda.empty_cache()

    # ========== 汇总对比 ==========
    base = PREVIOUS_RESULTS["base"]
    all_models = {**PREVIOUS_RESULTS, "shallow": shallow_results}

    print("\n" + "=" * 75)
    print("完整对比（含shallow）")
    print("=" * 75)
    print(f"\n{'模型':<12} {'数学':>8} {'空间':>8} {'规划':>8} {'逻辑':>8}  配置说明")
    print("-" * 70)

    configs = {
        "base":      "基础模型",
        "v2":        "全28层 q+v r=8",
        "r64":       "全28层 q+v r=64",
        "full_lora": "全28层 全模块 r=8",
        "shallow":   "前10层 q+v r=8  ← 本次",
    }

    for name, results in all_models.items():
        delta_str = ""
        if name != "base":
            deltas = [results[t] - base[t] for t in ["math","spatial","planning","logic"]]
            delta_str = f"  ({'+' if deltas[0]>=0 else ''}{deltas[0]:.2f} {'+' if deltas[1]>=0 else ''}{deltas[1]:.2f} {'+' if deltas[2]>=0 else ''}{deltas[2]:.2f} {'+' if deltas[3]>=0 else ''}{deltas[3]:.2f})"
        print(f"{name:<12} {results['math']:>8.3f} {results['spatial']:>8.3f} "
              f"{results['planning']:>8.3f} {results['logic']:>8.3f}  "
              f"{configs[name]}{delta_str}")

    # ========== 核心结论 ==========
    print("\n" + "=" * 65)
    print("核心结论")
    print("=" * 65)

    tasks = {"math": "数学推理", "spatial": "空间推理",
             "planning": "序列规划", "logic": "逻辑推理"}

    found = False
    for task, name in tasks.items():
        delta = shallow_results[task] - base[task]
        delta_v2 = PREVIOUS_RESULTS["v2"][task] - base[task]
        if delta > 0.05:
            found = True
            print(f"\n→ ✅ {name}出现正向迁移（{delta:+.3f}）")
            print(f"  shallow: {shallow_results[task]:.3f}  "
                  f"v2: {PREVIOUS_RESULTS['v2'][task]:.3f}  "
                  f"base: {base[task]:.3f}")
            print(f"  浅层LoRA在{name}上优于全层LoRA（{delta_v2:+.3f}）")
        elif delta < -0.05:
            print(f"\n→ ❌ {name}出现负向迁移（{delta:+.3f}）")
            print(f"  浅层LoRA对{name}有轻微损害")

    if not found:
        print("\n→ 浅层LoRA（前10层）仍为零迁移")
        print("\n  至此，已系统测试了所有主要变量：")
        print("  ┌─────────────────────────────────────────────┐")
        print("  │ 变量          │ 结论                         │")
        print("  ├─────────────────────────────────────────────┤")
        print("  │ LoRA rank     │ r=8 vs r=64 → 均零迁移       │")
        print("  │ 目标模块      │ q+v vs 全模块 → 均零迁移      │")
        print("  │ 更新层位置    │ 全层 vs 浅层 → 均零迁移       │")
        print("  │ 目标任务      │ 数学/空间/规划/逻辑 → 均零迁移 │")
        print("  │ 数据来源      │ 合成/真实/混合 → 均零迁移      │")
        print("  └─────────────────────────────────────────────┘")
        print("\n  结论：零迁移与LoRA配置无关，根本原因在于")
        print("  五子棋任务本身与通用推理任务缺乏足够的表示空间共性")
        print("  要产生迁移，需要换源任务（路线B：数学微调→逻辑推理）")

    # 保存
    save_path = os.path.join(CONFIG["output_dir"], "shallow_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"shallow": shallow_results,
                   "all_models": all_models}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()