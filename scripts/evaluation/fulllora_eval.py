"""
fulllora_eval.py

评测full-lora（包含MLP模块）在所有任务上的表现
对比：base / v2（q+v） / r64（q+v r=64） / full-lora（全模块含MLP）

核心假设：MLP是迁移瓶颈，加入MLP微调后应出现迁移
"""

import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "models": {
        "base":      None,
        "v2":        "./checkpoints/qwen-gomoku-v2/final_model",
        "r64":       "./checkpoints/qwen-gomoku-r64/final_model",
        "full_lora": "./checkpoints/qwen-gomoku-full-lora/final_model",
    },
    "output_dir": "./fulllora_eval_results",
    "max_new_tokens": 80,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 评测样本 ====================

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
    ("A is upper-left of B. B is lower-right of C. What is A relative to C?", "overlap"),
    ("P above Q. Q left of R. R below S. What is P relative to S?", "upper-left"),
    ("X is north of Y. Y is east of Z. Where is X relative to Z?", "upper-right"),
    ("A right of B. B right of C. C above D. D right of E. What is A relative to E?", "upper-right"),
    ("A above B. B right of C. C above D. D left of E. What is A relative to E?", "above"),
    ("A left of B. B above C. C right of D. D below E. E left of F. What is A relative to F?", "left"),
    ("Start facing East. Turn left. Turn left again. Which direction now?", "west"),
    ("Start facing North. Turn right three times. Which direction?", "west"),
    ("A is to the left of B. B is to the left of C. C is above D. What is A relative to D?", "upper-left"),
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


# ==================== 评测函数 ====================

def load_model(path, base_path, is_base=False):
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
    if is_base:
        base.eval()
        return base
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


def check(response, answer):
    return answer.lower() in response.lower()


def evaluate_task(model, tokenizer, samples, desc, device):
    correct = 0
    for prompt, answer in tqdm(samples, desc=desc, leave=False):
        response = generate(model, tokenizer, prompt, device)
        correct += int(check(response, answer))
    return correct / len(samples)


def main():
    print("=" * 65)
    print("full-lora完整评测")
    print("验证：加入MLP模块后是否产生推理迁移？")
    print("对比：base / v2(q+v) / r64(q+v,r64) / full-lora(全模块)")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True,
        trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}

    for model_name, model_path in CONFIG["models"].items():
        print(f"\n{'='*65}")
        print(f"评测模型：{model_name}")
        print(f"{'='*65}")

        model = load_model(model_path, CONFIG["base_model_path"],
                          is_base=(model_path is None))

        math_acc  = evaluate_task(model, tokenizer, MATH_SAMPLES,
                                  f"{model_name}×数学", device)
        spatial_acc = evaluate_task(model, tokenizer, SPATIAL_SAMPLES,
                                    f"{model_name}×空间", device)
        plan_acc  = evaluate_task(model, tokenizer, PLANNING_SAMPLES,
                                  f"{model_name}×规划", device)
        logic_acc = evaluate_task(model, tokenizer, LOGIC_SAMPLES,
                                  f"{model_name}×逻辑", device)

        all_results[model_name] = {
            "math": math_acc,
            "spatial": spatial_acc,
            "planning": plan_acc,
            "logic": logic_acc,
        }

        print(f"  数学:     {math_acc:.3f}")
        print(f"  空间推理: {spatial_acc:.3f}")
        print(f"  序列规划: {plan_acc:.3f}")
        print(f"  逻辑推理: {logic_acc:.3f}")

        del model
        torch.cuda.empty_cache()

    # ========== 汇总 ==========
    print("\n" + "=" * 70)
    print("汇总对比（括号内为与基础模型的差值）")
    print("=" * 70)

    base = all_results["base"]
    tasks = ["math", "spatial", "planning", "logic"]
    task_names = {"math": "数学推理", "spatial": "空间推理",
                  "planning": "序列规划", "logic": "逻辑推理"}

    print(f"\n{'任务':<12}", end="")
    for m in CONFIG["models"]:
        print(f"  {m:>12}", end="")
    print()
    print("-" * 70)

    for task in tasks:
        print(f"{task_names[task]:<12}", end="")
        for model_name in CONFIG["models"]:
            acc = all_results[model_name][task]
            if model_name == "base":
                print(f"  {acc:>12.3f}", end="")
            else:
                delta = acc - base[task]
                sign = "+" if delta >= 0 else ""
                print(f"  {acc:.3f}({sign}{delta:.3f})", end="")
        print()

    # ========== 核心结论 ==========
    print("\n" + "=" * 65)
    print("核心结论：MLP微调是否打破零迁移？")
    print("=" * 65)

    full_results = all_results["full_lora"]
    v2_results = all_results["v2"]
    found_transfer = False

    for task in tasks:
        delta_full = full_results[task] - base[task]
        delta_v2 = v2_results[task] - base[task]
        if delta_full > 0.05:
            found_transfer = True
            print(f"\n→ ✅ {task_names[task]}出现正向迁移！")
            print(f"  full-lora: {full_results[task]:.3f} "
                  f"（+{delta_full:.3f} vs 基础模型）")
            print(f"  v2(q+v):   {v2_results[task]:.3f} "
                  f"（{delta_v2:+.3f} vs 基础模型）")
            print(f"  MLP模块是该任务迁移的关键条件")

    if not found_transfer:
        print("\n→ full-lora在所有任务上仍为零迁移")
        print("\n  这是一个重要的负面结论：")
        print("  即使同时微调attention+MLP全模块，")
        print("  五子棋任务仍无法向数学/空间/规划/逻辑任务产生迁移")
        print("\n  结合残差流分析（MLP/Attn≈2x）的发现，")
        print("  可以得出更强的结论：")
        print("  零迁移不是因为LoRA覆盖模块不足，")
        print("  而是因为五子棋任务本身与这些目标任务")
        print("  在语义表示层面没有足够的结构共性")
        print("  LoRA无论覆盖哪些模块，都无法建立跨域的推理桥梁")
    else:
        print(f"\n→ 部分任务出现迁移，验证了MLP是迁移瓶颈的假设")

    # 保存
    save_path = os.path.join(CONFIG["output_dir"], "fulllora_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()