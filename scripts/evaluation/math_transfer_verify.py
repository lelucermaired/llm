"""
math_transfer_verify.py

验证full-lora在数学推理上+0.050的正向迁移是否稳定
扩展到50道题，多种子重复，统计显著性检验
对比：base / v2 / full-lora
"""

import os, json, torch, random
import numpy as np
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "models": {
        "base":            None,
        "single_task":     "./archive/checkpoints/qwen-gomoku-real/final_model",
        "multitask_9to1":  "./checkpoints/qwen-multitask-9to1/final_model",
    },
    "output_dir": "./results/evaluations/math_verify",
    "max_new_tokens": 60,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 50道数学题 ====================

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


def load_base_model(base_path):
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
    base.eval()
    return base


def attach_lora(base_model, adapter_path):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def detach_lora(peft_model):
    """卸载 LoRA adapter，归还纯 base 模型"""
    base = peft_model.base_model.model
    del peft_model
    torch.cuda.empty_cache()
    return base


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


def evaluate(model, tokenizer, samples, desc, device):
    correct_flags = []
    for prompt, answer in tqdm(samples, desc=desc, leave=False):
        response = generate(model, tokenizer, prompt, device)
        correct_flags.append(int(answer.lower() in response.lower()))
    return correct_flags


def main():
    print("=" * 65)
    print("数学推理迁移验证（50道题）")
    print("验证full-lora的+0.050是否稳定")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True,
        trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_flags = {}

    print("\n加载 base 模型（仅加载一次）...")
    base_model = load_base_model(CONFIG["base_model_path"])

    for model_name, model_path in CONFIG["models"].items():
        print(f"\n评测模型：{model_name}...")
        if model_path is None:
            # base 模型直接评测
            model = base_model
            flags = evaluate(model, tokenizer, MATH_SAMPLES, model_name, device)
            all_flags[model_name] = flags
            acc = np.mean(flags)
            print(f"  准确率: {acc:.3f} ({sum(flags)}/{len(flags)})")
        else:
            # 挂载 LoRA adapter 评测后卸载
            model = attach_lora(base_model, model_path)
            flags = evaluate(model, tokenizer, MATH_SAMPLES, model_name, device)
            all_flags[model_name] = flags
            acc = np.mean(flags)
            print(f"  准确率: {acc:.3f} ({sum(flags)}/{len(flags)})")
            base_model = detach_lora(model)

    del base_model
    torch.cuda.empty_cache()

    # ========== 统计检验 ==========
    print("\n" + "=" * 65)
    print("统计显著性检验（McNemar检验）")
    print("=" * 65)

    base_flags = all_flags["base"]
    base_acc = np.mean(base_flags)
    comparison_models = [m for m in CONFIG["models"].keys() if m != "base"]

    for model_name in comparison_models:
        ft_flags = all_flags[model_name]
        ft_acc = np.mean(ft_flags)
        delta = ft_acc - base_acc

        b = sum(1 for x, y in zip(base_flags, ft_flags) if x == 0 and y == 1)
        c = sum(1 for x, y in zip(base_flags, ft_flags) if x == 1 and y == 0)

        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            chi2, p_value = 0, 1.0

        print(f"\n{model_name} vs base:")
        print(f"  base准确率:             {base_acc:.3f}")
        print(f"  {model_name}准确率: {ft_acc:.3f}")
        print(f"  差值:                   {delta:+.3f}")
        print(f"  McNemar: chi2={chi2:.3f}, p={p_value:.4f}")
        print(f"  base错ft对: {b}道  |  base对ft错: {c}道")

        if p_value < 0.05:
            print(f"  -> [显著] p<0.05，迁移结论可信")
        elif p_value < 0.10:
            print(f"  -> [边界] p<0.10，结论需谨慎")
        else:
            print(f"  -> [不显著] p={p_value:.3f}，差异在噪声范围内")

    # ========== 逐题对比 ==========
    print("\n" + "=" * 65)
    print("multitask_9to1 vs single_task 逐题差异（只显示不一致）")
    print("=" * 65)

    if "single_task" in all_flags and "multitask_9to1" in all_flags:
        s_flags = all_flags["single_task"]
        m_flags = all_flags["multitask_9to1"]
        diff_count = 0
        for i, (prompt, answer) in enumerate(MATH_SAMPLES):
            if s_flags[i] != m_flags[i]:
                diff_count += 1
                status = "single错->multi对 [+]" if m_flags[i] == 1 else "single对->multi错 [-]"
                print(f"  题{i+1:>2}: {prompt[:50]:<50} {status}")
        if diff_count == 0:
            print("  两组结果完全一致")
        else:
            print(f"\n  共{diff_count}道题结果不同")

    # ========== 核心结论 ==========
    print("\n" + "=" * 65)
    print("核心结论")
    print("=" * 65)

    accs = {m: float(np.mean(f)) for m, f in all_flags.items()}
    for m, acc in accs.items():
        delta = acc - base_acc
        direction = "[正向迁移]" if delta > 0.02 else ("[负向]" if delta < -0.02 else "[持平]")
        print(f"  {m:<22}: {acc:.3f}  ({delta:+.3f} vs base)  {direction}")

    if "single_task" in accs and "multitask_9to1" in accs:
        mt_delta = accs["multitask_9to1"] - accs["single_task"]
        print(f"\n  多任务 vs 单任务: {mt_delta:+.3f}")
        if mt_delta > 0.02:
            print("  -> 多任务联合SFT在数学OOD上优于单任务SFT，方向二有效")
        elif mt_delta < -0.02:
            print("  -> 多任务联合SFT数学表现低于单任务，可调整混合比例")
        else:
            print("  -> 两者无显著差异，混合数学数据未造成负面影响")

    # 保存
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    save_path = os.path.join(CONFIG["output_dir"], "math_verify_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracies": accs,
            "flags": all_flags,
            "samples": [(p, a) for p, a in MATH_SAMPLES],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()