"""
Base Model BBH Evaluation
不加载任何LoRA，直接评测base model
用于和层消融结果对照
"""

import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ── 配置 ──────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_CACHE   = "/root/autodl-tmp/hf_cache"
OUTPUT_DIR = "/root/autodl-tmp/llm-project/results/layer_ablation"
MAX_SAMPLES = 50

TARGET_TASKS = [
    "object_counting",
    "tracking_shuffled_objects_three_objects",
]

PROMPT_TEMPLATE = """\
{input}

Think step by step and give your final answer after "the answer is".
"""

# ── 工具函数 ──────────────────────────────────────────────
def extract_answer(text):
    patterns = [
        r"the answer is[:\s]+([^\n\.]+)",
        r"答案[是为][:\s]+([^\n\.]+)",
        r"answer[:\s]+([^\n\.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""

def normalize(s):
    return s.lower().strip().rstrip(".")

def load_bbh_task(task_name):
    ds = load_dataset(
        "lukaemon/bbh",
        task_name,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
    return [{"input": row["input"], "target": row["target"]} for row in split]

def evaluate(model, tokenizer, examples, max_samples=50):
    model.eval()
    correct = 0
    total = min(len(examples), max_samples)
    wrong_cases = []

    for ex in tqdm(examples[:total], leave=False):
        question = ex.get("input", "")
        gold = normalize(str(ex.get("target", "")))

        prompt = PROMPT_TEMPLATE.format(input=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        pred = normalize(extract_answer(generated))

        if pred == gold or pred in gold or gold in pred:
            correct += 1
        else:
            wrong_cases.append({"gold": gold, "pred": pred})

    acc = correct / total
    return acc, wrong_cases

# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Base Model BBH Evaluation (no LoRA)")
    print(f"  Model: {BASE_MODEL}")
    print("=" * 55)

    print("\n[加载] base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("[加载] 完成（无LoRA）")

    results = {}

    for task in TARGET_TASKS:
        print(f"\n[评测] {task}...")
        examples = load_bbh_task(task)
        print(f"  共 {len(examples)} 条，取前 {MAX_SAMPLES} 条")

        acc, wrong = evaluate(model, tokenizer, examples, MAX_SAMPLES)
        results[task] = {
            "accuracy": acc,
            "correct": int(acc * MAX_SAMPLES),
            "total": MAX_SAMPLES,
        }
        print(f"  → 准确率: {acc:.1%}  ({int(acc*MAX_SAMPLES)}/{MAX_SAMPLES})")

    # 保存
    out_path = os.path.join(OUTPUT_DIR, "base_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[保存] {out_path}")

    # 汇总对比
    print("\n── 对照表（与层消融结果对比）──────────────────────")
    ablation_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation = json.load(f)

        header = f"{'组别':<16}" + "".join(f"{t[:22]:<24}" for t in TARGET_TASKS)
        print(header)
        print("-" * len(header))

        # base行
        row = f"{'base (no LoRA)':<16}"
        for task in TARGET_TASKS:
            row += f"{results[task]['accuracy']:.1%}{'':16}"
        print(row)

        # 消融各组
        groups = ["full", "early_only", "middle_only", "late_only",
                  "no_middle", "no_early", "no_late"]
        for g in groups:
            row = f"{g:<16}"
            for task in TARGET_TASKS:
                val = ablation.get(task, {}).get(g, None)
                if val is not None:
                    # 计算相对base的delta
                    delta = val - results[task]["accuracy"]
                    sign = "+" if delta >= 0 else ""
                    row += f"{val:.1%}({sign}{delta:.1%}){'':6}"
                else:
                    row += f"{'N/A':<24}"
            print(row)
    else:
        print("（找不到ablation_results.json，仅显示base结果）")

    print("\n✓ 完成！")

if __name__ == "__main__":
    main()