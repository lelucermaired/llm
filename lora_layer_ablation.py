"""
LoRA Layer-wise Ablation
依次屏蔽不同层的adapter，观察BBH分数变化
定位哪些层真正负责推理迁移

实验组：
  - full          : 所有层（baseline）
  - early_only    : 只保留 0-13
  - middle_only   : 只保留 14-22
  - late_only     : 只保留 23-27
  - no_middle     : 去掉 14-22，保留其余
  - no_early      : 去掉 0-13，保留其余
  - no_late       : 去掉 23-27，保留其余

评测任务：object_counting（正迁移）+ tracking_shuffled_objects（负迁移）
数据集：lukaemon/bbh
"""

import os
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# ── 路径配置 ──────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH  = "/root/autodl-tmp/llm-project/checkpoints/qwen-gomoku-maxlora/final_model"
OUTPUT_DIR = "/root/autodl-tmp/llm-project/results/layer_ablation"
HF_CACHE   = "/root/autodl-tmp/hf_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 目标任务（lukaemon/bbh子任务名）──────────────────────
TARGET_TASKS = [
    "object_counting",
    "tracking_shuffled_objects_three_objects",
]

# ── 层分组 ────────────────────────────────────────────────
GROUPS = {
    "full"       : list(range(0, 28)),
    "early_only" : list(range(0, 14)),
    "middle_only": list(range(14, 23)),
    "late_only"  : list(range(23, 28)),
    "no_middle"  : list(range(0, 14)) + list(range(23, 28)),
    "no_early"   : list(range(14, 28)),
    "no_late"    : list(range(0, 23)),
}

# ── Prompt ────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
{input}

Think step by step and give your final answer after "the answer is".
"""

# ── 工具：参数快照 / 置零 / 恢复 ─────────────────────────
def get_snapshot(model):
    snap = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            snap[name] = param.data.clone()
    return snap

def apply_mask(model, keep_layers):
    keep_set = set(keep_layers)
    zeroed = []
    for name, param in model.named_parameters():
        if ("lora_A" not in name and "lora_B" not in name):
            continue
        if "layers." not in name:
            continue
        layer_num = int(name.split("layers.")[1].split(".")[0])
        if layer_num not in keep_set:
            param.data.zero_()
            zeroed.append(name)
    return zeroed

def restore(model, snap):
    for name, param in model.named_parameters():
        if name in snap:
            param.data.copy_(snap[name])

# ── 数据加载 ──────────────────────────────────────────────
def load_bbh_task(task_name):
    ds = load_dataset(
        "lukaemon/bbh",
        task_name,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
    return [{"input": row["input"], "target": row["target"]} for row in split]

# ── 答案提取 ──────────────────────────────────────────────
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

# ── 评测 ──────────────────────────────────────────────────
def evaluate_task(model, tokenizer, examples, max_samples=50, max_new_tokens=256):
    model.eval()
    correct = 0
    total = min(len(examples), max_samples)

    for ex in tqdm(examples[:total], leave=False):
        question = ex.get("input", "")
        gold = normalize(str(ex.get("target", "")))

        prompt = PROMPT_TEMPLATE.format(input=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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

    return correct / total

# ── 绘图 ──────────────────────────────────────────────────
def plot_results(results):
    tasks = list(results.keys())
    groups = list(GROUPS.keys())
    n_tasks = len(tasks)

    colors = {
        "full"       : "#2c7bb6",
        "early_only" : "#abd9e9",
        "middle_only": "#d7191c",
        "late_only"  : "#fdae61",
        "no_middle"  : "#984ea3",
        "no_early"   : "#4daf4a",
        "no_late"    : "#ff7f00",
    }

    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        vals = [results[task].get(g, 0) * 100 for g in groups]
        bars = ax.bar(
            groups, vals,
            color=[colors.get(g, "#999") for g in groups],
            edgecolor="white", linewidth=0.8, alpha=0.9,
        )
        full_val = results[task].get("full", 0) * 100
        ax.axhline(full_val, color="#2c7bb6", linewidth=1.5,
                   linestyle="--", alpha=0.6, label=f"full={full_val:.1f}%")

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
            )

        ax.set_title(task.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.15 + 5)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "LoRA Layer Ablation: Which Layers Drive Transfer?\n"
        "(Qwen2.5-7B, r=64, Gomoku → BBH)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "layer_ablation_plot.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out_path}")
    plt.close()

# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  LoRA Layer-wise Ablation")
    print("  Dataset: lukaemon/bbh")
    print("=" * 60)

    # 加载模型
    print("\n[加载] base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("[加载] LoRA adapter...")
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    # 快照
    print("[准备] 保存LoRA参数快照...")
    snap = get_snapshot(model)
    print(f"  共 {len(snap)} 个LoRA参数张量")

    # 加载数据
    print("\n[数据] 从 lukaemon/bbh 加载...")
    task_data = {}
    for task in TARGET_TASKS:
        try:
            examples = load_bbh_task(task)
            task_data[task] = examples
            print(f"  {task}: {len(examples)} 条")
        except Exception as e:
            print(f"  [警告] {task} 加载失败: {e}")

    if not task_data:
        print("[错误] 没有加载到任何数据")
        return

    # 逐组消融
    results = {task: {} for task in task_data}

    for group_name, keep_layers in GROUPS.items():
        print(f"\n── {group_name} (保留层: {keep_layers[:4]}{'...' if len(keep_layers)>4 else ''}) ──")
        restore(model, snap)
        zeroed = apply_mask(model, keep_layers)
        print(f"  置零 {len(zeroed)} 个张量")

        for task, examples in task_data.items():
            print(f"  评测 {task}...", end="", flush=True)
            acc = evaluate_task(model, tokenizer, examples, max_samples=50)
            results[task][group_name] = acc
            print(f" {acc:.1%}")

    restore(model, snap)

    # 保存结果
    out_json = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[保存] {out_json}")

    # 打印表格
    print("\n── 结果汇总 ─────────────────────────────────────────")
    print(f"{'Group':<16}" + "".join(f"{t[:24]:<26}" for t in task_data))
    print("-" * (16 + 26 * len(task_data)))
    for group in GROUPS:
        row = f"{group:<16}"
        for task in task_data:
            row += f"{results[task].get(group, 0):.1%}{'':18}"
        print(row)

    plot_results(results)
    print("\n✓ 完成！")

if __name__ == "__main__":
    main()