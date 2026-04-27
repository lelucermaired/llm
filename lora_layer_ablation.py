"""
LoRA Layer-wise Ablation
依次屏蔽不同层的adapter，观察BBH分数变化
定位哪些层真正负责推理迁移

实验组：
  - full          : 所有层（baseline）
  - early_only    : 只保留 0–13
  - middle_only   : 只保留 14–22
  - late_only     : 只保留 23–27
  - no_middle     : 去掉 14–22，保留其余
  - no_early      : 去掉 0–13，保留其余
  - no_late       : 去掉 23–27，保留其余

评测任务：object_counting（正迁移）+ tracking_shuffled_objects（负迁移）
"""

import os
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── 路径配置 ──────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/llm-project/checkpoints/qwen-gomoku-maxlora/final_model"
BBH_DIR = "/root/autodl-tmp/llm-project/datasets/bbh"  # 每个子任务一个json
OUTPUT_DIR = "/root/autodl-tmp/llm-project/results/layer_ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 目标任务 ──────────────────────────────────────────────
TARGET_TASKS = [
    "object_counting",
    "tracking_shuffled_objects_three_objects",  # 确认文件名
]

# ── 层分组 ────────────────────────────────────────────────
NUM_LAYERS = 28
GROUPS = {
    "full": list(range(0, 28)),  # 所有层
    "early_only": list(range(0, 14)),  # 0–13
    "middle_only": list(range(14, 23)),  # 14–22
    "late_only": list(range(23, 28)),  # 23–27
    "no_middle": list(range(0, 14)) + list(range(23, 28)),
    "no_early": list(range(14, 28)),
    "no_late": list(range(0, 23)),
}

# ── LoRA模块列表 ──────────────────────────────────────────
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


# ── 工具：屏蔽指定层之外的adapter ─────────────────────────
def get_all_lora_params(model):
    """返回所有LoRA参数及其原始值的快照"""
    snapshot = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            snapshot[name] = param.data.clone()
    return snapshot


def apply_layer_mask(model, keep_layers: list):
    """
    将不在keep_layers中的层的lora_A/lora_B全部置零
    返回被置零的参数名列表（用于恢复）
    """
    zeroed = []
    keep_set = set(keep_layers)

    for name, param in model.named_parameters():
        if ("lora_A" not in name and "lora_B" not in name):
            continue
        # 解析层号
        if "layers." not in name:
            continue
        layer_num = int(name.split("layers.")[1].split(".")[0])
        if layer_num not in keep_set:
            param.data.zero_()
            zeroed.append(name)

    return zeroed


def restore_params(model, snapshot):
    """用快照恢复所有LoRA参数"""
    for name, param in model.named_parameters():
        if name in snapshot:
            param.data.copy_(snapshot[name])


# ── BBH评测 ───────────────────────────────────────────────
PROMPT_TEMPLATE = """\
{input}

Think step by step and give your final answer after "the answer is".
"""


def load_bbh_task(task_name):
    """尝试几种常见路径格式"""
    candidates = [
        os.path.join(BBH_DIR, task_name, "val_data.json"),
        os.path.join(BBH_DIR, task_name, "test.json"),
        os.path.join(BBH_DIR, f"{task_name}.json"),
        os.path.join(BBH_DIR, "data", f"{task_name}.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # 统一格式
            if isinstance(data, dict) and "examples" in data:
                return data["examples"]
            elif isinstance(data, list):
                return data
    raise FileNotFoundError(f"找不到 {task_name}，尝试过：\n" + "\n".join(candidates))


def extract_answer(text):
    """从模型输出中提取最终答案"""
    # 匹配 "the answer is X" 或 "答案是X"
    patterns = [
        r"the answer is[:\s]+([^\n\.]+)",
        r"答案[是为][:\s]+([^\n\.]+)",
        r"answer[:\s]+([^\n\.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")
    # 退而求其次：取最后一行非空内容
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def normalize(s):
    return s.lower().strip().rstrip(".")


def evaluate_task(model, tokenizer, examples, max_samples=50, max_new_tokens=256):
    model.eval()
    correct = 0
    total = min(len(examples), max_samples)

    for ex in tqdm(examples[:total], leave=False):
        question = ex.get("input", ex.get("question", ""))
        gold = normalize(str(ex.get("target", ex.get("answer", ""))))

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
            skip_special_tokens=True
        )
        pred = normalize(extract_answer(generated))

        if pred == gold or pred in gold or gold in pred:
            correct += 1

    return correct / total


# ── 主流程 ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  LoRA Layer-wise Ablation")
    print("  Tasks: object_counting / tracking_shuffled_objects")
    print("=" * 60)

    # 1. 加载模型
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

    # 2. 快照全部LoRA参数
    print("[准备] 保存adapter权重快照...")
    snapshot = get_all_lora_params(model)
    print(f"  共 {len(snapshot)} 个LoRA参数张量")

    # 3. 加载BBH数据
    print("\n[数据] 加载BBH任务...")
    task_data = {}
    for task in TARGET_TASKS:
        try:
            examples = load_bbh_task(task)
            task_data[task] = examples
            print(f"  {task}: {len(examples)} 条")
        except FileNotFoundError as e:
            print(f"  [警告] {e}")

    if not task_data:
        print("[错误] 没有找到任何BBH数据，请检查 BBH_DIR 路径")
        return

    # 4. 逐组消融
    results = {task: {} for task in task_data}

    for group_name, keep_layers in GROUPS.items():
        print(f"\n── 实验组: {group_name} (保留层: {keep_layers[:3]}{'...' if len(keep_layers) > 3 else ''}) ──")

        # 应用掩码
        restore_params(model, snapshot)  # 先恢复
        zeroed = apply_layer_mask(model, keep_layers)
        print(f"  置零了 {len(zeroed)} 个参数张量")

        # 评测每个任务
        for task, examples in task_data.items():
            print(f"  评测 {task}...", end="", flush=True)
            acc = evaluate_task(model, tokenizer, examples, max_samples=50)
            results[task][group_name] = acc
            print(f" → {acc:.1%}")

    # 恢复完整adapter
    restore_params(model, snapshot)

    # 5. 保存结果
    out_json = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[保存] {out_json}")

    # 6. 打印表格
    print("\n── 结果汇总 ─────────────────────────────────────────")
    header = f"{'Group':<16}" + "".join(f"{t[:20]:<22}" for t in task_data)
    print(header)
    print("-" * len(header))
    for group in GROUPS:
        row = f"{group:<16}"
        for task in task_data:
            val = results[task].get(group, float("nan"))
            row += f"{val:.1%}{'':14}"
        print(row)

    # 7. 画图
    plot_results(results)


def plot_results(results):
    tasks = list(results.keys())
    groups = list(GROUPS.keys())
    n_tasks = len(tasks)

    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 5), sharey=False)
    if n_tasks == 1:
        axes = [axes]

    colors = {
        "full": "#2c7bb6",
        "early_only": "#abd9e9",
        "middle_only": "#d7191c",
        "late_only": "#fdae61",
        "no_middle": "#984ea3",
        "no_early": "#4daf4a",
        "no_late": "#ff7f00",
    }

    for ax, task in zip(axes, tasks):
        vals = [results[task].get(g, 0) * 100 for g in groups]
        bars = ax.bar(groups, vals,
                      color=[colors.get(g, "#999") for g in groups],
                      edgecolor="white", linewidth=0.8, alpha=0.9)

        # full基线虚线
        full_val = results[task].get("full", 0) * 100
        ax.axhline(full_val, color="#2c7bb6", linewidth=1.5,
                   linestyle="--", alpha=0.6, label=f"full={full_val:.1f}%")

        # 标注数值
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_title(task.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.15 + 5)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("LoRA Layer Ablation: Which Layers Drive Transfer?\n"
                 "(Qwen2.5-7B, r=64, Gomoku → BBH)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "layer_ablation_plot.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out_path}")
    plt.close()


if __name__ == "__main__":
    main()