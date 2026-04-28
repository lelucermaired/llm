"""
Multi-task Attention Pattern Analysis
比较5个正向迁移任务 vs 1个负迁移任务的attention entropy模式
看浅层entropy下降是否是正向迁移任务的共同特征

任务：
  正向：object_counting(+24%), logical_deduction(+15%),
        geometric_shapes(+7%), temporal_sequences(+6%), multistep_arithmetic(+5%)
  负向：tracking_shuffled_objects_three_objects(-22%)
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import gc

# ── 配置 ──────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH  = "/root/autodl-tmp/llm-project/checkpoints/qwen-gomoku-maxlora/final_model"
HF_CACHE   = "/root/autodl-tmp/hf_cache"
OUTPUT_DIR = "/root/autodl-tmp/llm-project/results/attention_multitask"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 10

# 正向任务 + 负向任务
TASKS = {
    "object_counting"                         : ("正向", +24, "#1f77b4"),
    "logical_deduction_three_objects"         : ("正向", +15, "#2ca02c"),
    "geometric_shapes"                        : ("正向", +7,  "#9467bd"),
    "temporal_sequences"                      : ("正向", +6,  "#8c564b"),
    "multistep_arithmetic_two"                : ("正向", +5,  "#17becf"),
    "tracking_shuffled_objects_three_objects" : ("负向", -22, "#d62728"),
}

PROMPT_TEMPLATE = "{input}\n\nThink step by step and give your final answer after \"the answer is\"."

# ── 数据加载 ──────────────────────────────────────────────
def load_task(task_name):
    try:
        ds = load_dataset("lukaemon/bbh", task_name,
                          cache_dir=HF_CACHE, trust_remote_code=True)
        split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
        return [{"input": row["input"], "target": row["target"]} for row in split]
    except Exception as e:
        print(f"  [警告] {task_name} 加载失败: {e}")
        return []

# ── Attention收集 ─────────────────────────────────────────
def collect_entropy_per_layer(model, tokenizer, examples, n=10):
    """
    只收集attention entropy，不存储原始attention矩阵（省内存）
    返回：list of float，每层的平均entropy
    """
    model.eval()
    layer_entropies = {}  # layer_idx -> list of float

    for ex in examples[:n]:
        prompt = PROMPT_TEMPLATE.format(input=ex["input"])
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        if not outputs.attentions:
            print("  [警告] 没有收到attention输出，检查attn_implementation")
            return []

        for layer_idx, attn in enumerate(outputs.attentions):
            # attn: (1, heads, seq, seq)
            attn_f = attn[0].float()  # (heads, seq, seq)
            eps = 1e-12
            ent = -(attn_f * torch.log(attn_f + eps)).sum(dim=-1)  # (heads, seq)
            mean_ent = ent.mean().item()
            if layer_idx not in layer_entropies:
                layer_entropies[layer_idx] = []
            layer_entropies[layer_idx].append(mean_ent)

        # 立刻释放attention（省显存）
        del outputs
        torch.cuda.empty_cache()

    num_layers = len(layer_entropies)
    return [np.mean(layer_entropies[l]) for l in range(num_layers)]

# ── 收集所有任务的entropy ─────────────────────────────────
def collect_all_tasks(model, tokenizer, task_data, model_name):
    results = {}
    for task, examples in task_data.items():
        direction = TASKS[task][0]
        delta     = TASKS[task][1]
        print(f"  [{model_name}] {task} ({direction} {delta:+d}%)...")
        ents = collect_entropy_per_layer(model, tokenizer, examples, NUM_SAMPLES)
        results[task] = ents
        if ents:
            print(f"    → {len(ents)} 层，早层均值={np.mean(ents[:14]):.4f}")
    return results

# ── 绘图1：每任务折线图（base vs lora） ──────────────────
def plot_per_task(base_results, lora_results, num_layers):
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharex=True)
    axes = axes.flatten()
    fig.suptitle(
        "Attention Entropy per Layer: Base vs MaxLoRA\n"
        "Positive Transfer Tasks vs Negative Transfer Task",
        fontsize=14, fontweight="bold",
    )

    x = np.arange(num_layers)
    zone_kw = dict(alpha=0.05)

    for ax, (task, (direction, delta, color)) in zip(axes, TASKS.items()):
        base_ent = base_results.get(task, [])
        lora_ent = lora_results.get(task, [])

        if not base_ent or not lora_ent:
            ax.text(0.5, 0.5, f"{task}\n(数据加载失败)",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        ax.plot(x, base_ent, "o-", color="gray",  label="Base",
                lw=1.5, ms=3, alpha=0.8)
        ax.plot(x, lora_ent, "s-", color=color,   label="MaxLoRA",
                lw=2,   ms=4, alpha=0.9)
        ax.fill_between(x, base_ent, lora_ent,
                        alpha=0.15,
                        color="green" if direction == "正向" else "red")

        ax.axvspan(0,  14, color="blue",  **zone_kw)
        ax.axvspan(14, 23, color="red",   **zone_kw)
        ax.axvspan(23, 28, color="green", **zone_kw)

        # 计算浅层delta
        early_delta = np.mean(lora_ent[:14]) - np.mean(base_ent[:14])
        sign = "↓" if early_delta < 0 else "↑"
        arrow_color = "green" if direction == "正向" else "red"

        label_str = f"{'✓ 正向' if direction=='正向' else '✗ 负向'} {delta:+d}%"
        ax.set_title(
            f"{task.replace('_', ' ')}\n{label_str}  |  浅层Δentropy={early_delta:+.4f}{sign}",
            fontsize=9, color=arrow_color, fontweight="bold",
        )
        ax.set_ylabel("Attention Entropy", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(range(0, num_layers, 4))

    axes[-1].set_xlabel("Layer Index", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Layer Index", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "entropy_per_task.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 绘图2：核心对比图——浅层entropy delta柱状图 ──────────
def plot_early_delta_bar(base_results, lora_results):
    """
    每个任务的浅层（0-13层）entropy变化量
    正向任务应该都是负值（entropy下降，注意力更集中）
    负向任务如果和正向不同，就是规律
    """
    tasks = list(TASKS.keys())
    deltas = []
    colors = []
    labels = []

    for task in tasks:
        base_ent = base_results.get(task, [])
        lora_ent = lora_results.get(task, [])
        if not base_ent or not lora_ent:
            deltas.append(0)
        else:
            d = np.mean(lora_ent[:14]) - np.mean(base_ent[:14])
            deltas.append(d)

        direction = TASKS[task][0]
        color     = TASKS[task][2]
        delta_pct = TASKS[task][1]
        colors.append(color)
        short_name = task.replace("_three_objects", "").replace("_two", "")
        short_name = short_name.replace("_", "\n")
        labels.append(f"{short_name}\n({delta_pct:+d}%)")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(tasks)), deltas, color=colors,
                  alpha=0.85, edgecolor="white", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    ax.axvline(4.5, color="gray", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.text(2,   max(deltas) * 0.8 if max(deltas) > 0 else min(deltas) * 0.8,
            "正向迁移任务", ha="center", fontsize=11, color="steelblue", fontweight="bold")
    ax.text(5.0, max(deltas) * 0.8 if max(deltas) > 0 else min(deltas) * 0.8,
            "负向迁移", ha="center", fontsize=11, color="firebrick", fontweight="bold")

    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.0001 if val >= 0 else -0.0003),
                f"{val:+.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Early Layer (0-13) Entropy Change\n(MaxLoRA - Base)", fontsize=11)
    ax.set_title(
        "Attention Entropy Change in Early Layers: Positive vs Negative Transfer Tasks\n"
        "(Qwen2.5-7B, r=64, Gomoku MaxLoRA)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "early_entropy_delta_bar.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 绘图3：三区域对比热图 ─────────────────────────────────
def plot_zone_heatmap(base_results, lora_results):
    """
    行：6个任务
    列：early / middle / late
    颜色：entropy delta（lora - base）
    """
    zones = {
        "Early\n(0-13)":   list(range(0, 14)),
        "Middle\n(14-22)": list(range(14, 23)),
        "Late\n(23-27)":   list(range(23, 28)),
    }

    tasks = list(TASKS.keys())
    short_labels = []
    for t in tasks:
        direction, delta, _ = TASKS[t]
        s = t.replace("_three_objects","").replace("_two","").replace("_"," ")
        short_labels.append(f"{s}\n({direction} {delta:+d}%)")

    mat = np.zeros((len(tasks), len(zones)))
    for i, task in enumerate(tasks):
        base_ent = base_results.get(task, [])
        lora_ent = lora_results.get(task, [])
        if not base_ent or not lora_ent:
            mat[i, :] = np.nan
            continue
        for j, (zone, layers) in enumerate(zones.items()):
            b = np.mean([base_ent[l] for l in layers if l < len(base_ent)])
            l = np.mean([lora_ent[l] for l in layers if l < len(lora_ent)])
            mat[i, j] = l - b

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = np.nanmax(np.abs(mat))
    sns.heatmap(
        mat, ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        xticklabels=list(zones.keys()),
        yticklabels=short_labels,
        annot=True, fmt=".4f",
        annot_kws={"size": 9},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Entropy Change (MaxLoRA - Base)"},
    )
    ax.set_title(
        "Attention Entropy Change by Zone\n"
        "Blue = more focused (↓entropy), Red = more dispersed (↑entropy)",
        fontsize=11, fontweight="bold",
    )

    # 分隔线：正向 vs 负向任务
    n_positive = sum(1 for t in TASKS if TASKS[t][0] == "正向")
    ax.axhline(n_positive, color="black", linewidth=2, linestyle="--")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "zone_entropy_heatmap.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 打印数值摘要 ──────────────────────────────────────────
def print_summary(base_results, lora_results):
    print("\n── 浅层Entropy变化汇总（核心结论） ─────────────────")
    print(f"{'任务':<45} {'方向':>4} {'迁移幅度':>8} {'浅层Δentropy':>14} {'判断'}")
    print("-" * 80)

    positive_deltas = []
    negative_deltas = []

    for task, (direction, delta, _) in TASKS.items():
        base_ent = base_results.get(task, [])
        lora_ent = lora_results.get(task, [])
        if not base_ent or not lora_ent:
            continue
        d = np.mean(lora_ent[:14]) - np.mean(base_ent[:14])
        judgment = "↓更集中" if d < 0 else "↑更分散"
        print(f"{task:<45} {direction:>4} {delta:>+7d}%  {d:>+14.6f}  {judgment}")
        if direction == "正向":
            positive_deltas.append(d)
        else:
            negative_deltas.append(d)

    print()
    if positive_deltas:
        print(f"正向任务浅层Δentropy均值: {np.mean(positive_deltas):+.6f}")
    if negative_deltas:
        print(f"负向任务浅层Δentropy均值: {np.mean(negative_deltas):+.6f}")

    if positive_deltas and negative_deltas:
        all_positive_negative = all(d < 0 for d in positive_deltas)
        tracking_opposite = negative_deltas[0] > np.mean(positive_deltas)
        print()
        if all_positive_negative:
            print("✓ 所有正向任务浅层entropy均下降（注意力更集中）")
        else:
            print("✗ 并非所有正向任务entropy都下降，规律不一致")
        if tracking_opposite:
            print("✓ 负向任务(tracking)的entropy变化方向或幅度与正向任务不同")
        else:
            print("✗ 负向任务与正向任务pattern相似，难以区分")

    # 保存
    out = os.path.join(OUTPUT_DIR, "multitask_summary.json")
    summary = {}
    for task in TASKS:
        base_ent = base_results.get(task, [])
        lora_ent = lora_results.get(task, [])
        if base_ent and lora_ent:
            summary[task] = {
                "direction": TASKS[task][0],
                "transfer_delta_pct": TASKS[task][1],
                "early_entropy_delta": float(np.mean(lora_ent[:14]) - np.mean(base_ent[:14])),
                "middle_entropy_delta": float(np.mean(lora_ent[14:23]) - np.mean(base_ent[14:23])),
                "late_entropy_delta": float(np.mean(lora_ent[23:]) - np.mean(base_ent[23:])),
            }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] {out}")

# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Multi-task Attention Entropy Analysis")
    print("  5 positive + 1 negative transfer tasks")
    print("=" * 65)

    # 加载所有数据
    print("\n[数据] 加载BBH任务...")
    task_data = {}
    for task in TASKS:
        examples = load_task(task)
        if examples:
            task_data[task] = examples
            print(f"  {task}: {len(examples)} 条")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # ── 第一轮：base model ──
    print("\n[加载] base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("[收集] base attention entropy（6个任务）...")
    base_results = collect_all_tasks(model, tokenizer, task_data, "Base")
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── 第二轮：lora model ──
    print("\n[加载] lora model...")
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    lora_model = PeftModel.from_pretrained(base_for_lora, LORA_PATH)
    lora_model.eval()
    print("[收集] lora attention entropy（6个任务）...")
    lora_results = collect_all_tasks(lora_model, tokenizer, task_data, "LoRA")
    del lora_model; gc.collect(); torch.cuda.empty_cache()

    # 检测层数
    num_layers = max(
        len(v) for v in list(base_results.values()) + list(lora_results.values())
        if v
    )
    print(f"\n[信息] 检测到 {num_layers} 层")

    # 绘图
    print("\n[绘图] 逐任务折线图...")
    plot_per_task(base_results, lora_results, num_layers)

    print("[绘图] 浅层delta柱状图...")
    plot_early_delta_bar(base_results, lora_results)

    print("[绘图] 三区域热图...")
    plot_zone_heatmap(base_results, lora_results)

    print_summary(base_results, lora_results)

    print(f"\n✓ 完成！输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()