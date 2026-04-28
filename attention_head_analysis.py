"""
Attention Head-level Analysis
细粒度分析每个注意力头的行为差异

核心问题：
1. 哪些head在LoRA后变化最大？（Head重要性热力图）
2. 正向任务和负向任务是否激活不同的head集合？
3. 是否存在"迁移头"——只在正向任务上entropy下降的特定head？

指标：
- Head Entropy：每个head的注意力集中度
- Head Attention Shift：base vs lora的注意力模式差异（per head）
- Transfer Head Score：正向任务entropy变化 - 负向任务entropy变化
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
OUTPUT_DIR = "/root/autodl-tmp/llm-project/results/attention_head"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES  = 15
NUM_LAYERS   = 28
NUM_HEADS    = 28  # Qwen2.5-7B有28个head

TASKS = {
    "object_counting"                         : ("正向", +24, "#1f77b4"),
    "logical_deduction_three_objects"         : ("正向", +15, "#2ca02c"),
    "tracking_shuffled_objects_three_objects" : ("负向", -22, "#d62728"),
}

PROMPT_TEMPLATE = "{input}\n\nThink step by step and give your final answer after \"the answer is\"."

# ── 数据加载 ──────────────────────────────────────────────
def load_task(task_name, n=15):
    try:
        ds = load_dataset("lukaemon/bbh", task_name,
                          cache_dir=HF_CACHE, trust_remote_code=True)
        split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
        return [{"input": row["input"], "target": row["target"]} for row in split][:n]
    except Exception as e:
        print(f"  [警告] {task_name}: {e}")
        return []

# ── 收集每个head的entropy ─────────────────────────────────
def collect_head_entropy(model, tokenizer, examples, n=15):
    """
    返回：np.array (num_layers, num_heads)
    每个head在所有样本上的平均entropy
    """
    model.eval()
    # layer -> head -> list of entropy values
    head_entropies = {
        l: {h: [] for h in range(NUM_HEADS)}
        for l in range(NUM_LAYERS)
    }

    for ex in examples[:n]:
        prompt = PROMPT_TEMPLATE.format(input=ex["input"])
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        if not outputs.attentions:
            print("  [警告] 没有attention输出")
            del outputs
            continue

        for layer_idx, attn in enumerate(outputs.attentions):
            # attn: (1, heads, seq, seq)
            attn_f = attn[0].float()  # (heads, seq, seq)
            eps = 1e-12
            # entropy per head: (heads, seq) -> (heads,)
            ent = -(attn_f * torch.log(attn_f + eps)).sum(dim=-1).mean(dim=-1)
            for h in range(min(attn_f.shape[0], NUM_HEADS)):
                head_entropies[layer_idx][h].append(ent[h].item())

        del outputs
        torch.cuda.empty_cache()

    # 转为矩阵 (num_layers, num_heads)
    mat = np.full((NUM_LAYERS, NUM_HEADS), np.nan)
    for l in range(NUM_LAYERS):
        for h in range(NUM_HEADS):
            vals = head_entropies[l][h]
            if vals:
                mat[l, h] = np.mean(vals)
    return mat

# ── 收集所有任务的head entropy ────────────────────────────
def collect_all(model, tokenizer, task_data, model_tag):
    results = {}
    for task, examples in task_data.items():
        direction, delta, _ = TASKS[task]
        print(f"  [{model_tag}] {task} ({direction}{delta:+d}%)...")
        mat = collect_head_entropy(model, tokenizer, examples, NUM_SAMPLES)
        results[task] = mat
        print(f"    → shape={mat.shape}, "
              f"早层均值={np.nanmean(mat[:14,:]):.4f}")
    return results

# ── 绘图1：Head重要性热力图（LoRA前后变化最大的head）────
def plot_head_change_heatmap(base_results, lora_results):
    """
    对每个任务，计算每个head的entropy变化 (lora - base)
    画 (layer × head) 热力图
    """
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(8 * n_tasks, 10))
    if n_tasks == 1:
        axes = [axes]

    fig.suptitle(
        "Per-Head Attention Entropy Change: MaxLoRA vs Base\n"
        "Blue = more focused (↓entropy)  |  Red = more dispersed (↑entropy)",
        fontsize=13, fontweight="bold",
    )

    for ax, (task, (direction, delta, color)) in zip(axes, TASKS.items()):
        base_mat = base_results.get(task)
        lora_mat = lora_results.get(task)
        if base_mat is None or lora_mat is None:
            continue

        diff = lora_mat - base_mat  # (layers, heads)

        vmax = np.nanpercentile(np.abs(diff), 95)
        sns.heatmap(
            diff,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax, vmax=vmax,
            xticklabels=[str(h) if h % 4 == 0 else "" for h in range(NUM_HEADS)],
            yticklabels=[str(l) if l % 4 == 0 else "" for l in range(NUM_LAYERS)],
            cbar_kws={"label": "Δentropy (LoRA-Base)", "shrink": 0.8},
            linewidths=0,
        )

        # 区域分隔线
        ax.axhline(14, color="blue",  lw=1.5, ls="--", alpha=0.6)
        ax.axhline(23, color="green", lw=1.5, ls="--", alpha=0.6)
        ax.text(NUM_HEADS + 0.3, 7,  "Early",  va="center", color="blue",  fontsize=8)
        ax.text(NUM_HEADS + 0.3, 18, "Middle", va="center", color="red",   fontsize=8)
        ax.text(NUM_HEADS + 0.3, 25, "Late",   va="center", color="green", fontsize=8)

        short = task.replace("_three_objects","").replace("_two","")
        ax.set_title(
            f"{short}\n({direction} {delta:+d}%)",
            fontsize=10, color=color, fontweight="bold",
        )
        ax.set_xlabel("Head Index", fontsize=9)
        ax.set_ylabel("Layer Index", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "head_entropy_heatmap.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 绘图2：Transfer Head Score热力图 ──────────────────────
def plot_transfer_head_score(base_results, lora_results):
    """
    Transfer Head Score = mean(正向任务Δentropy) - 负向任务Δentropy
    正值 = 该head在正向任务上更集中，在负向任务上不变或更分散
         = 可能是"迁移头"
    """
    pos_tasks = [t for t in TASKS if TASKS[t][0] == "正向"]
    neg_tasks = [t for t in TASKS if TASKS[t][0] == "负向"]

    pos_diffs = []
    for task in pos_tasks:
        if task in base_results and task in lora_results:
            pos_diffs.append(lora_results[task] - base_results[task])

    neg_diffs = []
    for task in neg_tasks:
        if task in base_results and task in lora_results:
            neg_diffs.append(lora_results[task] - base_results[task])

    if not pos_diffs or not neg_diffs:
        print("[警告] Transfer Head Score缺少数据")
        return

    pos_mean = np.nanmean(pos_diffs, axis=0)  # (layers, heads)
    neg_mean = np.nanmean(neg_diffs, axis=0)

    # Transfer score: 负值更好（entropy下降更多）
    # 正向任务entropy下降 - 负向任务entropy变化
    # 如果正向下降更多（更负），score = pos_mean - neg_mean 更负
    transfer_score = pos_mean - neg_mean  # (layers, heads)

    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    fig.suptitle(
        "Attention Head Transfer Analysis\n"
        "Transfer Score = Positive Task Δentropy − Negative Task Δentropy\n"
        "(Negative score = head more focused on positive tasks → potential 'transfer head')",
        fontsize=12, fontweight="bold",
    )

    datasets = [
        (pos_mean,      "Mean Δentropy — Positive Tasks",  "RdBu_r"),
        (neg_mean,      "Mean Δentropy — Negative Tasks",  "RdBu_r"),
        (transfer_score,"Transfer Score (Pos − Neg)",       "RdBu_r"),
    ]

    for ax, (data, title, cmap) in zip(axes, datasets):
        vmax = np.nanpercentile(np.abs(data), 95)
        sns.heatmap(
            data, ax=ax,
            cmap=cmap, center=0,
            vmin=-vmax, vmax=vmax,
            xticklabels=[str(h) if h % 4 == 0 else "" for h in range(NUM_HEADS)],
            yticklabels=[str(l) if l % 4 == 0 else "" for l in range(NUM_LAYERS)],
            cbar_kws={"label": "Value", "shrink": 0.8},
            linewidths=0,
        )
        ax.axhline(14, color="blue",  lw=1.5, ls="--", alpha=0.5)
        ax.axhline(23, color="green", lw=1.5, ls="--", alpha=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Head Index", fontsize=9)
        ax.set_ylabel("Layer Index", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "transfer_head_score.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

    # 找top迁移头
    flat_score = transfer_score.copy()
    flat_score[np.isnan(flat_score)] = 0
    top_k = 10
    flat_idx = np.argsort(flat_score.ravel())[:top_k]  # 最负的 = 最像迁移头
    top_heads = [(idx // NUM_HEADS, idx % NUM_HEADS, flat_score.ravel()[idx])
                 for idx in flat_idx]

    print("\n── Top 迁移头（Transfer Score最低，最可能是迁移头）───")
    print(f"{'Layer':>6} {'Head':>6} {'Transfer Score':>16}")
    print("-" * 32)
    for layer, head, score in top_heads:
        print(f"{layer:>6} {head:>6} {score:>+16.4f}")

    return transfer_score, top_heads

# ── 绘图3：Top迁移头的逐层曲线 ───────────────────────────
def plot_top_head_curves(base_results, lora_results, top_heads):
    """
    对top5迁移头，画它们在各任务上的entropy曲线
    """
    top5 = top_heads[:5]
    fig, axes = plt.subplots(1, len(top5), figsize=(5 * len(top5), 5))
    if len(top5) == 1:
        axes = [axes]

    fig.suptitle(
        "Top Transfer Heads: Per-Task Entropy Trajectory\n"
        "(heads with largest positive-vs-negative task entropy difference)",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(NUM_LAYERS)

    for ax, (layer, head, score) in zip(axes, top5):
        for task, (direction, delta, color) in TASKS.items():
            base_mat = base_results.get(task)
            lora_mat = lora_results.get(task)
            if base_mat is None or lora_mat is None:
                continue

            base_val = base_mat[:, head]
            lora_val = lora_mat[:, head]

            ls = "-" if direction == "正向" else "--"
            short = task.replace("_three_objects","").replace("_two","")
            short = short.replace("tracking_shuffled_objects", "tracking")
            ax.plot(x, base_val, color=color, ls=":", lw=1, alpha=0.5)
            ax.plot(x, lora_val, color=color, ls=ls, lw=2,
                    label=f"{short}({direction}{delta:+d}%)", alpha=0.9)

        ax.axvline(layer, color="black", lw=1.5, ls="--",
                   alpha=0.7, label=f"Layer {layer}")
        ax.set_title(f"Layer {layer}, Head {head}\n(score={score:+.4f})",
                     fontsize=9)
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Head Entropy", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "top_transfer_heads.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 数值摘要 ──────────────────────────────────────────────
def print_summary(base_results, lora_results):
    print("\n── 各任务浅层head entropy变化（均值±std）──────────────")
    print(f"{'任务':<45} {'方向':>4} {'均值Δ':>10} {'std':>8}")
    print("-" * 72)
    for task, (direction, delta, _) in TASKS.items():
        base_mat = base_results.get(task)
        lora_mat = lora_results.get(task)
        if base_mat is None or lora_mat is None:
            continue
        diff = (lora_mat - base_mat)[:14, :]
        print(f"{task:<45} {direction:>4} "
              f"{np.nanmean(diff):>+10.4f} {np.nanstd(diff):>8.4f}")

# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Attention Head-level Analysis")
    print(f"  {NUM_LAYERS} layers × {NUM_HEADS} heads")
    print("=" * 65)

    print("\n[数据] 加载BBH任务...")
    task_data = {}
    for task in TASKS:
        examples = load_task(task, NUM_SAMPLES)
        if examples:
            task_data[task] = examples
            print(f"  {task}: {len(examples)} 条")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base_results = {}
    lora_results = {}

    # ── 第一轮：base ──
    print("\n[加载] base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    base_results = collect_all(model, tokenizer, task_data, "Base")
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── 第二轮：lora ──
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
    lora_results = collect_all(lora_model, tokenizer, task_data, "LoRA")
    del lora_model; gc.collect(); torch.cuda.empty_cache()

    # ── 绘图 ──
    print("\n[绘图] Head entropy变化热力图...")
    plot_head_change_heatmap(base_results, lora_results)

    print("[绘图] Transfer Head Score热力图...")
    result = plot_transfer_head_score(base_results, lora_results)
    if result:
        transfer_score, top_heads = result
        print("[绘图] Top迁移头曲线...")
        plot_top_head_curves(base_results, lora_results, top_heads)

    print_summary(base_results, lora_results)

    # 保存原始数据
    out = os.path.join(OUTPUT_DIR, "head_entropy_data.npz")
    save_dict = {}
    for task in task_data:
        if task in base_results:
            save_dict[f"base_{task}"] = base_results[task]
        if task in lora_results:
            save_dict[f"lora_{task}"] = lora_results[task]
    np.savez(out, **save_dict)
    print(f"[保存] {out}")

    print(f"\n✓ 完成！输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()