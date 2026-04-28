"""
CKA (Centered Kernel Alignment) Analysis
计算每层：五子棋数据 vs BBH任务 的表征相似度
比较 base 和 maxlora 下这个相似度的变化

核心问题：
  LoRA微调后，正向迁移任务的表征是否在浅层更接近五子棋？
  负向迁移任务是否相反？

CKA值域 [0,1]：
  1 = 两组数据的表征完全对齐
  0 = 完全不相关
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import gc

# ── 配置 ──────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH    = "/root/autodl-tmp/llm-project/checkpoints/qwen-gomoku-maxlora/final_model"
GOMOKU_DATA  = "/root/autodl-tmp/llm-project/datasets/real_games_v2/train.json"
HF_CACHE     = "/root/autodl-tmp/hf_cache"
OUTPUT_DIR   = "/root/autodl-tmp/llm-project/results/cka_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES  = 30   # 每个数据集取多少条（CKA需要样本数 > 隐层维度是不可能的，30条足够）

TASKS = {
    "object_counting"                         : ("正向", +24, "#1f77b4"),
    "logical_deduction_three_objects"         : ("正向", +15, "#2ca02c"),
    "geometric_shapes"                        : ("正向", +7,  "#9467bd"),
    "tracking_shuffled_objects_three_objects" : ("负向", -22, "#d62728"),
}

PROMPT_TEMPLATE = "{input}\n\nThink step by step and give your final answer after \"the answer is\"."

# ── CKA核心计算 ───────────────────────────────────────────
def centering(K):
    """对Gram矩阵做中心化"""
    n = K.shape[0]
    unit = np.ones([n, n]) / n
    I = np.eye(n)
    H = I - unit
    return H @ K @ H

def rbf_kernel(X, sigma=None):
    """RBF核，X: (n, d)"""
    GX = X @ X.T
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = np.sqrt(mdist) if mdist > 0 else 1.0
    KX = np.exp(-KX / (2 * sigma ** 2))
    return KX

def linear_CKA(X, Y):
    """
    线性CKA，X: (n, d1), Y: (n, d2)
    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    # 中心化
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X

    num   = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")
    return float(num / denom) if denom > 1e-10 else 0.0

def kernel_CKA(X, Y):
    """
    核CKA（RBF核）
    更能捕捉非线性结构
    """
    Kx = centering(rbf_kernel(X))
    Ky = centering(rbf_kernel(Y))
    num   = np.sum(Kx * Ky)
    denom = np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky))
    return float(num / denom) if denom > 1e-10 else 0.0

# ── 收集隐状态 ────────────────────────────────────────────
def collect_hidden_states(model, tokenizer, texts, n=30):
    """
    对每条文本做forward pass，收集每层最后一个token的隐状态
    返回：{layer_idx: np.array (n, hidden_dim)}
    """
    model.eval()
    all_hidden = {}

    for text in texts[:n]:
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of (batch, seq, hidden) per layer
        # 第0个是embedding层，1..28是transformer层
        for layer_idx, hidden in enumerate(outputs.hidden_states[1:]):
            vec = hidden[0, -1, :].float().cpu().numpy()  # (hidden_dim,)
            if layer_idx not in all_hidden:
                all_hidden[layer_idx] = []
            all_hidden[layer_idx].append(vec)

        del outputs
        torch.cuda.empty_cache()

    # 转为矩阵
    return {
        layer: np.stack(vecs, axis=0)  # (n, hidden_dim)
        for layer, vecs in all_hidden.items()
    }

# ── 加载数据 ──────────────────────────────────────────────
def load_gomoku_texts(path, n=30):
    with open(path) as f:
        data = json.load(f)
    texts = []
    for item in data[:n]:
        if isinstance(item, dict):
            text = item.get("input", item.get("prompt", str(item)))
        else:
            text = str(item)
        texts.append(text[:1000])  # 截断避免OOM
    print(f"  五子棋数据: {len(texts)} 条")
    return texts

def load_bbh_texts(task_name, n=30):
    try:
        ds = load_dataset("lukaemon/bbh", task_name,
                          cache_dir=HF_CACHE, trust_remote_code=True)
        split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
        texts = [PROMPT_TEMPLATE.format(input=row["input"]) for row in split]
        return texts[:n]
    except Exception as e:
        print(f"  [警告] {task_name} 加载失败: {e}")
        return []

# ── 计算CKA矩阵 ───────────────────────────────────────────
def compute_cka_per_layer(gomoku_hidden, task_hidden, num_layers):
    """
    对每层计算 CKA(gomoku表征, task表征)
    返回 list of float，长度=num_layers
    """
    linear_cka_vals = []
    kernel_cka_vals = []

    for layer in range(num_layers):
        if layer not in gomoku_hidden or layer not in task_hidden:
            linear_cka_vals.append(float("nan"))
            kernel_cka_vals.append(float("nan"))
            continue

        X = gomoku_hidden[layer]  # (n, hidden)
        Y = task_hidden[layer]    # (n, hidden)

        # 样本数需要一致
        n = min(len(X), len(Y))
        X, Y = X[:n], Y[:n]

        linear_cka_vals.append(linear_CKA(X, Y))
        kernel_cka_vals.append(kernel_CKA(X, Y))

    return linear_cka_vals, kernel_cka_vals

# ── 绘图：主图 ────────────────────────────────────────────
def plot_cka_curves(base_cka, lora_cka, num_layers):
    """
    base_cka / lora_cka: {task: (linear_vals, kernel_vals)}
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(
        "CKA Similarity: Gomoku Representations vs BBH Task Representations\n"
        "Base vs MaxLoRA — Does LoRA bring positive-transfer tasks closer to Gomoku?",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(num_layers)
    zone_kw = dict(alpha=0.05)

    for ax, cka_type, title in zip(
        axes,
        ["linear", "kernel"],
        ["Linear CKA", "Kernel CKA (RBF)"],
    ):
        for task, (direction, delta, color) in TASKS.items():
            base_vals = base_cka[task][0 if cka_type=="linear" else 1]
            lora_vals = lora_cka[task][0 if cka_type=="linear" else 1]

            short = task.replace("_three_objects","").replace("_two","")
            short = short.replace("tracking_shuffled_objects", "tracking")
            label = f"{short} ({direction} {delta:+d}%)"

            ls = "-" if direction == "正向" else "--"
            lw = 2 if direction == "正向" else 2.5

            # base：虚线
            ax.plot(x, base_vals, color=color, ls=":",
                    lw=1.2, alpha=0.5)
            # lora：实线
            ax.plot(x, lora_vals, color=color, ls=ls,
                    lw=lw, label=label, alpha=0.9)

            # 填充差异区域
            diff = np.array(lora_vals) - np.array(base_vals)
            ax.fill_between(x,
                            np.array(base_vals),
                            np.array(lora_vals),
                            alpha=0.08, color=color)

        ax.axvspan(0,  14, color="blue",  **zone_kw)
        ax.axvspan(14, 23, color="red",   **zone_kw)
        ax.axvspan(23, 28, color="green", **zone_kw)

        ymax = ax.get_ylim()[1]
        ax.text(7,  ymax * 0.97, "Early",  ha="center", color="blue",  fontsize=9, alpha=0.6)
        ax.text(18, ymax * 0.97, "Middle", ha="center", color="red",   fontsize=9, alpha=0.6)
        ax.text(25, ymax * 0.97, "Late",   ha="center", color="green", fontsize=9, alpha=0.6)

        ax.set_title(f"{title}\n(solid=MaxLoRA, dotted=Base)", fontsize=11)
        ax.set_xlabel("Layer Index", fontsize=10)
        ax.set_ylabel("CKA Similarity (Gomoku vs BBH Task)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xticks(range(0, num_layers, 2))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cka_curves.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

def plot_cka_delta_bar(base_cka, lora_cka):
    """
    每个任务在浅层(0-13)的CKA变化量（lora - base）
    正向任务应该CKA增大（表征更接近五子棋）
    """
    tasks  = list(TASKS.keys())
    deltas_linear = []
    deltas_kernel = []
    colors = []
    labels = []

    for task in tasks:
        bl, bk = base_cka[task]
        ll, lk = lora_cka[task]
        dl = np.nanmean(ll[:14]) - np.nanmean(bl[:14])
        dk = np.nanmean(lk[:14]) - np.nanmean(bk[:14])
        deltas_linear.append(dl)
        deltas_kernel.append(dk)
        direction, delta_pct, color = TASKS[task]
        colors.append(color)
        short = task.replace("_three_objects","").replace("_two","")
        short = short.replace("_", "\n")
        labels.append(f"{short}\n({direction}{delta_pct:+d}%)")

    x = np.arange(len(tasks))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "CKA Change in Early Layers (0-13): MaxLoRA vs Base\n"
        "Positive = representations moved closer to Gomoku after LoRA",
        fontsize=12, fontweight="bold",
    )

    for ax, deltas, title in zip(
        axes,
        [deltas_linear, deltas_kernel],
        ["Linear CKA Delta", "Kernel CKA Delta"],
    ):
        bars = ax.bar(x, deltas, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", lw=1)

        # 分隔线
        n_pos = sum(1 for t in TASKS if TASKS[t][0]=="正向")
        ax.axvline(n_pos - 0.5, color="gray", lw=1.5, ls="--", alpha=0.5)

        for bar, val in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.001 if val >= 0 else -0.003),
                f"{val:+.4f}", ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("ΔCKA (MaxLoRA - Base)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cka_delta_bar.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 数值摘要 ──────────────────────────────────────────────
def print_summary(base_cka, lora_cka):
    print("\n── CKA变化汇总（浅层0-13）─────────────────────────────")
    print(f"{'任务':<45} {'方向':>4} {'ΔCKA(linear)':>14} {'ΔCKA(kernel)':>14} {'判断'}")
    print("-" * 85)

    pos_linear, neg_linear = [], []
    for task, (direction, delta, _) in TASKS.items():
        bl, bk = base_cka[task]
        ll, lk = lora_cka[task]
        dl = np.nanmean(ll[:14]) - np.nanmean(bl[:14])
        dk = np.nanmean(lk[:14]) - np.nanmean(bk[:14])
        judgment = "↑更接近五子棋" if dl > 0 else "↓更远离五子棋"
        print(f"{task:<45} {direction:>4} {dl:>+14.6f} {dk:>+14.6f}  {judgment}")
        if direction == "正向":
            pos_linear.append(dl)
        else:
            neg_linear.append(dl)

    print()
    print(f"正向任务平均ΔCKA: {np.mean(pos_linear):+.6f}")
    print(f"负向任务平均ΔCKA: {np.mean(neg_linear):+.6f}")

    if np.mean(pos_linear) > np.mean(neg_linear):
        print("\n✓ 正向迁移任务的表征在浅层更接近五子棋，假说成立")
    else:
        print("\n✗ 假说不成立，需要重新审视机制解释")

    # 保存
    summary = {}
    for task in TASKS:
        bl, bk = base_cka[task]
        ll, lk = lora_cka[task]
        summary[task] = {
            "direction": TASKS[task][0],
            "transfer_pct": TASKS[task][1],
            "early_linear_cka_base": float(np.nanmean(bl[:14])),
            "early_linear_cka_lora": float(np.nanmean(ll[:14])),
            "early_linear_cka_delta": float(np.nanmean(ll[:14]) - np.nanmean(bl[:14])),
            "early_kernel_cka_delta": float(np.nanmean(lk[:14]) - np.nanmean(bk[:14])),
        }
    out = os.path.join(OUTPUT_DIR, "cka_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[保存] {out}")

# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  CKA Representation Similarity Analysis")
    print("  Gomoku vs BBH Tasks: Base vs MaxLoRA")
    print("=" * 65)

    # 加载文本
    print("\n[数据] 加载文本...")
    gomoku_texts = load_gomoku_texts(GOMOKU_DATA, NUM_SAMPLES)
    task_texts   = {}
    for task in TASKS:
        texts = load_bbh_texts(task, NUM_SAMPLES)
        if texts:
            task_texts[task] = texts
            print(f"  {task}: {len(texts)} 条")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base_cka = {}
    lora_cka = {}

    for model_name, loader in [
        ("Base",    lambda: AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL, dtype=torch.float16,
                        device_map="auto", trust_remote_code=True)),
        ("MaxLoRA", lambda: PeftModel.from_pretrained(
                        AutoModelForCausalLM.from_pretrained(
                            BASE_MODEL, dtype=torch.float16,
                            device_map="auto", trust_remote_code=True),
                        LORA_PATH)),
    ]:
        print(f"\n[加载] {model_name}...")
        model = loader()
        model.eval()

        print(f"  收集五子棋隐状态...")
        gomoku_hidden = collect_hidden_states(
            model, tokenizer, gomoku_texts, NUM_SAMPLES)
        num_layers = len(gomoku_hidden)
        print(f"  → {num_layers} 层, {list(gomoku_hidden.values())[0].shape}")

        cka_store = base_cka if model_name == "Base" else lora_cka

        for task, texts in task_texts.items():
            direction, delta, _ = TASKS[task]
            print(f"  收集 {task} ({direction}{delta:+d}%)...")
            task_hidden = collect_hidden_states(
                model, tokenizer, texts, NUM_SAMPLES)
            lin, ker = compute_cka_per_layer(gomoku_hidden, task_hidden, num_layers)
            cka_store[task] = (lin, ker)
            print(f"    浅层linear CKA均值: {np.nanmean(lin[:14]):.4f}")

        del model; gc.collect(); torch.cuda.empty_cache()

    # 绘图
    print("\n[绘图] CKA曲线...")
    plot_cka_curves(base_cka, lora_cka, num_layers)

    print("[绘图] CKA delta柱状图...")
    plot_cka_delta_bar(base_cka, lora_cka)

    print_summary(base_cka, lora_cka)

    print(f"\n✓ 完成！输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()