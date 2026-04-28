"""
Attention Pattern Analysis
比较base和maxlora在正向迁移任务上的注意力模式差异

分析内容：
1. 每层attention entropy（注意力是否更集中）
2. 关键词attention权重（模型是否更关注计数相关词）
3. base vs maxlora的attention差异热力图（选几个典型层）
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
OUTPUT_DIR = "/root/autodl-tmp/llm-project/results/attention_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES  = 10    # 用于entropy统计的样本数
TARGET_LAYER = [0, 4, 8, 13, 17, 22, 25, 27]  # 可视化的代表层

TASK = "object_counting"

PROMPT_TEMPLATE = """\
{input}

Think step by step and give your final answer after "the answer is".
"""

# ── 1. 数据加载 ───────────────────────────────────────────
def load_task(task_name):
    ds = load_dataset("lukaemon/bbh", task_name,
                      cache_dir=HF_CACHE, trust_remote_code=True)
    split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
    return [{"input": row["input"], "target": row["target"]} for row in split]

# ── 2. 收集attention权重 ──────────────────────────────────
def collect_attentions(model, tokenizer, examples, n=10):
    """
    返回：
      all_attn: list of dict {layer: attn_matrix (heads, seq, seq)}
      all_tokens: list of token列表
    """
    model.eval()
    all_attn   = []
    all_tokens = []

    for ex in examples[:n]:
        prompt = PROMPT_TEMPLATE.format(input=ex["input"])
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(model.device)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
            )

        # outputs.attentions: tuple of (batch, heads, seq, seq) per layer
        attn_per_layer = {}
        for layer_idx, attn in enumerate(outputs.attentions):
            # attn: (1, heads, seq, seq) → (heads, seq, seq)
            attn_per_layer[layer_idx] = attn[0].float().cpu()

        all_attn.append(attn_per_layer)
        all_tokens.append(tokens)

    return all_attn, all_tokens

# ── 3. Attention Entropy ──────────────────────────────────
def compute_entropy(attn_matrix):
    """
    attn_matrix: (heads, seq, seq)
    每个head每个query位置的entropy，返回 (heads, seq)
    entropy低 → 注意力集中；entropy高 → 注意力分散
    """
    # attn已经是softmax后的概率分布
    eps = 1e-12
    entropy = -(attn_matrix * torch.log(attn_matrix + eps)).sum(dim=-1)
    return entropy  # (heads, seq)

def compute_mean_entropy_per_layer(all_attn, num_layers):
    """
    返回每层的平均entropy（对所有样本、所有head、所有位置取均值）
    """
    layer_entropy = []
    for layer in range(num_layers):
        entropies = []
        for attn_dict in all_attn:
            if layer not in attn_dict:
                continue
            attn = attn_dict[layer]  # (heads, seq, seq)
            ent  = compute_entropy(attn)  # (heads, seq)
            entropies.append(ent.mean().item())
        layer_entropy.append(np.mean(entropies) if entropies else float("nan"))
    return layer_entropy

# ── 4. 关键词注意力权重 ───────────────────────────────────
COUNTING_KEYWORDS = [
    "how", "many", "count", "number", "total",
    "objects", "items", "things",
]

def find_keyword_positions(tokens):
    positions = []
    for i, tok in enumerate(tokens):
        clean = tok.replace("▁", "").replace("Ġ", "").lower()
        if any(kw in clean for kw in COUNTING_KEYWORDS):
            positions.append(i)
    return positions

def compute_keyword_attention(all_attn, all_tokens, num_layers):
    """
    对每层，计算最后一个token（预测位置）对关键词位置的平均attention
    """
    layer_kw_attn = []
    for layer in range(num_layers):
        kw_attns = []
        for attn_dict, tokens in zip(all_attn, all_tokens):
            if layer not in attn_dict:
                continue
            attn = attn_dict[layer]  # (heads, seq, seq)
            kw_pos = find_keyword_positions(tokens)
            if not kw_pos:
                continue
            # 最后一个token对关键词位置的attention，对所有head取均值
            last_attn = attn[:, -1, :]  # (heads, seq)
            kw_attn   = last_attn[:, kw_pos].mean().item()
            kw_attns.append(kw_attn)
        layer_kw_attn.append(np.mean(kw_attns) if kw_attns else float("nan"))
    return layer_kw_attn

# ── 5. Attention差异热力图（单样本，代表层）─────────────
def plot_attention_heatmaps(base_attn_dict, lora_attn_dict, tokens,
                             target_layers, title_prefix):
    """
    对选定的层，画base/maxlora/差值三张热力图
    只取第一个head（最具代表性的）
    """
    n_layers = len(target_layers)
    fig, axes = plt.subplots(
        n_layers, 3,
        figsize=(18, 3.5 * n_layers),
    )
    if n_layers == 1:
        axes = [axes]

    # 截断过长的token序列用于显示
    MAX_SHOW = 40
    show_tokens = [t.replace("▁", "").replace("Ġ", "") for t in tokens[:MAX_SHOW]]

    for row, layer in enumerate(target_layers):
        if layer not in base_attn_dict or layer not in lora_attn_dict:
            continue

        # 取head=0，截断序列
        base_map = base_attn_dict[layer][0, :MAX_SHOW, :MAX_SHOW].numpy()
        lora_map = lora_attn_dict[layer][0, :MAX_SHOW, :MAX_SHOW].numpy()
        diff_map = lora_map - base_map

        vmax = max(base_map.max(), lora_map.max())

        for col, (data, cmap, title, vmin_) in enumerate([
            (base_map, "Blues",    f"Base — Layer {layer}",    0),
            (lora_map, "Oranges",  f"MaxLoRA — Layer {layer}", 0),
            (diff_map, "RdBu_r",   f"Diff (LoRA−Base)",        -vmax/2),
        ]):
            ax = axes[row][col]
            vmax_ = vmax if col < 2 else vmax/2
            sns.heatmap(
                data, ax=ax,
                cmap=cmap,
                vmin=vmin_, vmax=vmax_,
                xticklabels=show_tokens,
                yticklabels=show_tokens,
                cbar=True,
                square=True,
                linewidths=0,
            )
            ax.set_title(title, fontsize=9)
            ax.tick_params(axis="both", labelsize=6)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle(
        f"Attention Pattern: Base vs MaxLoRA\n{title_prefix}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"attn_heatmap_{title_prefix.replace(' ','_')}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 6. 综合折线图 ─────────────────────────────────────────
def plot_summary(base_entropy, lora_entropy,
                 base_kw, lora_kw,
                 num_layers):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(
        "Attention Analysis: Base vs MaxLoRA\n"
        f"Task: {TASK} (Positive Transfer +24%)",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(num_layers)
    zone_kw = dict(alpha=0.04)

    # ── 图1：Entropy ──
    ax = axes[0]
    ax.plot(x, base_entropy, "o-", color="#2c7bb6", label="Base",    lw=2, ms=4)
    ax.plot(x, lora_entropy, "s-", color="#d7191c", label="MaxLoRA", lw=2, ms=4)
    ax.fill_between(x, base_entropy, lora_entropy, alpha=0.12, color="gray")
    ax.axvspan(0,  14, color="blue",  **zone_kw)
    ax.axvspan(14, 23, color="red",   **zone_kw)
    ax.axvspan(23, 28, color="green", **zone_kw)
    ax.set_ylabel("Mean Attention Entropy\n(lower = more focused)", fontsize=10)
    ax.set_title("Attention Entropy per Layer  (lower = model attends more selectively)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    # ── 图2：关键词attention权重 ──
    ax = axes[1]
    ax.plot(x, base_kw, "o-", color="#2c7bb6", label="Base",    lw=2, ms=4)
    ax.plot(x, lora_kw, "s-", color="#d7191c", label="MaxLoRA", lw=2, ms=4)
    ax.fill_between(x, base_kw, lora_kw, alpha=0.12, color="gray")
    ax.axvspan(0,  14, color="blue",  **zone_kw)
    ax.axvspan(14, 23, color="red",   **zone_kw)
    ax.axvspan(23, 28, color="green", **zone_kw)
    ax.set_ylabel("Keyword Attention Weight\n(higher = more focus on count words)", fontsize=10)
    ax.set_title("Last-token Attention to Counting Keywords per Layer", fontsize=11)
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xticks(range(0, num_layers, 2))

    # 区域标注
    for ax_ in axes:
        ymax = ax_.get_ylim()[1]
        ax_.text(7,  ymax * 0.95, "Early",  ha="center", color="blue",  fontsize=9, alpha=0.6)
        ax_.text(18, ymax * 0.95, "Middle", ha="center", color="red",   fontsize=9, alpha=0.6)
        ax_.text(25, ymax * 0.95, "Late",   ha="center", color="green", fontsize=9, alpha=0.6)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "attention_summary.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  Attention Pattern Analysis — {TASK}")
    print("=" * 60)

    examples = load_task(TASK)
    print(f"[数据] {len(examples)} 条，取前 {NUM_SAMPLES} 条")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # ── 加载base，收集attention ──
    print("\n[加载] base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("[收集] base attention...")
    base_attn, base_tokens = collect_attentions(model, tokenizer, examples, NUM_SAMPLES)
    num_layers = len(base_attn[0])
    print(f"  → {num_layers} 层, {len(base_tokens[0])} tokens (第1条)")

    base_entropy = compute_mean_entropy_per_layer(base_attn, num_layers)
    base_kw      = compute_keyword_attention(base_attn, base_tokens, num_layers)

    # 保存第一条样本的attn用于热力图
    base_attn_sample  = base_attn[0]
    sample_tokens     = base_tokens[0]

    del model; gc.collect(); torch.cuda.empty_cache()

    # ── 加载lora，收集attention ──
    print("\n[加载] lora model...")
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_model = PeftModel.from_pretrained(base_for_lora, LORA_PATH)
    lora_model.eval()

    print("[收集] lora attention...")
    lora_attn, _ = collect_attentions(lora_model, tokenizer, examples, NUM_SAMPLES)

    lora_entropy = compute_mean_entropy_per_layer(lora_attn, num_layers)
    lora_kw      = compute_keyword_attention(lora_attn, base_tokens, num_layers)

    lora_attn_sample = lora_attn[0]

    del lora_model; gc.collect(); torch.cuda.empty_cache()

    # ── 绘图 ──
    print("\n[绘图] 综合折线图...")
    plot_summary(base_entropy, lora_entropy, base_kw, lora_kw, num_layers)

    print("[绘图] attention热力图（代表层）...")
    plot_attention_heatmaps(
        base_attn_sample, lora_attn_sample,
        sample_tokens, TARGET_LAYER,
        title_prefix="object_counting",
    )

    # ── 数值摘要 ──
    print("\n── 关键对比 ─────────────────────────────────────────")
    zones = {
        "early (0-13)" : list(range(0, 14)),
        "middle (14-22)": list(range(14, 23)),
        "late (23-27)" : list(range(23, 28)),
    }
    print(f"{'区域':<16} {'Base熵':>10} {'LoRA熵':>10} {'Δ熵':>10} "
          f"{'Base关键词':>12} {'LoRA关键词':>12} {'Δ关键词':>10}")
    print("-" * 82)
    for zone, layers in zones.items():
        be = np.nanmean([base_entropy[l] for l in layers])
        le = np.nanmean([lora_entropy[l] for l in layers])
        bk = np.nanmean([base_kw[l] for l in layers])
        lk = np.nanmean([lora_kw[l] for l in layers])
        print(f"{zone:<16} {be:>10.4f} {le:>10.4f} {le-be:>+10.4f} "
              f"{bk:>12.6f} {lk:>12.6f} {lk-bk:>+10.6f}")

    # 保存数值
    out = os.path.join(OUTPUT_DIR, "attention_stats.json")
    with open(out, "w") as f:
        json.dump({
            "base_entropy": base_entropy,
            "lora_entropy": lora_entropy,
            "base_keyword_attn": base_kw,
            "lora_keyword_attn": lora_kw,
        }, f, indent=2)
    print(f"\n[保存] {out}")
    print(f"\n✓ 完成！输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()