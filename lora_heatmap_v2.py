"""
LoRA Heatmap v2 - 双指标分析
1. Frobenius范数热力图（权重变化幅度）
2. 有效秩热力图（更新的方向集中度）

有效秩低 → 更新集中在少数奇异方向 → 任务特定记忆
有效秩高 → 更新分散在多个方向   → 通用特征调整

两张图对比：如果中间层范数大但有效秩低，说明中间层在记忆源任务
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from safetensors.torch import load_file

# ── 路径配置 ──────────────────────────────────────────────
LORA_PATH  = "/root/autodl-tmp/llm-project/checkpoints/qwen-gomoku-maxlora/final_model"
OUTPUT_DIR = "/root/autodl-tmp/llm-project/results/heatmaps_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES  = ["gate_proj", "up_proj", "down_proj"]
ALL_MODULES  = ATTN_MODULES + MLP_MODULES

# ── 1. 加载adapter权重 ────────────────────────────────────
def load_adapter_weights(lora_path):
    path = Path(lora_path)
    sf_files  = sorted(path.glob("adapter_model*.safetensors"))
    bin_files = sorted(path.glob("adapter_model*.bin"))

    weights = {}
    if sf_files:
        print(f"[加载] safetensors: {[f.name for f in sf_files]}")
        for f in sf_files:
            weights.update(load_file(f, device="cpu"))
    elif bin_files:
        print(f"[加载] bin: {[f.name for f in bin_files]}")
        for f in bin_files:
            weights.update(torch.load(f, map_location="cpu"))
    else:
        raise FileNotFoundError(f"找不到adapter权重: {lora_path}")

    print(f"[加载] 共 {len(weights)} 个张量")
    return weights

# ── 2. 解析层号和模块 ─────────────────────────────────────
def get_key(layer, module):
    """构建lora_A/B的key，兼容两种命名前缀"""
    if module in ATTN_MODULES:
        sub = f"self_attn.{module}"
    else:
        sub = f"mlp.{module}"
    return (
        f"base_model.model.model.layers.{layer}.{sub}.lora_{{X}}.weight",
        f"model.layers.{layer}.{sub}.lora_{{X}}.weight",
    )

def find_key(weights, layer, module, suffix):
    templates, templates2 = get_key(layer, module)
    k1 = templates.replace("{X}", suffix)
    k2 = templates2.replace("{X}", suffix)
    if k1 in weights:
        return k1
    if k2 in weights:
        return k2
    return None

# ── 3. 有效秩计算 ─────────────────────────────────────────
def effective_rank(matrix: torch.Tensor) -> float:
    """
    Roy & Vetterli (2007) 有效秩定义：
    er = exp( -sum(p_i * log(p_i)) )
    其中 p_i = sigma_i / sum(sigma)（奇异值归一化概率分布）

    范围：[1, min(d_out, d_in)]
    值越小说明更新越集中（更像单一方向的记忆）
    值越大说明更新越分散（更像通用调整）
    """
    try:
        S = torch.linalg.svdvals(matrix.float())
        S = S[S > 1e-10]  # 过滤数值噪声
        if len(S) == 0:
            return 0.0
        p = S / S.sum()
        entropy = -(p * torch.log(p + 1e-12)).sum().item()
        return float(np.exp(entropy))
    except Exception:
        return float("nan")

def normalized_effective_rank(matrix: torch.Tensor) -> float:
    """归一化到[0,1]：er / min(d_out, d_in)"""
    er = effective_rank(matrix)
    max_rank = min(matrix.shape)
    return er / max_rank if max_rank > 0 else float("nan")

# ── 4. 计算所有指标 ───────────────────────────────────────
def compute_metrics(weights):
    # 检测层数
    layers = set()
    for key in weights:
        if "lora_A" in key and "layers." in key:
            try:
                n = int(key.split("layers.")[1].split(".")[0])
                layers.add(n)
            except Exception:
                pass
    num_layers = max(layers) + 1
    print(f"[分析] 检测到 {num_layers} 层，{len(ALL_MODULES)} 个模块")

    frob  = {l: {} for l in range(num_layers)}  # 归一化Frobenius范数
    er    = {l: {} for l in range(num_layers)}  # 归一化有效秩
    rank_r = {}  # 每个模块的实际rank r

    for layer in range(num_layers):
        for module in ALL_MODULES:
            ka = find_key(weights, layer, module, "A")
            kb = find_key(weights, layer, module, "B")

            if ka is None or kb is None:
                frob[layer][module] = float("nan")
                er[layer][module]   = float("nan")
                continue

            A = weights[ka].float()  # (r, d_in)
            B = weights[kb].float()  # (d_out, r)
            rank_r[module] = A.shape[0]

            delta_W = B @ A  # (d_out, d_in)

            # Frobenius范数归一化
            d_out, d_in = delta_W.shape
            frob[layer][module] = torch.norm(delta_W, p="fro").item() / np.sqrt(d_out * d_in)

            # 有效秩（归一化）
            er[layer][module] = normalized_effective_rank(delta_W)

    print(f"[分析] LoRA rank r = {list(rank_r.values())[0] if rank_r else 'unknown'}")
    return frob, er, num_layers

# ── 5. 转矩阵 ─────────────────────────────────────────────
def to_mat(metrics, num_layers, modules):
    mat = np.full((len(modules), num_layers), np.nan)
    for j in range(num_layers):
        for i, mod in enumerate(modules):
            v = metrics[j].get(mod, np.nan)
            mat[i, j] = v
    return mat

# ── 6. 主热力图：Frobenius + 有效秩并排 ──────────────────
def plot_dual_heatmap(frob, er, num_layers):
    fig, axes = plt.subplots(2, 1, figsize=(22, 10))
    fig.suptitle(
        "LoRA Update Analysis: Qwen2.5-7B (r=64, Gomoku)\n"
        "Top: Weight Update Magnitude (Frobenius Norm)  |  "
        "Bottom: Update Diversity (Normalized Effective Rank)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    layer_ticks = list(range(0, num_layers, 4))
    xtick_labels = [str(i) if i in layer_ticks else "" for i in range(num_layers)]

    # ── 图1：Frobenius范数 ──
    ax1 = axes[0]
    mat_frob = to_mat(frob, num_layers, ALL_MODULES)
    sns.heatmap(
        mat_frob, ax=ax1,
        cmap="YlOrRd",
        xticklabels=xtick_labels,
        yticklabels=ALL_MODULES,
        cbar_kws={"label": r"$\||BA\||_F / \sqrt{d_{out} \cdot d_{in}}$", "shrink": 0.8},
        linewidths=0.3, linecolor="white",
    )
    ax1.set_title("Frobenius Norm  (larger = more weight change)", fontsize=11)
    ax1.set_xlabel("Layer Index", fontsize=10)
    ax1.axhline(y=len(ATTN_MODULES), color="royalblue", lw=2, ls="--")
    ax1.text(num_layers + 0.2, len(ATTN_MODULES)/2, "Attn",
             va="center", color="royalblue", fontsize=9, fontweight="bold")
    ax1.text(num_layers + 0.2, len(ATTN_MODULES) + len(MLP_MODULES)/2, "MLP",
             va="center", color="firebrick", fontsize=9, fontweight="bold")

    # ── 图2：有效秩 ──
    ax2 = axes[1]
    mat_er = to_mat(er, num_layers, ALL_MODULES)
    sns.heatmap(
        mat_er, ax=ax2,
        cmap="YlGnBu",
        xticklabels=xtick_labels,
        yticklabels=ALL_MODULES,
        vmin=0, vmax=1,
        cbar_kws={"label": "Normalized Effective Rank [0,1]", "shrink": 0.8},
        linewidths=0.3, linecolor="white",
    )
    ax2.set_title("Normalized Effective Rank  (larger = more diverse/general update)", fontsize=11)
    ax2.set_xlabel("Layer Index", fontsize=10)
    ax2.axhline(y=len(ATTN_MODULES), color="royalblue", lw=2, ls="--")
    ax2.text(num_layers + 0.2, len(ATTN_MODULES)/2, "Attn",
             va="center", color="royalblue", fontsize=9, fontweight="bold")
    ax2.text(num_layers + 0.2, len(ATTN_MODULES) + len(MLP_MODULES)/2, "MLP",
             va="center", color="firebrick", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "heatmap_frob_vs_er.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 7. 折线图：浅/中/深层的两个指标均值对比 ──────────────
def plot_layer_profile(frob, er, num_layers):
    zones = {
        "early (0-13)" : list(range(0, 14)),
        "middle (14-22)": list(range(14, 23)),
        "late (23-27)" : list(range(23, 28)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Per-Layer Profile: Frobenius Norm vs Effective Rank\n"
                 "(Qwen2.5-7B, r=64, Gomoku LoRA)",
                 fontsize=13, fontweight="bold")

    x = np.arange(num_layers)
    colors_mod = {
        "q_proj": "#1f77b4", "k_proj": "#aec7e8",
        "v_proj": "#ff7f0e", "o_proj": "#ffbb78",
        "gate_proj": "#d62728", "up_proj": "#ff9896", "down_proj": "#9467bd",
    }

    for ax, (metrics, title, ylabel) in zip(axes, [
        (frob, "Frobenius Norm per Layer", r"$\||BA\||_F / \sqrt{d \cdot d}$"),
        (er,   "Effective Rank per Layer", "Normalized Effective Rank"),
    ]):
        for mod in ALL_MODULES:
            vals = [metrics[l].get(mod, np.nan) for l in range(num_layers)]
            ax.plot(x, vals, "o-", color=colors_mod[mod], label=mod,
                    linewidth=1.5, markersize=3, alpha=0.8)

        # 分区背景
        ax.axvspan(0,  14, alpha=0.05, color="blue",   label="_early")
        ax.axvspan(14, 23, alpha=0.05, color="red",    label="_middle")
        ax.axvspan(23, 28, alpha=0.05, color="green",  label="_late")

        # 分区标注
        ax.text(7,  ax.get_ylim()[1] * 0.95 if not np.isnan(ax.get_ylim()[1]) else 0.9,
                "Early\n(0-13)", ha="center", color="blue", fontsize=9, alpha=0.7)
        ax.text(18, 0, "Middle\n(14-22)", ha="center", color="red", fontsize=9, alpha=0.7)
        ax.text(25, 0, "Late\n(23-27)", ha="center", color="green", fontsize=9, alpha=0.7)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer Index", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.set_xticks(range(0, num_layers, 2))
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "layer_profile_frob_er.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 8. 区域均值统计表 ─────────────────────────────────────
def print_zone_stats(frob, er, num_layers):
    zones = {
        "early (0-13)" : list(range(0, 14)),
        "middle (14-22)": list(range(14, 23)),
        "late (23-27)" : list(range(23, 28)),
    }

    stats = {}
    for zone, layers in zones.items():
        frob_vals = [frob[l][m] for l in layers for m in ALL_MODULES
                     if not np.isnan(frob[l].get(m, np.nan))]
        er_vals   = [er[l][m]   for l in layers for m in ALL_MODULES
                     if not np.isnan(er[l].get(m, np.nan))]
        stats[zone] = {
            "frob_mean": np.mean(frob_vals),
            "er_mean":   np.mean(er_vals),
        }

    print("\n── 区域统计 ─────────────────────────────────────────")
    print(f"{'区域':<18} {'Frob均值':>12} {'有效秩均值':>14} {'解读'}")
    print("-" * 65)
    for zone, s in stats.items():
        if s["frob_mean"] == max(v["frob_mean"] for v in stats.values()):
            frob_tag = "← 最大"
        else:
            frob_tag = ""
        if s["er_mean"] == min(v["er_mean"] for v in stats.values()):
            er_tag = "← 最小（最集中）"
        else:
            er_tag = ""
        print(f"{zone:<18} {s['frob_mean']:>12.6f}{frob_tag:8} "
              f"{s['er_mean']:>10.4f} {er_tag}")

    # 保存
    out = os.path.join(OUTPUT_DIR, "zone_stats.json")
    with open(out, "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in stats.items()}, f, indent=2)
    print(f"\n[保存] {out}")

    return stats

# ── 9. 关键论点验证 ───────────────────────────────────────
def verify_hypothesis(stats):
    print("\n── 假说验证 ─────────────────────────────────────────")

    mid_frob = stats["middle (14-22)"]["frob_mean"]
    early_frob = stats["early (0-13)"]["frob_mean"]
    mid_er   = stats["middle (14-22)"]["er_mean"]
    early_er = stats["early (0-13)"]["er_mean"]

    h1 = mid_frob > early_frob
    h2 = mid_er < early_er

    print(f"假说1: 中间层Frobenius范数 > 浅层  →  {'✓ 成立' if h1 else '✗ 不成立'}")
    print(f"       中间层={mid_frob:.6f}, 浅层={early_frob:.6f}")
    print(f"假说2: 中间层有效秩 < 浅层（更集中）→  {'✓ 成立' if h2 else '✗ 不成立'}")
    print(f"       中间层={mid_er:.4f}, 浅层={early_er:.4f}")

    if h1 and h2:
        print("\n→ 两个假说均成立：中间层权重变化大但方向集中，")
        print("  支持「中间层吸收源任务特定模式，浅层做通用调整」的解释。")
    elif h1 and not h2:
        print("\n→ 假说1成立，假说2不成立：中间层变化大且分散，")
        print("  需要重新审视解释框架。")
    else:
        print("\n→ 结果不支持预设假说，需要重新解释。")

# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  LoRA Heatmap v2: Frobenius + Effective Rank")
    print("=" * 55)

    weights = load_adapter_weights(LORA_PATH)
    frob, er, num_layers = compute_metrics(weights)

    print("\n[绘图] 双指标热力图...")
    plot_dual_heatmap(frob, er, num_layers)

    print("[绘图] 逐层折线图...")
    plot_layer_profile(frob, er, num_layers)

    stats = print_zone_stats(frob, er, num_layers)
    verify_hypothesis(stats)

    print(f"\n✓ 完成！输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()