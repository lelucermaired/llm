"""
LoRA Attention vs MLP Heatmap Analysis
计算每层每模块的 ||B @ A||_F，生成热力图
不需要加载base模型，直接读adapter权重
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
LORA_PATH = "./checkpoints/qwen-gomoku-maxlora/final_model"
OUTPUT_DIR = "./results/heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 模块分组 ──────────────────────────────────────────────
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
ALL_MODULES = ATTN_MODULES + MLP_MODULES


# ── 1. 加载adapter权重 ────────────────────────────────────
def load_adapter_weights(lora_path):
    """自动识别safetensors或pytorch_model.bin"""
    path = Path(lora_path)

    sf_files = list(path.glob("adapter_model*.safetensors"))
    bin_files = list(path.glob("adapter_model*.bin"))

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
        raise FileNotFoundError(f"在 {lora_path} 找不到adapter权重文件")

    print(f"[加载] 共 {len(weights)} 个张量")
    return weights


# ── 2. 计算每层每模块的 ||BA||_F ──────────────────────────
def compute_lora_norms(weights):
    """
    返回两个dict：
      norms[layer][module] = ||BA||_F (标量)
      norms_normalized[layer][module] = ||BA||_F / sqrt(d_out * d_in) (归一化)
    """
    # 先找出有多少层
    layers = set()
    for key in weights:
        if "lora_A" in key and "layers." in key:
            parts = key.split("layers.")
            layer_num = int(parts[1].split(".")[0])
            layers.add(layer_num)

    num_layers = max(layers) + 1
    print(f"[分析] 检测到 {num_layers} 层")

    norms = {l: {} for l in range(num_layers)}
    norms_norm = {l: {} for l in range(num_layers)}

    for layer in range(num_layers):
        for module in ALL_MODULES:
            # 构建key（兼容不同命名风格）
            if module in ATTN_MODULES:
                key_a = f"base_model.model.model.layers.{layer}.self_attn.{module}.lora_A.weight"
                key_b = f"base_model.model.model.layers.{layer}.self_attn.{module}.lora_B.weight"
            else:
                key_a = f"base_model.model.model.layers.{layer}.mlp.{module}.lora_A.weight"
                key_b = f"base_model.model.model.layers.{layer}.mlp.{module}.lora_B.weight"

            if key_a not in weights or key_b not in weights:
                # 尝试不带 base_model.model 前缀
                key_a2 = key_a.replace("base_model.model.", "")
                key_b2 = key_b.replace("base_model.model.", "")
                if key_a2 in weights:
                    key_a, key_b = key_a2, key_b2
                else:
                    norms[layer][module] = np.nan
                    norms_norm[layer][module] = np.nan
                    continue

            A = weights[key_a].float()  # (r, d_in)
            B = weights[key_b].float()  # (d_out, r)

            # ΔW = B @ A，形状 (d_out, d_in)
            delta_W = B @ A

            frob = torch.norm(delta_W, p="fro").item()
            d_out, d_in = delta_W.shape
            frob_normalized = frob / np.sqrt(d_out * d_in)

            norms[layer][module] = frob
            norms_norm[layer][module] = frob_normalized

    return norms, norms_norm, num_layers


# ── 3. 转为矩阵 ───────────────────────────────────────────
def to_matrix(norms_dict, num_layers, modules):
    mat = np.zeros((len(modules), num_layers))
    for j, layer in enumerate(range(num_layers)):
        for i, mod in enumerate(modules):
            v = norms_dict[layer].get(mod, np.nan)
            mat[i, j] = v if v is not None else np.nan
    return mat


# ── 4. 绘图 ───────────────────────────────────────────────
def plot_heatmaps(norms, norms_norm, num_layers):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("LoRA Weight Update Analysis: Qwen2.5-7B (r=64, Gomoku)\n"
                 r"Color: $\||\Delta W\||_F = \||BA\||_F$",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    layer_ticks = list(range(0, num_layers, 4))

    # ── 图1：全模块原始范数 ──
    ax1 = fig.add_subplot(gs[0, :])
    mat_all = to_matrix(norms_norm, num_layers, ALL_MODULES)

    sns.heatmap(
        mat_all,
        ax=ax1,
        cmap="YlOrRd",
        xticklabels=[str(i) if i in layer_ticks else "" for i in range(num_layers)],
        yticklabels=ALL_MODULES,
        cbar_kws={"label": r"$\||BA\||_F / \sqrt{d_{out} \cdot d_{in}}$", "shrink": 0.8},
        linewidths=0.3,
        linecolor="white",
    )
    ax1.set_title("All Modules — Normalized Frobenius Norm per Layer", fontsize=12, pad=8)
    ax1.set_xlabel("Layer Index", fontsize=10)
    ax1.set_ylabel("")

    # 分隔线：attention vs MLP
    ax1.axhline(y=len(ATTN_MODULES), color="royalblue", linewidth=2.5, linestyle="--")
    ax1.text(num_layers + 0.3, len(ATTN_MODULES) / 2, "Attn",
             va="center", color="royalblue", fontsize=10, fontweight="bold")
    ax1.text(num_layers + 0.3, len(ATTN_MODULES) + len(MLP_MODULES) / 2, "MLP",
             va="center", color="firebrick", fontsize=10, fontweight="bold")

    # ── 图2：Attention模块 ──
    ax2 = fig.add_subplot(gs[1, 0])
    mat_attn = to_matrix(norms_norm, num_layers, ATTN_MODULES)

    vmax = np.nanmax([to_matrix(norms_norm, num_layers, ALL_MODULES)])
    sns.heatmap(
        mat_attn,
        ax=ax2,
        cmap="Blues",
        xticklabels=[str(i) if i in layer_ticks else "" for i in range(num_layers)],
        yticklabels=ATTN_MODULES,
        vmin=0, vmax=vmax,
        cbar_kws={"label": "Norm", "shrink": 0.8},
        linewidths=0.3,
        linecolor="white",
    )
    ax2.set_title("Attention Modules", fontsize=12, pad=8)
    ax2.set_xlabel("Layer Index", fontsize=10)

    # ── 图3：MLP模块 ──
    ax3 = fig.add_subplot(gs[1, 1])
    mat_mlp = to_matrix(norms_norm, num_layers, MLP_MODULES)

    sns.heatmap(
        mat_mlp,
        ax=ax3,
        cmap="Reds",
        xticklabels=[str(i) if i in layer_ticks else "" for i in range(num_layers)],
        yticklabels=MLP_MODULES,
        vmin=0, vmax=vmax,
        cbar_kws={"label": "Norm", "shrink": 0.8},
        linewidths=0.3,
        linecolor="white",
    )
    ax3.set_title("MLP Modules", fontsize=12, pad=8)
    ax3.set_xlabel("Layer Index", fontsize=10)

    out_path = os.path.join(OUTPUT_DIR, "lora_heatmap_all.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out_path}")
    plt.close()


def plot_layer_profile(norms_norm, num_layers):
    """额外：每层的attn_sum vs mlp_sum折线图，直观看哪些层贡献更大"""
    attn_sums = []
    mlp_sums = []

    for layer in range(num_layers):
        a = np.nanmean([norms_norm[layer].get(m, np.nan) for m in ATTN_MODULES])
        m = np.nanmean([norms_norm[layer].get(m, np.nan) for m in MLP_MODULES])
        attn_sums.append(a)
        mlp_sums.append(m)

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(num_layers)
    ax.plot(x, attn_sums, "o-", color="royalblue", label="Attention (mean)", linewidth=2, markersize=5)
    ax.plot(x, mlp_sums, "s-", color="firebrick", label="MLP (mean)", linewidth=2, markersize=5)
    ax.fill_between(x, attn_sums, alpha=0.15, color="royalblue")
    ax.fill_between(x, mlp_sums, alpha=0.15, color="firebrick")

    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel(r"Mean $\||BA\||_F / \sqrt{d \cdot d}$", fontsize=11)
    ax.set_title("Per-Layer LoRA Update Magnitude: Attention vs MLP\n(Qwen2.5-7B, r=64, Gomoku)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xticks(range(0, num_layers, 2))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    out_path = os.path.join(OUTPUT_DIR, "lora_layer_profile.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out_path}")
    plt.close()


# ── 5. 保存数值结果 ───────────────────────────────────────
def save_stats(norms, norms_norm, num_layers):
    summary = {
        "per_module_mean_normalized": {},
        "top5_layers_by_total_norm": [],
    }

    all_norm_mat = to_matrix(norms_norm, num_layers, ALL_MODULES)

    for i, mod in enumerate(ALL_MODULES):
        summary["per_module_mean_normalized"][mod] = float(np.nanmean(all_norm_mat[i]))

    # 每层总norm（所有模块求和）
    layer_totals = np.nansum(all_norm_mat, axis=0)
    top5 = np.argsort(layer_totals)[::-1][:5]
    summary["top5_layers_by_total_norm"] = [
        {"layer": int(l), "total_norm": float(layer_totals[l])} for l in top5
    ]

    out_path = os.path.join(OUTPUT_DIR, "lora_norm_stats.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[保存] {out_path}")

    # 打印摘要
    print("\n── 模块平均归一化范数 ──────────────────")
    for mod, val in sorted(summary["per_module_mean_normalized"].items(),
                           key=lambda x: -x[1]):
        bar = "█" * int(val * 500)
        print(f"  {mod:12s}  {val:.6f}  {bar}")

    print("\n── 更新幅度最大的5层 ───────────────────")
    for entry in summary["top5_layers_by_total_norm"]:
        print(f"  Layer {entry['layer']:2d}  total_norm={entry['total_norm']:.4f}")


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  LoRA Heatmap Analysis")
    print("  Model: Qwen2.5-7B, r=64, Gomoku MaxLoRA")
    print("=" * 55)

    weights = load_adapter_weights(LORA_PATH)
    norms, norms_norm, num_layers = compute_lora_norms(weights)

    print("\n[绘图] 生成热力图...")
    plot_heatmaps(norms, norms_norm, num_layers)
    plot_layer_profile(norms_norm, num_layers)
    save_stats(norms, norms_norm, num_layers)

    print(f"\n✓ 完成！输出目录：{OUTPUT_DIR}")