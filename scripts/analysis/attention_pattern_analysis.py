"""
attention_pattern_analysis.py

注意力模式分析（Attention Pattern）
核心问题：LoRA主要改变了q_proj（71%更新量），attention的查询方向变了吗？

对比：
  - 五子棋输入：微调模型是否更关注棋盘位置token？
  - 数学输入：attention pattern是否几乎没变？

预期结论：
  LoRA改变了模型"看什么"（attention pattern），
  但没有改变模型"想输出什么"（logit lens），
  这解释了为什么五子棋任务性能提升，但数学推理没有迁移。

用法：
    python attention_pattern_analysis.py
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "finetuned_model_path": "./checkpoints/qwen-gomoku-v2/final_model",
    "output_dir": "./attention_results",
    "num_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,
    "num_layers": 28,
    # 分析哪些层（选浅/中/深各几层）
    "target_layers": [0, 4, 8, 12, 16, 20, 24, 27],
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 探针 ====================

MATH_PROMPT = """Solve this math problem step by step.
Question: If a train travels at 60 miles per hour for 2.5 hours, how far does it travel?
Answer:"""

GOMOKU_PROMPT = """You are a Gomoku master. Rules: players alternate placing stones, first to connect five in a row wins.
Analyze the board:
Board state (B=Black, W=White, .=Empty):
  G H I J K L M
7 . . . . . . .
8 . . B B B . .
9 . . W W . . .
10 . . . . . . .
It is Black's turn. What is the optimal move?"""


# ==================== 提取Attention ====================

def get_attention_weights(model, tokenizer, prompt, target_layers, device):
    """
    提取指定层的attention权重矩阵
    返回：{layer_idx: attention_matrix (num_heads, seq_len, seq_len)}
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
    input_ids = inputs["input_ids"].to(device)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]

    attention_weights = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output[1] 是attention weights，shape: (batch, num_heads, seq, seq)
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn = output[1].detach().float().cpu()
                attention_weights[layer_idx] = attn[0]  # (num_heads, seq, seq)
        return hook_fn

    # 兼容base model和PeftModel
    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.base_model.model.model.layers

    for layer_idx in target_layers:
        h = layers[layer_idx].self_attn.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids, output_attentions=True)

    for h in hooks:
        h.remove()

    return attention_weights, tokens


def compute_attention_diff(base_attn, ft_attn):
    """
    计算两个attention矩阵的差异
    base_attn, ft_attn: (num_heads, seq_len, seq_len)
    返回：逐head的L2差异、平均差异矩阵
    """
    diff = ft_attn - base_attn  # (num_heads, seq, seq)
    head_diffs = diff.norm(dim=(-2, -1)).numpy()  # (num_heads,)
    avg_diff = diff.abs().mean(dim=0).numpy()     # (seq, seq)
    avg_base = base_attn.mean(dim=0).numpy()
    avg_ft = ft_attn.mean(dim=0).numpy()
    return head_diffs, avg_diff, avg_base, avg_ft


def load_model(path, base_path, is_base=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    if is_base:
        base.eval()
        return base
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    return model


# ==================== 可视化 ====================

def plot_attention_comparison(base_attn, ft_attn, tokens, layer_idx,
                               domain, save_dir, max_tokens=30):
    """绘制基础模型 vs 微调模型的attention热力图对比"""
    seq_len = min(len(tokens), max_tokens)
    tokens_show = [t.replace('Ġ', ' ').replace('Ċ', '\\n')[:6] for t in tokens[:seq_len]]

    base_avg = base_attn.mean(dim=0).numpy()[:seq_len, :seq_len]
    ft_avg = ft_attn.mean(dim=0).numpy()[:seq_len, :seq_len]
    diff = (ft_avg - base_avg)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Attention Pattern - Layer {layer_idx} - {domain}', fontsize=13)

    # 基础模型
    im0 = axes[0].imshow(base_avg, cmap='Blues', aspect='auto', vmin=0, vmax=base_avg.max())
    axes[0].set_title('Base Model')
    axes[0].set_xticks(range(seq_len))
    axes[0].set_yticks(range(seq_len))
    axes[0].set_xticklabels(tokens_show, rotation=90, fontsize=6)
    axes[0].set_yticklabels(tokens_show, fontsize=6)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # 微调模型
    im1 = axes[1].imshow(ft_avg, cmap='Blues', aspect='auto', vmin=0, vmax=ft_avg.max())
    axes[1].set_title('Finetuned Model')
    axes[1].set_xticks(range(seq_len))
    axes[1].set_yticks(range(seq_len))
    axes[1].set_xticklabels(tokens_show, rotation=90, fontsize=6)
    axes[1].set_yticklabels(tokens_show, fontsize=6)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # 差异图（微调-基础）
    vmax = max(abs(diff.min()), abs(diff.max())) + 1e-8
    im2 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Difference (FT - Base)')
    axes[2].set_xticks(range(seq_len))
    axes[2].set_yticks(range(seq_len))
    axes[2].set_xticklabels(tokens_show, rotation=90, fontsize=6)
    axes[2].set_yticklabels(tokens_show, fontsize=6)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'attn_layer{layer_idx:02d}_{domain}.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    return fname


def plot_layer_diff_summary(math_layer_diffs, gomoku_layer_diffs, save_dir):
    """绘制各层attention差异汇总图：数学 vs 五子棋"""
    layers = sorted(math_layer_diffs.keys())
    math_vals = [math_layer_diffs[l] for l in layers]
    gomoku_vals = [gomoku_layer_diffs[l] for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, math_vals, width, label='Math Input', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, gomoku_vals, width, label='Gomoku Input', color='coral', alpha=0.8)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Difference (avg L2 norm across heads)')
    ax.set_title('Attention Pattern Change: Base vs Finetuned\nMath Input vs Gomoku Input')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, 'attn_diff_summary.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ 汇总图已保存: {fname}")
    return fname


def plot_head_diff_heatmap(math_head_diffs, gomoku_head_diffs, target_layers, save_dir):
    """绘制各层各head的attention差异热力图"""
    n_layers = len(target_layers)
    n_heads = len(list(math_head_diffs.values())[0])

    math_matrix = np.array([math_head_diffs[l] for l in target_layers])   # (n_layers, n_heads)
    gomoku_matrix = np.array([gomoku_head_diffs[l] for l in target_layers])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    vmax = max(math_matrix.max(), gomoku_matrix.max())

    im0 = axes[0].imshow(math_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
    axes[0].set_title('Math Input: Per-Head Attention Diff')
    axes[0].set_xlabel('Head Index')
    axes[0].set_ylabel('Layer')
    axes[0].set_yticks(range(n_layers))
    axes[0].set_yticklabels([f'L{l}' for l in target_layers])
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(gomoku_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
    axes[1].set_title('Gomoku Input: Per-Head Attention Diff')
    axes[1].set_xlabel('Head Index')
    axes[1].set_ylabel('Layer')
    axes[1].set_yticks(range(n_layers))
    axes[1].set_yticklabels([f'L{l}' for l in target_layers])
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle('Per-Head Attention Pattern Change (Base vs Finetuned)', fontsize=13)
    plt.tight_layout()
    fname = os.path.join(save_dir, 'attn_head_heatmap.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ Head热力图已保存: {fname}")
    return fname


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("Attention Pattern 分析")
    print("LoRA改变了模型'看什么'（attention）但没改变'想什么'（logit）？")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_layers = CONFIG["target_layers"]

    # ========== 基础模型 ==========
    print("\n[1/2] 基础模型提取Attention...")
    base_model = load_model(None, CONFIG["base_model_path"], is_base=True)

    base_math_attn, math_tokens = get_attention_weights(
        base_model, tokenizer, MATH_PROMPT, target_layers, device)
    base_gomoku_attn, gomoku_tokens = get_attention_weights(
        base_model, tokenizer, GOMOKU_PROMPT, target_layers, device)
    print(f"  数学输入序列长度: {len(math_tokens)} tokens")
    print(f"  五子棋输入序列长度: {len(gomoku_tokens)} tokens")

    del base_model
    torch.cuda.empty_cache()

    # ========== 微调模型 ==========
    print("\n[2/2] 微调模型提取Attention...")
    ft_model = load_model(CONFIG["finetuned_model_path"], CONFIG["base_model_path"])

    ft_math_attn, _ = get_attention_weights(
        ft_model, tokenizer, MATH_PROMPT, target_layers, device)
    ft_gomoku_attn, _ = get_attention_weights(
        ft_model, tokenizer, GOMOKU_PROMPT, target_layers, device)

    del ft_model
    torch.cuda.empty_cache()

    # ========== 计算差异 ==========
    print("\n计算Attention差异...")

    math_layer_diffs = {}     # layer -> avg L2 diff across heads
    gomoku_layer_diffs = {}
    math_head_diffs = {}      # layer -> per-head L2 diff
    gomoku_head_diffs = {}

    print(f"\n{'层':>4}  {'数学Attn差异':>14}  {'五子棋Attn差异':>14}  {'倍数':>8}")
    print("-" * 50)

    for layer_idx in target_layers:
        if layer_idx not in base_math_attn or layer_idx not in ft_math_attn:
            continue

        # 数学
        math_head_diff, math_avg_diff, math_base_avg, math_ft_avg = compute_attention_diff(
            base_math_attn[layer_idx], ft_math_attn[layer_idx])
        math_layer_diffs[layer_idx] = float(np.mean(math_head_diff))
        math_head_diffs[layer_idx] = math_head_diff

        # 五子棋
        gomoku_head_diff, gomoku_avg_diff, gomoku_base_avg, gomoku_ft_avg = compute_attention_diff(
            base_gomoku_attn[layer_idx], ft_gomoku_attn[layer_idx])
        gomoku_layer_diffs[layer_idx] = float(np.mean(gomoku_head_diff))
        gomoku_head_diffs[layer_idx] = gomoku_head_diff

        ratio = gomoku_layer_diffs[layer_idx] / (math_layer_diffs[layer_idx] + 1e-8)
        flag = " ◀ 五子棋更大" if ratio > 1.5 else ""
        print(f"  {layer_idx:>2}  {math_layer_diffs[layer_idx]:>14.4f}  "
              f"{gomoku_layer_diffs[layer_idx]:>14.4f}  {ratio:>7.2f}x{flag}")

        # 保存热力图（选几个代表层）
        if layer_idx in [0, 8, 16, 27]:
            plot_attention_comparison(
                base_math_attn[layer_idx], ft_math_attn[layer_idx],
                math_tokens, layer_idx, "Math", CONFIG["output_dir"])
            plot_attention_comparison(
                base_gomoku_attn[layer_idx], ft_gomoku_attn[layer_idx],
                gomoku_tokens, layer_idx, "Gomoku", CONFIG["output_dir"])

    # ========== 汇总 ==========
    avg_math = np.mean(list(math_layer_diffs.values()))
    avg_gomoku = np.mean(list(gomoku_layer_diffs.values()))
    overall_ratio = avg_gomoku / (avg_math + 1e-8)

    print(f"\n数学输入平均Attention差异：   {avg_math:.4f}")
    print(f"五子棋输入平均Attention差异：  {avg_gomoku:.4f}")
    print(f"五子棋是数学的：               {overall_ratio:.2f}x")

    # ========== 生成汇总图 ==========
    print("\n生成可视化图表...")
    plot_layer_diff_summary(math_layer_diffs, gomoku_layer_diffs, CONFIG["output_dir"])
    plot_head_diff_heatmap(math_head_diffs, gomoku_head_diffs, target_layers, CONFIG["output_dir"])

    # ========== 核心结论 ==========
    print("\n" + "=" * 60)
    print("核心结论")
    print("=" * 60)

    if overall_ratio > 2.0:
        print(f"\n→ ✅ 强领域特异性（{overall_ratio:.2f}x）：")
        print("  五子棋输入的attention pattern变化远大于数学输入")
        print("  LoRA确实改变了模型'看什么'，且改变具有领域特异性")
        print("  但这种attention变化未能传播到logit层（Logit Lens一致率98%）")
        print("  解释：LoRA改变了attention查询方向，但v_proj和输出层未充分适配")
    elif overall_ratio > 1.3:
        print(f"\n→ ⚠ 弱领域特异性（{overall_ratio:.2f}x）：")
        print("  五子棋输入attention变化略大于数学输入，但差距不显著")
        print("  结合Logit Lens和CKA结论，LoRA的整体影响规模过小")
    else:
        print(f"\n→ ℹ 无明显领域特异性（{overall_ratio:.2f}x）：")
        print("  两类输入的attention变化程度相近")
        print("  LoRA r=8的低秩约束限制了对任何类型输入的影响规模")

    # ========== 保存数值结果 ==========
    save_path = os.path.join(CONFIG["output_dir"], "attention_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "math_layer_diffs": math_layer_diffs,
            "gomoku_layer_diffs": gomoku_layer_diffs,
            "avg_math": avg_math,
            "avg_gomoku": avg_gomoku,
            "overall_ratio": overall_ratio,
            "math_tokens": math_tokens,
            "gomoku_tokens": gomoku_tokens,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 数值结果已保存至: {save_path}")
    print(f"✅ 热力图已保存至: {CONFIG['output_dir']}/")


if __name__ == "__main__":
    main()