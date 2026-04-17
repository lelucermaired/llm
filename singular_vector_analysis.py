"""
singular_vector_analysis.py

奇异向量旋转分析：验证SFT导致OOD下降的谱机制
理论依据：Jin et al. (2025) arXiv:2509.12235
  SFT不改变奇异值大小，但旋转奇异向量方向
  奇异向量旋转 → 预训练泛化子空间被破坏 → OOD下降

分析内容：
1. 逐层奇异值对比（base vs v2）：验证奇异值几乎不变
2. 逐层奇异向量cosine similarity（base vs v2）：量化旋转程度
3. top-k vs bottom-k奇异向量的差异：是否顶部方向旋转更大
4. 不同模块（q_proj/v_proj/mlp）的旋转差异

对比：base vs v2(CE) vs cot_detailed
"""

import os, json, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "models": {
        "v2":           "./archive/checkpoints/qwen-gomoku-real/final_model",
        "cot_detailed": "./checkpoints/qwen-gomoku-cot-detailed/final_model",
    },
    "output_dir": "./results/singular_vector_analysis",
    "n_layers": 28,
    "top_k": 32,   # 分析top-k个奇异向量
    # 分析的目标模块
    "target_modules": ["self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj"],
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def load_model_weights(adapter_path, base_path):
    """加载模型并提取权重（不需要推理，直接读权重）"""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path, quantization_config=bnb,
        device_map="auto", local_files_only=True,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
        # 合并LoRA权重到base（用于SVD分析）
        model = model.merge_and_unload()
    else:
        model = base
    return model


def extract_weight_matrix(model, layer_idx, module_name):
    """提取指定层指定模块的权重矩阵"""
    try:
        layer = model.model.layers[layer_idx]
        # 按模块名逐级访问
        parts = module_name.split('.')
        m = layer
        for p in parts:
            m = getattr(m, p)
        # 返回float32权重
        return m.weight.data.float().cpu()
    except AttributeError:
        return None


def compute_singular_alignment(W1, W2, top_k=32):
    """
    计算两个权重矩阵的奇异向量对齐程度
    返回：
      sv_ratio: 奇异值比值（W2/W1），接近1说明奇异值不变
      top_alignment: top-k左奇异向量的平均cosine similarity
      bottom_alignment: bottom-k左奇异向量的平均cosine similarity
    """
    # SVD分解（只取U矩阵，左奇异向量）
    try:
        U1, S1, V1 = torch.linalg.svd(W1, full_matrices=False)
        U2, S2, V2 = torch.linalg.svd(W2, full_matrices=False)
    except Exception:
        return None

    k = min(top_k, U1.shape[1], U2.shape[1])

    # 奇异值比值（W2的奇异值/W1的奇异值）
    sv_ratio = (S2[:k] / (S1[:k] + 1e-8)).mean().item()

    # top-k奇异向量对齐：U1[:,:k]^T @ U2[:,:k]的对角线绝对值均值
    # 完全对齐=1，完全旋转=0
    overlap_top = torch.abs(U1[:, :k].T @ U2[:, :k])
    top_alignment = overlap_top.diag().mean().item()

    # bottom-k奇异向量（最小的k个）
    n = U1.shape[1]
    bottom_k = min(top_k, n)
    overlap_bot = torch.abs(U1[:, -bottom_k:].T @ U2[:, -bottom_k:])
    bottom_alignment = overlap_bot.diag().mean().item()

    return {
        "sv_ratio": sv_ratio,
        "top_alignment": top_alignment,
        "bottom_alignment": bottom_alignment,
        "sv_mean_base": S1[:k].mean().item(),
        "sv_mean_ft": S2[:k].mean().item(),
    }


def analyze_model(model_name, adapter_path):
    """分析一个模型相对base的奇异向量变化"""
    print(f"\n分析模型：{model_name}")

    # 加载base
    print("  加载base模型...")
    base_model = load_model_weights(None, CONFIG["base_model_path"])

    # 加载微调模型
    print(f"  加载{model_name}...")
    ft_model = load_model_weights(adapter_path, CONFIG["base_model_path"])

    results = {}
    for module in CONFIG["target_modules"]:
        results[module] = {}
        layer_results = []

        for layer_idx in tqdm(range(CONFIG["n_layers"]),
                              desc=f"  {module}", leave=False):
            W_base = extract_weight_matrix(base_model, layer_idx, module)
            W_ft   = extract_weight_matrix(ft_model,   layer_idx, module)

            if W_base is None or W_ft is None:
                layer_results.append(None)
                continue

            alignment = compute_singular_alignment(
                W_base, W_ft, top_k=CONFIG["top_k"])
            layer_results.append(alignment)

        results[module] = layer_results

    del base_model, ft_model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return results


def plot_results(all_results, output_dir):
    """绘制逐层奇异向量对齐程度"""
    layers = list(range(CONFIG["n_layers"]))
    modules = CONFIG["target_modules"]
    model_names = list(all_results.keys())
    colors = {"v2": "#FF5722", "cot_detailed": "#2196F3"}

    fig, axes = plt.subplots(len(modules), 3, figsize=(15, 4*len(modules)))
    fig.suptitle("Singular Vector Analysis: base vs fine-tuned models\n"
                 "Top: singular value ratio | Mid: top-k alignment | Bot: bottom-k alignment",
                 fontsize=11)

    metric_keys = ["sv_ratio", "top_alignment", "bottom_alignment"]
    metric_labels = ["Singular Value Ratio (ft/base)",
                     "Top-k Singular Vector Alignment",
                     "Bottom-k Singular Vector Alignment"]

    for mi, module in enumerate(modules):
        short_name = module.split('.')[-1]
        for ki, (metric_key, metric_label) in enumerate(
                zip(metric_keys, metric_labels)):
            ax = axes[mi, ki]
            for model_name in model_names:
                vals = []
                for layer_res in all_results[model_name][module]:
                    if layer_res is not None:
                        vals.append(layer_res[metric_key])
                    else:
                        vals.append(None)
                vals_clean = [v if v is not None else float('nan') for v in vals]
                ax.plot(layers, vals_clean,
                        label=model_name,
                        color=colors.get(model_name, "gray"),
                        linewidth=1.5)

            ax.axhline(1.0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_title(f"{short_name} - {metric_label}", fontsize=9)
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_label.split('(')[0])
            ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "singular_vector_alignment.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表保存至: {save_path}")


def main():
    print("=" * 65)
    print("奇异向量旋转分析")
    print("验证：SFT保持奇异值 but 旋转奇异向量方向")
    print("对比：base vs v2 vs cot_detailed")
    print("=" * 65)

    cache_path = os.path.join(CONFIG["output_dir"], "sv_analysis.json")
    all_results = {}

    if os.path.exists(cache_path):
        print("[缓存] 读取已有分析结果")
        with open(cache_path, encoding="utf-8") as f:
            all_results_serializable = json.load(f)
        # 转回nested dict
        for model_name, mod_results in all_results_serializable.items():
            all_results[model_name] = {}
            for module, layer_list in mod_results.items():
                all_results[model_name][module] = layer_list
    else:
        for model_name, adapter_path in CONFIG["models"].items():
            all_results[model_name] = analyze_model(model_name, adapter_path)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # ==================== 数值汇总 ====================
    print("\n" + "=" * 65)
    print("奇异向量分析结果")
    print("=" * 65)

    for model_name, mod_results in all_results.items():
        print(f"\n【{model_name}】（相对base）")
        for module, layer_list in mod_results.items():
            valid = [r for r in layer_list if r is not None]
            if not valid:
                continue
            sv_ratios   = [r["sv_ratio"]        for r in valid]
            top_aligns  = [r["top_alignment"]   for r in valid]
            bot_aligns  = [r["bottom_alignment"] for r in valid]

            print(f"  {module.split('.')[-1]:<12}:"
                  f"  sv_ratio={np.mean(sv_ratios):.4f}±{np.std(sv_ratios):.4f}"
                  f"  top_align={np.mean(top_aligns):.4f}±{np.std(top_aligns):.4f}"
                  f"  bot_align={np.mean(bot_aligns):.4f}±{np.std(bot_aligns):.4f}")

    # ==================== 关键发现 ====================
    print("\n" + "=" * 65)
    print("关键发现")
    print("=" * 65)

    for model_name, mod_results in all_results.items():
        print(f"\n{model_name}:")
        for module, layer_list in mod_results.items():
            valid = [r for r in layer_list if r is not None]
            if not valid:
                continue
            sv_mean = np.mean([r["sv_ratio"] for r in valid])
            top_mean = np.mean([r["top_alignment"] for r in valid])

            sv_stable = abs(sv_mean - 1.0) < 0.05
            vec_rotated = top_mean < 0.95

            if sv_stable and vec_rotated:
                print(f"  {module.split('.')[-1]}: "
                      f"奇异值稳定(ratio={sv_mean:.4f}) + "
                      f"奇异向量旋转(top_align={top_mean:.4f}) "
                      f"→ 符合SFT Generalization Paradox理论")
            elif sv_stable:
                print(f"  {module.split('.')[-1]}: "
                      f"奇异值稳定(ratio={sv_mean:.4f}) + "
                      f"奇异向量保持(top_align={top_mean:.4f}) "
                      f"→ SFT几乎未改变该模块")
            else:
                print(f"  {module.split('.')[-1]}: "
                      f"奇异值变化(ratio={sv_mean:.4f}) "
                      f"top_align={top_mean:.4f}")

    # 绘图
    plot_results(all_results, CONFIG["output_dir"])

    print("\n✅ 分析完成")
    print(f"数据: {cache_path}")
    print(f"图表: {CONFIG['output_dir']}/singular_vector_alignment.png")


if __name__ == "__main__":
    main()