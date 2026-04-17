"""
lora_weight_analysis.py

分析LoRA权重更新量（Frobenius范数）
探究：LoRA的改变集中在哪些层、哪些模块？
并与Logit Lens的不一致层（8-16）进行对比

用法：
    python lora_weight_analysis.py
"""

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from collections import defaultdict

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "finetuned_model_path": "./checkpoints/qwen-gomoku-v2/final_model",
    "output_dir": "./lora_weight_results",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Logit Lens中出现不一致的层（供对比）
LOGIT_LENS_INCONSISTENT_LAYERS = [8, 12, 14, 15, 16]


def main():
    print("=" * 60)
    print("LoRA 权重更新量分析")
    print("探究：更新量分布与Logit Lens不一致层是否吻合？")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("\n加载微调模型...")
    base = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    ft_model = PeftModel.from_pretrained(base, CONFIG["finetuned_model_path"])
    ft_model.eval()

    # ==================== 提取LoRA参数 ====================
    print("\n提取LoRA权重更新量（ΔW = B × A）...")

    # 按层、按模块统计
    layer_module_norms = defaultdict(dict)   # layer_idx -> {module: norm}
    layer_total_norms = defaultdict(float)   # layer_idx -> total norm
    module_total_norms = defaultdict(float)  # module_name -> total norm

    all_lora_params = {}
    for name, param in ft_model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            all_lora_params[name] = param.detach().float()

    # 计算每对(lora_A, lora_B)的ΔW范数
    # 名称格式：base_model.model.model.layers.{i}.self_attn.{q/k/v/o}_proj.lora_{A/B}.default.weight
    processed = set()
    delta_w_records = []

    for name, param in all_lora_params.items():
        if 'lora_A' not in name:
            continue

        base_name = name.replace('lora_A.default', 'lora_B.default')
        if base_name not in all_lora_params:
            continue

        if name in processed:
            continue
        processed.add(name)
        processed.add(base_name)

        lora_A = param                          # (r, in_features)
        lora_B = all_lora_params[base_name]     # (out_features, r)
        delta_W = lora_B @ lora_A               # (out_features, in_features)
        frobenius_norm = delta_W.norm(p='fro').item()

        # 解析层号和模块名
        parts = name.split('.')
        layer_idx = None
        module_name = None
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
            if p in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                     'gate_proj', 'up_proj', 'down_proj']:
                module_name = p

        if layer_idx is not None and module_name is not None:
            layer_module_norms[layer_idx][module_name] = frobenius_norm
            layer_total_norms[layer_idx] += frobenius_norm
            module_total_norms[module_name] += frobenius_norm
            delta_w_records.append({
                "layer": layer_idx,
                "module": module_name,
                "frobenius_norm": frobenius_norm,
                "rank": lora_A.shape[0],
            })

    # ==================== 打印结果 ====================

    # 1. 按层汇总
    print("\n" + "=" * 60)
    print("逐层LoRA更新量（Frobenius范数之和）")
    print("=" * 60)
    sorted_layers = sorted(layer_total_norms.keys())
    max_norm = max(layer_total_norms.values())

    print(f"{'层':>4}  {'总范数':>10}  {'相对强度':>8}  {'是否Logit不一致层':<18}  模块分布")
    print("-" * 80)
    for layer_idx in sorted_layers:
        total = layer_total_norms[layer_idx]
        relative = total / max_norm
        bar = "█" * int(relative * 20)
        is_inconsistent = "⚠ Logit不一致" if layer_idx in LOGIT_LENS_INCONSISTENT_LAYERS else ""
        modules = layer_module_norms[layer_idx]
        module_str = "  ".join([f"{m}:{v:.2f}" for m, v in sorted(modules.items())])
        print(f"  {layer_idx:>2}  {total:>10.4f}  {bar:<20}  {is_inconsistent:<18}  {module_str}")

    # 2. 按模块汇总
    print("\n" + "=" * 60)
    print("各模块LoRA更新量汇总（跨所有层）")
    print("=" * 60)
    sorted_modules = sorted(module_total_norms.items(), key=lambda x: x[1], reverse=True)
    for module, norm in sorted_modules:
        print(f"  {module:<12}  {norm:>10.4f}")

    # 3. 更新量最大的Top5层
    top5_layers = sorted(layer_total_norms.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n" + "=" * 60)
    print("更新量最大的5层")
    print("=" * 60)
    for layer_idx, norm in top5_layers:
        is_inconsistent = "← Logit Lens不一致层" if layer_idx in LOGIT_LENS_INCONSISTENT_LAYERS else ""
        print(f"  Layer {layer_idx:>2}：{norm:.4f}  {is_inconsistent}")

    # 4. 更新量与Logit Lens不一致的相关性分析
    print("\n" + "=" * 60)
    print("更新量 vs Logit Lens 不一致层 对比分析")
    print("=" * 60)
    inconsistent_norms = [layer_total_norms[l] for l in LOGIT_LENS_INCONSISTENT_LAYERS
                          if l in layer_total_norms]
    consistent_norms = [layer_total_norms[l] for l in sorted_layers
                        if l not in LOGIT_LENS_INCONSISTENT_LAYERS]

    avg_inconsistent = np.mean(inconsistent_norms) if inconsistent_norms else 0
    avg_consistent = np.mean(consistent_norms) if consistent_norms else 0

    print(f"Logit不一致层（{LOGIT_LENS_INCONSISTENT_LAYERS}）平均更新量：{avg_inconsistent:.4f}")
    print(f"Logit一致层平均更新量：                              {avg_consistent:.4f}")
    print(f"比值（不一致/一致）：                                {avg_inconsistent/avg_consistent:.3f}x")

    if avg_inconsistent > avg_consistent * 1.2:
        print("\n→ 结论：更新量较大的层与Logit Lens出现不一致的层高度吻合")
        print("  说明LoRA的权重更新确实在这些层引起了token预测的轻微偏移")
        print("  但偏移量极小（Top1一致率仍>90%），被深层网络修正")
    elif avg_inconsistent > avg_consistent * 0.8:
        print("\n→ 结论：更新量分布与Logit Lens不一致层无明显相关性")
        print("  说明LoRA更新量的大小不直接决定token预测的改变")
        print("  即使某层更新量较大，也未必影响该层的词表预测")
    else:
        print("\n→ 结论：Logit不一致层的更新量反而低于平均水平")
        print("  不一致可能来自上下层的联合效应，而非单层更新量决定")

    # 5. 浅层vs深层更新量对比
    n_layers = len(sorted_layers)
    mid = n_layers // 2
    shallow_norm = np.mean([layer_total_norms[l] for l in sorted_layers[:mid]])
    deep_norm = np.mean([layer_total_norms[l] for l in sorted_layers[mid:]])
    print(f"\n浅层（0-{mid-1}）平均更新量：{shallow_norm:.4f}")
    print(f"深层（{mid}-{sorted_layers[-1]}）平均更新量：{deep_norm:.4f}")
    if deep_norm > shallow_norm * 1.1:
        print("→ 更新量集中在深层，与Logit Lens深层一致率=100%形成对比")
        print("  深层更新量大但预测不变，说明深层LoRA改变的是表示方向而非输出预测")
    elif shallow_norm > deep_norm * 1.1:
        print("→ 更新量集中在浅层")
    else:
        print("→ 更新量在各层分布较均匀")

    # ==================== 保存结果 ====================
    save_path = os.path.join(CONFIG["output_dir"], "lora_weight_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "layer_total_norms": dict(layer_total_norms),
            "module_total_norms": dict(module_total_norms),
            "layer_module_norms": {str(k): v for k, v in layer_module_norms.items()},
            "delta_w_records": delta_w_records,
            "logit_lens_inconsistent_layers": LOGIT_LENS_INCONSISTENT_LAYERS,
            "avg_inconsistent_norm": avg_inconsistent,
            "avg_consistent_norm": avg_consistent,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()