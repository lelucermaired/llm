"""
logit_lens_analysis.py

Logit Lens 分析：在每一层提取隐层状态，投影回词表空间
比较基础模型与五子棋微调模型在数学题上的中间层预测差异

核心问题：LoRA的改变究竟落在哪里？

用法：
    python logit_lens_analysis.py
"""

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "finetuned_model_path": "./checkpoints/qwen-gomoku-v2/final_model",
    "output_dir": "./logit_lens_results",
    "n_probes": 30,       # 使用多少道数学题作为探针
    "task": "math",       # "math" 或 "gomoku"
    "top_k": 5,           # 每层显示top-k预测token
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 探针题目（数学推理题）====================
MATH_PROBES = [
    "What is 15 + 27?",
    "If a train travels 60 miles per hour for 2 hours, how far does it travel?",
    "What is the square root of 144?",
    "A rectangle has length 8 and width 5. What is its area?",
    "If x + 3 = 10, what is x?",
    "What is 7 multiplied by 8?",
    "A store sells apples for $2 each. If you buy 6 apples, how much do you spend?",
    "What is 100 divided by 4?",
    "If a triangle has base 6 and height 4, what is its area?",
    "What is 3 to the power of 4?",
    "A car travels 150 miles on 5 gallons of gas. How many miles per gallon does it get?",
    "If 3 workers can complete a job in 6 days, how long will it take 9 workers?",
    "What is 25% of 80?",
    "A circle has radius 7. What is its circumference? Use pi = 3.14.",
    "If a shirt costs $40 and is on sale for 20% off, what is the sale price?",
    "What is the average of 10, 20, 30, 40, and 50?",
    "How many seconds are in 3 hours?",
    "If 2x - 4 = 10, what is x?",
    "A box contains 5 red and 3 blue balls. What fraction are red?",
    "What is the perimeter of a square with side length 9?",
    "John has 24 cookies and shares them equally among 6 friends. How many does each get?",
    "If a temperature drops from 15 degrees to -5 degrees, by how much did it drop?",
    "A recipe needs 2.5 cups of flour for 12 cookies. How much flour for 36 cookies?",
    "What is 15% of 200?",
    "If a = 4 and b = 3, what is a squared plus b squared?",
    "A train leaves at 9am and arrives at 2pm. How long is the journey?",
    "What is the median of 3, 7, 9, 1, 5?",
    "Divide 360 by 12. What do you get?",
    "A pool holds 500 liters. It fills at 25 liters per minute. How long to fill?",
    "What is the next prime number after 13?",
]


GOMOKU_PROBES = [
    "你是一个五子棋大师。棋盘状态（●黑子，○白子，·空位）：\n  C D E F G H I J K L M N O\n1 · · · · · · · · · · · · ·\n2 · · · · · · · · · · · · ·\n3 · · · · · · · · · · · · ·\n4 · · · · · · · · · · · · ·\n5 · · · · · ● ○ ○ ● · ● · ●\n6 · · · · ○ · ● ● ● ○ ○ ○ ·\n7 · · · · · ● ○ ○ ○ ● ● ● ·\n8 · · · ● ● ○ · ● ● ○ · · ·\n请问最佳落子是？",
    "五子棋当前局面，黑棋已在H5, I5, J5, K5四个位置，白棋需要阻止。最佳落子位置是？",
    "五子棋棋盘：黑棋在G7, H7, I7, J7，白棋在H6, I6。黑棋下一步最佳落子？",
    "五子棋分析：棋盘中央黑棋已形成三连，H8, I8, J8均为黑子，下一步黑棋最佳落子坐标是？",
    "五子棋局面分析，当前轮到黑棋，黑棋在I5, I6, I7三个位置已连成竖向三连，最佳落子是？",
    "五子棋棋盘状态：黑子占据H8, I9, J10（斜线方向），如何继续？最佳落子？",
    "五子棋对局，白棋在G6, H6, I6三连，黑棋应如何阻止？给出最佳落子坐标。",
    "当前五子棋局面，黑棋在J5, J6, J7, J8纵向四连，白棋最紧急的防守落子是？",
    "五子棋分析：黑棋已形成活三（两端均空），位于H7, I7, J7，白棋最佳应对？",
    "五子棋棋盘，黑棋在I5, J6, K7斜向三连，继续该方向的最佳落子是？",
]


# ==================== Logit Lens 核心函数 ====================

def get_logit_lens(model, tokenizer, prompt, device):
    """
    对单个prompt做logit lens分析
    返回每一层在最后一个token位置上的top-k预测
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(device)

    hidden_states_per_layer = []

    # 注册hook提取每层输出
    hooks = []
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # 取最后一个token的hidden state，detach避免梯度
            hidden_states_per_layer.append(hidden[:, -1, :].detach().float())
        return hook_fn

    # 兼容base model和PeftModel两种结构
    try:
        layers = model.model.layers
        norm = model.model.norm
        lm_head = model.lm_head
    except AttributeError:
        layers = model.base_model.model.model.layers
        norm = model.base_model.model.model.norm
        lm_head = model.base_model.model.lm_head

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # 移除hooks
    for h in hooks:
        h.remove()

    # 获取lm_head的dtype，确保投影时dtype一致
    lm_head_dtype = next(lm_head.parameters()).dtype

    # 对每层hidden state做logit lens投影
    layer_predictions = []
    for layer_idx, hidden in enumerate(hidden_states_per_layer):
        # 先过norm，再过lm_head，转换dtype避免float32/bfloat16不匹配
        hidden = hidden.to(lm_head_dtype)
        normed = norm(hidden)
        logits = lm_head(normed)  # (1, vocab_size)
        probs = torch.softmax(logits[0], dim=-1)

        top_k_probs, top_k_ids = torch.topk(probs, CONFIG["top_k"])
        top_k_tokens = [tokenizer.decode([tid.item()]).strip() for tid in top_k_ids]
        top_k_probs_list = top_k_probs.detach().float().cpu().numpy().tolist()

        layer_predictions.append({
            "layer": layer_idx,
            "top_tokens": top_k_tokens,
            "top_probs": top_k_probs_list,
            "entropy": float(-torch.sum(probs * torch.log(probs + 1e-10)).item()),
        })

    return layer_predictions


def compute_prediction_divergence(base_preds, ft_preds):
    """
    计算每层基础模型和微调模型预测的差异
    用top-1 token是否一致 + top-1 token概率差来衡量
    """
    divergences = []
    for b, f in zip(base_preds, ft_preds):
        top1_match = int(b["top_tokens"][0] == f["top_tokens"][0])
        prob_diff = abs(b["top_probs"][0] - f["top_probs"][0])
        entropy_diff = abs(b["entropy"] - f["entropy"])
        divergences.append({
            "layer": b["layer"],
            "top1_match": top1_match,
            "prob_diff": prob_diff,
            "entropy_diff": entropy_diff,
            "base_top1": b["top_tokens"][0],
            "ft_top1": f["top_tokens"][0],
            "base_top1_prob": b["top_probs"][0],
            "ft_top1_prob": f["top_probs"][0],
        })
    return divergences


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
    )
    if is_base:
        base.eval()
        return base
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    return model


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("Logit Lens 分析")
    print("探究：LoRA的改变究竟落在哪些层？")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )

    if CONFIG["task"] == "gomoku":
        probes = GOMOKU_PROBES[:CONFIG["n_probes"]]
    else:
        probes = MATH_PROBES[:CONFIG["n_probes"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 基础模型 ==========
    print("\n[1/2] 加载基础模型，提取Logit Lens...")
    base_model = load_model(None, CONFIG["base_model_path"], is_base=True)

    all_base_preds = []
    for prompt in tqdm(probes, desc="基础模型"):
        preds = get_logit_lens(base_model, tokenizer, prompt, device)
        all_base_preds.append(preds)

    del base_model
    torch.cuda.empty_cache()

    # ========== 微调模型 ==========
    print("\n[2/2] 加载微调模型，提取Logit Lens...")
    ft_model = load_model(
        CONFIG["finetuned_model_path"],
        CONFIG["base_model_path"],
        is_base=False
    )

    all_ft_preds = []
    for prompt in tqdm(probes, desc="微调模型"):
        preds = get_logit_lens(ft_model, tokenizer, prompt, device)
        all_ft_preds.append(preds)

    del ft_model
    torch.cuda.empty_cache()

    # ========== 计算差异 ==========
    print("\n计算层间预测差异...")
    n_layers = len(all_base_preds[0])

    # 每层聚合：top1_match率、avg prob_diff、avg entropy_diff
    layer_stats = []
    for layer_idx in range(n_layers):
        top1_matches = []
        prob_diffs = []
        entropy_diffs = []
        base_top1_changes = []  # 该层base top1是否和上一层不同（预测演化速度）

        for probe_idx in range(len(probes)):
            div = compute_prediction_divergence(
                [all_base_preds[probe_idx][layer_idx]],
                [all_ft_preds[probe_idx][layer_idx]]
            )[0]
            top1_matches.append(div["top1_match"])
            prob_diffs.append(div["prob_diff"])
            entropy_diffs.append(div["entropy_diff"])

        layer_stats.append({
            "layer": layer_idx,
            "top1_agreement": np.mean(top1_matches),       # 基础和微调模型top1一致率
            "avg_prob_diff": np.mean(prob_diffs),           # 概率差
            "avg_entropy_diff": np.mean(entropy_diffs),     # 熵差（不确定性变化）
        })

    # ========== 打印结果 ==========
    print("\n" + "=" * 65)
    print("逐层 Logit Lens 差异分析（数学推理探针）")
    print("=" * 65)
    print(f"{'层':>5} {'Top1一致率':>10} {'概率差(avg)':>12} {'熵差(avg)':>12}  {'变化程度'}")
    print("-" * 65)

    for s in layer_stats:
        agreement = s["top1_agreement"]
        prob_diff = s["avg_prob_diff"]
        # 用一致率判断变化程度
        if agreement >= 0.9:
            change_label = "极小 ▪"
        elif agreement >= 0.7:
            change_label = "小   ▪▪"
        elif agreement >= 0.5:
            change_label = "中   ▪▪▪"
        else:
            change_label = "大   ▪▪▪▪"

        print(f"  {s['layer']:>3}  {agreement:>10.3f}  {prob_diff:>12.4f}  "
              f"{s['avg_entropy_diff']:>12.4f}  {change_label}")

    # ========== 找出差异最大的层 ==========
    sorted_by_diff = sorted(layer_stats, key=lambda x: x["avg_prob_diff"], reverse=True)
    print("\n差异最大的5层：")
    for s in sorted_by_diff[:5]:
        print(f"  Layer {s['layer']:>2}：top1一致率={s['top1_agreement']:.3f}，"
              f"概率差={s['avg_prob_diff']:.4f}")

    # ========== 打印典型案例 ==========
    print("\n" + "=" * 60)
    print("典型探针逐层预测演化（第1道题）")
    print("=" * 60)
    print(f"题目：{probes[0]}")
    print(f"\n{'层':>4}  {'基础模型Top1':>15} {'概率':>7}  {'微调模型Top1':>15} {'概率':>7}  {'一致'}")
    print("-" * 65)
    for layer_idx in range(0, n_layers, 2):  # 每隔一层打印
        b = all_base_preds[0][layer_idx]
        f = all_ft_preds[0][layer_idx]
        match = "✓" if b["top_tokens"][0] == f["top_tokens"][0] else "✗"
        print(f"  {layer_idx:>2}  {b['top_tokens'][0]:>15} {b['top_probs'][0]:>7.3f}"
              f"  {f['top_tokens'][0]:>15} {f['top_probs'][0]:>7.3f}  {match}")

    # ========== 保存结果 ==========
    save_data = {
        "config": CONFIG,
        "probes": probes,
        "layer_stats": layer_stats,
        "base_predictions": all_base_preds,
        "ft_predictions": all_ft_preds,
    }
    save_path = os.path.join(CONFIG["output_dir"], f"logit_lens_{CONFIG['task']}_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细结果已保存至: {save_path}")

    # ========== 核心结论 ==========
    avg_agreement = np.mean([s["top1_agreement"] for s in layer_stats])
    deep_agreement = np.mean([s["top1_agreement"] for s in layer_stats[20:]])
    shallow_agreement = np.mean([s["top1_agreement"] for s in layer_stats[:10]])

    print("\n" + "=" * 60)
    print("核心结论")
    print("=" * 60)
    print(f"全层平均Top1一致率:   {avg_agreement:.3f}")
    print(f"浅层（0-9）一致率:    {shallow_agreement:.3f}")
    print(f"深层（20-末）一致率:  {deep_agreement:.3f}")

    if avg_agreement > 0.85:
        print("\n→ 微调模型与基础模型在数学推理上的逐层预测高度一致")
        print("  LoRA的改变几乎未影响模型处理数学题时的中间层表示")
        print("  与CKA=0.9999的结论互相印证")
    if shallow_agreement > deep_agreement + 0.05:
        print("\n→ 浅层一致率高于深层，说明LoRA的影响集中在深层")
        print("  深层负责任务特异性表示，浅层负责通用语义编码")
    elif deep_agreement > shallow_agreement + 0.05:
        print("\n→ 深层一致率高于浅层，说明LoRA的影响更多集中在浅层")


if __name__ == "__main__":
    main()