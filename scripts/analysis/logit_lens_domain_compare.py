"""
logit_lens_domain_compare.py

Logit Lens 领域对比分析
核心问题：LoRA的改变是"领域特异性"的吗？

对比两类输入：
  - 数学题：基础模型 vs 微调模型（预期差异极小）
  - 五子棋题：基础模型 vs 微调模型（预期差异更大）

如果五子棋输入的差异远大于数学输入，说明LoRA的改变
专门针对五子棋相关的token预测路径，具有领域特异性。

用法：
    python logit_lens_domain_compare.py
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
    "output_dir": "./logit_lens_domain_results",
    "n_probes": 30,
    "top_k": 5,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 探针题目 ====================

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
    "You are a Gomoku master. Board state (B=Black, W=White, .=Empty):\n  A B C D E\n1 . . . . .\n2 . B B . .\n3 . W W . .\n4 . . . . .\nBlack to move. Best move?",
    "Analyze: Black has three in a row at H8 I8 J8, White has two at H9 I9. Black to move. Best choice?",
    "In Gomoku, what is an open four and how do you defend against it?",
    "Black stones at H7 I7 J7 K7. Black to move. Where to play to win?",
    "White has four stones at E5 F5 G5 H5 with both ends open. Black to move. How to respond?",
    "You are a Gomoku expert. Black has a blocked four. Analyze the best defensive move for White.",
    "Black stones at D4 E5 F6 form a diagonal open three. Best attacking direction?",
    "If Black has an open three in the center, should White prioritize defense or offense?",
    "Gomoku rule: five in a row wins. Black has two simultaneous open threes. What is this pattern called?",
    "Black at J8 J9 J10 forms a vertical open three. Better extension: J7 or J11?",
    "Black has stones at E5 F6 G7 forming a diagonal. White must defend. Best move?",
    "Analyze: Black threatens to win at K10. White has one move to block. Where?",
    "Black has open four at C3 D3 E3 F3. White to move. Only correct defense?",
    "In Gomoku, explain the difference between a live three and a dead three.",
    "Black at G7 H8 forming diagonal two. Best next move to build toward five?",
    "White stones at H5 H6 H7 H8. Black must block. Where to play?",
    "Black has double threat at J7 and J9. White can only block one. Which is more urgent?",
    "Analyze this Gomoku endgame: Black needs one more stone to complete five at row 8.",
    "Black at D4 E4 F4, open both sides. What is the strongest follow-up move?",
    "White has a hidden threat at diagonal B2 C3 D4 E5. Should Black block now?",
    "Black forms a fork with threats at both K8 and H11. How should White respond?",
    "Explain why playing at the center (H8) is generally stronger in Gomoku openings.",
    "Black at F6 G7 H8 I9. One more stone completes diagonal five. Where to play?",
    "White blocks at J8. Does Black still have a winning continuation? Analyze.",
    "Black has three separate pairs on the board. Which pair to develop first?",
    "Gomoku tactics: Black threatens five at row 12. Can White both block and counter-attack?",
    "Analyze: open board, Black plays center. What are the top three follow-up moves?",
    "Black at E5 E6 E7 vertical three. White at F5 F6. Black to move. Best play?",
    "White has stones at G8 H8 I8 J8. Black must prevent five. Where to play?",
    "Black creates a double open three at moves K7 and K9 simultaneously. Name this tactic.",
]

CHINESE_MATH_PROBES = [
    "15加27等于多少？",
    "一列火车每小时行驶60公里，行驶2小时后走了多远？",
    "144的平方根是多少？",
    "一个长方形长8宽5，面积是多少？",
    "如果x加3等于10，x等于多少？",
    "7乘以8等于多少？",
    "一家商店苹果每个2元，买6个要花多少钱？",
    "100除以4等于多少？",
    "一个三角形底边为6高为4，面积是多少？",
    "3的4次方等于多少？",
    "一辆汽车行驶150公里消耗5升油，百公里油耗是多少？",
    "3个工人6天完成一项工作，9个工人需要几天？",
    "80的25%是多少？",
    "一个圆半径为7，周长是多少？取π=3.14。",
    "一件衬衫原价40元，打八折后售价是多少？",
    "10、20、30、40、50的平均数是多少？",
    "3小时有多少秒？",
    "如果2x减4等于10，x等于多少？",
    "一个盒子里有5个红球和3个蓝球，红球占几分之几？",
    "一个正方形边长为9，周长是多少？",
    "小明有24块饼干，平均分给6个朋友，每人几块？",
    "气温从15度降到零下5度，降了多少度？",
    "一个食谱做12块饼干需要2.5杯面粉，做36块需要多少杯？",
    "200的15%是多少？",
    "如果a等于4，b等于3，a的平方加b的平方等于多少？",
    "火车早上9点出发，下午2点到达，行程多长时间？",
    "3、7、9、1、5的中位数是多少？",
    "360除以12等于多少？",
    "一个水池容量500升，每分钟注入25升，几分钟注满？",
    "13之后的下一个质数是多少？",
]



# ==================== 核心函数 ====================

def get_logit_lens(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
    input_ids = inputs["input_ids"].to(device)

    hidden_states_per_layer = []

    hooks = []
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            hidden_states_per_layer.append(hidden[:, -1, :].detach().float())
        return hook_fn

    # 兼容base model和PeftModel
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
        model(input_ids)

    for h in hooks:
        h.remove()

    lm_head_dtype = next(lm_head.parameters()).dtype
    layer_predictions = []

    for layer_idx, hidden in enumerate(hidden_states_per_layer):
        hidden = hidden.to(lm_head_dtype)
        normed = norm(hidden)
        logits = lm_head(normed)
        probs = torch.softmax(logits[0], dim=-1)

        top_k_probs, top_k_ids = torch.topk(probs, CONFIG["top_k"])
        top_k_tokens = [tokenizer.decode([tid.item()]).strip() for tid in top_k_ids]
        top_k_probs_list = top_k_probs.detach().float().cpu().numpy().tolist()

        layer_predictions.append({
            "layer": layer_idx,
            "top_tokens": top_k_tokens,
            "top_probs": top_k_probs_list,
            "entropy": float(-torch.sum(probs * torch.log(probs + 1e-10)).detach().float().item()),
        })

    return layer_predictions


def compute_divergence(base_preds, ft_preds):
    """计算两个模型逐层的预测差异"""
    results = []
    for b, f in zip(base_preds, ft_preds):
        top1_match = int(b["top_tokens"][0] == f["top_tokens"][0])
        prob_diff = abs(b["top_probs"][0] - f["top_probs"][0])
        # top-k token集合的Jaccard相似度
        base_set = set(b["top_tokens"])
        ft_set = set(f["top_tokens"])
        jaccard = len(base_set & ft_set) / len(base_set | ft_set)
        entropy_diff = abs(b["entropy"] - f["entropy"])
        results.append({
            "layer": b["layer"],
            "top1_match": top1_match,
            "prob_diff": prob_diff,
            "jaccard": jaccard,
            "entropy_diff": entropy_diff,
        })
    return results


def aggregate_divergence(all_divergences):
    """跨多个probe聚合每层的差异统计"""
    n_layers = len(all_divergences[0])
    layer_stats = []
    for layer_idx in range(n_layers):
        top1_matches = [d[layer_idx]["top1_match"] for d in all_divergences]
        prob_diffs = [d[layer_idx]["prob_diff"] for d in all_divergences]
        jaccards = [d[layer_idx]["jaccard"] for d in all_divergences]
        entropy_diffs = [d[layer_idx]["entropy_diff"] for d in all_divergences]
        layer_stats.append({
            "layer": layer_idx,
            "top1_agreement": np.mean(top1_matches),
            "avg_prob_diff": np.mean(prob_diffs),
            "avg_jaccard": np.mean(jaccards),
            "avg_entropy_diff": np.mean(entropy_diffs),
        })
    return layer_stats


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

def run_probes(model, tokenizer, probes, desc, device):
    all_preds = []
    for prompt in tqdm(probes, desc=desc):
        preds = get_logit_lens(model, tokenizer, prompt, device)
        all_preds.append(preds)
    return all_preds


def main():
    print("=" * 65)
    print("Logit Lens 领域特异性分析")
    print("数学输入 vs 五子棋输入：LoRA的改变是领域特异性的吗？")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    math_probes = MATH_PROBES[:CONFIG["n_probes"]]
    gomoku_probes = GOMOKU_PROBES[:CONFIG["n_probes"]]
    cn_math_probes = CHINESE_MATH_PROBES[:CONFIG["n_probes"]]

    # ========== 基础模型 ==========
    print("\n[1/2] 基础模型推理...")
    base_model = load_model(None, CONFIG["base_model_path"], is_base=True)
    base_math = run_probes(base_model, tokenizer, math_probes, "基础模型×数学(EN)", device)
    base_gomoku = run_probes(base_model, tokenizer, gomoku_probes, "基础模型×五子棋", device)
    base_cn_math = run_probes(base_model, tokenizer, cn_math_probes, "基础模型×数学(CN)", device)
    del base_model
    torch.cuda.empty_cache()

    # ========== 微调模型 ==========
    print("\n[2/2] 微调模型推理...")
    ft_model = load_model(CONFIG["finetuned_model_path"], CONFIG["base_model_path"])
    ft_math = run_probes(ft_model, tokenizer, math_probes, "微调模型×数学(EN)", device)
    ft_gomoku = run_probes(ft_model, tokenizer, gomoku_probes, "微调模型×五子棋", device)
    ft_cn_math = run_probes(ft_model, tokenizer, cn_math_probes, "微调模型×数学(CN)", device)
    del ft_model
    torch.cuda.empty_cache()

    # ========== 计算差异 ==========
    print("\n计算领域差异...")

    math_divs = [compute_divergence(base_math[i], ft_math[i]) for i in range(len(math_probes))]
    gomoku_divs = [compute_divergence(base_gomoku[i], ft_gomoku[i]) for i in range(len(gomoku_probes))]
    cn_math_divs = [compute_divergence(base_cn_math[i], ft_cn_math[i]) for i in range(len(cn_math_probes))]

    math_stats = aggregate_divergence(math_divs)
    gomoku_stats = aggregate_divergence(gomoku_divs)
    cn_math_stats = aggregate_divergence(cn_math_divs)

    # ========== 打印对比表 ==========
    print("\n" + "=" * 90)
    print("逐层差异对比：英文数学 vs 中文数学 vs 五子棋（基础模型 vs 微调模型）")
    print("语言控制实验：排除语言差异，验证纯领域特异性")
    print("=" * 90)
    print(f"{'层':>4}  {'EN数学概率差':>12}  {'CN数学概率差':>12}  {'五子棋概率差':>12}  {'语言倍数CN/EN':>13}  {'领域倍数Go/EN':>13}")
    print("-" * 90)

    for m, c, g in zip(math_stats, cn_math_stats, gomoku_stats):
        lang_ratio = c["avg_prob_diff"] / (m["avg_prob_diff"] + 1e-8)
        domain_ratio = g["avg_prob_diff"] / (m["avg_prob_diff"] + 1e-8)
        flag = " ◀" if domain_ratio > lang_ratio * 1.5 else ""
        print(f"  {m['layer']:>2}  {m['avg_prob_diff']:>12.4f}  {c['avg_prob_diff']:>12.4f}  "
              f"{g['avg_prob_diff']:>12.4f}  {lang_ratio:>13.2f}x  {domain_ratio:>13.2f}x{flag}")

    # ========== 汇总统计 ==========
    math_avg_prob = np.mean([s["avg_prob_diff"] for s in math_stats])
    cn_math_avg_prob = np.mean([s["avg_prob_diff"] for s in cn_math_stats])
    gomoku_avg_prob = np.mean([s["avg_prob_diff"] for s in gomoku_stats])

    lang_ratio = cn_math_avg_prob / (math_avg_prob + 1e-8)
    domain_ratio = gomoku_avg_prob / (math_avg_prob + 1e-8)
    pure_domain_ratio = gomoku_avg_prob / (cn_math_avg_prob + 1e-8)

    print("\n" + "=" * 65)
    print("汇总：语言差异 vs 领域差异")
    print("=" * 65)
    print(f"  英文数学平均概率差：{math_avg_prob:.4f}（基准）")
    print(f"  中文数学平均概率差：{cn_math_avg_prob:.4f}（语言差异倍数：{lang_ratio:.2f}x）")
    print(f"  五子棋平均概率差：  {gomoku_avg_prob:.4f}（领域差异倍数：{domain_ratio:.2f}x）")
    print(f"  纯领域差异（五子棋/中文数学）：{pure_domain_ratio:.2f}x")

    print("\n" + "=" * 65)
    print("核心结论：语言控制实验")
    print("=" * 65)

    if lang_ratio > 0.8 and lang_ratio < 1.2:
        print(f"\n→ ✅ 语言差异极小（CN/EN={lang_ratio:.2f}x）：")
        print("  中英文数学题对LoRA的影响几乎相同，语言不是混淆变量")
    else:
        print(f"\n→ ⚠ 存在语言差异（CN/EN={lang_ratio:.2f}x）：")
        print("  中英文数学题对LoRA的影响存在差异，需在论文中说明")

    if pure_domain_ratio > 1.5:
        print(f"\n→ ✅ 纯领域特异性显著（五子棋/中文数学={pure_domain_ratio:.2f}x）：")
        print("  排除语言差异后，五子棋输入的LoRA影响仍明显大于数学输入")
        print("  领域特异性结论成立，与attention pattern分析一致")
    else:
        print(f"\n→ ⚠ 排除语言差异后领域特异性减弱（{pure_domain_ratio:.2f}x）：")
        print("  之前的部分差异可能来自语言差异而非领域差异")

    # ========== 保存结果 ==========
    save_path = os.path.join(CONFIG["output_dir"], "domain_compare_results_v2.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "math_stats": math_stats,
            "cn_math_stats": cn_math_stats,
            "gomoku_stats": gomoku_stats,
            "summary": {
                "math_avg_prob_diff": math_avg_prob,
                "cn_math_avg_prob_diff": cn_math_avg_prob,
                "gomoku_avg_prob_diff": gomoku_avg_prob,
                "lang_ratio": lang_ratio,
                "domain_ratio": domain_ratio,
                "pure_domain_ratio": pure_domain_ratio,
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()