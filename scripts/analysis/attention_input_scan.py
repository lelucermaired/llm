"""
attention_input_scan.py

对15类不同输入做attention差异扫描
找出哪类输入在五子棋LoRA微调前后变化最大
→ 变化最大的输入类型，最可能是迁移发生的目标任务

用法：
    python attention_input_scan.py
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "finetuned_model_path": "./checkpoints/qwen-gomoku-v2/final_model",
    "output_dir": "./attention_scan_results",
    "target_layers": [0, 4, 8, 12, 16, 20, 24, 27],
    "n_probes": 5,   # 每类输入5个探针，取平均
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 15类输入探针 ====================

INPUT_CATEGORIES = {

    "gomoku": [
        "You are a Gomoku master. Board (B=Black, W=White, .=Empty):\n  G H I J K\n8 . . B B B\nBlack to move. Best move?",
        "Gomoku: Black at H7 I7 J7 K7. One more to win. Where to play?",
        "Analyze Gomoku board: White has open three at E5 F5 G5. Black to block where?",
        "Gomoku tactics: Black forms double open three. What is this pattern called?",
        "Gomoku: Black at D4 E5 F6 diagonal open three. Best next move?",
    ],

    "math_arithmetic": [
        "What is 15 + 27?",
        "What is 7 multiplied by 8?",
        "What is 144 divided by 12?",
        "What is 3 to the power of 4?",
        "What is the square root of 256?",
    ],

    "math_word_problem": [
        "A train travels 60 mph for 2.5 hours. How far does it travel?",
        "A store sells apples for $2 each. If you buy 6, how much do you spend?",
        "If a rectangle has length 8 and width 5, what is its area?",
        "John has 24 cookies shared equally among 6 friends. How many each?",
        "A pool holds 500 liters and fills at 25 liters per minute. How long to fill?",
    ],

    "logical_reasoning": [
        "All cats are mammals. All mammals are animals. Is a cat an animal?",
        "If it rains, the ground is wet. The ground is dry. Did it rain?",
        "All A are B. Some B are C. Must some A be C?",
        "Either P or Q is true. P is false. What about Q?",
        "If all doctors are scientists and John is a doctor, is John a scientist?",
    ],

    "spatial_reasoning": [
        "A is to the left of B. B is above C. What is the relation of A to C?",
        "A is above B. B is to the right of C. C is below D. What is relation of A to D?",
        "Object X is north of Y. Y is east of Z. Where is X relative to Z?",
        "A is left of B. B is left of C. C is left of D. Where is A relative to D?",
        "If you face north and turn right twice, which direction do you face?",
    ],

    "sequential_planning": [
        "Blocks: A on B, B on table. Goal: B on A. Minimum moves?",
        "Tower of Hanoi: 3 disks on peg A. Move to peg C. How many moves?",
        "Grid: start (1,1), goal (3,3), no obstacles. Minimum moves?",
        "Pancake sort: [3,1,2] top to bottom. Goal: [1,2,3]. Minimum flips?",
        "Blocks: C on B, B on A, A on table. Goal: all separate. First move?",
    ],

    "code_generation": [
        "Write a Python function to check if a number is prime.",
        "Write a function to reverse a string in Python.",
        "Write Python code to find the maximum element in a list.",
        "Write a function to compute factorial recursively.",
        "Write Python code to sort a list of integers.",
    ],

    "creative_writing": [
        "Write a short poem about the ocean.",
        "Write the opening sentence of a mystery novel.",
        "Describe a sunset in three sentences.",
        "Write a haiku about autumn leaves.",
        "Create a short metaphor for time passing.",
    ],

    "factual_qa": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical formula of water?",
        "In what year did World War II end?",
        "What is the speed of light in meters per second?",
    ],

    "commonsense": [
        "If you drop a glass on a hard floor, what happens?",
        "Why do people wear coats in winter?",
        "What happens to ice when you heat it?",
        "Why do we need to eat food?",
        "If you plant a seed and water it, what usually happens?",
    ],

    "dialogue": [
        "Hi, how are you doing today?",
        "What do you think about artificial intelligence?",
        "Can you recommend a good book to read?",
        "What is your favorite season and why?",
        "Tell me something interesting about space.",
    ],

    "chess": [
        "In chess, what is the value of a queen?",
        "Explain the en passant rule in chess.",
        "White has king at e1, rook at a1. Black king at e8. How to checkmate?",
        "What is castling in chess and when can you do it?",
        "In chess, which piece moves in an L-shape?",
    ],

    "scientific_reasoning": [
        "Why does the moon orbit the Earth?",
        "Explain why objects fall at the same rate regardless of mass.",
        "What causes ocean tides?",
        "Why is the sky blue?",
        "Explain how photosynthesis works.",
    ],

    "analogical_reasoning": [
        "Doctor is to hospital as teacher is to ___.",
        "Hot is to cold as fast is to ___.",
        "Book is to library as painting is to ___.",
        "Fish is to water as bird is to ___.",
        "Pen is to write as knife is to ___.",
    ],

    "multi_step_math": [
        "If x + 3 = 10 and y = 2x, what is y?",
        "A train leaves at 9am at 60mph. Another leaves at 10am at 80mph. When does the second catch up?",
        "If 2x - 4 = 10, and y = x^2, what is y?",
        "A rectangle's perimeter is 30. Its length is twice its width. What is the area?",
        "If compound interest rate is 10% per year, how much does $100 grow to in 2 years?",
    ],
}


# ==================== Attention 提取 ====================

def get_attention_diff(base_model, ft_model, tokenizer, prompts, target_layers, device):
    """
    对一组prompts计算基础模型和微调模型的attention差异
    返回每层的平均L2差异
    """
    layer_diffs = {l: [] for l in target_layers}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(device)

        for model, model_name in [(base_model, "base"), (ft_model, "ft")]:
            attns = {}

            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple) and out[1] is not None:
                        attns[layer_idx] = out[1].detach().float().cpu()[0]
                return hook_fn

            try:
                layers = model.model.layers
            except AttributeError:
                layers = model.base_model.model.model.layers

            hooks = []
            for l in target_layers:
                h = layers[l].self_attn.register_forward_hook(make_hook(l))
                hooks.append(h)

            with torch.no_grad():
                model(input_ids, output_attentions=True)

            for h in hooks:
                h.remove()

            if model_name == "base":
                base_attns = attns.copy()
            else:
                ft_attns = attns.copy()

        # 计算每层差异
        for l in target_layers:
            if l in base_attns and l in ft_attns:
                diff = (ft_attns[l] - base_attns[l]).norm(dim=(-2, -1)).mean().item()
                layer_diffs[l].append(diff)

    # 每层取平均
    avg_diffs = {l: np.mean(v) if v else 0.0 for l, v in layer_diffs.items()}
    overall_avg = np.mean(list(avg_diffs.values()))
    return overall_avg, avg_diffs


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


# ==================== 主流程 ====================

def main():
    print("=" * 65)
    print("多类输入 Attention 差异扫描")
    print("找出哪类输入在五子棋LoRA微调前后变化最大")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n加载模型...")
    base_model = load_model(None, CONFIG["base_model_path"], is_base=True)
    ft_model = load_model(CONFIG["finetuned_model_path"], CONFIG["base_model_path"])

    # ========== 扫描每类输入 ==========
    print("\n开始扫描...")
    results = {}

    for category, prompts in tqdm(INPUT_CATEGORIES.items(), desc="扫描类别"):
        probes = prompts[:CONFIG["n_probes"]]
        overall, by_layer = get_attention_diff(
            base_model, ft_model, tokenizer, probes,
            CONFIG["target_layers"], device)
        results[category] = {
            "overall_diff": overall,
            "by_layer": {str(l): v for l, v in by_layer.items()},
        }

    # ========== 排序输出 ==========
    sorted_results = sorted(results.items(), key=lambda x: x[1]["overall_diff"], reverse=True)

    print("\n" + "=" * 65)
    print("各类输入的Attention差异排名（从大到小）")
    print("五子棋微调对哪类输入改变最大？")
    print("=" * 65)

    gomoku_diff = results["gomoku"]["overall_diff"]
    print(f"\n{'排名':<6} {'输入类型':<22} {'Attention差异':>14} {'vs五子棋':>10}")
    print("-" * 58)

    for rank, (cat, data) in enumerate(sorted_results, 1):
        ratio = data["overall_diff"] / (gomoku_diff + 1e-8)
        flag = " ← 基准" if cat == "gomoku" else ""
        flag = " ✅ 候选迁移目标" if ratio > 0.8 and cat != "gomoku" else flag
        print(f"  {rank:<4} {cat:<22} {data['overall_diff']:>14.4f} {ratio:>9.2f}x{flag}")

    # ========== 找出候选迁移目标 ==========
    print("\n" + "=" * 65)
    print("候选迁移目标任务（attention变化接近或超过五子棋输入）")
    print("=" * 65)

    candidates = [(cat, data) for cat, data in sorted_results
                  if cat != "gomoku" and data["overall_diff"] >= gomoku_diff * 0.7]

    if candidates:
        print(f"\n以下类型输入的attention变化达到五子棋的70%以上：")
        for cat, data in candidates:
            ratio = data["overall_diff"] / gomoku_diff
            print(f"  → {cat:<22} 差异={data['overall_diff']:.4f}，"
                  f"为五子棋的{ratio:.2f}x")
        print("\n这些类型的任务最可能从五子棋微调中获得迁移")
        print("建议优先在这些任务上做迁移评测")
    else:
        print("\n没有类型达到五子棋attention变化的70%")
        print("说明LoRA的改变高度特异于五子棋输入，其他类型输入几乎不受影响")
        print("这从attention层面解释了跨任务零迁移的普遍性")

    # ========== 逐层分析：哪类输入在深层变化最大 ==========
    print("\n" + "=" * 65)
    print("深层（L20-L27）Attention差异排名")
    print("深层与任务特异性表示相关，变化大的类型最可能有迁移")
    print("=" * 65)

    deep_layers = [str(l) for l in CONFIG["target_layers"] if l >= 20]
    deep_results = {}
    for cat, data in results.items():
        deep_avg = np.mean([data["by_layer"].get(l, 0) for l in deep_layers])
        deep_results[cat] = deep_avg

    sorted_deep = sorted(deep_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'排名':<6} {'输入类型':<22} {'深层差异':>12}")
    print("-" * 44)
    for rank, (cat, val) in enumerate(sorted_deep, 1):
        flag = " ← 基准" if cat == "gomoku" else ""
        print(f"  {rank:<4} {cat:<22} {val:>12.4f}{flag}")

    # ========== 可视化 ==========
    categories = [cat for cat, _ in sorted_results]
    overall_diffs = [data["overall_diff"] for _, data in sorted_results]
    colors = ["coral" if cat == "gomoku" else
              "steelblue" if deep_results[cat] >= gomoku_diff * 0.7 else
              "lightgray" for cat in categories]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 图1：总体差异排名
    axes[0].barh(range(len(categories)), overall_diffs, color=colors, alpha=0.85)
    axes[0].set_yticks(range(len(categories)))
    axes[0].set_yticklabels(categories, fontsize=9)
    axes[0].set_xlabel("Attention Difference (avg L2 norm)")
    axes[0].set_title("Attention Pattern Change by Input Type\n(Base vs Finetuned, all layers)")
    axes[0].axvline(x=gomoku_diff, color='red', linestyle='--', alpha=0.5, label='Gomoku baseline')
    axes[0].axvline(x=gomoku_diff * 0.7, color='orange', linestyle=':', alpha=0.5, label='70% threshold')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='x', alpha=0.3)

    # 图2：逐层热力图（各类输入×各层）
    layer_labels = [f"L{l}" for l in CONFIG["target_layers"]]
    matrix = np.array([[results[cat]["by_layer"].get(str(l), 0)
                        for l in CONFIG["target_layers"]]
                       for cat, _ in sorted_results])

    im = axes[1].imshow(matrix, cmap='YlOrRd', aspect='auto')
    axes[1].set_xticks(range(len(CONFIG["target_layers"])))
    axes[1].set_xticklabels(layer_labels, fontsize=9)
    axes[1].set_yticks(range(len(categories)))
    axes[1].set_yticklabels(categories, fontsize=9)
    axes[1].set_title("Attention Diff Heatmap\n(Input Type × Layer)")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    fig_path = os.path.join(CONFIG["output_dir"], "attention_scan.png")
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 可视化图表已保存: {fig_path}")

    # ========== 保存结果 ==========
    save_path = os.path.join(CONFIG["output_dir"], "attention_scan_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "sorted_ranking": [(cat, data["overall_diff"]) for cat, data in sorted_results],
            "deep_layer_ranking": sorted_deep,
            "gomoku_diff": gomoku_diff,
            "candidates": [(cat, data["overall_diff"]) for cat, data in candidates],
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ 数值结果已保存: {save_path}")

    print("\n" + "=" * 65)
    print("下一步建议")
    print("=" * 65)
    if candidates:
        top_candidate = candidates[0][0]
        print(f"\n→ 优先对 '{top_candidate}' 类任务做迁移评测")
        print("  这类输入的attention变化最接近五子棋，最可能产生迁移")
    else:
        print("\n→ 进行方向三：只微调浅层（前10层q+v）")
        print("  让更新落在通用语义层，理论上更容易产生跨域迁移")


if __name__ == "__main__":
    main()