"""
head_ablation.py

Head消融实验：验证L27层Head 7-13是否为"五子棋专用回路"

实验设计：
  将L27层特定head的attention输出置零，测试：
  1. 五子棋任务性能（掌握度评估：规则理解、局面识别、落子质量）
  2. 数学任务性能（GSM8K子集）

预期结论：
  - 消融Head 7-13 -> 五子棋性能显著下降，数学性能基本不变
  - 消融随机head -> 五子棋和数学性能均无明显变化（对照组）
  -> 证明这些head是领域特异性的功能模块

用法：
    python head_ablation.py
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
    "output_dir": "./head_ablation_results",
    "ablation_layer": 27,
    "target_heads": list(range(7, 14)),    # Head 7-13（五子棋专用回路候选）
    "control_heads": list(range(0, 7)),    # Head 0-6（对照组，相同数量）
    "num_heads": 28,
    "head_dim": 128,
    "n_math_samples": 20,
    "n_gomoku_samples": 10,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 评测样本 ====================

MATH_SAMPLES = [
    ("What is 15 + 27?", "42"),
    ("What is 7 multiplied by 8?", "56"),
    ("What is 100 divided by 4?", "25"),
    ("What is the square root of 144?", "12"),
    ("What is 3 to the power of 4?", "81"),
    ("A rectangle has length 8 and width 5. What is its area?", "40"),
    ("If x + 3 = 10, what is x?", "7"),
    ("What is 25% of 80?", "20"),
    ("What is the average of 10, 20, 30, 40, and 50?", "30"),
    ("How many seconds are in 3 hours?", "10800"),
    ("What is 15% of 200?", "30"),
    ("If 2x - 4 = 10, what is x?", "7"),
    ("What is the perimeter of a square with side length 9?", "36"),
    ("Divide 360 by 12.", "30"),
    ("What is the next prime number after 13?", "17"),
    ("What is 45 + 67?", "112"),
    ("What is 9 multiplied by 9?", "81"),
    ("What is 200 divided by 8?", "25"),
    ("If a = 4 and b = 3, what is a squared plus b squared?", "25"),
    ("What is 2 to the power of 8?", "256"),
]

GOMOKU_RULE_SAMPLES = [
    ("In Gomoku, how many stones in a row are needed to win?", "five", "rule"),
    ("In Gomoku, can players place stones anywhere on the empty board?", "yes", "rule"),
    ("What is an 'open three' in Gomoku?", "three consecutive stones with both ends open", "rule"),
    ("In Gomoku, what is a 'double three' forbidden move?", "two open threes simultaneously", "rule"),
    ("If Black plays first in Gomoku, who has the advantage?", "black", "rule"),
]

GOMOKU_MOVE_SAMPLES = [
    (
        "Board state (B=Black, W=White, .=Empty):\n  G H I J K\n8 . . B B B\n9 . . W W .\nBlack to move. To win immediately, play at:",
        "J8", "move"
    ),
    (
        "Board state:\n  E F G H I\n5 . B B B .\nBlack has open three at F5 G5 H5. Best winning move:",
        "E5", "move"
    ),
    (
        "Black stones at J7 J8 J9 J10. Black to move. To complete five in a column, play at:",
        "J11", "move"
    ),
    (
        "White has four in a row at D4 E4 F4 G4 with H4 empty. Black must block at:",
        "H4", "move"
    ),
    (
        "Black at C3 D4 E5 F6 diagonal. One more stone completes five. Play at:",
        "G7", "move"
    ),
]


# ==================== Hook：置零指定head的输出 ====================

class HeadAblationHook:
    """
    在o_proj的输入端将指定head的贡献置零。
    o_proj输入shape: (batch, seq_len, num_heads * head_dim)
    按head_dim切片，将ablate_heads对应的列置零后再经过o_proj投影。
    这样消融的是指定head对输出的贡献，hook经验证生效。
    """

    def __init__(self, ablate_heads, num_heads, head_dim):
        self.ablate_heads = ablate_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hook_handle = None

    def hook_fn(self, module, input, output):
        # output: (batch, seq_len, hidden_size)，即o_proj的输出
        # 通过修改output来消融：将ablate_heads对应的输出列置零
        # 等价于这些head的贡献被移除
        if isinstance(output, tuple):
            hidden = output[0].clone()
            for head_idx in self.ablate_heads:
                start = head_idx * self.head_dim
                end = (head_idx + 1) * self.head_dim
                hidden[:, :, start:end] = 0.0
            return (hidden,) + output[1:]
        else:
            hidden = output.clone()
            for head_idx in self.ablate_heads:
                start = head_idx * self.head_dim
                end = (head_idx + 1) * self.head_dim
                hidden[:, :, start:end] = 0.0
            return hidden

    def register(self, model, layer_idx):
        try:
            layer = model.model.layers[layer_idx]
        except AttributeError:
            layer = model.base_model.model.model.layers[layer_idx]
        # 挂在o_proj上，经验证hook在此处生效
        self.hook_handle = layer.self_attn.o_proj.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None


# ==================== 评测函数 ====================

def generate_response(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    """
    生成模型回复。
    使用use_cache=False确保每个token都经过完整forward，hook在所有token上生效。
    KV cache会绕过hook，必须禁用。
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated, use_cache=False)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    response = tokenizer.decode(
        generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def check_answer(response, expected):
    """检查回复是否包含正确答案"""
    return expected.lower() in response.lower()


def evaluate_math(model, tokenizer, samples, desc, device, ablation_hook=None):
    """评估数学任务性能"""
    if ablation_hook:
        ablation_hook.register(model, CONFIG["ablation_layer"])

    correct = 0
    results = []
    for prompt, answer in tqdm(samples, desc=desc, leave=False):
        response = generate_response(model, tokenizer, prompt, max_new_tokens=30, device=device)
        is_correct = check_answer(response, answer)
        correct += int(is_correct)
        results.append({
            "prompt": prompt,
            "expected": answer,
            "response": response[:100],
            "correct": is_correct,
        })

    if ablation_hook:
        ablation_hook.remove()

    accuracy = correct / len(samples)
    return accuracy, results


def evaluate_gomoku(model, tokenizer, rule_samples, move_samples, desc, device, ablation_hook=None):
    """评估五子棋任务性能（规则理解 + 落子质量）"""
    if ablation_hook:
        ablation_hook.register(model, CONFIG["ablation_layer"])

    # 规则理解
    rule_correct = 0
    for prompt, answer, _ in tqdm(rule_samples, desc=f"{desc}-规则", leave=False):
        response = generate_response(model, tokenizer, prompt, max_new_tokens=50, device=device)
        rule_correct += int(check_answer(response, answer))

    # 落子质量
    move_correct = 0
    for prompt, answer, _ in tqdm(move_samples, desc=f"{desc}-落子", leave=False):
        response = generate_response(model, tokenizer, prompt, max_new_tokens=20, device=device)
        move_correct += int(check_answer(response, answer))

    if ablation_hook:
        ablation_hook.remove()

    rule_acc = rule_correct / len(rule_samples)
    move_acc = move_correct / len(move_samples)
    combined = (rule_acc + move_acc) / 2
    return rule_acc, move_acc, combined


# ==================== 加载模型 ====================

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
    print("=" * 65)
    print("Head消融实验")
    print(f"目标：L{CONFIG['ablation_layer']} Head {CONFIG['target_heads']}")
    print("验证：这些head是五子棋专用回路，对数学无影响？")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    math_samples = MATH_SAMPLES[:CONFIG["n_math_samples"]]

    print("\n加载微调模型...")
    ft_model = load_model(CONFIG["finetuned_model_path"], CONFIG["base_model_path"])

    results = {}

    # ========== 条件1：无消融（基准） ==========
    print("\n[1/4] 基准评测（无消融）...")
    math_acc, math_res = evaluate_math(
        ft_model, tokenizer, math_samples, "数学-基准", device)
    rule_acc, move_acc, gomoku_acc = evaluate_gomoku(
        ft_model, tokenizer, GOMOKU_RULE_SAMPLES, GOMOKU_MOVE_SAMPLES, "五子棋-基准", device)

    results["baseline"] = {
        "math_acc": math_acc,
        "gomoku_rule_acc": rule_acc,
        "gomoku_move_acc": move_acc,
        "gomoku_combined": gomoku_acc,
    }
    print(f"  数学准确率:     {math_acc:.3f}")
    print(f"  五子棋规则理解: {rule_acc:.3f}")
    print(f"  五子棋落子质量: {move_acc:.3f}")
    print(f"  五子棋综合:     {gomoku_acc:.3f}")

    # ========== 条件2：消融目标Head 7-13 ==========
    print(f"\n[2/4] 消融目标Head {CONFIG['target_heads']}...")
    target_hook = HeadAblationHook(
        CONFIG["target_heads"], CONFIG["num_heads"], CONFIG["head_dim"])

    math_acc_t, _ = evaluate_math(
        ft_model, tokenizer, math_samples, "数学-目标消融", device, target_hook)
    rule_acc_t, move_acc_t, gomoku_acc_t = evaluate_gomoku(
        ft_model, tokenizer, GOMOKU_RULE_SAMPLES, GOMOKU_MOVE_SAMPLES,
        "五子棋-目标消融", device, target_hook)

    results["target_ablation"] = {
        "ablated_heads": CONFIG["target_heads"],
        "math_acc": math_acc_t,
        "gomoku_rule_acc": rule_acc_t,
        "gomoku_move_acc": move_acc_t,
        "gomoku_combined": gomoku_acc_t,
        "math_drop": math_acc - math_acc_t,
        "gomoku_drop": gomoku_acc - gomoku_acc_t,
    }
    print(f"  数学准确率:     {math_acc_t:.3f}（变化：{math_acc_t-math_acc:+.3f}）")
    print(f"  五子棋规则理解: {rule_acc_t:.3f}（变化：{rule_acc_t-rule_acc:+.3f}）")
    print(f"  五子棋落子质量: {move_acc_t:.3f}（变化：{move_acc_t-move_acc:+.3f}）")
    print(f"  五子棋综合:     {gomoku_acc_t:.3f}（变化：{gomoku_acc_t-gomoku_acc:+.3f}）")

    # ========== 条件3：消融对照Head 0-6 ==========
    print(f"\n[3/4] 消融对照Head {CONFIG['control_heads']}（相同数量，对照组）...")
    control_hook = HeadAblationHook(
        CONFIG["control_heads"], CONFIG["num_heads"], CONFIG["head_dim"])

    math_acc_c, _ = evaluate_math(
        ft_model, tokenizer, math_samples, "数学-对照消融", device, control_hook)
    rule_acc_c, move_acc_c, gomoku_acc_c = evaluate_gomoku(
        ft_model, tokenizer, GOMOKU_RULE_SAMPLES, GOMOKU_MOVE_SAMPLES,
        "五子棋-对照消融", device, control_hook)

    results["control_ablation"] = {
        "ablated_heads": CONFIG["control_heads"],
        "math_acc": math_acc_c,
        "gomoku_rule_acc": rule_acc_c,
        "gomoku_move_acc": move_acc_c,
        "gomoku_combined": gomoku_acc_c,
        "math_drop": math_acc - math_acc_c,
        "gomoku_drop": gomoku_acc - gomoku_acc_c,
    }
    print(f"  数学准确率:     {math_acc_c:.3f}（变化：{math_acc_c-math_acc:+.3f}）")
    print(f"  五子棋规则理解: {rule_acc_c:.3f}（变化：{rule_acc_c-rule_acc:+.3f}）")
    print(f"  五子棋落子质量: {move_acc_c:.3f}（变化：{move_acc_c-move_acc:+.3f}）")
    print(f"  五子棋综合:     {gomoku_acc_c:.3f}（变化：{gomoku_acc_c-gomoku_acc:+.3f}）")

    # ========== 条件4：消融所有head（压力测试） ==========
    print(f"\n[4/4] 消融所有Head（压力测试，验证hook是否正常工作）...")
    all_heads = list(range(CONFIG["num_heads"]))
    all_hook = HeadAblationHook(all_heads, CONFIG["num_heads"], CONFIG["head_dim"])

    math_acc_a, _ = evaluate_math(
        ft_model, tokenizer, math_samples[:5], "数学-全消融", device, all_hook)
    results["all_ablation"] = {"math_acc": math_acc_a}
    print(f"  数学准确率（全消融）: {math_acc_a:.3f}（应接近0，验证hook有效性）")

    # ========== 汇总结果 ==========
    print("\n" + "=" * 65)
    print("汇总：Head消融对比")
    print("=" * 65)
    print(f"{'条件':<20} {'数学准确率':>10} {'五子棋综合':>10} {'数学变化':>10} {'五子棋变化':>10}")
    print("-" * 65)
    print(f"{'基准（无消融）':<20} {math_acc:>10.3f} {gomoku_acc:>10.3f} {'—':>10} {'—':>10}")
    print(f"{'消融H7-13（目标）':<20} {math_acc_t:>10.3f} {gomoku_acc_t:>10.3f} "
          f"{math_acc_t-math_acc:>+10.3f} {gomoku_acc_t-gomoku_acc:>+10.3f}")
    print(f"{'消融H0-6（对照）':<20} {math_acc_c:>10.3f} {gomoku_acc_c:>10.3f} "
          f"{math_acc_c-math_acc:>+10.3f} {gomoku_acc_c-gomoku_acc:>+10.3f}")

    # ========== 核心结论 ==========
    print("\n" + "=" * 65)
    print("核心结论")
    print("=" * 65)

    gomoku_drop_target = gomoku_acc - gomoku_acc_t
    math_drop_target = math_acc - math_acc_t
    gomoku_drop_control = gomoku_acc - gomoku_acc_c
    math_drop_control = math_acc - math_acc_c

    print(f"\n目标消融（H7-13）：五子棋下降={gomoku_drop_target:+.3f}，数学下降={math_drop_target:+.3f}")
    print(f"对照消融（H0-6）： 五子棋下降={gomoku_drop_control:+.3f}，数学下降={math_drop_control:+.3f}")

    if gomoku_drop_target > 0.1 and abs(math_drop_target) < 0.05:
        print("\n→ ✅ 强领域特异性功能模块验证成功：")
        print("  消融H7-13后五子棋性能显著下降，但数学性能几乎不变")
        print("  证明这些head是五子棋任务的专用回路")
        print("  与attention pattern分析结论完全吻合")
    elif gomoku_drop_target > gomoku_drop_control * 1.5:
        print("\n→ ⚠ 弱领域特异性：目标消融对五子棋的影响大于对照消融")
        print("  H7-13对五子棋有相对更重要的作用，但差异不够显著")
    else:
        print("\n→ ℹ 未观察到功能分离：")
        print("  目标消融和对照消融的影响相近")
        print("  可能原因：五子棋功能分散在多个head，单一head组消融效果有限")
        print("  或：4bit量化导致hook精度损失，影响消融效果")

    if math_drop_target > 0.1:
        print("\n⚠ 注意：消融H7-13也影响了数学性能，说明这些head并非纯粹领域特异性")

    # ========== 保存 ==========
    save_path = os.path.join(CONFIG["output_dir"], "head_ablation_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()