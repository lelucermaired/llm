"""
chess_prior_knowledge_eval.py

测试基础模型对多种棋类的先验知识水平
验证"预训练数据包含大量棋类知识"假说
对比：五子棋 vs 围棋 vs 中国象棋 vs 国际象棋

用法:
    python chess_prior_knowledge_eval.py
"""

import os
import re
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "output_dir": "./prior_knowledge_results",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 各棋类规则理解题 ====================

GOMOKU_QUESTIONS = [
    {"q": "五子棋中，以下哪种说法是正确的？\nA. 白棋先手\nB. 黑棋先手\nC. 随机决定先手\nD. 双方同时落子", "a": "B"},
    {"q": "五子棋的胜利条件是什么？\nA. 占领棋盘中心\nB. 在横竖斜任意方向连成5个同色棋子\nC. 使对方无法落子\nD. 连成4个同色棋子", "a": "B"},
    {"q": "什么叫做'活三'？\nA. 己方已连成三子，且两端都有空位可以延伸\nB. 己方已连成三子，只有一端有空位\nC. 对方连成三子\nD. 棋盘上有三个空位", "a": "A"},
    {"q": "什么叫做'冲四'？\nA. 己方连成四子，两端都开放\nB. 己方连成四子，只有一端有空位可以延伸成五\nC. 己方四子被对方阻断\nD. 强制对方在某位置落子", "a": "B"},
    {"q": "面对对方的'活四'，正确的应对是？\nA. 继续进攻，不必理会\nB. 必须立即在活四的一端落子防守\nC. 在棋盘中心落子\nD. 随意落子", "a": "B"},
]

GO_QUESTIONS = [
    {"q": "围棋中，以下哪种说法是正确的？\nA. 白棋先手\nB. 黑棋先手\nC. 随机决定先手\nD. 年长者先手", "a": "B"},
    {"q": "围棋的胜利条件是什么？\nA. 吃掉对方所有棋子\nB. 占领更多地盘（围住更多空间）\nC. 连成五子\nD. 到达对方棋盘边缘", "a": "B"},
    {"q": "围棋中什么叫'气'？\nA. 棋子的移动能力\nB. 棋子或棋块紧邻的空交叉点\nC. 棋子的价值\nD. 一种进攻方式", "a": "B"},
    {"q": "围棋中，一块棋被提走的条件是？\nA. 被对方棋子包围超过一圈\nB. 所有的气都被对方占据（气尽）\nC. 棋子数量少于对方\nD. 位于棋盘边缘", "a": "B"},
    {"q": "围棋中'劫'是指什么？\nA. 一种开局策略\nB. 双方可以无限循环提子的局面，规则禁止立即回提\nC. 终局计分方式\nD. 一种防守阵型", "a": "B"},
]

CHINESE_CHESS_QUESTIONS = [
    {"q": "中国象棋中，'将'（帅）的移动规则是？\nA. 可以移动到棋盘任意位置\nB. 只能在九宫格内移动，每次走一步\nC. 可以跳过其他棋子\nD. 只能直线移动不限步数", "a": "B"},
    {"q": "中国象棋中，'马'的走法是？\nA. 直线走任意步\nB. 走'日'字形，先直后斜\nC. 走'田'字形\nD. 只能斜走", "a": "B"},
    {"q": "中国象棋中，'炮'吃子的规则是？\nA. 直线走任意步直接吃子\nB. 需要跳过恰好一个棋子才能吃子\nC. 像马一样走日字吃子\nD. 只能吃相邻的子", "a": "B"},
    {"q": "中国象棋中，'象'（相）的限制是？\nA. 不能过河\nB. 不能吃子\nC. 只能在己方半场活动，且走'田'字\nD. 每次只能走一步", "a": "C"},
    {"q": "中国象棋的胜利条件是？\nA. 消灭对方所有棋子\nB. 将死对方的将（帅）\nC. 占领棋盘中心\nD. 率先走完100步", "a": "B"},
]

CHESS_QUESTIONS = [
    {"q": "国际象棋中，'皇后'的移动规则是？\nA. 只能斜走\nB. 可以直线和斜线走任意步数\nC. 走'L'形\nD. 只能走一格", "a": "B"},
    {"q": "国际象棋中，'马'（骑士）的走法是？\nA. 直线走任意步\nB. 走'L'形，可以跳过其他棋子\nC. 斜走任意步\nD. 只能向前走", "a": "B"},
    {"q": "国际象棋的'王车易位'条件包括？\nA. 王和车都未移动过，且路径无阻挡\nB. 只需王未移动过\nC. 任何时候都可以\nD. 只在开局第一手", "a": "A"},
    {"q": "国际象棋中，'过路兵'（En passant）是指？\nA. 兵可以横向移动\nB. 当对方兵走两格经过己方兵时，可以斜吃该兵\nC. 兵到达底线可以升变\nD. 兵可以后退", "a": "B"},
    {"q": "国际象棋的胜利条件是？\nA. 消灭对方所有棋子\nB. 将死对方的王（King无法逃脱将军）\nC. 占领棋盘中心四格\nD. 率先升变一个兵", "a": "B"},
]

ALL_GAMES = {
    "五子棋": GOMOKU_QUESTIONS,
    "围棋": GO_QUESTIONS,
    "中国象棋": CHINESE_CHESS_QUESTIONS,
    "国际象棋": CHESS_QUESTIONS,
}


# ==================== 推理函数 ====================

def generate(model, tokenizer, prompt, max_new_tokens=20):
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512, padding=False
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].size(1)
    return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()


def extract_answer(response):
    response_upper = response.strip().upper()
    if response_upper and response_upper[0] in "ABCD":
        return response_upper[0]
    match = re.search(r'\b([ABCD])\b', response_upper)
    if match:
        return match.group(1)
    return None


def eval_game(model, tokenizer, game_name, questions):
    correct = 0
    results = []
    for q in tqdm(questions, desc=game_name):
        prompt = f"请回答以下{game_name}规则问题，只需给出选项字母（A/B/C/D）：\n\n{q['q']}\n\n答案："
        response = generate(model, tokenizer, prompt)
        pred = extract_answer(response)
        is_correct = (pred == q["a"])
        correct += is_correct
        results.append({
            "expected": q["a"],
            "predicted": pred,
            "correct": is_correct,
            "response": response[:50]
        })
    accuracy = correct / len(questions)
    print(f"  {game_name}: {accuracy:.1%} ({correct}/{len(questions)})")
    return {"accuracy": accuracy, "correct": correct, "total": len(questions), "results": results}


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("棋类先验知识测试")
    print("验证：预训练模型对各棋类的知识储备")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n加载基础模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    all_results = {}
    print("\n开始测试各棋类先验知识...")
    for game_name, questions in ALL_GAMES.items():
        all_results[game_name] = eval_game(model, tokenizer, game_name, questions)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 汇总
    print("\n" + "=" * 60)
    print("先验知识汇总")
    print("=" * 60)
    print(f"{'棋类':<12} {'准确率':>8} {'正确/总计':>10}")
    print("-" * 35)
    for game, res in all_results.items():
        print(f"{game:<12} {res['accuracy']:>7.1%} {res['correct']:>5}/{res['total']}")

    accuracies = {g: r["accuracy"] for g, r in all_results.items()}
    avg = sum(accuracies.values()) / len(accuracies)
    best = max(accuracies, key=accuracies.get)
    worst = min(accuracies, key=accuracies.get)

    print(f"\n平均准确率: {avg:.1%}")
    print(f"最强: {best} ({accuracies[best]:.1%})")
    print(f"最弱: {worst} ({accuracies[worst]:.1%})")

    gomoku_acc = accuracies.get("五子棋", 0)
    print(f"\n五子棋先验知识 ({gomoku_acc:.1%}) vs 平均 ({avg:.1%})")
    if gomoku_acc >= avg:
        print("→ 基础模型对五子棋的先验知识与其他棋类相当")
        print("→ 说明微调未能在规则理解上超越基础模型，模型学到的是格式而非新知识")
    else:
        print("→ 基础模型对五子棋了解相对较少")
        print("→ 即便如此，微调依然未能提升规则理解，说明LoRA表达能力不足")

    save_path = os.path.join(CONFIG["output_dir"], "prior_knowledge_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()