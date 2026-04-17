"""
gomoku_mastery_eval.py

五子棋任务掌握度评估
测试微调模型是否真正学会了五子棋知识，以及学到的是具体知识还是通用推理模式

评估三个维度：
1. 规则理解：能否正确回答五子棋规则相关问题
2. 局面识别：能否识别活三、冲四、必胜局面
3. 落子质量：在明确局面下能否做出合理落子

用法:
    python gomoku_mastery_eval.py
"""

import os
import re
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "models": {
        "base":        None,  # 只用基础模型
        "synthetic":   "./checkpoints/qwen-gomoku-v2/final_model",
        "real":        "./checkpoints/qwen-gomoku-real/final_model",
        "merged":      "./checkpoints/qwen-gomoku-merged/final_model",
    },
    "output_dir": "./mastery_results",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ==================== 评估题目 ====================

# 维度1：规则理解（选择题，有明确正确答案）
RULE_QUESTIONS = [
    {
        "question": "五子棋中，以下哪种说法是正确的？\nA. 白棋先手\nB. 黑棋先手\nC. 随机决定先手\nD. 双方同时落子",
        "answer": "B",
        "category": "规则理解"
    },
    {
        "question": "五子棋的胜利条件是什么？\nA. 占领棋盘中心\nB. 在横竖斜任意方向连成5个同色棋子\nC. 使对方无法落子\nD. 连成4个同色棋子",
        "answer": "B",
        "category": "规则理解"
    },
    {
        "question": "什么叫做'活三'？\nA. 己方已连成三子，且两端都有空位可以延伸\nB. 己方已连成三子，只有一端有空位\nC. 对方连成三子\nD. 棋盘上有三个空位",
        "answer": "A",
        "category": "规则理解"
    },
    {
        "question": "什么叫做'冲四'？\nA. 己方连成四子，两端都开放\nB. 己方连成四子，只有一端有空位可以延伸成五\nC. 己方四子被对方阻断\nD. 强制对方在某位置落子",
        "answer": "B",
        "category": "规则理解"
    },
    {
        "question": "面对对方的'活四'（四子连珠，两端均开放），正确的应对是？\nA. 继续进攻，不必理会\nB. 必须立即在活四的一端落子防守\nC. 在棋盘中心落子\nD. 随意落子",
        "answer": "B",
        "category": "规则理解"
    },
]

# 维度2：局面识别（给出棋盘，判断局面类型）
POSITION_QUESTIONS = [
    {
        "question": """分析以下棋盘局面，黑棋（●）是否形成了"活三"？
棋盘：
  A B C D E F G
4 · · · · · · ·
5 · · ● ● ● · ·
6 · · · · · · ·
请回答：是或否，并说明原因。""",
        "answer": "是",
        "keywords": ["是", "活三", "两端", "空位", "开放"],
        "category": "局面识别"
    },
    {
        "question": """分析以下棋盘局面，白棋（○）下一手能否立即获胜？
棋盘：
  A B C D E F
5 · ○ ○ ○ ○ ·
请回答：是或否，并说明原因。""",
        "answer": "是",
        "keywords": ["是", "活四", "获胜", "五子", "A5", "F5"],
        "category": "局面识别"
    },
    {
        "question": """分析以下棋盘局面，黑棋（●）是否处于必须防守的紧急情况？
棋盘：
  A B C D E F
7 · ○ ○ ○ ○ ·
请回答：是或否，并说明原因。""",
        "answer": "是",
        "keywords": ["是", "活四", "防守", "紧急", "必须", "A7", "F7"],
        "category": "局面识别"
    },
    {
        "question": """以下局面中，黑棋（●）连成了几子？
棋盘：
  E F G H I
8 · ● ● ● ·
请回答具体数字。""",
        "answer": "3",
        "keywords": ["3", "三", "三子"],
        "category": "局面识别"
    },
    {
        "question": """分析以下棋盘，黑棋（●）是否已经获胜？
棋盘：
  D E F G H I
6 · ● ● ● ● ● ·
请回答：是或否。""",
        "answer": "是",
        "keywords": ["是", "获胜", "五子", "连珠", "赢"],
        "category": "局面识别"
    },
]

# 维度3：落子质量（明确局面下的最优落子）
MOVE_QUESTIONS = [
    {
        "question": """你是五子棋专家。以下局面轮到黑棋（●）落子：
棋盘：
  E F G H I J
7 · ● ● ● ● ·
轮到黑棋（●）走。黑棋应该落在哪里立即获胜？请直接给出坐标。""",
        "correct_moves": ["E7", "J7"],
        "wrong_moves": ["F7", "G7", "H7", "I7"],
        "category": "落子质量",
        "type": "winning_move"
    },
    {
        "question": """你是五子棋专家。以下局面轮到黑棋（●）落子：
棋盘：
  D E F G H I
8 · ○ ○ ○ ○ ·
轮到黑棋（●）走。白棋有活四威胁，黑棋必须防守。应该落在哪里？请直接给出坐标。""",
        "correct_moves": ["D8", "I8"],
        "wrong_moves": ["E8", "F8", "G8", "H8"],
        "category": "落子质量",
        "type": "defensive_move"
    },
    {
        "question": """你是五子棋专家。以下局面轮到黑棋（●）落子：
棋盘：
  G H I J K
5 · · ● · ·
6 · ● · · ·
7 ● · · · ·
这是一条斜线上的三子，黑棋应该在哪里继续延伸形成活四？请给出坐标。""",
        "correct_moves": ["J4", "F8", "K4", "E9"],
        "wrong_moves": [],
        "category": "落子质量",
        "type": "attack_move"
    },
    {
        "question": """你是五子棋专家。以下局面轮到黑棋（●）落子：
棋盘：
  G H I J K
8 · · · · ·
9 · ● ● · ·
10 · · · · ·
黑棋有活二，应该如何延伸？给出最佳落子坐标。""",
        "correct_moves": ["G9", "J9", "K9", "F9"],
        "wrong_moves": [],
        "category": "落子质量",
        "type": "attack_move"
    },
]


# ==================== 模型加载与推理 ====================

def load_model(model_path, base_model_path, is_base=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if is_base:
        base.eval()
        return base
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens=150):
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


# ==================== 评估函数 ====================

def eval_rule_questions(model, tokenizer, questions):
    """评估规则理解：选择题"""
    correct = 0
    results = []
    for q in tqdm(questions, desc="规则理解"):
        prompt = f"请回答以下五子棋规则问题，只需给出选项字母（A/B/C/D）：\n\n{q['question']}\n\n答案："
        response = generate(model, tokenizer, prompt, max_new_tokens=20)

        # 提取答案字母
        pred = None
        response_upper = response.strip().upper()
        if response_upper and response_upper[0] in "ABCD":
            pred = response_upper[0]
        else:
            match = re.search(r'\b([ABCD])\b', response_upper)
            if match:
                pred = match.group(1)

        is_correct = (pred == q["answer"])
        correct += is_correct
        results.append({
            "question": q["question"][:50] + "...",
            "expected": q["answer"],
            "predicted": pred,
            "correct": is_correct,
            "response": response[:100]
        })

    return {
        "accuracy": correct / len(questions),
        "correct": correct,
        "total": len(questions),
        "results": results
    }


def eval_position_questions(model, tokenizer, questions):
    """评估局面识别"""
    correct = 0
    results = []
    for q in tqdm(questions, desc="局面识别"):
        prompt = f"你是五子棋专家。{q['question']}"
        response = generate(model, tokenizer, prompt, max_new_tokens=100)

        # 检查关键词匹配
        response_lower = response.lower()
        keyword_hits = sum(1 for kw in q["keywords"] if kw.lower() in response_lower)
        answer_present = q["answer"].lower() in response_lower

        # 判断正确性：答案词出现且关键词至少命中1个
        is_correct = answer_present and keyword_hits >= 1
        correct += is_correct
        results.append({
            "category": q["category"],
            "expected": q["answer"],
            "keyword_hits": keyword_hits,
            "correct": is_correct,
            "response": response[:150]
        })

    return {
        "accuracy": correct / len(questions),
        "correct": correct,
        "total": len(questions),
        "results": results
    }


def eval_move_questions(model, tokenizer, questions):
    """评估落子质量"""
    correct = 0
    results = []
    for q in tqdm(questions, desc="落子质量"):
        prompt = q["question"]
        response = generate(model, tokenizer, prompt, max_new_tokens=80)

        # 提取坐标（如H7, J4等）
        coords = re.findall(r'\b([A-O](?:1[0-5]|[1-9]))\b', response.upper())

        # 判断是否落在正确位置
        is_correct = False
        pred_coord = coords[0] if coords else None

        if q["type"] == "winning_move" or q["type"] == "defensive_move":
            # 必须落在指定位置
            if coords:
                is_correct = any(c in q["correct_moves"] for c in coords)
        else:
            # 进攻类：落在非己方已有子的空位即可
            if coords:
                is_correct = not any(c in q.get("wrong_moves", []) for c in coords)

        correct += is_correct
        results.append({
            "type": q["type"],
            "correct_moves": q["correct_moves"],
            "predicted": pred_coord,
            "all_coords": coords,
            "correct": is_correct,
            "response": response[:150]
        })

    return {
        "accuracy": correct / len(questions),
        "correct": correct,
        "total": len(questions),
        "results": results
    }


# ==================== 主流程 ====================

def evaluate_model(model_name, model_path, tokenizer, is_base=False):
    print(f"\n{'='*50}")
    print(f"评估模型：{model_name}")
    print(f"{'='*50}")

    model = load_model(model_path, CONFIG["base_model_path"], is_base=is_base)

    results = {}

    # 维度1：规则理解
    print("\n[1/3] 规则理解测试...")
    results["rule"] = eval_rule_questions(model, tokenizer, RULE_QUESTIONS)
    print(f"  准确率: {results['rule']['accuracy']:.1%} ({results['rule']['correct']}/{results['rule']['total']})")

    # 维度2：局面识别
    print("\n[2/3] 局面识别测试...")
    results["position"] = eval_position_questions(model, tokenizer, POSITION_QUESTIONS)
    print(f"  准确率: {results['position']['accuracy']:.1%} ({results['position']['correct']}/{results['position']['total']})")

    # 维度3：落子质量
    print("\n[3/3] 落子质量测试...")
    results["move"] = eval_move_questions(model, tokenizer, MOVE_QUESTIONS)
    print(f"  准确率: {results['move']['accuracy']:.1%} ({results['move']['correct']}/{results['move']['total']})")

    # 综合掌握度
    overall = (results["rule"]["accuracy"] + results["position"]["accuracy"] + results["move"]["accuracy"]) / 3
    results["overall_mastery"] = overall
    print(f"\n综合掌握度: {overall:.1%}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    print("=" * 60)
    print("五子棋任务掌握度评估")
    print("区分：具体知识习得 vs 通用推理模式迁移")
    print("=" * 60)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # 评估基础模型
    all_results["base"] = evaluate_model(
        "基础模型（未微调）",
        CONFIG["base_model_path"],
        tokenizer,
        is_base=True
    )

    # 评估各微调模型
    for model_name, model_path in CONFIG["models"].items():
        if model_path is None:
            continue
        if not os.path.exists(model_path):
            print(f"\n跳过 {model_name}：路径不存在 {model_path}")
            continue
        all_results[model_name] = evaluate_model(model_name, model_path, tokenizer)

    # 汇总输出
    print("\n" + "=" * 60)
    print("掌握度评估汇总")
    print("=" * 60)
    print(f"{'模型':<15} {'规则理解':>8} {'局面识别':>8} {'落子质量':>8} {'综合':>8}")
    print("-" * 55)
    for name, res in all_results.items():
        print(f"{name:<15} "
              f"{res['rule']['accuracy']:>7.1%} "
              f"{res['position']['accuracy']:>8.1%} "
              f"{res['move']['accuracy']:>8.1%} "
              f"{res['overall_mastery']:>7.1%}")

    # 分析结论
    print("\n" + "=" * 60)
    print("分析结论")
    print("=" * 60)
    base_mastery = all_results["base"]["overall_mastery"]
    for name, res in all_results.items():
        if name == "base":
            continue
        improvement = res["overall_mastery"] - base_mastery
        rule_imp = res["rule"]["accuracy"] - all_results["base"]["rule"]["accuracy"]
        pos_imp = res["position"]["accuracy"] - all_results["base"]["position"]["accuracy"]
        move_imp = res["move"]["accuracy"] - all_results["base"]["move"]["accuracy"]
        print(f"\n{name} vs 基础模型:")
        print(f"  规则理解提升: {rule_imp:+.1%}")
        print(f"  局面识别提升: {pos_imp:+.1%}")
        print(f"  落子质量提升: {move_imp:+.1%}")
        print(f"  综合提升:     {improvement:+.1%}")
        if improvement > 0.1:
            print(f"  → 模型习得了显著的五子棋领域知识")
        elif improvement > 0.0:
            print(f"  → 模型习得了少量五子棋领域知识")
        else:
            print(f"  → 模型几乎未习得五子棋领域知识（仅学到输出格式）")

    # 保存结果
    save_path = os.path.join(CONFIG["output_dir"], "mastery_results.json")

    # 转换为可序列化格式
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, bool):
            return obj
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(all_results), f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细结果已保存至: {save_path}")


if __name__ == "__main__":
    main()