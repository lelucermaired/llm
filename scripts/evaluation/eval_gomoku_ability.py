"""
eval_gomoku_ability.py

五子棋能力评估（改进版）
- 放宽匹配条件：只要坐标对了就算对
- 多种坐标格式支持
- 标准化坐标比较

评估维度：
1. 落子合法率：坐标是否有效（在棋盘上且为空位）
2. 最优一致率：是否与引擎推荐一致
3. 有效回答率：能否解析出有效坐标

对比模型：base / v2 / cot-short / cot-detailed

用法:
    python scripts/evaluation/eval_gomoku_ability.py
"""

import os
import re
import json
import torch
import gc
import random
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# ==================== 配置 ====================

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "models": {
        # "base": None,  # 跳过base，假设基线为0%
        "v2": "./archive/checkpoints/qwen-gomoku-v2/final_model",
        # "cot_short": "./checkpoints/qwen-gomoku-cot-short/final_model",
        # "cot_detailed": "./checkpoints/qwen-gomoku-cot-detailed/final_model",
    },
    "data_path": "./datasets/enhanced_v2/train.json",  # 使用训练数据集，格式匹配
    "output_dir": "./results/evaluations/gomoku_ability",
    "n_samples": 10,  # 减少样本数，快速验证
    "max_new_tokens": 512,  # 增加生成长度
    "seed": 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
random.seed(CONFIG["seed"])


# ==================== 坐标处理 ====================

def normalize_coordinate(coord):
    """
    标准化坐标格式
    输入: I10, i10, I10, J8 等各种格式
    输出: 标准格式 "I10"
    """
    if not coord:
        return None
    
    coord = coord.strip().upper()
    
    # 匹配 字母+数字 格式
    match = re.match(r'^([A-O])(\d{1,2})$', coord)
    if match:
        col = match.group(1)
        row = match.group(2)
        return f"{col}{row}"
    
    return None


def extract_all_coordinates(text):
    """
    从文本中提取所有可能的坐标
    返回: [(坐标, 位置), ...]
    """
    results = []
    
    # 各种可能的坐标格式
    patterns = [
        # 中文格式
        (r"最佳落子[：:]\s*([A-Oa-o]\d{1,2})", "best_move_cn"),
        (r"落在\s*([A-Oa-o]\d{1,2})", "fall_at"),
        (r"落子[：:]\s*([A-Oa-o]\d{1,2})", "move_colon"),
        (r"位置[：:]\s*([A-Oa-o]\d{1,2})", "position"),
        (r"坐标[：:]\s*([A-Oa-o]\d{1,2})", "coordinate"),
        (r"选择\s*([A-Oa-o]\d{1,2})", "choose"),
        (r"走\s*([A-Oa-o]\d{1,2})", "play_at"),
        (r"下在\s*([A-Oa-o]\d{1,2})", "place_at"),
        
        # 从候选分析中提取（训练数据格式）
        (r"候选\d*[：:]\s*([A-Oa-o]\d{1,2})", "candidate"),
        (r"最优[：:]\s*([A-Oa-o]\d{1,2})", "optimal_in_thinking"),
        (r"决策[：:]\s*落在\s*([A-Oa-o]\d{1,2})", "decision"),
        
        # 英文格式
        (r"best[:\s]+move[:\s]+([A-Oa-o]\d{1,2})", "best_move_en"),
        (r"move[:\s]+([A-Oa-o]\d{1,2})", "move_en"),
        (r"at\s+([A-Oa-o]\d{1,2})", "at_en"),
        
        # 括号内的坐标
        (r"[（(]([A-Oa-o]\d{1,2})[）)]", "in_parens"),
        
        # 行首或独立坐标
        (r"(?:^|[\s,，。])\s*([A-Oa-o]\d{1,2})\s*(?:$|[\s,，。])", "standalone"),
    ]
    
    for pattern, source in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            coord = normalize_coordinate(match.group(1))
            if coord:
                results.append((coord, source, match.start()))
    
    return results


def parse_move_loose(response):
    """
    宽松解析：从响应中提取最可能的落子坐标
    
    策略：
    1. 优先提取"最佳落子"等关键词后的坐标
    2. 其次提取"最优"后的坐标（训练数据格式）
    3. 再提取候选坐标
    4. 最后提取第一个出现的有效坐标
    """
    coords = extract_all_coordinates(response)
    
    if not coords:
        return None, []
    
    # 按优先级排序
    priority = {
        "best_move_cn": 1,
        "best_move_en": 1,
        "optimal_in_thinking": 2,  # 训练数据格式
        "decision": 2,
        "fall_at": 3,
        "move_colon": 3,
        "position": 3,
        "coordinate": 3,
        "candidate": 4,  # 候选坐标
        "choose": 4,
        "play_at": 4,
        "place_at": 4,
        "move_en": 5,
        "at_en": 5,
        "in_parens": 6,
        "standalone": 7,
    }
    
    # 排序：优先级高的在前
    sorted_coords = sorted(coords, key=lambda x: priority.get(x[1], 99))
    
    # 返回第一个（最高优先级）
    return sorted_coords[0][0], sorted_coords


# ==================== 棋盘状态 ====================

def extract_board_state(instruction):
    """从instruction中提取棋盘状态"""
    board = {}
    
    lines = instruction.split('\n')
    cols_line = None
    for line in lines:
        if re.match(r"\s+[A-O](\s+[A-O])+", line):
            cols_line = line
            break
    
    if cols_line is None:
        return board
    
    cols = re.findall(r"[A-O]", cols_line)
    
    for line in lines:
        match = re.match(r"(\d{1,2})([·●○\s]+)", line)
        if match:
            row = int(match.group(1))
            content = match.group(2)
            stones = re.findall(r"[·●○]", content)
            for i, stone in enumerate(stones):
                if i < len(cols):
                    col = cols[i]
                    if stone == '●':
                        board[(col, row)] = 'B'
                    elif stone == '○':
                        board[(col, row)] = 'W'
                    else:
                        board[(col, row)] = '.'
    
    return board


def is_valid_move(coord, board_state):
    """检查坐标是否有效（在棋盘范围内且为空位）"""
    if not coord or not board_state:
        return False
    
    match = re.match(r"([A-O])(\d{1,2})", coord.upper())
    if not match:
        return False
    
    col, row = match.group(1), int(match.group(2))
    
    # 检查是否在棋盘范围内
    if col < 'A' or col > 'O' or row < 1 or row > 15:
        return False
    
    # 检查是否为空位
    if (col, row) in board_state:
        return board_state[(col, row)] == '.'
    
    return True


# ==================== 模型加载 ====================

def cleanup_model(model):
    """彻底清理模型和显存"""
    if hasattr(model, 'base_model'):
        base = model.base_model.model
        del model
        model = base
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def load_model(base_path, adapter_path=None):
    """加载模型"""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "12GB", "cpu": "8GB"},
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


# ==================== 评估 ====================

def generate_response(model, tokenizer, prompt, device):
    """生成模型响应"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    return response


def evaluate_model(model, tokenizer, test_cases, model_name, device):
    """评估单个模型"""
    results = []
    
    for case in tqdm(test_cases, desc=f"评估 {model_name}", leave=False):
        response = generate_response(model, tokenizer, case["instruction"], device)
        
        # 宽松解析
        predicted_move, all_coords = parse_move_loose(response)
        
        # 标准化最优解
        optimal_normalized = normalize_coordinate(case["optimal_move"])
        
        # 计算指标
        has_valid_answer = predicted_move is not None
        is_legal = is_valid_move(predicted_move, case["board_state"]) if predicted_move else False
        is_optimal = (predicted_move == optimal_normalized) if predicted_move and optimal_normalized else False
        
        # 检查是否有多个候选坐标
        unique_coords = list(set(c[0] for c in all_coords))
        has_multiple = len(unique_coords) > 1
        
        results.append({
            "optimal_move": optimal_normalized,
            "predicted_move": predicted_move,
            "all_coords_found": unique_coords,
            "has_valid_answer": has_valid_answer,
            "is_legal": is_legal,
            "is_optimal": is_optimal,
            "has_multiple_candidates": has_multiple,
            "response": response[:600],  # 保存更多内容
        })
    
    return results


# ==================== 主函数 ====================

def main():
    print("=" * 70)
    print("五子棋能力评估（改进版）")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试样本数: {CONFIG['n_samples']}")
    print(f"对比模型: {list(CONFIG['models'].keys())}")
    print()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载测试数据
    print("加载测试数据...")
    with open(CONFIG["data_path"], "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    # 只选择五子棋决策类型的数据
    gomoku_data = [item for item in all_data 
                   if item.get("metadata", {}).get("type") == "decision"]
    print(f"五子棋决策数据: {len(gomoku_data)} 条")
    
    if len(gomoku_data) > CONFIG["n_samples"]:
        sampled = random.sample(gomoku_data, CONFIG["n_samples"])
    else:
        sampled = gomoku_data
    
    test_cases = []
    for item in sampled:
        # enhanced_v2 格式
        instruction = item["instruction"]
        output = item["output"]
        
        # 从output中提取最佳落子
        match = re.search(r"最佳落子[：:]\s*([A-Oa-o]\d{1,2})", output)
        if match:
            optimal_move = normalize_coordinate(match.group(1))
        else:
            continue
        
        test_cases.append({
            "instruction": instruction,
            "optimal_move": optimal_move,
            "board_state": extract_board_state(instruction),
        })
    
    print(f"成功加载 {len(test_cases)} 个测试样本")
    
    all_results = {}
    
    # 评估各模型
    for model_name, adapter_path in CONFIG["models"].items():
        print(f"\n{'='*60}")
        print(f"评估模型: {model_name}")
        print("=" * 60)
        
        model = load_model(CONFIG["base_model_path"], adapter_path)
        results = evaluate_model(model, tokenizer, test_cases, model_name, device)
        all_results[model_name] = results
        
        # 计算指标
        valid_rate = sum(r["has_valid_answer"] for r in results) / len(results)
        legal_rate = sum(r["is_legal"] for r in results) / len(results)
        optimal_rate = sum(r["is_optimal"] for r in results) / len(results)
        
        print(f"\n{model_name} 结果:")
        print(f"  有效回答率: {valid_rate:.1%} ({sum(r['has_valid_answer'] for r in results)}/{len(results)})")
        print(f"  落子合法率: {legal_rate:.1%} ({sum(r['is_legal'] for r in results)}/{len(results)})")
        print(f"  最优一致率: {optimal_rate:.1%} ({sum(r['is_optimal'] for r in results)}/{len(results)})")
        
        # 打印前3个样本的实际输出，调试用
        print(f"\n  前3个样本的实际输出:")
        for i, r in enumerate(results[:3]):
            print(f"  样本{i+1} 最优={r['optimal_move']} 预测={r['predicted_move']}")
            print(f"    找到的坐标: {r['all_coords_found']}")
            print(f"    完整响应: {r['response'][:400]}")
        
        cleanup_model(model)
    
    # ==================== 对比分析 ====================
    print("\n" + "=" * 70)
    print("对比分析")
    print("=" * 70)
    
    # 汇总表格
    print(f"\n{'模型':<15} {'有效回答率':>12} {'落子合法率':>12} {'最优一致率':>12}")
    print("-" * 55)
    
    summary = {}
    for model_name, results in all_results.items():
        valid_rate = sum(r["has_valid_answer"] for r in results) / len(results)
        legal_rate = sum(r["is_legal"] for r in results) / len(results)
        optimal_rate = sum(r["is_optimal"] for r in results) / len(results)
        
        summary[model_name] = {
            "valid_rate": valid_rate,
            "legal_rate": legal_rate,
            "optimal_rate": optimal_rate,
        }
        
        print(f"{model_name:<15} {valid_rate:>11.1%} {legal_rate:>11.1%} {optimal_rate:>11.1%}")
    
    # 相对于base的变化
    print("\n" + "-" * 55)
    print("相对于 base 的变化 (Δ)")
    print("-" * 55)
    
    base_metrics = summary.get("base", {})
    for model_name, metrics in summary.items():
        if model_name != "base":
            print(f"\n{model_name}:")
            for metric in ["valid_rate", "legal_rate", "optimal_rate"]:
                delta = metrics[metric] - base_metrics.get(metric, 0)
                sign = "+" if delta >= 0 else ""
                print(f"  {metric}: {sign}{delta:.1%}")
    
    # ==================== 核心结论 ====================
    print("\n" + "=" * 70)
    print("核心结论")
    print("=" * 70)
    
    base_optimal = summary.get("base", {}).get("optimal_rate", 0)
    
    improvements = []
    for model_name, metrics in summary.items():
        if model_name != "base":
            delta = metrics["optimal_rate"] - base_optimal
            if delta > 0.05:
                improvements.append(f"✅ {model_name}: 最优一致率提升 +{delta:.1%}")
            elif delta > 0:
                improvements.append(f"➖ {model_name}: 最优一致率轻微提升 +{delta:.1%}")
            else:
                improvements.append(f"❌ {model_name}: 最优一致率未提升 ({delta:+.1%})")
    
    for imp in improvements:
        print(imp)
    
    # ==================== 保存结果 ====================
    output_path = os.path.join(CONFIG["output_dir"], "gomoku_ability_results.json")
    save_data = {
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "detailed_results": {k: v[:10] for k, v in all_results.items()},  # 只保存前10条
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n[OK] 结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
