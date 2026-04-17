"""
eval_cot_quality_move.py

评估推理链质量提升后五子棋落子质量的变化
对比模型：v2 / cot-short / cot-detailed

核心问题验证：
- 推理链质量提升是否对应五子棋任务内性能的提升？
- 如果是，说明推理链质量确实影响了任务学习，但不影响OOD迁移
- 如果不是，说明不同推理链对任务学习也没有区别

评测指标：
1. 落子合法率：模型落子是否在空位上
2. 格式正确率：是否输出"最佳落子：XX"格式  
3. 与引擎最优解一致率：落子是否和引擎给出的最优解相同

用法:
    python scripts/evaluation/eval_cot_quality_move.py
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
        # v2: 基础版推理链（伪推理链）
        "v2": "./archive/checkpoints/qwen-gomoku-v2/final_model",
        # cot-short: 结构化简短推理链
        "cot-short": "./checkpoints/qwen-gomoku-cot-short/final_model",
        # cot-detailed: 详细因果推理链
        "cot-detailed": "./checkpoints/qwen-gomoku-cot-detailed/final_model",
    },
    "data_path": "./datasets/real_games_v2/train.json",
    "output_dir": "./results/evaluations/cot_quality_move",
    "n_samples": 100,  # 测试样本数
    "max_new_tokens": 256,  # 推理链可能较长
    "seed": 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
random.seed(CONFIG["seed"])


# ==================== 数据处理 ====================

def load_test_data(data_path, n_samples=100):
    """加载测试数据，提取棋盘、问题、引擎最优解"""
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    # 随机采样
    if len(all_data) > n_samples:
        sampled = random.sample(all_data, n_samples)
    else:
        sampled = all_data
    
    test_cases = []
    for item in sampled:
        # 从instruction中提取棋盘信息
        instruction = item["instruction"]
        
        # 从output中提取引擎最优解（格式：最佳落子：XX）
        output = item["output"]
        match = re.search(r"最佳落子[：:]\s*([A-O]\d{1,2})", output)
        if match:
            optimal_move = match.group(1)
        else:
            # 尝试其他格式
            match2 = re.search(r"落在([A-O]\d{1,2})", output)
            if match2:
                optimal_move = match2.group(1)
            else:
                continue  # 跳过无法解析的样本
        
        # 提取棋盘状态用于合法性验证
        board_state = extract_board_state(instruction)
        
        test_cases.append({
            "instruction": instruction,
            "optimal_move": optimal_move,
            "board_state": board_state,
            "game_file": item.get("game_file", ""),
            "move_idx": item.get("move_idx", -1),
        })
    
    return test_cases


def extract_board_state(instruction):
    """从instruction中提取棋盘状态，用于验证落子合法性"""
    # 棋盘格式：
    #   C D E F G H I J K L M N O
    # 1 · · · · · · ○ · · · · · ·
    # 2 · · · · · · · ● ○ · · · ·
    # ...
    
    board = {}  # {(col, row): 'B'/'W'/'.'}
    
    lines = instruction.split('\n')
    cols_line = None
    for line in lines:
        # 找列标题行
        if re.match(r"\s+[A-O](\s+[A-O])+", line):
            cols_line = line
            break
    
    if cols_line is None:
        return board
    
    # 解析列名
    cols = re.findall(r"[A-O]", cols_line)
    
    # 解析棋盘行
    for line in lines:
        # 匹配：行号 + 棋盘内容
        match = re.match(r"(\d{1,2})([·●○\s]+)", line)
        if match:
            row = int(match.group(1))
            content = match.group(2)
            # 提取每个位置的棋子
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


def is_valid_move(move, board_state):
    """检查落子是否合法（在空位上）"""
    if not move or not board_state:
        return False
    
    # 解析坐标（如 I10 -> (I, 10)）
    match = re.match(r"([A-O])(\d{1,2})", move.upper())
    if not match:
        return False
    
    col, row = match.group(1), int(match.group(2))
    
    # 检查是否在棋盘范围内且为空位
    if (col, row) in board_state:
        return board_state[(col, row)] == '.'
    
    # 如果棋盘状态中没有这个位置，假设在范围内
    return True


# ==================== 模型加载 ====================

def load_base_model(base_path):
    """加载4bit量化基础模型"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def attach_lora(base_model, adapter_path):
    """挂载LoRA适配器"""
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def detach_lora(peft_model):
    """卸载LoRA适配器"""
    base = peft_model.base_model.model
    del peft_model
    torch.cuda.empty_cache()
    gc.collect()
    return base


# ==================== 推理与评估 ====================

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


def parse_move(response):
    """从响应中解析落子坐标"""
    # 优先匹配 "最佳落子[：:]\s*XX" 格式
    patterns = [
        r"最佳落子[：:]\s*([A-O]\d{1,2})",
        r"落在([A-O]\d{1,2})",
        r"落子[：:]\s*([A-O]\d{1,2})",
        r"位置[：:]\s*([A-O]\d{1,2})",
        r"坐标[：:]\s*([A-O]\d{1,2})",
        r"\b([A-O]\d{1,2})\b",  # 最后尝试任意坐标格式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None


def evaluate_model(model, tokenizer, test_cases, model_name, device):
    """评估单个模型"""
    results = []
    
    for case in tqdm(test_cases, desc=f"评估 {model_name}", leave=False):
        response = generate_response(model, tokenizer, case["instruction"], device)
        
        # 解析落子
        predicted_move = parse_move(response)
        
        # 计算各项指标
        has_valid_format = "最佳落子" in response or "落在" in response
        is_legal = is_valid_move(predicted_move, case["board_state"]) if predicted_move else False
        is_optimal = predicted_move == case["optimal_move"] if predicted_move else False
        
        results.append({
            "instruction": case["instruction"][:200] + "...",  # 截断保存
            "optimal_move": case["optimal_move"],
            "predicted_move": predicted_move,
            "has_valid_format": has_valid_format,
            "is_legal": is_legal,
            "is_optimal": is_optimal,
            "response": response[:500],  # 截断保存
        })
    
    return results


# ==================== 主函数 ====================

def main():
    print("=" * 70)
    print("推理链质量 vs 五子棋落子质量评估")
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
    test_cases = load_test_data(CONFIG["data_path"], CONFIG["n_samples"])
    print(f"成功加载 {len(test_cases)} 个测试样本")
    
    # 加载基础模型
    print("\n加载基础模型...")
    base_model = load_base_model(CONFIG["base_model_path"])
    
    all_results = {}
    
    # 评估各模型
    for model_name, model_path in CONFIG["models"].items():
        print(f"\n{'='*60}")
        print(f"评估模型: {model_name}")
        print(f"路径: {model_path}")
        print("=" * 60)
        
        # 挂载LoRA
        model = attach_lora(base_model, model_path)
        
        # 评估
        results = evaluate_model(model, tokenizer, test_cases, model_name, device)
        all_results[model_name] = results
        
        # 计算指标
        format_rate = sum(r["has_valid_format"] for r in results) / len(results)
        legal_rate = sum(r["is_legal"] for r in results) / len(results)
        optimal_rate = sum(r["is_optimal"] for r in results) / len(results)
        
        print(f"\n{model_name} 结果:")
        print(f"  格式正确率: {format_rate:.1%} ({sum(r['has_valid_format'] for r in results)}/{len(results)})")
        print(f"  落子合法率: {legal_rate:.1%} ({sum(r['is_legal'] for r in results)}/{len(results)})")
        print(f"  最优一致率: {optimal_rate:.1%} ({sum(r['is_optimal'] for r in results)}/{len(results)})")
        
        # 卸载LoRA
        base_model = detach_lora(model)
    
    # 清理
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # ==================== 对比分析 ====================
    print("\n" + "=" * 70)
    print("对比分析")
    print("=" * 70)
    
    # 汇总表格
    print(f"\n{'模型':<20} {'格式正确率':>12} {'落子合法率':>12} {'最优一致率':>12}")
    print("-" * 60)
    
    summary = {}
    for model_name, results in all_results.items():
        format_rate = sum(r["has_valid_format"] for r in results) / len(results)
        legal_rate = sum(r["is_legal"] for r in results) / len(results)
        optimal_rate = sum(r["is_optimal"] for r in results) / len(results)
        
        summary[model_name] = {
            "format_rate": format_rate,
            "legal_rate": legal_rate,
            "optimal_rate": optimal_rate,
        }
        
        print(f"{model_name:<20} {format_rate:>11.1%} {legal_rate:>11.1%} {optimal_rate:>11.1%}")
    
    # 差异分析
    print("\n" + "-" * 60)
    print("相对于 v2 的差异:")
    print("-" * 60)
    
    v2_metrics = summary.get("v2", {})
    for model_name, metrics in summary.items():
        if model_name != "v2":
            print(f"\n{model_name} vs v2:")
            for metric in ["format_rate", "legal_rate", "optimal_rate"]:
                delta = metrics[metric] - v2_metrics.get(metric, 0)
                sign = "+" if delta >= 0 else ""
                print(f"  {metric}: {sign}{delta:.1%}")
    
    # ==================== 核心结论 ====================
    print("\n" + "=" * 70)
    print("核心结论")
    print("=" * 70)
    
    # 检查是否有性能提升
    v2_optimal = summary.get("v2", {}).get("optimal_rate", 0)
    cot_short_optimal = summary.get("cot-short", {}).get("optimal_rate", 0)
    cot_detailed_optimal = summary.get("cot-detailed", {}).get("optimal_rate", 0)
    
    if cot_detailed_optimal > v2_optimal + 0.05:
        print("\n[结论] 详细推理链训练显著提升了五子棋落子质量")
        print(f"  v2 -> cot-detailed: +{(cot_detailed_optimal - v2_optimal):.1%}")
        print("  这说明推理链质量确实影响了任务内学习效果")
        print("  但OOD迁移为零可能是因为其他原因，需要进一步分析")
    elif cot_detailed_optimal > v2_optimal:
        print("\n[结论] 详细推理链训练对五子棋落子质量有轻微提升")
        print(f"  v2 -> cot-detailed: +{(cot_detailed_optimal - v2_optimal):.1%}")
        print("  提升幅度有限，可能需要更多训练数据或更长训练")
    else:
        print("\n[结论] 推理链质量提升未带来五子棋落子质量的提升")
        print("  这说明:")
        print("  1. 推理链质量对任务内学习影响有限")
        print("  2. 模型可能已经达到了当前数据规模下的性能上限")
        print("  3. OOD零迁移可能不是SFT框架的问题，而是数据质量的问题")
    
    # ==================== 保存结果 ====================
    output_path = os.path.join(CONFIG["output_dir"], "cot_quality_move_results.json")
    save_data = {
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "detailed_results": all_results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] 结果已保存至: {output_path}")
    
    # 保存易读的汇总报告
    report_path = os.path.join(CONFIG["output_dir"], "cot_quality_move_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("推理链质量 vs 五子棋落子质量评估报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试样本数: {CONFIG['n_samples']}\n\n")
        
        f.write("结果汇总:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'模型':<20} {'格式正确率':>12} {'落子合法率':>12} {'最优一致率':>12}\n")
        f.write("-" * 60 + "\n")
        for model_name, metrics in summary.items():
            f.write(f"{model_name:<20} {metrics['format_rate']:>11.1%} "
                   f"{metrics['legal_rate']:>11.1%} {metrics['optimal_rate']:>11.1%}\n")
        
        f.write("\n核心发现:\n")
        f.write("-" * 60 + "\n")
        if cot_detailed_optimal > v2_optimal + 0.05:
            f.write("详细推理链训练显著提升了五子棋落子质量\n")
        elif cot_detailed_optimal > v2_optimal:
            f.write("详细推理链训练对五子棋落子质量有轻微提升\n")
        else:
            f.write("推理链质量提升未带来五子棋落子质量的提升\n")
    
    print(f"[OK] 报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
