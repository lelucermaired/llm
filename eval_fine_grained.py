"""
eval_fine_grained.py

细粒度评测脚本，不只看对错，还看：
1. 推理步数：模型生成了几步推理
2. 中间步骤正确率：汉诺塔/积木题检查中间步骤
3. 答案置信度：answer token的logit值

对比：base / v2 / cot_detailed
"""

import os, json, sys, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "models": {
        "base":         None,
        "v2":           "./archive/checkpoints/qwen-gomoku-real/final_model",
        "cot_detailed": "./checkpoints/qwen-gomoku-cot-detailed/final_model",
    },
    "output_dir": "./results/evaluations/fine_grained",
    "max_new_tokens": 200,  # 更长，允许模型展开推理步骤
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 题库（带中间步骤标注）====================

# 汉诺塔：有标准中间步骤
HANOI_SAMPLES = [
    {
        "prompt": "Tower of Hanoi with 2 disks, pegs A B C. Move all from A to C. List each move.",
        "answer": "2",  # 最终步数
        "steps": ["move disk 1 from a to b", "move disk 2 from a to c", "move disk 1 from b to c"],
        "min_moves": 3,
    },
    {
        "prompt": "Tower of Hanoi with 3 disks, pegs A B C. Move all from A to C. How many moves minimum?",
        "answer": "7",
        "steps": ["7"],
        "min_moves": 7,
    },
    {
        "prompt": "Tower of Hanoi: move disk 1 first. Where does it go if moving from A to C using B?",
        "answer": "b",
        "steps": ["b"],
        "min_moves": 1,
    },
    {
        "prompt": "Tower of Hanoi 2 disks A to C via B. First move: disk 1 from A to?",
        "answer": "b",
        "steps": ["b"],
        "min_moves": 1,
    },
    {
        "prompt": "Tower of Hanoi 3 disks. After minimum moves, which peg has all disks?",
        "answer": "c",
        "steps": ["c"],
        "min_moves": 1,
    },
]

# 积木：有标准步骤顺序
BLOCKS_SAMPLES = [
    {
        "prompt": "Blocks: C on B, B on A, A on table. Goal: all on table. List moves in order.",
        "answer": "move c",  # 第一步
        "steps": ["move c", "move b", "move a"],
        "min_moves": 3,
    },
    {
        "prompt": "Blocks: A on table, B on table. Goal: B on A. How many moves?",
        "answer": "1",
        "steps": ["1"],
        "min_moves": 1,
    },
    {
        "prompt": "Blocks: D on C, C on B, B on A. Goal: all separate. First move?",
        "answer": "move d",
        "steps": ["move d"],
        "min_moves": 1,
    },
    {
        "prompt": "Blocks: A on B, B on table. Goal: B on A. Minimum moves?",
        "answer": "2",
        "steps": ["2"],
        "min_moves": 2,
    },
    {
        "prompt": "Blocks: A B C all on table. Goal: C on B on A. What is the last move?",
        "answer": "move c",
        "steps": ["move c on b"],
        "min_moves": 1,
    },
]

# 数学：有明确中间计算步骤
MATH_STEPPED = [
    {
        "prompt": "Solve step by step: If 3x + 6 = 21, what is x?",
        "answer": "5",
        "steps": ["3x = 15", "x = 5"],
        "min_moves": 2,
    },
    {
        "prompt": "Solve step by step: A train travels 120 miles in 2 hours. How far in 5 hours?",
        "answer": "300",
        "steps": ["60 miles per hour", "300"],
        "min_moves": 2,
    },
    {
        "prompt": "Solve step by step: What is 15% of 240?",
        "answer": "36",
        "steps": ["0.15", "36"],
        "min_moves": 2,
    },
    {
        "prompt": "Solve step by step: If a rectangle has perimeter 36 and length 10, what is the width?",
        "answer": "8",
        "steps": ["2 * (10 + width) = 36", "width = 8"],
        "min_moves": 2,
    },
    {
        "prompt": "Solve step by step: What is 2 to the power of 10?",
        "answer": "1024",
        "steps": ["512", "1024"],
        "min_moves": 2,
    },
]

ALL_TASKS = {
    "hanoi":  HANOI_SAMPLES,
    "blocks": BLOCKS_SAMPLES,
    "math_stepped": MATH_STEPPED,
}


# ==================== 细粒度评测函数 ====================

def count_reasoning_steps(response):
    """统计推理步数（按行、句号、编号列表计）"""
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    # 排除纯答案行
    reasoning_lines = [l for l in lines
                       if len(l) > 5 and not l.lower().startswith(('answer', 'result', 'final'))]
    # 按句号分句
    sentences = response.replace('\n', ' ').split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return max(len(reasoning_lines), len(sentences) // 2)


def check_intermediate_steps(response, steps):
    """检查中间步骤命中率"""
    response_lower = response.lower()
    hits = sum(1 for step in steps if step.lower() in response_lower)
    return hits / len(steps) if steps else 0


def get_answer_logit(model, tokenizer, prompt, answer, device):
    """获取answer首token的logit值（置信度）"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_tokens:
        return 0.0
    answer_token_id = answer_tokens[0]

    with torch.no_grad():
        output = model(**inputs)
        last_logits = output.logits[0, -1, :]  # 最后一个位置的logits
        probs = torch.softmax(last_logits.float(), dim=-1)
        answer_prob = probs[answer_token_id].item()
        answer_logit = last_logits[answer_token_id].item()

    return answer_logit, answer_prob


def generate_response(model, tokenizer, prompt, device, max_new_tokens=200):
    """生成完整回答"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()


def evaluate_model_fine(model, tokenizer, device, model_name):
    """细粒度评测一个模型"""
    results = {}

    for task_name, samples in ALL_TASKS.items():
        task_results = []
        for sample in tqdm(samples, desc=f"  {model_name}/{task_name}", leave=False):
            prompt  = sample["prompt"]
            answer  = sample["answer"]
            steps   = sample["steps"]

            # 生成回答
            response = generate_response(model, tokenizer, prompt, device)

            # 1. 最终答案对错
            correct = int(answer.lower() in response.lower())

            # 2. 推理步数
            n_steps = count_reasoning_steps(response)

            # 3. 中间步骤命中率
            step_hit = check_intermediate_steps(response, steps)

            # 4. 答案置信度（logit）
            logit, prob = get_answer_logit(model, tokenizer, prompt, answer, device)

            task_results.append({
                "correct": correct,
                "n_steps": n_steps,
                "step_hit": step_hit,
                "answer_logit": logit,
                "answer_prob": prob,
                "response_len": len(response),
            })

        # 汇总
        results[task_name] = {
            "accuracy":       np.mean([r["correct"]       for r in task_results]),
            "avg_steps":      np.mean([r["n_steps"]       for r in task_results]),
            "step_hit_rate":  np.mean([r["step_hit"]      for r in task_results]),
            "avg_logit":      np.mean([r["answer_logit"]  for r in task_results]),
            "avg_prob":       np.mean([r["answer_prob"]   for r in task_results]),
            "avg_resp_len":   np.mean([r["response_len"]  for r in task_results]),
            "detail":         task_results,
        }

    return results


# ==================== 主流程 ====================

def main():
    print("=" * 65)
    print("细粒度评测：推理步数 / 中间步骤命中率 / 答案置信度")
    print("对比：base / v2 / cot_detailed")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}

    for model_name, adapter_path in CONFIG["models"].items():
        cache_file = os.path.join(CONFIG["output_dir"], f"{model_name}_fine.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                all_results[model_name] = json.load(f)
            print(f"\n[缓存] {model_name}")
            continue

        print(f"\n{'='*50}")
        print(f"加载模型：{model_name}")
        print(f"{'='*50}")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model_path"], quantization_config=bnb,
            device_map="auto", local_files_only=True,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        if adapter_path:
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model
        model.eval()

        results = evaluate_model_fine(model, tokenizer, device, model_name)
        all_results[model_name] = results

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        del model, base_model
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # ==================== 汇总对比 ====================
    print("\n" + "=" * 75)
    print("细粒度评测结果对比")
    print("=" * 75)

    tasks = list(ALL_TASKS.keys())
    metrics = [
        ("accuracy",      "准确率"),
        ("avg_steps",     "平均推理步数"),
        ("step_hit_rate", "中间步骤命中率"),
        ("avg_logit",     "答案token logit"),
        ("avg_prob",      "答案token 概率"),
    ]

    for task in tasks:
        print(f"\n【{task}】")
        print(f"  {'指标':<20}", end="")
        for model_name in all_results:
            print(f"  {model_name:>14}", end="")
        print()
        print("  " + "-" * 65)

        for metric_key, metric_name in metrics:
            print(f"  {metric_name:<20}", end="")
            vals = []
            for model_name, res in all_results.items():
                v = res[task][metric_key]
                vals.append(v)
                if metric_key in ("avg_logit",):
                    print(f"  {v:>14.2f}", end="")
                elif metric_key in ("avg_prob",):
                    print(f"  {v:>14.4f}", end="")
                else:
                    print(f"  {v:>14.3f}", end="")
            print()

    # ==================== 关键发现 ====================
    print("\n" + "=" * 75)
    print("关键发现：相同答案准确率下的内部差异")
    print("=" * 75)

    base_res = all_results.get("base", {})
    for model_name, res in all_results.items():
        if model_name == "base":
            continue
        print(f"\n{model_name} vs base：")
        for task in tasks:
            if task not in res or task not in base_res:
                continue
            acc_delta  = res[task]["accuracy"]      - base_res[task]["accuracy"]
            step_delta = res[task]["avg_steps"]     - base_res[task]["avg_steps"]
            hit_delta  = res[task]["step_hit_rate"] - base_res[task]["step_hit_rate"]
            prob_delta = res[task]["avg_prob"]      - base_res[task]["avg_prob"]
            print(f"  {task:<15}: acc={acc_delta:+.3f}  "
                  f"steps={step_delta:+.2f}  "
                  f"step_hit={hit_delta:+.3f}  "
                  f"prob={prob_delta:+.4f}")

    print("\n解读：")
    print("  steps↑  = 模型生成更长推理链（不一定更准确）")
    print("  step_hit↑ = 推理过程更接近标准步骤（更像真正在推理）")
    print("  prob↑   = 对答案更有把握（即使答案相同，置信度也可能不同）")
    print("  若 acc持平 但 step_hit↑ prob↑，说明模型推理质量在提升但评测题太简单")

    # 保存完整结果
    save_path = os.path.join(CONFIG["output_dir"], "fine_grained_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        # 去掉detail字段（太长）
        save_data = {
            mn: {tn: {k: v for k, v in tv.items() if k != "detail"}
                 for tn, tv in mr.items()}
            for mn, mr in all_results.items()
        }
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()