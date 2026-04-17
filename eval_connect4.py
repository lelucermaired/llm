"""
Connect4 Benchmark 评测脚本 (兼容v1/v2)

v1: 用 difficulty 字段 (easy/medium/hard)
v2: 用 tier 字段 (winning/defensive/double_threat)
脚本自动检测使用哪个字段。

默认配置已包含所有adapter路径,匹配实际checkpoint目录名。
"""

import os
import re
import json
import argparse
import time
import gc
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ========== 默认配置 ==========
DEFAULT_CONFIG = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "benchmark_path": "./datasets/connect4_benchmark_v2/test.json",
    "results_dir": "./results/evaluations/connect4_v2",
    "models": [
        ("base", None),
        ("gomoku_cot_short",    "./checkpoints/qwen-gomoku-cot-short/final_model"),
        ("gomoku_cot_detailed", "./checkpoints/qwen-gomoku-cot-detailed/final_model"),
        ("gomoku_dft",          "./checkpoints/qwen-gomoku-dft/final_model"),
        ("gomoku_deeplora",     "./checkpoints/qwen-gomoku-deeplora/final_model"),
        ("gomoku_grpo",         "./checkpoints/qwen-gomoku-grpo/final_model"),
        ("gomoku_ood_monitor",  "./checkpoints/qwen-gomoku-ood-monitor/final_model"),
        ("gsm8k_sft",           "./checkpoints/qwen-gsm8k-sft/final_model"),
        ("gsm8k_dft",           "./checkpoints/qwen-gsm8k-dft/final_model"),
        ("multitask_9to1",      "./checkpoints/qwen-multitask-9to1/final_model"),
    ],
    "max_new_tokens": 20,
    "temperature": 0.0,
    "batch_size": 1,
}


def get_category(q):
    if "tier" in q:
        return q["tier"]
    if "difficulty" in q:
        return q["difficulty"]
    return "unknown"


def detect_benchmark_version(questions):
    if not questions:
        return "unknown", []
    sample = questions[0]
    if "tier" in sample:
        categories = sorted(set(q["tier"] for q in questions))
        return "tier", categories
    if "difficulty" in sample:
        cats = set(q["difficulty"] for q in questions)
        ordered = [c for c in ["easy", "medium", "hard"] if c in cats]
        ordered += [c for c in cats if c not in ["easy", "medium", "hard"]]
        return "difficulty", ordered
    return "unknown", []


def extract_column(text):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.isdigit():
        n = int(cleaned)
        if 0 <= n <= 6:
            return n
    match = re.search(r'\b([0-6])\b', cleaned)
    if match:
        return int(match.group(1))
    match = re.search(r'[0-6]', cleaned)
    if match:
        return int(match.group(0))
    return None


def load_model(base_model_name, adapter_path=None):
    print(f"  加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  加载base model (4bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    if adapter_path is not None:
        print(f"  加载adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answer(model, tokenizer, instruction, max_new_tokens, temperature):
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def evaluate_model(model, tokenizer, questions, cat_field, categories,
                   max_new_tokens, temperature, model_name):
    results = []
    correct_by_cat = {c: 0 for c in categories}
    total_by_cat = {c: 0 for c in categories}
    invalid_count = 0

    t0 = time.time()
    for i, q in enumerate(questions):
        raw_output = generate_answer(
            model, tokenizer, q["instruction"], max_new_tokens, temperature
        )
        pred_col = extract_column(raw_output)
        optimal = q["optimal_columns"]
        cat = q.get(cat_field, "unknown")

        is_valid = pred_col is not None
        is_correct = is_valid and (pred_col in optimal)

        if cat in total_by_cat:
            total_by_cat[cat] += 1
            if is_correct:
                correct_by_cat[cat] += 1
        if not is_valid:
            invalid_count += 1

        results.append({
            "id": q["id"],
            cat_field: cat,
            "raw_output": raw_output,
            "predicted_col": pred_col,
            "optimal_columns": optimal,
            "is_valid": is_valid,
            "is_correct": is_correct,
        })

        if (i + 1) % 20 == 0 or i == len(questions) - 1:
            elapsed = time.time() - t0
            correct_total = sum(correct_by_cat.values())
            print(f"    [{model_name}] 进度 {i+1}/{len(questions)} | "
                  f"当前准确率: {correct_total/(i+1):.1%} | 耗时 {elapsed:.0f}s")

    total = len(questions)
    correct_total = sum(correct_by_cat.values())

    summary = {
        "model_name": model_name,
        "total_questions": total,
        "correct": correct_total,
        "accuracy": correct_total / total if total > 0 else 0,
        "invalid_outputs": invalid_count,
        "invalid_rate": invalid_count / total if total > 0 else 0,
        "category_field": cat_field,
        "by_category": {
            c: {
                "correct": correct_by_cat[c],
                "total": total_by_cat[c],
                "accuracy": (correct_by_cat[c] / total_by_cat[c]
                             if total_by_cat[c] > 0 else 0),
            }
            for c in categories
        },
    }
    return summary, results


def main(cfg):
    bench_path = Path(cfg["benchmark_path"])
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Benchmark文件不存在: {bench_path}\n"
            f"请先运行benchmark生成脚本"
        )
    with open(bench_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    cat_field, categories = detect_benchmark_version(questions)
    print(f"加载benchmark: {len(questions)} 题 ({bench_path})")
    print(f"版本字段: {cat_field}, 类别: {categories}\n")

    os.makedirs(cfg["results_dir"], exist_ok=True)
    all_summaries = {}

    for model_name, adapter_path in cfg["models"]:
        print(f"\n{'=' * 60}")
        print(f"评测模型: {model_name}")
        print(f"{'=' * 60}")

        if adapter_path is not None and not Path(adapter_path).exists():
            print(f"  [跳过] adapter路径不存在: {adapter_path}")
            continue

        detail_path = Path(cfg["results_dir"]) / f"{model_name}_detailed.json"

        if detail_path.exists():
                print(f"  [跳过] 已存在结果: {detail_path}")
                with open(detail_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                all_summaries[model_name] = prev["summary"]
                continue

        model, tokenizer = load_model(cfg["base_model"], adapter_path)

        summary, detailed = evaluate_model(
            model, tokenizer, questions, cat_field, categories,
            cfg["max_new_tokens"], cfg["temperature"], model_name
        )
        all_summaries[model_name] = summary

        detail_path = Path(cfg["results_dir"]) / f"{model_name}_detailed.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "detailed": detailed,
            }, f, ensure_ascii=False, indent=2)

        print(f"\n  >> {model_name} 总准确率: {summary['accuracy']:.1%} "
              f"({summary['correct']}/{summary['total_questions']})")
        for c in categories:
            bd = summary["by_category"][c]
            print(f"     {c:16s}: {bd['accuracy']:.1%} ({bd['correct']}/{bd['total']})")
        print(f"     无效输出: {summary['invalid_outputs']}/{summary['total_questions']}")

        del model
        del tokenizer
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    summary_path = Path(cfg["results_dir"]) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # 对比表
    print(f"\n\n{'=' * 80}")
    print("模型对比汇总")
    print(f"{'=' * 80}")
    header = f"{'模型':<22} {'总准确率':>10}"
    for c in categories:
        header += f" {c[:12]:>13}"
    header += f" {'无效率':>8}"
    print(header)
    print("-" * len(header))

    for name, s in all_summaries.items():
        row = f"{name:<22} {s['accuracy']:>9.1%}"
        for c in categories:
            row += f" {s['by_category'][c]['accuracy']:>12.1%}"
        row += f" {s['invalid_rate']:>7.1%}"
        print(row)

    if "base" in all_summaries:
        base_acc = all_summaries["base"]["accuracy"]
        base_by_cat = {c: all_summaries["base"]["by_category"][c]["accuracy"]
                       for c in categories}
        print(f"\n相对base的delta:")
        print(f"{'模型':<22} {'总delta':>10}", end="")
        for c in categories:
            print(f" {c[:12]:>13}", end="")
        print()
        print("-" * 80)
        for name, s in all_summaries.items():
            if name == "base":
                continue
            row = f"{name:<22} {s['accuracy']-base_acc:+9.1%}"
            for c in categories:
                d = s["by_category"][c]["accuracy"] - base_by_cat[c]
                row += f" {d:+12.1%}"
            print(row)

    print(f"\n结果已保存至: {cfg['results_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=DEFAULT_CONFIG["benchmark_path"])
    parser.add_argument("--base_model", default=DEFAULT_CONFIG["base_model"])
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--name", default="custom")
    parser.add_argument("--results_dir", default=DEFAULT_CONFIG["results_dir"])
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg["benchmark_path"] = args.benchmark
    cfg["base_model"] = args.base_model
    cfg["results_dir"] = args.results_dir

    if args.adapter:
        cfg["models"] = [
            ("base", None),
            (args.name, args.adapter),
        ]

    main(cfg)