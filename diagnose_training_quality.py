"""
LoRA训练质量诊断脚本

回答三个核心问题:
1. 权重层面: LoRA adapter到底动了多少? (Frobenius范数、奇异值)
2. 行为层面: 模型在训练任务上表现如何? (训练集复现、loss)
3. 格式层面: 是真学到内容还是只学到输出格式?

用法:
    python diagnose_training_quality.py

输出:
    ./results/training_quality/<adapter_name>.json (每个adapter的详细诊断)
    ./results/training_quality/summary.json (汇总对比表)
"""

import os
import json
import gc
import re
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ========== 配置 ==========
CONFIG = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "output_dir": "./results/training_quality",
    "adapters": [
        ("gomoku_cot_short",    "./checkpoints/qwen-gomoku-cot-short"),
        ("gomoku_cot_detailed", "./checkpoints/qwen-gomoku-cot-detailed"),
        ("gomoku_dft",          "./checkpoints/qwen-gomoku-dft"),
        ("gomoku_deeplora",     "./checkpoints/qwen-gomoku-deeplora"),
        ("gomoku_grpo",         "./checkpoints/qwen-gomoku-grpo"),
        ("gomoku_ood_monitor",  "./checkpoints/qwen-gomoku-ood-monitor"),
        ("gsm8k_sft",           "./checkpoints/qwen-gsm8k-sft"),
        ("gsm8k_dft",           "./checkpoints/qwen-gsm8k-dft"),
        ("multitask_9to1",      "./checkpoints/qwen-multitask-9to1"),
    ],
    # 训练数据路径(用于采样做复现测试)
    "train_data_candidates": {
        "gomoku": [
            "./datasets/real_games_v2/train.json",
            "./datasets/real_games_v2.json",
        ],
        "gomoku_cot": [
            "./datasets/real_games_detailed_cot/train.json",
            "./datasets/real_games_detailed_cot.json",
        ],
        "gsm8k": [
            "./datasets/gsm8k_sft/train.json",
        ],
    },
    "n_train_samples": 20,  # 从训练集抽样多少条做复现测试
    "max_new_tokens": 256,
    "temperature": 0.0,
}


# ========== 第一层: 权重层面诊断 ==========
def analyze_adapter_weights(adapter_dir):
    """
    直接读取LoRA adapter的权重文件,不加载base model,快速分析:
    - 参数量
    - LoRA A/B矩阵的Frobenius范数
    - BA乘积的范数(实际权重变化量)
    - 每层的"激活强度"分布
    """
    # 找到adapter文件
    adapter_dir = Path(adapter_dir)
    candidates = [
        adapter_dir / "final_model" / "adapter_model.safetensors",
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "final_model" / "adapter_model.bin",
        adapter_dir / "adapter_model.bin",
    ]
    weight_path = next((p for p in candidates if p.exists()), None)
    if weight_path is None:
        return {"error": f"未找到adapter权重文件 in {adapter_dir}"}

    # 加载权重
    if weight_path.suffix == ".safetensors":
        weights = load_file(str(weight_path))
    else:
        weights = torch.load(str(weight_path), map_location="cpu")

    # 读config得到rank
    config_path = weight_path.parent / "adapter_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            adapter_config = json.load(f)
        rank = adapter_config.get("r", None)
        alpha = adapter_config.get("lora_alpha", None)
        target_modules = adapter_config.get("target_modules", [])
    else:
        rank = alpha = None
        target_modules = []

    # 统计lora_A和lora_B
    lora_A_keys = [k for k in weights.keys() if "lora_A" in k]
    lora_B_keys = [k for k in weights.keys() if "lora_B" in k]

    total_params = 0
    norms_A = []
    norms_B = []
    norms_BA = []    # B @ A 的Frobenius范数
    layer_info = []  # 每层的BA范数,用于看深浅分布

    # 按层聚合
    for a_key in sorted(lora_A_keys):
        a_tensor = weights[a_key].float()
        # 对应的B key
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in weights:
            continue
        b_tensor = weights[b_key].float()

        # Frobenius范数
        norm_A = a_tensor.norm().item()
        norm_B = b_tensor.norm().item()
        # 实际权重变化delta = (alpha/r) * B @ A
        scale = (alpha / rank) if (alpha and rank) else 1.0
        delta = scale * (b_tensor @ a_tensor)
        norm_BA = delta.norm().item()

        total_params += a_tensor.numel() + b_tensor.numel()
        norms_A.append(norm_A)
        norms_B.append(norm_B)
        norms_BA.append(norm_BA)

        # 提取层号 (例如 base_model.model.model.layers.12.self_attn.q_proj.lora_A.weight)
        match = re.search(r"layers\.(\d+)\.", a_key)
        layer_idx = int(match.group(1)) if match else -1
        module_match = re.search(r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", a_key)
        module_type = module_match.group(1) if module_match else "unknown"

        layer_info.append({
            "key": a_key,
            "layer": layer_idx,
            "module": module_type,
            "norm_A": norm_A,
            "norm_B": norm_B,
            "norm_BA": norm_BA,
        })

    # 统计
    result = {
        "weight_path": str(weight_path),
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "total_lora_params": total_params,
        "n_lora_modules": len(lora_A_keys),
        "norm_BA_mean": float(np.mean(norms_BA)) if norms_BA else 0,
        "norm_BA_max": float(np.max(norms_BA)) if norms_BA else 0,
        "norm_BA_min": float(np.min(norms_BA)) if norms_BA else 0,
        "norm_BA_std": float(np.std(norms_BA)) if norms_BA else 0,
        "norm_B_mean": float(np.mean(norms_B)) if norms_B else 0,
        # norm_B ≈ 0 说明训练几乎没动(B初始化为0)
        "norm_B_all_near_zero": bool(all(n < 1e-4 for n in norms_B)),
        "layer_info": layer_info,
    }

    # 按层聚合BA范数(看深浅分布)
    by_layer = {}
    for info in layer_info:
        layer = info["layer"]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(info["norm_BA"])
    result["layer_norms"] = {
        str(k): {"mean": float(np.mean(v)), "n_modules": len(v)}
        for k, v in sorted(by_layer.items())
    }

    return result


# ========== 第二层: 训练集复现测试 ==========
def find_training_data(adapter_name, candidates_map):
    """根据adapter名字推断用的是哪个训练集"""
    if "gsm8k" in adapter_name.lower():
        key = "gsm8k"
    elif "cot_detailed" in adapter_name.lower() or "cot-detailed" in adapter_name.lower():
        key = "gomoku_cot"
    elif "gomoku" in adapter_name.lower() or "multitask" in adapter_name.lower():
        key = "gomoku"
    else:
        return None

    for path in candidates_map.get(key, []):
        if Path(path).exists():
            return path, key
    return None, key


def load_training_samples(train_path, n_samples, seed=42):
    """从训练集抽样"""
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), min(n_samples, len(data)), replace=False)
    return [data[i] for i in indices]


def load_model_for_inference(base_model_name, adapter_path=None):
    """加载模型用于推理"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def compute_sample_loss(model, tokenizer, sample):
    """
    对一条训练样本计算loss (label=输出部分的交叉熵)
    这是最能说明"模型是否学到"的指标
    """
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")

    messages = [{"role": "user", "content": instruction}]
    prompt_only = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_only + output + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_only, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    # label中prompt部分mask掉
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    outputs = model(input_ids=full_ids, labels=labels)
    return outputs.loss.item()


@torch.no_grad()
def generate_for_sample(model, tokenizer, sample, max_new_tokens=256):
    """生成并返回模型输出"""
    instruction = sample.get("instruction", "")
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def compare_outputs(predicted, expected, task_type):
    """
    简单的输出比对,返回几个指标:
    - exact_match: 完全匹配
    - contains_answer: 预测中包含期望答案
    - length_ratio: 预测长度/期望长度(检测格式学习)
    """
    pred = predicted.strip()
    exp = expected.strip()

    exact = (pred == exp)
    contains = (exp in pred) if exp else False
    length_ratio = len(pred) / max(len(exp), 1)

    # 任务特定: gomoku看坐标,gsm8k看最终数字
    task_match = False
    if task_type == "gomoku":
        # 提取坐标 (字母+数字 或 数字,数字)
        pred_coords = re.findall(r'[a-oA-O]\d+|\d+,\s*\d+', pred)
        exp_coords = re.findall(r'[a-oA-O]\d+|\d+,\s*\d+', exp)
        task_match = (pred_coords and exp_coords and pred_coords[0].lower() == exp_coords[0].lower())
    elif task_type == "gsm8k":
        # GSM8K答案通常是"#### 数字"
        pred_nums = re.findall(r'####\s*(-?\d+(?:\.\d+)?)', pred)
        exp_nums = re.findall(r'####\s*(-?\d+(?:\.\d+)?)', exp)
        if not pred_nums:
            # fallback: 最后一个数字
            pred_nums = re.findall(r'(-?\d+(?:\.\d+)?)', pred)
        if not exp_nums:
            exp_nums = re.findall(r'(-?\d+(?:\.\d+)?)', exp)
        task_match = (pred_nums and exp_nums and pred_nums[-1] == exp_nums[-1])

    return {
        "exact_match": exact,
        "contains_answer": contains,
        "length_ratio": length_ratio,
        "task_match": task_match,
    }


def evaluate_training_reproduction(model, tokenizer, samples, task_type, max_new_tokens):
    """在训练样本上跑推理并计算指标"""
    losses = []
    match_stats = []
    examples = []

    for i, sample in enumerate(samples):
        try:
            loss = compute_sample_loss(model, tokenizer, sample)
            losses.append(loss)
        except Exception as e:
            print(f"      [loss失败] {e}")
            loss = None

        try:
            pred = generate_for_sample(model, tokenizer, sample, max_new_tokens)
            cmp = compare_outputs(pred, sample.get("output", ""), task_type)
            match_stats.append(cmp)
            if i < 3:  # 保存前3个样例
                examples.append({
                    "instruction": sample.get("instruction", "")[:200],
                    "expected": sample.get("output", "")[:200],
                    "predicted": pred[:200],
                    "loss": loss,
                    "match": cmp,
                })
        except Exception as e:
            print(f"      [生成失败] {e}")

    avg_loss = float(np.mean(losses)) if losses else None
    if match_stats:
        summary = {
            "avg_loss": avg_loss,
            "n_samples": len(match_stats),
            "exact_match_rate": float(np.mean([s["exact_match"] for s in match_stats])),
            "contains_answer_rate": float(np.mean([s["contains_answer"] for s in match_stats])),
            "task_match_rate": float(np.mean([s["task_match"] for s in match_stats])),
            "avg_length_ratio": float(np.mean([s["length_ratio"] for s in match_stats])),
            "examples": examples,
        }
    else:
        summary = {"avg_loss": avg_loss, "error": "no valid samples"}
    return summary


# ========== 主流程 ==========
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    all_results = {}

    # --- 第一阶段: 仅分析权重(不用GPU,很快) ---
    print("=" * 70)
    print("阶段1: 权重层面分析 (无需GPU)")
    print("=" * 70)
    for name, adapter_dir in CONFIG["adapters"]:
        print(f"\n[{name}]")
        if not Path(adapter_dir).exists():
            print(f"  [跳过] 目录不存在: {adapter_dir}")
            continue
        try:
            weight_result = analyze_adapter_weights(adapter_dir)
            if "error" in weight_result:
                print(f"  [错误] {weight_result['error']}")
                continue

            print(f"  rank={weight_result['rank']}, alpha={weight_result['alpha']}")
            print(f"  target_modules={weight_result['target_modules']}")
            print(f"  LoRA模块数: {weight_result['n_lora_modules']}")
            print(f"  总参数量: {weight_result['total_lora_params']:,}")
            print(f"  ||BA||_F: mean={weight_result['norm_BA_mean']:.4f}, "
                  f"max={weight_result['norm_BA_max']:.4f}, "
                  f"std={weight_result['norm_BA_std']:.4f}")
            print(f"  ||B||_F mean: {weight_result['norm_B_mean']:.4f}")

            if weight_result['norm_B_all_near_zero']:
                print(f"  ⚠️  警告: 所有B矩阵接近0,可能未训练!")
            elif weight_result['norm_BA_mean'] < 0.01:
                print(f"  ⚠️  警告: BA范数很小,训练可能不充分")
            else:
                print(f"  ✓ 权重已变化")

            all_results[name] = {"weights": weight_result}
        except Exception as e:
            print(f"  [异常] {e}")
            all_results[name] = {"error": str(e)}

    # --- 第二阶段: 训练集复现测试 ---
    print("\n\n" + "=" * 70)
    print("阶段2: 训练集复现测试 (需要GPU, 较慢)")
    print("=" * 70)

    for name, adapter_dir in CONFIG["adapters"]:
        if name not in all_results or "error" in all_results.get(name, {}):
            continue

        adapter_path = Path(adapter_dir) / "final_model"
        if not adapter_path.exists():
            adapter_path = Path(adapter_dir)

        train_info = find_training_data(name, CONFIG["train_data_candidates"])
        if train_info[0] is None:
            print(f"\n[{name}] 未找到训练数据({train_info[1]}),跳过复现测试")
            all_results[name]["reproduction"] = {"skipped": "no training data found"}
            continue
        train_path, task_type = train_info

        print(f"\n[{name}] 训练数据: {train_path} (task={task_type})")
        samples = load_training_samples(train_path, CONFIG["n_train_samples"])
        print(f"  抽样 {len(samples)} 条训练数据...")

        try:
            model, tokenizer = load_model_for_inference(
                CONFIG["base_model"], str(adapter_path)
            )
            print(f"  跑训练集复现 (max_new_tokens={CONFIG['max_new_tokens']})...")
            repro = evaluate_training_reproduction(
                model, tokenizer, samples, task_type, CONFIG["max_new_tokens"]
            )
            all_results[name]["reproduction"] = repro
            all_results[name]["task_type"] = task_type

            print(f"  平均loss: {repro.get('avg_loss'):.4f}")
            print(f"  任务级匹配率: {repro.get('task_match_rate', 0):.1%}")
            print(f"  字符包含率: {repro.get('contains_answer_rate', 0):.1%}")
            print(f"  平均长度比(pred/exp): {repro.get('avg_length_ratio', 0):.2f}")

            del model
            del tokenizer
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  [复现失败] {e}")
            all_results[name]["reproduction"] = {"error": str(e)}

    # --- 对比base (仅loss) ---
    print("\n\n" + "=" * 70)
    print("阶段3: Base模型在训练数据上的loss (对照组)")
    print("=" * 70)
    base_losses = {}
    model, tokenizer = load_model_for_inference(CONFIG["base_model"], None)
    for task_type in ["gomoku", "gomoku_cot", "gsm8k"]:
        for path in CONFIG["train_data_candidates"].get(task_type, []):
            if Path(path).exists():
                print(f"\n[base on {task_type}] 数据: {path}")
                samples = load_training_samples(path, CONFIG["n_train_samples"])
                losses = []
                for s in samples:
                    try:
                        l = compute_sample_loss(model, tokenizer, s)
                        losses.append(l)
                    except Exception:
                        pass
                if losses:
                    avg = float(np.mean(losses))
                    base_losses[task_type] = avg
                    print(f"  平均loss: {avg:.4f}")
                break
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- 保存结果 ---
    summary_path = Path(CONFIG["output_dir"]) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_losses": base_losses,
            "adapters": all_results,
        }, f, ensure_ascii=False, indent=2)

    # --- 生成对比表 ---
    print("\n\n" + "=" * 90)
    print("训练质量对比表")
    print("=" * 90)
    print(f"{'模型':<22} {'rank':>5} {'|BA|均值':>10} {'任务数据':>12} "
          f"{'base_loss':>10} {'ft_loss':>10} {'loss降':>9} {'任务匹配':>9}")
    print("-" * 90)

    for name, result in all_results.items():
        w = result.get("weights", {})
        r = result.get("reproduction", {})
        task = result.get("task_type", "?")
        base_l = base_losses.get(task, None)
        ft_l = r.get("avg_loss", None)
        loss_drop = (base_l - ft_l) if (base_l and ft_l) else None
        task_match = r.get("task_match_rate", None)

        print(f"{name:<22} "
              f"{w.get('rank', '?'):>5} "
              f"{w.get('norm_BA_mean', 0):>10.4f} "
              f"{task:>12} "
              f"{base_l if base_l else 'N/A':>10}{'':>0} "
              f"{ft_l if ft_l else 'N/A':>10}{'':>0} "
              f"{loss_drop if loss_drop else 'N/A':>9}{'':>0} "
              f"{f'{task_match:.1%}' if task_match is not None else 'N/A':>9}")

    # --- 诊断建议 ---
    print("\n\n" + "=" * 70)
    print("诊断建议")
    print("=" * 70)
    for name, result in all_results.items():
        w = result.get("weights", {})
        r = result.get("reproduction", {})
        issues = []

        if w.get("norm_B_all_near_zero"):
            issues.append("❌ B矩阵接近0,训练基本无效")
        elif w.get("norm_BA_mean", 0) < 0.001:
            issues.append("⚠️ 权重变化极小,训练可能不充分")

        ft_loss = r.get("avg_loss")
        base_loss = base_losses.get(result.get("task_type"))
        if ft_loss and base_loss:
            drop = base_loss - ft_loss
            if drop < 0.1:
                issues.append(f"⚠️ Loss几乎没降(Δ={drop:.3f}),模型没学到")
            elif drop > 0.5:
                issues.append(f"✓ Loss明显下降(Δ={drop:.3f})")

        task_m = r.get("task_match_rate")
        if task_m is not None:
            if task_m < 0.1:
                issues.append(f"⚠️ 训练集任务匹配率极低({task_m:.1%}),可能只学到格式")
            elif task_m > 0.5:
                issues.append(f"✓ 训练集任务匹配率高({task_m:.1%})")

        length_ratio = r.get("avg_length_ratio")
        if length_ratio is not None:
            if length_ratio < 0.3:
                issues.append(f"⚠️ 输出过短({length_ratio:.2f}),可能退化")
            elif length_ratio > 3:
                issues.append(f"⚠️ 输出过长({length_ratio:.2f}),可能重复")

        print(f"\n[{name}]")
        if not issues:
            print("  (无明显问题)")
        else:
            for s in issues:
                print(f"  {s}")

    print(f"\n\n详细结果保存至: {summary_path}")


if __name__ == "__main__":
    main()