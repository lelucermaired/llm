# ==================== Soft-Strict 五子棋评估完整脚本 ====================
# 在你【现有评估流程】基础上，仅替换评估逻辑，不改模型加载、不改数据结构

import torch
import json
import time
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==================== 配置 ====================
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "./checkpoints/qwen7b-gomoku-lora/final_model"
TEST_CASE_FILE = "./datasets/gomoku_diagnostic_dataset_val.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHAPES = ["活二", "活三", "冲四"]
SIDES = ["black", "white"]

# ==================== 模型加载 ====================
def load_model(with_lora=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0} if DEVICE == "cuda" else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )

    if with_lora:
        model = PeftModel.from_pretrained(model, LORA_PATH, is_trainable=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model.eval()
    return model, tokenizer


# ==================== 输出解析 ====================
def parse_prediction(text):
    pred = {
        "black": {k: None for k in SHAPES},
        "white": {k: None for k in SHAPES},
        "winner": None
    }

    current = None
    for line in text.splitlines():
        line = line.strip()

        if line.startswith("1.") and "黑" in line:
            current = "black"
        elif line.startswith("2.") and "白" in line:
            current = "white"

        if current:
            for k in SHAPES:
                if k in line:
                    m = re.search(r"(\d+)", line)
                    if m:
                        pred[current][k] = int(m.group(1))

        if "黑棋获胜" in line:
            pred["winner"] = "BLACK"
        elif "白棋获胜" in line:
            pred["winner"] = "WHITE"
        elif "未分胜负" in line:
            pred["winner"] = "NONE"

    return pred


# ==================== Soft-Strict 评分 ====================
def soft_score(pred, gold):
    score = 0.0
    max_score = 0.0

    for side in SIDES:
        for shape in SHAPES:
            max_score += 1.0
            p = pred[side].get(shape)
            g = gold[f"patterns_{side}"][shape]

            if p is None:
                continue

            # 存在性
            if (p > 0) == (g > 0):
                score += 0.5

            # 数量误差
            score += 0.5 * max(0.0, 1 - abs(p - g) / max(g, 1))

    max_score += 1
    if pred.get("winner") == gold.get("winner"):
        score += 1

    return score / max_score


# ==================== 评估主逻辑 ====================
def build_prompt(item):
    # 兼容 gomoku_diagnostic_dataset 的真实结构
    board = item.get("board", "")
    question = item.get("question", "")
    instruction = item.get("instruction", "")

    return f"""你是一个五子棋裁判，请根据棋盘回答问题。

棋盘：
{board}

问题：
{question or instruction}

请给出黑白双方的棋形分析。
"""


def evaluate(model, tokenizer, data, name):
    print(f"\n{'='*60}")
    print(f"Soft-Strict 评估：{name}")
    print(f"{'='*60}")

    total_score = 0.0
    valid_n = 0

    for i, item in enumerate(data):
        # ===== 核心修复：不再用 item["input"] =====
        prompt = build_prompt(item).strip()

        if not prompt:
            print(f"[{i+1}] ⚠️ prompt 构造失败，跳过")
            continue

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        )

        if inputs["input_ids"].shape[1] == 0:
            print(f"[{i+1}] ⚠️ tokenizer 为空，跳过")
            continue

        if DEVICE == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        pred = parse_prediction(text)
        gold = item["metadata"]

        s = soft_score(pred, gold)
        total_score += s
        valid_n += 1

        print(f"[{i+1}] score={s:.3f}")

    acc = total_score / max(valid_n, 1)
    print(f"\n{name} Soft-Strict 平均得分: {acc:.4f}")
    return acc
# ==================== 主程序 ====================
def main():
    with open(TEST_CASE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_model, base_tok = load_model(with_lora=False)
    base_acc = evaluate(base_model, base_tok, data, "Base")
    del base_model
    torch.cuda.empty_cache()

    lora_model, lora_tok = load_model(with_lora=True)
    lora_acc = evaluate(lora_model, lora_tok, data, "LoRA")

    print("\n" + "="*60)
    print("最终对比结果")
    print("="*60)
    print(f"Base : {base_acc:.4f}")
    print(f"LoRA : {lora_acc:.4f}")
    print(f"Δ提升: {lora_acc - base_acc:+.4f}")


if __name__ == "__main__":
    main()
