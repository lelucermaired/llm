"""
eval_hard_gsm8k.py

从GSM8K测试集筛选需要多步推理的难题
对比：base / v2 / cot_detailed

筛选标准：
- 答案步骤数 >= 4步（solution中换行数）
- 数字答案（便于字符串匹配）
- 取前50题

评测指标：
- 准确率（最终答案对错）
- 推理步数（模型生成了几步）
- 答案置信度（answer token logit）
"""

import os, json, re, torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "models": {
        "base": None,
        "ood_final": "./checkpoints/qwen-gomoku-ood-monitor/final_model",
    },
    "output_dir": "./results/evaluations/hard_gsm8k",
    "max_new_tokens": 300,
    "n_questions": 200,
    "min_steps": 4,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def extract_answer(solution):
    """提取GSM8K答案（####后的数字）"""
    m = re.search(r'####\s*([\d,]+)', solution)
    if m:
        return m.group(1).replace(',', '')
    return None


def count_solution_steps(solution):
    """统计标准解题步数"""
    lines = [l.strip() for l in solution.split('\n') if l.strip()]
    return len(lines)


def load_hard_questions(n=50, min_steps=4):
    """从GSM8K测试集筛选难题"""
    print("加载GSM8K测试集...")
    ds = load_dataset("gsm8k", "main")
    test = ds["test"]

    hard = []
    for item in test:
        answer = extract_answer(item["answer"])
        if answer is None:
            continue
        steps = count_solution_steps(item["answer"])
        if steps < min_steps:
            continue
        hard.append({
            "question": item["question"],
            "answer": answer,
            "n_steps": steps,
            "solution": item["answer"],
        })

    # 按步数降序，取最难的n题
    hard.sort(key=lambda x: x["n_steps"], reverse=True)
    selected = hard[:n]

    print(f"✅ 筛选出 {len(selected)} 道难题")
    print(f"   步数范围：{selected[-1]['n_steps']} - {selected[0]['n_steps']} 步")
    print(f"   平均步数：{np.mean([s['n_steps'] for s in selected]):.1f}")

    return selected


def count_reasoning_steps(response):
    """统计模型回答的推理步数"""
    lines = [l.strip() for l in response.split('\n') if l.strip() and len(l.strip()) > 5]
    return len(lines)


def generate_and_eval(model, tokenizer, question, answer, device):
    """生成回答并计算各项指标"""
    prompt = (f"Solve the following math problem step by step.\n\n"
              f"Problem: {question}\n\n"
              f"Show your work and give the final answer.")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    response = tokenizer.decode(
        out.sequences[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()

    # 准确率：答案是否出现在回答中
    correct = int(answer in response.replace(',', ''))

    # 推理步数
    n_steps = count_reasoning_steps(response)

    # 答案token的logit（在生成序列中找答案出现的位置）
    answer_logit = 0.0
    answer_prob = 0.0
    try:
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        if answer_tokens and out.scores:
            # 在生成的token序列中查找答案token位置
            gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:].tolist()
            for pos, tok_id in enumerate(gen_ids):
                if tok_id == answer_tokens[0] and pos < len(out.scores):
                    logits = out.scores[pos][0]
                    probs = torch.softmax(logits.float(), dim=-1)
                    answer_logit = logits[tok_id].item()
                    answer_prob = probs[tok_id].item()
                    break
    except Exception:
        pass

    return {
        "correct": correct,
        "n_steps": n_steps,
        "answer_logit": answer_logit,
        "answer_prob": answer_prob,
        "response": response[:300],  # 只保留前300字符
    }


def evaluate_model(model_name, adapter_path, questions, tokenizer, device):
    """评测单个模型"""
    cache_file = os.path.join(CONFIG["output_dir"], f"{model_name}_hard.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            result = json.load(f)
        print(f"  [缓存] {model_name}: acc={result['accuracy']:.3f}")
        return result

    print(f"\n{'='*50}")
    print(f"加载模型：{model_name}")

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

    details = []
    for q in tqdm(questions, desc=f"  {model_name}"):
        res = generate_and_eval(
            model, tokenizer, q["question"], q["answer"], device)
        details.append(res)

    result = {
        "accuracy":      float(np.mean([d["correct"]      for d in details])),
        "avg_steps":     float(np.mean([d["n_steps"]      for d in details])),
        "avg_logit":     float(np.mean([d["answer_logit"] for d in details])),
        "avg_prob":      float(np.mean([d["answer_prob"]  for d in details])),
        "details":       details,
    }

    print(f"  ✅ {model_name}: acc={result['accuracy']:.3f}  "
          f"steps={result['avg_steps']:.1f}  "
          f"logit={result['avg_logit']:.2f}  "
          f"prob={result['avg_prob']:.4f}")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    del model, base_model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    print("=" * 65)
    print("GSM8K难题评测（多步推理，≥4步）")
    print("对比：base / v2 / cot_detailed")
    print("=" * 65)

    # 加载难题
    questions_cache = os.path.join(CONFIG["output_dir"], "questions.json")
    if os.path.exists(questions_cache):
        with open(questions_cache, encoding="utf-8") as f:
            questions = json.load(f)
        print(f"[缓存] 已加载 {len(questions)} 道难题")
    else:
        questions = load_hard_questions(
            n=CONFIG["n_questions"], min_steps=CONFIG["min_steps"])
        with open(questions_cache, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}
    for model_name, adapter_path in CONFIG["models"].items():
        result = evaluate_model(model_name, adapter_path, questions, tokenizer, device)
        all_results[model_name] = result

    # ==================== 汇总 ====================
    print("\n" + "=" * 65)
    print("GSM8K难题评测结果（50题，≥4步推理）")
    print("=" * 65)

    base = all_results.get("base", {})
    metrics = [
        ("accuracy",  "准确率"),
        ("avg_steps", "平均推理步数"),
        ("avg_logit", "答案logit"),
        ("avg_prob",  "答案概率"),
    ]

    print(f"\n{'指标':<16}", end="")
    for name in all_results:
        print(f"  {name:>14}", end="")
    print()
    print("-" * 65)

    for key, label in metrics:
        print(f"{label:<16}", end="")
        for name, res in all_results.items():
            v = res.get(key, 0)
            if key == "avg_prob":
                print(f"  {v:>14.4f}", end="")
            elif key == "avg_logit":
                print(f"  {v:>14.2f}", end="")
            else:
                print(f"  {v:>14.3f}", end="")
        print()

    print("\n【delta vs base】")
    for name, res in all_results.items():
        if name == "base":
            continue
        acc_d  = res["accuracy"]  - base.get("accuracy", 0)
        step_d = res["avg_steps"] - base.get("avg_steps", 0)
        prob_d = res["avg_prob"]  - base.get("avg_prob", 0)
        print(f"  {name}: acc={acc_d:+.3f}  steps={step_d:+.1f}  prob={prob_d:+.4f}")

    print("\n【结论】")
    for name, res in all_results.items():
        if name == "base":
            continue
        acc_d = res["accuracy"] - base.get("accuracy", 0)
        prob_d = res["avg_prob"] - base.get("avg_prob", 0)
        if acc_d > 0.03:
            print(f"  {name}: ✅ 准确率正向迁移 {acc_d:+.3f}")
        elif prob_d > 0.01:
            print(f"  {name}: 〜 准确率持平但置信度提升 {prob_d:+.4f}，推理质量内部改善")
        else:
            print(f"  {name}: = 无显著变化")

    save_path = os.path.join(CONFIG["output_dir"], "hard_gsm8k_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        without_details = {
            n: {k: v for k, v in r.items() if k != "details"}
            for n, r in all_results.items()
        }
        json.dump(without_details, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()