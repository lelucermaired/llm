import argparse
import json
import os
import re
from typing import Iterable, List

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "./checkpoints/mistral7b-gomoku-cot-maxlora/final_model"
DEFAULT_CACHE_DIR = "/root/autodl-tmp/hf_cache"
DEFAULT_OUTPUT = "./results/cross_model/mistral7b_gomoku_cot_representative.json"

TASKS = {
    "object_counting": "object_counting",
    "logical_deduction_3": "logical_deduction_three_objects",
    "dyck_languages": "dyck_languages",
    "tracking_shuffled": "tracking_shuffled_objects_three_objects",
    "colored_objects": "reasoning_about_colored_objects",
}

TASK_MAX_NEW = {
    "object_counting": 256,
    "logical_deduction_3": 256,
    "dyck_languages": 256,
    "tracking_shuffled": 256,
    "colored_objects": 256,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mistral BASE vs Gomoku CoT adapter on representative BBH tasks.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--models", nargs="+", default=["BASE", "GOMOKU_COT"])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(base_model, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name, base_model, adapter, cache_dir):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=make_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if model_name == "GOMOKU_COT":
        model = PeftModel.from_pretrained(model, adapter)
    elif model_name != "BASE":
        raise ValueError(f"Unknown model name: {model_name}")
    model.eval()
    return model


def load_task(task_label, cache_dir):
    return load_dataset("lukaemon/bbh", TASKS[task_label], split="test", cache_dir=cache_dir)


def load_train_examples(task_label, n_shot, cache_dir):
    task_name = TASKS[task_label]
    try:
        train_ds = load_dataset("lukaemon/bbh", task_name, split="train", cache_dir=cache_dir)
        if len(train_ds) >= n_shot:
            return train_ds.select(range(n_shot))
    except Exception:
        pass
    test_ds = load_task(task_label, cache_dir)
    return test_ds.select(range(min(n_shot, len(test_ds))))


def build_fewshot_messages(task_label, n_shot, cache_dir):
    ds = load_train_examples(task_label, n_shot, cache_dir)
    messages = []
    for ex in ds:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": str(ex["target"])})
    return messages


def extract_answer(resp, task_label):
    match = re.search(r"\(([A-Z])\)", resp)
    if match:
        return match.group(1)

    if task_label.startswith("logical_deduction"):
        match = re.search(r"\b([A-G])\b", resp)
        if match:
            return match.group(1)

    if task_label in {"tracking_shuffled", "colored_objects"}:
        matches = re.findall(r"\b([A-E])\b", resp)
        if matches:
            return matches[-1]

    match = re.search(r"\b(Valid|Invalid)\b", resp, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    numbers = re.findall(r"\b\d+\b", resp)
    if numbers:
        return numbers[-1]

    return resp.strip()[:24]


def gold_answer(example, task_label):
    gold = str(example["target"]).strip()
    return re.sub(r"[()]", "", gold).strip()


def generate_response(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return resp


def evaluate_tasks(model, tokenizer, task_labels: List[str], n_shot, limit, cache_dir):
    results = {}
    for task_label in task_labels:
        fewshot = build_fewshot_messages(task_label, n_shot, cache_dir)
        ds = load_task(task_label, cache_dir)
        subset = ds.select(range(n_shot, min(n_shot + limit, len(ds))))
        correct = 0
        records = []

        for local_idx, ex in enumerate(subset):
            messages = fewshot + [{"role": "user", "content": ex["input"]}]
            resp = generate_response(
                model,
                tokenizer,
                messages,
                max_new_tokens=TASK_MAX_NEW.get(task_label, 256),
            )
            pred = re.sub(r"[()]", "", extract_answer(resp, task_label)).strip()
            gold = gold_answer(ex, task_label)
            is_correct = pred == gold
            correct += int(is_correct)
            records.append(
                {
                    "example_index": n_shot + local_idx,
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                    "response": resp,
                }
            )

        results[task_label] = {
            "correct": correct,
            "total": len(subset),
            "acc": correct / max(len(subset), 1),
            "records": records,
        }
    return results


def ensure_dir(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_json(path, data):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_delta_table(all_results):
    if "BASE" not in all_results or "GOMOKU_COT" not in all_results:
        return
    print("\nDelta table")
    print("-" * 72)
    for task in TASKS:
        if task not in all_results["BASE"] or task not in all_results["GOMOKU_COT"]:
            continue
        base_acc = all_results["BASE"][task]["acc"]
        tuned_acc = all_results["GOMOKU_COT"][task]["acc"]
        print(f"{task:<24} BASE={base_acc*100:5.1f}%  GOMOKU_COT={tuned_acc*100:5.1f}%  delta={(tuned_acc-base_acc)*100:+5.1f}%")


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.base_model, args.cache_dir)
    all_results = {}

    for model_name in args.models:
        print("=" * 72)
        print("Evaluating", model_name)
        model = load_model(model_name, args.base_model, args.adapter, args.cache_dir)
        results = evaluate_tasks(
            model,
            tokenizer,
            task_labels=args.tasks,
            n_shot=args.n_shot,
            limit=args.limit,
            cache_dir=args.cache_dir,
        )
        all_results[model_name] = results
        for task, info in results.items():
            print(f"  {task:<24} {info['correct']}/{info['total']} = {info['acc']*100:.1f}%")
        del model

    print_delta_table(all_results)
    write_json(args.output, all_results)
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
