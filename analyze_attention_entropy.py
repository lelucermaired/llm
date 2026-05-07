import argparse
import math

import torch

from mechanism_common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    REPRESENTATIVE_TASKS,
    build_fewshot_messages,
    load_model,
    load_task,
    load_tokenizer,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute attention entropy on representative tasks.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--models", nargs="+", default=["BASE", "GOMOKU_COT", "GO_COT", "GO_NOCOT"])
    parser.add_argument("--tasks", nargs="+", default=REPRESENTATIVE_TASKS)
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output_dir", default=f"{DEFAULT_OUTPUT_DIR}/attention_entropy")
    return parser.parse_args()


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=-1)


def analyze_task(model, tokenizer, task_label: str, n_shot: int, limit: int):
    fewshot = build_fewshot_messages(task_label, n_shot=n_shot)
    ds = load_task(task_label)
    subset = ds.select(range(n_shot, min(n_shot + limit, len(ds))))
    per_layer_sum = None
    count = 0

    for ex in subset:
        messages = fewshot + [{"role": "user", "content": ex["input"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        attentions = outputs.attentions
        sample_values = []
        for attn in attentions:
            last_token_attn = attn[:, :, -1, :]
            entropy = entropy_from_probs(last_token_attn).mean().item()
            sample_values.append(entropy)
        if per_layer_sum is None:
            per_layer_sum = [0.0 for _ in sample_values]
        for i, value in enumerate(sample_values):
            per_layer_sum[i] += value
        count += 1

    if count == 0:
        return []
    return [value / count for value in per_layer_sum]


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.base_model)
    summary = {}

    for model_name in args.models:
        print("=" * 60)
        print("Entropy", model_name)
        model = load_model(model_name, base_model=args.base_model)
        summary[model_name] = {}
        rows = []
        for task_label in args.tasks:
            values = analyze_task(model, tokenizer, task_label, args.n_shot, args.limit)
            summary[model_name][task_label] = values
            for layer_idx, value in enumerate(values):
                rows.append([task_label, layer_idx, f"{value:.8f}"])
            print(f"  {task_label}: {len(values)} layers")
        write_csv(
            f"{args.output_dir}/{model_name.lower()}.csv",
            ["task", "layer", "entropy"],
            rows,
        )
        del model

    write_json(f"{args.output_dir}/summary.json", summary)
    print("\nSaved:", f"{args.output_dir}/summary.json")


if __name__ == "__main__":
    main()