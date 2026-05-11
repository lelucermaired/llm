import argparse
import csv
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional

import torch
from datasets import load_dataset

from mechanism_common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    generate_response,
    load_model,
    load_tokenizer,
    write_json,
)


PROMPT_VARIANTS = ["cot", "answer_only"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K as a math-transfer and output-format control experiment."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["BASE", "GOMOKU_COT", "GOMOKU_NOCOT", "GO_COT", "GO_NOCOT"],
    )
    parser.add_argument("--variants", nargs="+", default=PROMPT_VARIANTS, choices=PROMPT_VARIANTS)
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument(
        "--output",
        default=f"{DEFAULT_OUTPUT_DIR}/gsm8k_format_control/results.json",
    )
    parser.add_argument(
        "--summary_csv",
        default=f"{DEFAULT_OUTPUT_DIR}/gsm8k_format_control/summary.csv",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Store per-example generations in the JSON output.",
    )
    return parser.parse_args()


def load_gsm8k(split: str, cache_dir: str):
    return load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)


def extract_gsm8k_gold(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)", answer)
    if match:
        return normalize_number(match.group(1))
    numbers = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", answer)
    return normalize_number(numbers[-1]) if numbers else "NONE"


def normalize_number(text: str) -> str:
    cleaned = text.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(".")
    try:
        value = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return cleaned
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def extract_prediction(response: str) -> Dict[str, Optional[str]]:
    marker_match = re.findall(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)", response)
    if marker_match:
        value = normalize_number(marker_match[-1])
        return {
            "answer": value,
            "has_final_marker": True,
            "parse_method": "marker",
        }

    final_match = re.findall(
        r"(?:answer|therefore|so|result|equals|=)\D{0,20}([-+]?\$?[\d,]+(?:\.\d+)?)",
        response,
        flags=re.IGNORECASE,
    )
    if final_match:
        return {
            "answer": normalize_number(final_match[-1]),
            "has_final_marker": False,
            "parse_method": "answer_phrase",
        }

    numbers = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", response)
    if numbers:
        return {
            "answer": normalize_number(numbers[-1]),
            "has_final_marker": False,
            "parse_method": "last_number",
        }

    return {
        "answer": None,
        "has_final_marker": False,
        "parse_method": "unparsed",
    }


def build_fewshot_messages(train_ds, n_shot: int, variant: str):
    messages = []
    for ex in train_ds.select(range(min(n_shot, len(train_ds)))):
        messages.append({"role": "user", "content": build_user_prompt(ex["question"], variant)})
        messages.append({"role": "assistant", "content": build_assistant_target(ex["answer"], variant)})
    return messages


def build_user_prompt(question: str, variant: str) -> str:
    if variant == "cot":
        return (
            "Solve the math word problem step by step. "
            "End the response with a final line in the exact format '#### <answer>'.\n\n"
            f"Problem: {question}"
        )
    if variant == "answer_only":
        return (
            "Solve the math word problem. Reply with only one final line in the exact format "
            "'#### <answer>'.\n\n"
            f"Problem: {question}"
        )
    raise ValueError(f"Unknown prompt variant: {variant}")


def build_assistant_target(answer: str, variant: str) -> str:
    gold = extract_gsm8k_gold(answer)
    if variant == "cot":
        return answer.strip()
    if variant == "answer_only":
        return f"#### {gold}"
    raise ValueError(f"Unknown prompt variant: {variant}")


def evaluate_gsm8k(
    model,
    tokenizer,
    train_ds,
    test_ds,
    variant: str,
    n_shot: int,
    limit: int,
    max_new_tokens: int,
    save_predictions: bool,
):
    fewshot = build_fewshot_messages(train_ds, n_shot=n_shot, variant=variant)
    total = min(limit, len(test_ds))
    correct = 0
    parsed = 0
    final_marker = 0
    predictions: List[Dict[str, str]] = []

    for idx, ex in enumerate(test_ds.select(range(total))):
        messages = fewshot + [{"role": "user", "content": build_user_prompt(ex["question"], variant)}]
        _, response = generate_response(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
        )
        pred_info = extract_prediction(response)
        gold = extract_gsm8k_gold(ex["answer"])
        pred = pred_info["answer"]

        if pred is not None:
            parsed += 1
        if pred_info["has_final_marker"]:
            final_marker += 1
        if pred == gold:
            correct += 1

        if save_predictions:
            predictions.append(
                {
                    "idx": idx,
                    "question": ex["question"],
                    "gold": gold,
                    "pred": pred or "NONE",
                    "correct": pred == gold,
                    "has_final_marker": pred_info["has_final_marker"],
                    "parse_method": pred_info["parse_method"],
                    "response": response,
                }
            )

    result = {
        "correct": correct,
        "total": total,
        "acc": correct / max(total, 1),
        "parse_rate": parsed / max(total, 1),
        "final_marker_rate": final_marker / max(total, 1),
    }
    if save_predictions:
        result["predictions"] = predictions
    return result


def maybe_shuffle(ds, seed: Optional[int]):
    if seed is None:
        return ds
    return ds.shuffle(seed=seed)


def write_summary_csv(path: str, all_results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "variant", "correct", "total", "acc", "parse_rate", "final_marker_rate"])
        for model_name, model_results in all_results.items():
            for variant, info in model_results.items():
                writer.writerow(
                    [
                        model_name,
                        variant,
                        info["correct"],
                        info["total"],
                        f"{info['acc'] * 100:.1f}",
                        f"{info['parse_rate'] * 100:.1f}",
                        f"{info['final_marker_rate'] * 100:.1f}",
                    ]
                )


def print_summary(all_results):
    print("\n" + "=" * 92)
    print("GSM8K summary")
    print("=" * 92)
    print(
        "Model".ljust(14)
        + "Variant".ljust(14)
        + "Acc".rjust(10)
        + "Parsed".rjust(12)
        + "Marker".rjust(12)
        + "Correct".rjust(12)
    )
    print("-" * 92)
    for model_name, model_results in all_results.items():
        for variant, info in model_results.items():
            print(
                model_name.ljust(14)
                + variant.ljust(14)
                + f"{info['acc'] * 100:.1f}%".rjust(10)
                + f"{info['parse_rate'] * 100:.1f}%".rjust(12)
                + f"{info['final_marker_rate'] * 100:.1f}%".rjust(12)
                + f"{info['correct']}/{info['total']}".rjust(12)
            )


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.base_model)
    train_ds = load_gsm8k("train", cache_dir=args.cache_dir)
    test_ds = maybe_shuffle(load_gsm8k("test", cache_dir=args.cache_dir), args.sample_seed)
    all_results = {}

    for model_name in args.models:
        print("=" * 60)
        print("Evaluating", model_name)
        model = load_model(model_name, base_model=args.base_model)
        model_results = {}

        for variant in args.variants:
            result = evaluate_gsm8k(
                model,
                tokenizer,
                train_ds=train_ds,
                test_ds=test_ds,
                variant=variant,
                n_shot=args.n_shot,
                limit=args.limit,
                max_new_tokens=args.max_new_tokens,
                save_predictions=args.save_predictions,
            )
            model_results[variant] = result
            print(
                f"  {variant:<14}"
                f"{result['correct']}/{result['total']} = {result['acc'] * 100:.1f}%"
                f"  parsed={result['parse_rate'] * 100:.1f}%"
                f"  marker={result['final_marker_rate'] * 100:.1f}%"
            )

        all_results[model_name] = model_results
        del model
        torch.cuda.empty_cache()

    write_json(args.output, all_results)
    write_summary_csv(args.summary_csv, all_results)
    print_summary(all_results)
    print("\nSaved JSON:", args.output)
    print("Saved CSV: ", args.summary_csv)


if __name__ == "__main__":
    main()
