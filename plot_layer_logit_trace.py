import argparse
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from mechanism_common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    build_fewshot_messages,
    load_model,
    load_task,
    load_tokenizer,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Trace candidate logit scores across layers and plot layer curves.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--models", nargs="+", default=["BASE", "GOMOKU_COT", "GO_COT", "GO_NOCOT"])
    parser.add_argument("--task", default="object_counting")
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--window", type=int, default=2, help="Numeric candidate window for object_counting.")
    parser.add_argument("--output_dir", default=f"{DEFAULT_OUTPUT_DIR}/layer_logit_trace")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    base = model.get_base_model()
    if hasattr(base, "lm_head"):
        return base.lm_head
    if hasattr(base, "model") and hasattr(base.model, "lm_head"):
        return base.model.lm_head
    raise AttributeError("Could not locate lm_head on model.")


def normalize_gold(raw: str, task_label: str) -> str:
    text = str(raw).strip()
    text = re.sub(r"[()]", "", text).strip()

    if task_label == "object_counting":
        match = re.search(r"\b(\d+)\b", text)
        return match.group(1) if match else ""

    if task_label.startswith("logical_deduction"):
        match = re.search(r"\b([A-Z])\b", text)
        return match.group(1) if match else text

    if task_label in {"navigate", "causal_judgement", "web_of_lies"}:
        match = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
        return match.group(1).capitalize() if match else text

    match = re.search(r"\b(Valid|Invalid)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    return text


def candidate_answers(task_label: str, prompt_text: str, gold: str, window: int) -> List[str]:
    if task_label == "object_counting":
        value = int(gold)
        return [str(x) for x in range(max(0, value - window), value + window + 1)]

    if task_label.startswith("logical_deduction"):
        labels = sorted(set(re.findall(r"\(([A-Z])\)", prompt_text)))
        return labels or ["A", "B", "C", "D", "E", "F", "G"]

    if task_label in {"navigate", "causal_judgement", "web_of_lies"}:
        return ["Yes", "No"]

    if task_label == "dyck_languages":
        return ["Valid", "Invalid"]

    return [gold]


def encode_variants(tokenizer, answer: str) -> List[List[int]]:
    variants = []
    for text in [answer, " " + answer]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and ids not in variants:
            variants.append(ids)
    return variants


def score_sequence_per_layer(model, tokenizer, lm_head, prompt: str, answer: str) -> List[float]:
    variants = encode_variants(tokenizer, answer)
    best_scores = None

    for answer_ids in variants:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer_ids
        input_ids = torch.tensor([full_ids], device=model.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states[1:]
        start = len(prompt_ids) - 1
        end = len(full_ids) - 1
        scores = []

        for hidden in hidden_states:
            target_hidden = hidden[:, start:end, :]
            logits = lm_head(target_hidden)
            log_probs = torch.log_softmax(logits, dim=-1)
            gold_ids = torch.tensor(answer_ids, device=log_probs.device).view(1, -1, 1)
            token_log_probs = torch.gather(log_probs, 2, gold_ids).squeeze(0).squeeze(-1)
            scores.append(token_log_probs.mean().item())

        if best_scores is None or scores[-1] > best_scores[-1]:
            best_scores = scores

    return best_scores or []


def trace_model(model, tokenizer, task_label: str, n_shot: int, limit: int, window: int):
    fewshot = build_fewshot_messages(task_label, n_shot=n_shot)
    ds = load_task(task_label)
    subset = ds.select(range(n_shot, min(n_shot + limit, len(ds))))
    lm_head = get_lm_head(model)
    traces = []

    for local_idx, ex in enumerate(subset):
        gold = normalize_gold(ex["target"], task_label)
        if not gold:
            print(f"[skip] example={local_idx} empty gold={repr(ex['target'])}")
            continue

        messages = fewshot + [{"role": "user", "content": ex["input"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        candidates = candidate_answers(task_label, ex["input"], gold, window)
        scores_by_candidate = {}

        for candidate in candidates:
            scores = score_sequence_per_layer(model, tokenizer, lm_head, prompt, candidate)
            if scores:
                scores_by_candidate[candidate] = scores

        if gold not in scores_by_candidate:
            print(f"[skip] example={local_idx} missing gold score: {gold}")
            continue

        n_layers = len(scores_by_candidate[gold])
        for layer in range(n_layers):
            ranked = sorted(
                ((cand, scores[layer]) for cand, scores in scores_by_candidate.items()),
                key=lambda x: x[1],
                reverse=True,
            )
            top_answer, top_score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else None
            gold_score = scores_by_candidate[gold][layer]
            margin = gold_score - runner_up if runner_up is not None else 0.0
            traces.append(
                {
                    "example_index": local_idx,
                    "gold": gold,
                    "candidates": candidates,
                    "layer": layer,
                    "gold_score": gold_score,
                    "top_answer": top_answer,
                    "top_score": top_score,
                    "margin_vs_runner_up": margin,
                    "top_is_gold": 1 if top_answer == gold else 0,
                }
            )

    return traces


def summarize_traces(traces):
    by_layer: Dict[int, List[dict]] = {}
    for row in traces:
        by_layer.setdefault(row["layer"], []).append(row)

    summary = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        n = len(rows)
        summary.append(
            {
                "layer": layer,
                "n": n,
                "gold_score": sum(r["gold_score"] for r in rows) / n,
                "top_score": sum(r["top_score"] for r in rows) / n,
                "margin_vs_runner_up": sum(r["margin_vs_runner_up"] for r in rows) / n,
                "top_is_gold_rate": sum(r["top_is_gold"] for r in rows) / n,
            }
        )
    return summary


def write_trace_outputs(output_dir, model_name, task_label, traces, summary):
    os.makedirs(output_dir, exist_ok=True)
    detail_rows = [
        [
            r["example_index"],
            r["gold"],
            "|".join(r["candidates"]),
            r["layer"],
            f"{r['gold_score']:.8f}",
            r["top_answer"],
            f"{r['top_score']:.8f}",
            f"{r['margin_vs_runner_up']:.8f}",
            r["top_is_gold"],
        ]
        for r in traces
    ]
    write_csv(
        os.path.join(output_dir, f"{model_name.lower()}_{task_label}_detail.csv"),
        [
            "example_index",
            "gold",
            "candidates",
            "layer",
            "gold_score",
            "top_answer",
            "top_score",
            "margin_vs_runner_up",
            "top_is_gold",
        ],
        detail_rows,
    )

    summary_rows = [
        [
            r["layer"],
            r["n"],
            f"{r['gold_score']:.8f}",
            f"{r['top_score']:.8f}",
            f"{r['margin_vs_runner_up']:.8f}",
            f"{r['top_is_gold_rate']:.8f}",
        ]
        for r in summary
    ]
    write_csv(
        os.path.join(output_dir, f"{model_name.lower()}_{task_label}_summary.csv"),
        ["layer", "n", "gold_score", "top_score", "margin_vs_runner_up", "top_is_gold_rate"],
        summary_rows,
    )


def plot_metric(all_summaries, task_label, metric, ylabel, output_path, dpi):
    plt.figure(figsize=(8, 4.5))
    for model_name, summary in all_summaries.items():
        layers = [r["layer"] for r in summary]
        values = [r[metric] for r in summary]
        plt.plot(layers, values, marker="o", linewidth=1.8, markersize=3.5, label=model_name)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(f"{task_label}: {ylabel} across layers")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = load_tokenizer(args.base_model)
    all_summaries = {}
    all_json = {}

    for model_name in args.models:
        print("=" * 60)
        print("Tracing", model_name, args.task)
        model = load_model(model_name, base_model=args.base_model)
        traces = trace_model(model, tokenizer, args.task, args.n_shot, args.limit, args.window)
        summary = summarize_traces(traces)
        write_trace_outputs(args.output_dir, model_name, args.task, traces, summary)
        all_summaries[model_name] = summary
        all_json[model_name] = {"summary": summary, "detail": traces}
        print(f"  traced_rows={len(traces)} layers={len(summary)}")
        del model

    write_json(os.path.join(args.output_dir, f"{args.task}_summary.json"), all_json)

    plot_metric(
        all_summaries,
        args.task,
        "gold_score",
        "Gold answer log probability",
        os.path.join(args.output_dir, f"{args.task}_gold_score_by_layer.png"),
        args.dpi,
    )
    plot_metric(
        all_summaries,
        args.task,
        "margin_vs_runner_up",
        "Gold vs runner-up margin",
        os.path.join(args.output_dir, f"{args.task}_margin_by_layer.png"),
        args.dpi,
    )
    plot_metric(
        all_summaries,
        args.task,
        "top_is_gold_rate",
        "Top-is-gold rate",
        os.path.join(args.output_dir, f"{args.task}_top_is_gold_by_layer.png"),
        args.dpi,
    )
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
