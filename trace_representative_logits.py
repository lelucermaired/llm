import argparse
import re
from typing import Dict, List

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
    parser = argparse.ArgumentParser(description="Trace constrained answer scores across layers.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--models", nargs="+", default=["BASE", "GOMOKU_COT", "GO_COT", "GO_NOCOT"])
    parser.add_argument("--tasks", nargs="+", default=REPRESENTATIVE_TASKS)
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output_dir", default=f"{DEFAULT_OUTPUT_DIR}/logit_trace")
    return parser.parse_args()


def normalize_gold_answer(raw: str, task_label: str) -> str:
    text = str(raw).strip()
    text = re.sub(r"[()]", "", text).strip()

    if not text:
        return ""

    if task_label in {"navigate", "causal_judgement", "web_of_lies"}:
        match = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
        return match.group(1).capitalize() if match else text

    if task_label.startswith("logical_deduction"):
        match = re.search(r"\b([A-Z])\b", text)
        return match.group(1) if match else text

    return text


def get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    base = model.get_base_model()
    if hasattr(base, "lm_head"):
        return base.lm_head
    if hasattr(base, "model") and hasattr(base.model, "lm_head"):
        return base.model.lm_head
    raise AttributeError("Could not locate lm_head on model.")


def candidate_answers(task_label: str, prompt_text: str, gold_answer: str) -> List[str]:
    if task_label.startswith("logical_deduction"):
        labels = sorted(set(re.findall(r"\(([A-Z])\)", prompt_text)))
        if labels:
            return labels
        return ["A", "B", "C", "D", "E", "F", "G"]

    if task_label in {"navigate", "causal_judgement", "web_of_lies"}:
        return ["Yes", "No"]

    return [gold_answer]


def encode_answer_variants(tokenizer, answer: str) -> List[List[int]]:
    variants = []
    for text in [answer, " " + answer]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and ids not in variants:
            variants.append(ids)
    return variants


def score_answer_sequence_per_layer(model, tokenizer, prompt: str, answer: str) -> List[float]:
    variants = encode_answer_variants(tokenizer, answer)
    if not variants:
        return []

    lm_head = get_lm_head(model)
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
        seq_scores = []
        start = len(prompt_ids) - 1
        end = len(full_ids) - 1

        for hidden in hidden_states:
            target_hidden = hidden[:, start:end, :]
            logits = lm_head(target_hidden)
            log_probs = torch.log_softmax(logits, dim=-1)
            gold = torch.tensor(answer_ids, device=log_probs.device).unsqueeze(0).unsqueeze(-1)
            token_log_probs = torch.gather(log_probs, 2, gold).squeeze(0).squeeze(-1)
            seq_scores.append(token_log_probs.mean().item())

        if best_scores is None or seq_scores[-1] > best_scores[-1]:
            best_scores = seq_scores

    return best_scores or []


def score_candidates_per_layer(model, tokenizer, prompt: str, answers: List[str]) -> Dict[str, List[float]]:
    scores = {}
    for answer in answers:
        seq_scores = score_answer_sequence_per_layer(model, tokenizer, prompt, answer)
        if seq_scores:
            scores[answer] = seq_scores
    return scores


def trace_task(model, tokenizer, task_label: str, n_shot: int, limit: int):
    fewshot = build_fewshot_messages(task_label, n_shot=n_shot)
    ds = load_task(task_label)
    subset = ds.select(range(n_shot, min(n_shot + limit, len(ds))))
    traces = []

    for idx, ex in enumerate(subset):
        gold_answer = normalize_gold_answer(ex["target"], task_label)
        if not gold_answer:
            print(f"[skip] task={task_label} example={idx} empty gold answer")
            continue

        messages = fewshot + [{"role": "user", "content": ex["input"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        candidates = candidate_answers(task_label, ex["input"], gold_answer)
        candidate_scores = score_candidates_per_layer(model, tokenizer, prompt, candidates)

        if gold_answer not in candidate_scores:
            print(f"[skip] task={task_label} example={idx} missing gold score for {repr(gold_answer)}")
            continue

        n_layers = len(candidate_scores[gold_answer])
        layers = []
        for layer_idx in range(n_layers):
            gold_score = candidate_scores[gold_answer][layer_idx]
            ranked = sorted(
                ((ans, scores[layer_idx]) for ans, scores in candidate_scores.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            top_answer, top_score = ranked[0]
            competitor_score = ranked[1][1] if len(ranked) > 1 else None
            margin = gold_score - competitor_score if competitor_score is not None else None
            layers.append(
                {
                    "layer": layer_idx,
                    "gold_score": gold_score,
                    "top_answer": top_answer,
                    "top_score": top_score,
                    "margin_vs_runner_up": margin,
                }
            )

        traces.append(
            {
                "example_index": idx,
                "gold_answer": gold_answer,
                "candidates": list(candidate_scores.keys()),
                "layers": layers,
            }
        )

    return traces


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.base_model)
    overall = {}

    for model_name in args.models:
        print("=" * 60)
        print("Tracing", model_name)
        model = load_model(model_name, base_model=args.base_model)
        overall[model_name] = {}

        for task_label in args.tasks:
            traces = trace_task(model, tokenizer, task_label, args.n_shot, args.limit)
            overall[model_name][task_label] = traces

            rows = []
            for trace in traces:
                for layer_info in trace["layers"]:
                    rows.append(
                        [
                            trace["example_index"],
                            trace["gold_answer"],
                            "|".join(trace["candidates"]),
                            layer_info["layer"],
                            f"{layer_info['gold_score']:.8f}",
                            layer_info["top_answer"],
                            f"{layer_info['top_score']:.8f}",
                            "" if layer_info["margin_vs_runner_up"] is None else f"{layer_info['margin_vs_runner_up']:.8f}",
                        ]
                    )

            write_csv(
                f"{args.output_dir}/{model_name.lower()}_{task_label}.csv",
                [
                    "example_index",
                    "gold_answer",
                    "candidates",
                    "layer",
                    "gold_score",
                    "top_answer",
                    "top_score",
                    "margin_vs_runner_up",
                ],
                rows,
            )
            print(f"  {task_label}: traced {len(traces)} examples")

        del model

    write_json(f"{args.output_dir}/summary.json", overall)
    print("\nSaved:", f"{args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
