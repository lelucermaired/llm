import argparse
import re
from typing import List

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


TASK_LABEL = "object_counting"


def parse_args():
    parser = argparse.ArgumentParser(description="Trace object_counting with numeric candidate sets.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--models", nargs="+", default=["BASE", "GOMOKU_COT", "GO_COT", "GO_NOCOT"])
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--window", type=int, default=2, help="Compare gold against [gold-window, gold+window].")
    parser.add_argument("--output_dir", default=f"{DEFAULT_OUTPUT_DIR}/object_counting_trace")
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


def normalize_gold(raw: str) -> str:
    text = str(raw).strip()
    match = re.search(r"\b(\d+)\b", text)
    return match.group(1) if match else ""


def build_candidates(gold_text: str, window: int) -> List[str]:
    gold = int(gold_text)
    values = list(range(max(0, gold - window), gold + window + 1))
    return [str(v) for v in values]


def encode_variants(tokenizer, answer: str) -> List[List[int]]:
    variants = []
    for text in [answer, " " + answer]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and ids not in variants:
            variants.append(ids)
    return variants


def score_sequence_per_layer(model, tokenizer, prompt: str, answer: str) -> List[float]:
    variants = encode_variants(tokenizer, answer)
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


def trace_examples(model, tokenizer, n_shot: int, limit: int, window: int):
    fewshot = build_fewshot_messages(TASK_LABEL, n_shot=n_shot)
    ds = load_task(TASK_LABEL)
    subset = ds.select(range(n_shot, min(n_shot + limit, len(ds))))
    traces = []

    for idx, ex in enumerate(subset):
        gold = normalize_gold(ex["target"])
        if not gold:
            print(f"[skip] example={idx} invalid gold={repr(ex['target'])}")
            continue

        messages = fewshot + [{"role": "user", "content": ex["input"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        candidates = build_candidates(gold, window)
        candidate_scores = {}
        for candidate in candidates:
            seq_scores = score_sequence_per_layer(model, tokenizer, prompt, candidate)
            if seq_scores:
                candidate_scores[candidate] = seq_scores

        if gold not in candidate_scores:
            print(f"[skip] example={idx} missing gold candidate {gold}")
            continue

        n_layers = len(candidate_scores[gold])
        layers = []
        for layer_idx in range(n_layers):
            ranked = sorted(
                ((cand, scores[layer_idx]) for cand, scores in candidate_scores.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            gold_score = candidate_scores[gold][layer_idx]
            top_answer, top_score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else None
            layers.append(
                {
                    "layer": layer_idx,
                    "gold_score": gold_score,
                    "top_answer": top_answer,
                    "top_score": top_score,
                    "margin_vs_runner_up": None if runner_up is None else gold_score - runner_up,
                }
            )

        traces.append(
            {
                "example_index": idx,
                "gold_answer": gold,
                "candidates": candidates,
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
        traces = trace_examples(model, tokenizer, args.n_shot, args.limit, args.window)
        overall[model_name] = traces

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
            f"{args.output_dir}/{model_name.lower()}.csv",
            ["example_index", "gold_answer", "candidates", "layer", "gold_score", "top_answer", "top_score", "margin_vs_runner_up"],
            rows,
        )
        print(f"  traced {len(traces)} examples")
        del model

    write_json(f"{args.output_dir}/summary.json", overall)
    print("\nSaved:", f"{args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
