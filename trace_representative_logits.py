import argparse

import torch

from mechanism_common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    REPRESENTATIVE_TASKS,
    build_fewshot_messages,
    gold_answer,
    load_model,
    load_task,
    load_tokenizer,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Trace gold first-token probability across layers.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--models", nargs="+", default=["BASE", "GOMOKU_COT", "GO_COT", "GO_NOCOT"])
    parser.add_argument("--tasks", nargs="+", default=REPRESENTATIVE_TASKS)
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output_dir", default=f"{DEFAULT_OUTPUT_DIR}/logit_trace")
    return parser.parse_args()


def first_answer_token_id(tokenizer, answer: str) -> int:
    ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Could not tokenize answer: {answer}")
    return ids[0]


def get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    base = model.get_base_model()
    if hasattr(base, "lm_head"):
        return base.lm_head
    if hasattr(base, "model") and hasattr(base.model, "lm_head"):
        return base.model.lm_head
    raise AttributeError("Could not locate lm_head on model.")


def trace_task(model, tokenizer, task_label: str, n_shot: int, limit: int):
    fewshot = build_fewshot_messages(task_label, n_shot=n_shot)
    ds = load_task(task_label)
    subset = ds.select(range(n_shot, min(n_shot + limit, len(ds))))
    traces = []
    lm_head = get_lm_head(model)

    for idx, ex in enumerate(subset):
        answer = gold_answer(ex, task_label)
        gold_token = first_answer_token_id(tokenizer, answer)
        messages = fewshot + [{"role": "user", "content": ex["input"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        per_layer = []
        for layer_idx, hidden in enumerate(hidden_states[1:]):
            last_hidden = hidden[:, -1, :]
            logits = lm_head(last_hidden)
            probs = torch.softmax(logits, dim=-1)
            prob = probs[0, gold_token].item()
            top_token = int(torch.argmax(probs, dim=-1)[0].item())
            per_layer.append(
                {
                    "layer": layer_idx,
                    "gold_token_prob": prob,
                    "top_token_id": top_token,
                    "top_token_text": tokenizer.decode([top_token]),
                }
            )
        traces.append({"example_index": idx, "gold_answer": answer, "layers": per_layer})
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
                            layer_info["layer"],
                            f"{layer_info['gold_token_prob']:.8f}",
                            layer_info["top_token_id"],
                            layer_info["top_token_text"].replace("\n", "\\n"),
                        ]
                    )
            write_csv(
                f"{args.output_dir}/{model_name.lower()}_{task_label}.csv",
                ["example_index", "gold_answer", "layer", "gold_token_prob", "top_token_id", "top_token_text"],
                rows,
            )
            print(f"  {task_label}: traced {len(traces)} examples")
        del model

    write_json(f"{args.output_dir}/summary.json", overall)
    print("\nSaved:", f"{args.output_dir}/summary.json")


if __name__ == "__main__":
    main()