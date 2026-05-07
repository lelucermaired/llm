import argparse

from mechanism_common import (
    ATTENTION_MODULES,
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    MLP_MODULES,
    REPRESENTATIVE_TASKS,
    evaluate_tasks,
    load_model,
    load_tokenizer,
    parse_leaf_name,
    write_json,
    zero_selected_lora_scaling,
)


ABLATIONS = ["full", "attention_only", "mlp_only"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run module ablation on LoRA adapters.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model", default="GOMOKU_COT")
    parser.add_argument("--tasks", nargs="+", default=REPRESENTATIVE_TASKS)
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default=f"{DEFAULT_OUTPUT_DIR}/module_ablation/results.json")
    return parser.parse_args()


def predicate_for_mode(mode: str):
    if mode == "attention_only":
        return lambda name, module: parse_leaf_name(name) in MLP_MODULES
    if mode == "mlp_only":
        return lambda name, module: parse_leaf_name(name) in ATTENTION_MODULES
    return lambda name, module: False


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.base_model)
    model = load_model(args.model, base_model=args.base_model)
    output = {}

    for mode in ABLATIONS:
        print("=" * 60)
        print(f"{args.model} :: {mode}")
        with zero_selected_lora_scaling(model, predicate_for_mode(mode)):
            results = evaluate_tasks(
                model,
                tokenizer,
                task_labels=args.tasks,
                n_shot=args.n_shot,
                limit=args.limit,
            )
        output[mode] = results
        for task, info in results.items():
            print(f"  {task:<24} {info['correct']}/{info['total']} = {info['acc']*100:.1f}%")

    write_json(args.output, {"model": args.model, "results": output})
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()