import argparse

from mechanism_common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    REPRESENTATIVE_TASKS,
    evaluate_tasks,
    layer_band,
    load_model,
    load_tokenizer,
    model_num_layers,
    parse_layer_index,
    write_json,
    zero_selected_lora_scaling,
)


MODES = [
    "full",
    "keep_shallow",
    "keep_middle",
    "keep_deep",
    "drop_shallow",
    "drop_middle",
    "drop_deep",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run layer-band LoRA ablation.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model", default="GOMOKU_COT")
    parser.add_argument("--tasks", nargs="+", default=REPRESENTATIVE_TASKS)
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default=f"{DEFAULT_OUTPUT_DIR}/layer_band_ablation/results.json")
    return parser.parse_args()


def make_predicate(mode: str, n_layers: int):
    if mode == "full":
        return lambda name, module: False

    if mode.startswith("keep_"):
        keep_band = mode.split("_", 1)[1]

        def predicate(name, module):
            layer_idx = parse_layer_index(name)
            if layer_idx is None:
                return False
            return layer_band(layer_idx, n_layers) != keep_band

        return predicate

    if mode.startswith("drop_"):
        drop_band = mode.split("_", 1)[1]

        def predicate(name, module):
            layer_idx = parse_layer_index(name)
            if layer_idx is None:
                return False
            return layer_band(layer_idx, n_layers) == drop_band

        return predicate

    raise ValueError(f"Unknown mode: {mode}")


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.base_model)
    model = load_model(args.model, base_model=args.base_model)
    n_layers = model_num_layers(model)
    output = {}

    for mode in MODES:
        print("=" * 60)
        print(f"{args.model} :: {mode}")
        with zero_selected_lora_scaling(model, make_predicate(mode, n_layers)):
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

    write_json(
        args.output,
        {"model": args.model, "num_layers": n_layers, "results": output},
    )
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()