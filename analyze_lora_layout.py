import argparse
from collections import defaultdict

from mechanism_common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    MODEL_SPECS,
    collect_lora_records,
    load_model,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LoRA weight layout across layers/modules.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--models", nargs="+", default=["GOMOKU_COT", "GOMOKU_NOCOT", "GO_COT", "GO_NOCOT"])
    parser.add_argument("--output_dir", default=f"{DEFAULT_OUTPUT_DIR}/lora_layout")
    return parser.parse_args()


def aggregate(records):
    by_layer = defaultdict(float)
    by_kind = defaultdict(float)
    by_band = defaultdict(float)
    for record in records:
        by_layer[record.layer_idx] += record.delta_norm
        by_kind[record.kind] += record.delta_norm
        by_band[record.band] += record.delta_norm
    return by_layer, by_kind, by_band


def main():
    args = parse_args()
    summary = {}

    for model_name in args.models:
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_name}")
        print(f"Analyzing {model_name} ...")
        model = load_model(model_name, base_model=args.base_model)
        records = collect_lora_records(model)

        by_layer, by_kind, by_band = aggregate(records)
        record_rows = [
            [
                r.module_name,
                r.layer_idx,
                r.band,
                r.leaf_name,
                r.kind,
                f"{r.delta_norm:.8f}",
                f"{r.base_norm:.8f}",
                f"{r.ratio:.8f}",
            ]
            for r in records
        ]
        write_csv(
            f"{args.output_dir}/{model_name.lower()}_records.csv",
            ["module_name", "layer_idx", "band", "leaf_name", "kind", "delta_norm", "base_norm", "ratio"],
            record_rows,
        )

        summary[model_name] = {
            "by_layer": {str(k): v for k, v in sorted(by_layer.items())},
            "by_kind": dict(by_kind),
            "by_band": dict(by_band),
            "num_records": len(records),
        }
        del model
        print(f"  collected {len(records)} LoRA records")

    write_json(f"{args.output_dir}/summary.json", summary)

    for model_name, data in summary.items():
        print()
        print(f"[{model_name}]")
        print("  by_kind:", data["by_kind"])
        print("  by_band:", data["by_band"])


if __name__ == "__main__":
    main()