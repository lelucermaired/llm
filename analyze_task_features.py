import argparse
from collections import Counter, defaultdict

from mechanism_common import write_csv


TASK_FEATURES = {
    "object_counting": {"trend": "positive", "SR": 1, "LP": 1, "CC": 0, "DU": 0, "AB": 0, "BJ": 0},
    "logical_deduction_3": {"trend": "positive", "SR": 1, "LP": 0, "CC": 1, "DU": 0, "AB": 1, "BJ": 0},
    "logical_deduction_5": {"trend": "mixed", "SR": 1, "LP": 0, "CC": 1, "DU": 0, "AB": 1, "BJ": 0},
    "logical_deduction_7": {"trend": "mixed", "SR": 1, "LP": 0, "CC": 1, "DU": 0, "AB": 1, "BJ": 0},
    "geometric_shapes": {"trend": "mixed", "SR": 1, "LP": 1, "CC": 1, "DU": 0, "AB": 0, "BJ": 0},
    "temporal_sequences": {"trend": "mixed", "SR": 1, "LP": 0, "CC": 1, "DU": 1, "AB": 1, "BJ": 0},
    "multistep_arithmetic": {"trend": "neutral", "SR": 1, "LP": 0, "CC": 0, "DU": 1, "AB": 0, "BJ": 0},
    "navigate": {"trend": "mixed", "SR": 1, "LP": 0, "CC": 0, "DU": 1, "AB": 0, "BJ": 1},
    "colored_objects": {"trend": "negative", "SR": 1, "LP": 1, "CC": 0, "DU": 0, "AB": 1, "BJ": 0},
    "tracking_shuffled": {"trend": "negative", "SR": 1, "LP": 0, "CC": 0, "DU": 1, "AB": 1, "BJ": 0},
    "causal_judgement": {"trend": "mixed", "SR": 1, "LP": 0, "CC": 1, "DU": 0, "AB": 0, "BJ": 1},
    "web_of_lies": {"trend": "exclude", "SR": 1, "LP": 0, "CC": 1, "DU": 1, "AB": 1, "BJ": 1},
    "ruin_names": {"trend": "positive", "SR": 1, "LP": 1, "CC": 1, "DU": 0, "AB": 0, "BJ": 0},
    "movie_recommendation": {"trend": "positive", "SR": 1, "LP": 1, "CC": 1, "DU": 0, "AB": 1, "BJ": 0},
    "word_sorting": {"trend": "neutral", "SR": 1, "LP": 1, "CC": 1, "DU": 1, "AB": 0, "BJ": 0},
    "dyck_languages": {"trend": "positive", "SR": 1, "LP": 1, "CC": 0, "DU": 1, "AB": 0, "BJ": 0},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize task structural features.")
    parser.add_argument("--output_csv", default="./results/mechanism/task_feature_matrix.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    header = ["task", "trend", "SR", "LP", "CC", "DU", "AB", "BJ"]
    rows = []
    grouped = defaultdict(list)
    for task, feats in TASK_FEATURES.items():
        grouped[feats["trend"]].append(task)
        rows.append([task, feats["trend"], feats["SR"], feats["LP"], feats["CC"], feats["DU"], feats["AB"], feats["BJ"]])
    write_csv(args.output_csv, header, rows)

    print("Task feature matrix saved:", args.output_csv)
    print()

    for trend in ["positive", "negative", "mixed", "neutral", "exclude"]:
        tasks = grouped.get(trend, [])
        if not tasks:
            continue
        counter = Counter()
        for task in tasks:
            for feat_name, feat_value in TASK_FEATURES[task].items():
                if feat_name == "trend":
                    continue
                counter[feat_name] += feat_value
        print(f"[{trend}] {len(tasks)} tasks")
        print("  tasks:", ", ".join(tasks))
        print(
            "  feature frequency:",
            ", ".join(f"{name}={counter[name]}/{len(tasks)}" for name in ["SR", "LP", "CC", "DU", "AB", "BJ"]),
        )
        print()


if __name__ == "__main__":
    main()