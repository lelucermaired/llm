"""
多任务数据集混合脚本
将五子棋数据（real_games_v2）与 GSM8K 数学数据按 9:1 比例混合，
随机打乱后保存至 datasets/multitask/train_mixed.json

混合逻辑：
- 五子棋 3168 条，取 3168 条（全部）
- GSM8K 500 条，按比例取 ~352 条（3168 / 9 ≈ 352）
- 合计约 3520 条，随机打乱
"""

import os
import json
import random

# ===== 配置 =====
GOMOKU_PATH = "./datasets/real_games_v2/train.json"
GSM8K_PATH = "./datasets/gsm8k/gsm8k_500.json"
OUTPUT_DIR = "./datasets/multitask"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_mixed.json")
RATIO = 9          # 五子棋:数学 = 9:1
SEED = 42
# ================


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("多任务数据集混合脚本（五子棋 9 : 数学 1）")
    print("=" * 60)

    # 检查输入文件
    for p in [GOMOKU_PATH, GSM8K_PATH]:
        if not os.path.exists(p):
            print(f"[ERROR] 文件不存在: {p}")
            if p == GSM8K_PATH:
                print("请先运行: python scripts/data_processing/prepare_gsm8k.py")
            return

    # 加载数据
    gomoku_data = load_json(GOMOKU_PATH)
    gsm8k_data = load_json(GSM8K_PATH)

    # 为五子棋数据打上 task_type 标记（如果没有）
    for item in gomoku_data:
        if "task_type" not in item:
            item["task_type"] = "gomoku"

    print(f"五子棋数据量: {len(gomoku_data)} 条")
    print(f"GSM8K 数据量: {len(gsm8k_data)} 条")

    # 计算 9:1 比例下需要的数学数据量
    n_gomoku = len(gomoku_data)
    n_math_target = n_gomoku // RATIO  # ~352 条
    n_math_actual = min(n_math_target, len(gsm8k_data))

    print(f"\n按 {RATIO}:1 比例：")
    print(f"  五子棋使用: {n_gomoku} 条")
    print(f"  数学使用:   {n_math_actual} 条（目标 {n_math_target}）")

    # 固定随机种子，保证可复现
    random.seed(SEED)
    math_sampled = random.sample(gsm8k_data, n_math_actual)

    # 合并并打乱
    mixed = gomoku_data + math_sampled
    random.shuffle(mixed)

    print(f"\n混合后总量: {len(mixed)} 条")
    print(f"  其中五子棋: {sum(1 for x in mixed if x.get('task_type') == 'gomoku')} 条")
    print(f"  其中数学:   {sum(1 for x in mixed if x.get('task_type') == 'math_reasoning')} 条")

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(mixed, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\n[OK] 已保存至: {OUTPUT_FILE}  ({size_kb:.0f} KB)")

    # 保存混合统计信息
    stats = {
        "total": len(mixed),
        "gomoku": sum(1 for x in mixed if x.get("task_type") == "gomoku"),
        "math_reasoning": sum(1 for x in mixed if x.get("task_type") == "math_reasoning"),
        "ratio": f"{RATIO}:1",
        "seed": SEED,
        "gomoku_source": GOMOKU_PATH,
        "math_source": GSM8K_PATH,
    }
    stats_file = os.path.join(OUTPUT_DIR, "mix_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] 统计信息: {stats_file}")
    print("\n下一步运行: python scripts/training/finetune_multitask.py")


if __name__ == "__main__":
    main()
