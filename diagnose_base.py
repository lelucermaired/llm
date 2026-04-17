"""
诊断脚本: 分析base模型在Connect4各难度的错误模式
用法: python diagnose_base.py
"""
import json
from collections import Counter

DETAILED_PATH = "./results/evaluations/connect4/base_detailed.json"
BENCHMARK_PATH = "./datasets/connect4_benchmark/test.json"

# 加载评测结果和benchmark
with open(DETAILED_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    bench = json.load(f)

# 建立id到题目meta的映射
bench_map = {q["id"]: q for q in bench}

# 总体统计
detailed = data["detailed"]
print("=" * 70)
print("Base模型错题分析")
print("=" * 70)

# 按难度分组
for diff in ["easy", "medium", "hard"]:
    print(f"\n【{diff.upper()}】")
    diff_items = [r for r in detailed if r["difficulty"] == diff]
    wrong = [r for r in diff_items if not r["is_correct"]]
    print(f"  总题: {len(diff_items)}, 错题: {len(wrong)}")

    if not wrong:
        continue

    # 分析错题
    print(f"\n  错题详情:")
    for r in wrong:
        q_meta = bench_map.get(r["id"], {})
        all_scores = q_meta.get("meta", {}).get("all_scores", {})
        pred = r["predicted_col"]
        pred_score = all_scores.get(str(pred), "N/A") if pred is not None else "N/A"
        best_score = q_meta.get("meta", {}).get("best_score", "N/A")

        # 看预测列的相对位置
        if all_scores and pred is not None:
            sorted_scores = sorted(
                [(int(k), v) for k, v in all_scores.items()],
                key=lambda x: -x[1]
            )
            pred_rank = next(
                (i + 1 for i, (c, s) in enumerate(sorted_scores) if c == pred),
                "N/A"
            )
        else:
            pred_rank = "N/A"

        print(f"    {r['id']}: 预测列={pred}(分数={pred_score}, 排名={pred_rank}/{len(all_scores)}), "
              f"最优列={r['optimal_columns']}(分数={best_score})")
        print(f"              输出: {r['raw_output'][:60]}")

    # 错题的预测列分布
    preds = [r["predicted_col"] for r in wrong if r["predicted_col"] is not None]
    print(f"\n  错题预测列分布: {Counter(preds)}")

# 整体统计: 模型预测列是否有偏好?
all_preds = [r["predicted_col"] for r in detailed if r["predicted_col"] is not None]
print(f"\n\n【所有题的预测列分布】")
print(f"  {Counter(all_preds)}")

# 所有题的最优列分布(看benchmark本身是否均衡)
all_optimal = []
for r in detailed:
    all_optimal.extend(r["optimal_columns"])
print(f"\n【所有题的最优列分布】(benchmark本身)")
print(f"  {Counter(all_optimal)}")

# 关键指标: 错题中,模型选的列是不是"看起来合理但不最优"?
print(f"\n\n【错题的'次优率'】")
for diff in ["easy", "medium", "hard"]:
    wrong = [r for r in detailed if r["difficulty"] == diff and not r["is_correct"]]
    if not wrong:
        continue
    # "合理"错误:预测列在前2名
    reasonable = 0
    terrible = 0
    for r in wrong:
        q_meta = bench_map.get(r["id"], {})
        all_scores = q_meta.get("meta", {}).get("all_scores", {})
        pred = r["predicted_col"]
        if pred is None or not all_scores:
            continue
        sorted_cols = sorted(
            [(int(k), v) for k, v in all_scores.items()],
            key=lambda x: -x[1]
        )
        top2_cols = [c for c, _ in sorted_cols[:2]]
        if pred in top2_cols:
            reasonable += 1
        else:
            terrible += 1
    total = reasonable + terrible
    if total > 0:
        print(f"  {diff}: {reasonable}/{total} ({reasonable/total:.0%}) 错题选的是'次优列(前2名)'")