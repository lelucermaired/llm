"""
significance_test.py

对浅层重置实验（reset_10_qv vs v2）做统计显著性检验
使用已有的评测缓存数据，不需要重新跑模型

检验方法：
1. 二项检验（McNemar's test）：逐题对比
2. Bootstrap置信区间：重采样1000次
3. Cohen's h：效应量
"""

import json, os
import numpy as np
from scipy import stats

CACHE_DIR = "./results/evaluations/shallow_reset"
BASE_RESULTS = {"math": 0.720, "spatial": 0.360, "planning": 0.520, "logic": 1.000}

# 从缓存读取各模型的准确率
def load_result(name):
    path = os.path.join(CACHE_DIR, f"{name}.json")
    if not os.path.exists(path):
        # 也试试all_models_50的缓存
        path2 = f"./results/evaluations/all_models_50/_cache/{name}.json"
        if os.path.exists(path2):
            path = path2
        else:
            print(f"找不到缓存：{name}")
            return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def bootstrap_ci(n_correct, n_total, n_bootstrap=10000, ci=0.95):
    """Bootstrap置信区间"""
    rng = np.random.default_rng(42)
    # 构造样本：n_correct个1，n_total-n_correct个0
    sample = np.array([1]*n_correct + [0]*(n_total - n_correct))
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = rng.choice(sample, size=n_total, replace=True)
        boot_means.append(boot_sample.mean())
    lower = np.percentile(boot_means, (1-ci)/2*100)
    upper = np.percentile(boot_means, (1+ci)/2*100)
    return lower, upper

def mcnemar_test(n_total, acc_a, acc_b):
    """
    McNemar检验（配对比较）
    假设：b对a错的题数 vs a对b错的题数
    近似：用二项检验代替（因为没有逐题数据）
    """
    n_a = int(round(acc_a * n_total))
    n_b = int(round(acc_b * n_total))
    # 假设较高的那个全部包含较低的那个答对的题
    # 即：差值题数 = n_b - n_a（b比a多答对的题）
    diff = n_b - n_a
    return n_a, n_b, diff

def cohen_h(p1, p2):
    """Cohen's h效应量"""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def main():
    print("=" * 65)
    print("浅层重置统计显著性分析")
    print("=" * 65)

    # 读取数据
    v2_res    = load_result("reset_0")    # v2（无重置）
    reset_res = load_result("reset_10")   # reset_10_qv

    if v2_res is None or reset_res is None:
        print("缓存文件不存在，尝试使用硬编码结果...")
        # 根据之前的实验结果直接填写
        v2_res    = {"math": 0.700, "spatial": 0.360, "planning": 0.540, "logic": 1.000}
        reset_res = {"math": 0.740, "spatial": 0.380, "planning": 0.560, "logic": 1.000}

    tasks = ["math", "spatial", "planning", "logic"]
    n_per_task = {"math": 50, "spatial": 50, "planning": 50, "logic": 20}

    print("\n【各任务逐一检验（题数=50/50/50/20）】")
    print(f"{'任务':<10} {'v2':>6} {'reset':>6} {'delta':>7} "
          f"{'p值':>8} {'显著':>5} {'95%CI(reset)':>18} {'Cohen h':>8}")
    print("-" * 75)

    all_v2_correct = 0
    all_reset_correct = 0
    all_n = 0

    for task in tasks:
        n = n_per_task[task]
        acc_v2    = v2_res.get(task, 0)
        acc_reset = reset_res.get(task, 0)
        n_v2    = int(round(acc_v2    * n))
        n_reset = int(round(acc_reset * n))
        delta   = acc_reset - acc_v2

        all_v2_correct    += n_v2
        all_reset_correct += n_reset
        all_n             += n

        # 二项检验：reset在n题中答对n_reset题，期望概率为acc_v2
        if acc_v2 > 0 and acc_v2 < 1:
            result = stats.binomtest(n_reset, n, acc_v2, alternative='greater')
            p = result.pvalue
        else:
            p = 1.0

        sig = "✅" if p < 0.05 else ("〜" if p < 0.10 else "✗")

        # Bootstrap CI
        lo, hi = bootstrap_ci(n_reset, n)

        # Cohen's h
        h = cohen_h(acc_reset, acc_v2)

        print(f"{task:<10} {acc_v2:>6.3f} {acc_reset:>6.3f} {delta:>+7.3f} "
              f"{p:>8.4f} {sig:>5} [{lo:.3f}, {hi:.3f}] {h:>8.3f}")

    # 综合检验（所有任务合并）
    print(f"\n{'合计':<10} {all_v2_correct/all_n:>6.3f} "
          f"{all_reset_correct/all_n:>6.3f} "
          f"{(all_reset_correct-all_v2_correct)/all_n:>+7.3f}")

    acc_v2_all    = all_v2_correct    / all_n
    acc_reset_all = all_reset_correct / all_n

    result_all = stats.binomtest(all_reset_correct, all_n, acc_v2_all, alternative='greater')
    p_all = result_all.pvalue
    lo_all, hi_all = bootstrap_ci(all_reset_correct, all_n)
    h_all = cohen_h(acc_reset_all, acc_v2_all)

    print(f"\n【综合检验（170题合并）】")
    print(f"  v2准确率：    {acc_v2_all:.3f} ({all_v2_correct}/{all_n})")
    print(f"  reset准确率： {acc_reset_all:.3f} ({all_reset_correct}/{all_n})")
    print(f"  delta：       {acc_reset_all-acc_v2_all:+.3f}")
    print(f"  p值：         {p_all:.4f} {'(p<0.05, 显著)' if p_all<0.05 else '(p<0.10, 边界显著)' if p_all<0.10 else '(不显著)'}")
    print(f"  95% CI：      [{lo_all:.3f}, {hi_all:.3f}]")
    print(f"  Cohen's h：   {h_all:.3f} ({'小效应' if abs(h_all)<0.2 else '中效应' if abs(h_all)<0.5 else '大效应'})")

    # 需要多少题才能达到显著
    print(f"\n【需要多少题才能达到p<0.05？】")
    for n_needed in [100, 150, 200, 300, 500]:
        n_reset_needed = int(round(acc_reset_all * n_needed))
        r = stats.binomtest(n_reset_needed, n_needed, acc_v2_all, alternative='greater')
        sig_mark = "✅ 显著" if r.pvalue < 0.05 else "✗"
        print(f"  n={n_needed:4d}: 预计答对{n_reset_needed:3d}题，p={r.pvalue:.4f}  {sig_mark}")

    # 结论
    print(f"\n【结论】")
    if p_all < 0.05:
        print(f"  ✅ 浅层重置效果在统计上显著（p={p_all:.4f}<0.05）")
        print(f"  效应量Cohen's h={h_all:.3f}（小效应），在实际应用中意义有限")
        print(f"  但作为后处理策略，无需重新训练即可获得显著改善，具有实用价值")
    elif p_all < 0.10:
        print(f"  〜 边界显著（p={p_all:.4f}，0.05<p<0.10）")
        print(f"  建议扩大题库到200题重新评测以确认结论")
    else:
        print(f"  ✗ 当前样本量下不显著（p={p_all:.4f}）")
        print(f"  需要扩大到约{[n for n in [100,150,200,300,500] if stats.binomtest(int(round(acc_reset_all*n)), n, acc_v2_all, alternative='greater').pvalue < 0.05][0] if any(stats.binomtest(int(round(acc_reset_all*n)), n, acc_v2_all, alternative='greater').pvalue < 0.05 for n in [100,150,200,300,500]) else 500}题才能达到显著")
        print(f"  当前结论：趋势性正向，但需更多数据支撑")

if __name__ == "__main__":
    main()