#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果分析和可视化脚本
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")


class MigrationResultsAnalyzer:
    """迁移结果分析器"""

    def __init__(self, results_path: str):
        """
        初始化分析器

        Args:
            results_path: 结果文件路径
        """
        self.results_path = results_path
        self.results = self._load_results()

    def _load_results(self) -> dict:
        """加载结果文件"""
        with open(self.results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results

    def create_summary_table(self, output_path: str = None) -> pd.DataFrame:
        """创建摘要表格"""
        if "comparison" not in self.results:
            print("结果中没有对比数据")
            return pd.DataFrame()

        data = []
        for bench, comp in self.results["comparison"].items():
            data.append({
                "基准": bench,
                "Base模型": f"{comp['base']:.4f}",
                "LoRA模型": f"{comp['lora']:.4f}",
                "Δ值": f"{comp['delta']:+.4f}",
                "变化": "↑提升" if comp["improvement"] else "↓下降",
                "提升百分比": f"{comp['improvement_percent']:.1f}%"
            })

        df = pd.DataFrame(data)

        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"摘要表格已保存到: {output_path}")

        return df

    def plot_migration_effects(self, output_path: str = "./migration_effects.png"):
        """绘制迁移效果图"""
        if "comparison" not in self.results:
            print("结果中没有对比数据")
            return

        comparison = self.results["comparison"]
        benchmarks = list(comparison.keys())
        base_scores = [comparison[b]["base"] for b in benchmarks]
        lora_scores = [comparison[b]["lora"] for b in benchmarks]
        deltas = [comparison[b]["delta"] for b in benchmarks]

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 柱状图：Base vs LoRA
        ax1 = axes[0, 0]
        x = np.arange(len(benchmarks))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, base_scores, width, label='Base模型', alpha=0.7, color='skyblue')
        bars2 = ax1.bar(x + width / 2, lora_scores, width, label='LoRA模型', alpha=0.7, color='lightcoral')

        ax1.set_xlabel('评估基准')
        ax1.set_ylabel('准确率')
        ax1.set_title('Base模型 vs LoRA模型性能对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax1.legend()

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)

        # 2. 迁移效果条形图
        ax2 = axes[0, 1]
        colors = ['green' if d > 0 else 'red' for d in deltas]
        bars = ax2.barh(benchmarks, deltas, color=colors, alpha=0.7)
        ax2.set_xlabel('Δ准确率 (LoRA - Base)')
        ax2.set_title('迁移效果 (Δ值)')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            label_x = width if width >= 0 else width
            ax2.annotate(f'{width:+.3f}',
                         xy=(label_x, bar.get_y() + bar.get_height() / 2),
                         xytext=(5 if width >= 0 else -5, 0),
                         textcoords="offset points",
                         ha='left' if width >= 0 else 'right',
                         va='center')

        # 3. 提升百分比饼图
        ax3 = axes[1, 0]
        positive_count = sum(1 for d in deltas if d > 0)
        negative_count = len(deltas) - positive_count

        if positive_count + negative_count > 0:
            sizes = [positive_count, negative_count]
            labels = [f'正迁移\n{positive_count}个', f'负迁移\n{negative_count}个']
            colors = ['lightgreen', 'lightcoral']
            explode = (0.1, 0) if positive_count > 0 else (0, 0.1)

            ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            ax3.axis('equal')
            ax3.set_title('正负迁移比例')

        # 4. 性能提升散点图
        ax4 = axes[1, 1]
        scatter = ax4.scatter(base_scores, lora_scores, c=deltas, cmap='RdYlGn',
                              s=100, alpha=0.7, edgecolors='black')

        # 添加对角线
        min_val = min(min(base_scores), min(lora_scores)) - 0.05
        max_val = max(max(base_scores), max(lora_scores)) + 0.05
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='y=x')

        # 添加基准标签
        for i, bench in enumerate(benchmarks):
            ax4.annotate(bench, (base_scores[i], lora_scores[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax4.set_xlabel('Base模型准确率')
        ax4.set_ylabel('LoRA模型准确率')
        ax4.set_title('性能提升散点图')
        ax4.set_xlim(min_val, max_val)
        ax4.set_ylim(min_val, max_val)
        ax4.legend()

        # 添加颜色条
        plt.colorbar(scatter, ax=ax4, label='Δ值')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"迁移效果图已保存到: {output_path}")

        return fig

    def plot_detailed_analysis(self, output_path: str = "./detailed_analysis.png"):
        """绘制详细分析图"""
        if "evaluations" not in self.results:
            print("结果中没有详细评估数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 各基准性能对比
        ax1 = axes[0, 0]
        if "base" in self.results["evaluations"] and "lora" in self.results["evaluations"]:
            benchmarks = list(self.results["evaluations"]["base"].keys())
            base_acc = [self.results["evaluations"]["base"][b]["accuracy"] for b in benchmarks]
            lora_acc = [self.results["evaluations"]["lora"][b]["accuracy"] for b in benchmarks]

            x = np.arange(len(benchmarks))
            ax1.bar(x - 0.2, base_acc, 0.4, label='Base', alpha=0.7)
            ax1.bar(x + 0.2, lora_acc, 0.4, label='LoRA', alpha=0.7)

            ax1.set_xlabel('基准')
            ax1.set_ylabel('准确率')
            ax1.set_title('各基准性能对比')
            ax1.set_xticks(x)
            ax1.set_xticklabels(benchmarks, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. 性能提升分布
        ax2 = axes[0, 1]
        if "comparison" in self.results:
            deltas = [self.results["comparison"][b]["delta"] for b in self.results["comparison"]]
            ax2.hist(deltas, bins=10, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.axvline(x=np.mean(deltas), color='red', linestyle='--', label=f'均值: {np.mean(deltas):.3f}')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Δ值')
            ax2.set_ylabel('频数')
            ax2.set_title('性能提升分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. 相关性分析
        ax3 = axes[1, 0]
        if "comparison" in self.results:
            base_scores = [self.results["comparison"][b]["base"] for b in self.results["comparison"]]
            improvements = [self.results["comparison"][b]["improvement_percent"] for b in self.results["comparison"]]

            scatter = ax3.scatter(base_scores, improvements, s=100, alpha=0.7)

            # 添加回归线
            if len(base_scores) > 1:
                z = np.polyfit(base_scores, improvements, 1)
                p = np.poly1d(z)
                ax3.plot(np.sort(base_scores), p(np.sort(base_scores)), "r--", alpha=0.5)

            # 添加标签
            for i, bench in enumerate(self.results["comparison"].keys()):
                ax3.annotate(bench, (base_scores[i], improvements[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax3.set_xlabel('Base模型准确率')
            ax3.set_ylabel('提升百分比 (%)')
            ax3.set_title('基础性能 vs 提升幅度')
            ax3.grid(True, alpha=0.3)

        # 4. 基准难度 vs 提升效果
        ax4 = axes[1, 1]
        if "comparison" in self.results:
            # 使用Base准确率作为难度指标（准确率越低越难）
            difficulty = [1 - self.results["comparison"][b]["base"] for b in self.results["comparison"]]
            deltas = [self.results["comparison"][b]["delta"] for b in self.results["comparison"]]

            ax4.scatter(difficulty, deltas, s=100, alpha=0.7)

            # 添加标签
            for i, bench in enumerate(self.results["comparison"].keys()):
                ax4.annotate(bench, (difficulty[i], deltas[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_xlabel('任务难度 (1 - Base准确率)')
            ax4.set_ylabel('Δ值')
            ax4.set_title('任务难度 vs 迁移效果')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"详细分析图已保存到: {output_path}")

        return fig

    def generate_report(self, output_path: str = "./migration_analysis_report.txt"):
        """生成分析报告"""
        report_lines = []

        report_lines.append("=" * 70)
        report_lines.append("推理能力迁移分析报告")
        report_lines.append("=" * 70)
        report_lines.append("")

        # 实验信息
        if "config" in self.results:
            report_lines.append("实验配置:")
            report_lines.append(f"  模型: {self.results['config'].get('model', '未知')}")
            report_lines.append(f"  基准: {', '.join(self.results['config'].get('benchmarks', []))}")
            report_lines.append("")

        # 性能对比
        if "comparison" in self.results:
            report_lines.append("性能对比结果:")
            report_lines.append("-" * 70)
            report_lines.append(f"{'基准':<15} {'Base':<10} {'LoRA':<10} {'Δ':<10} {'提升%':<10}")
            report_lines.append("-" * 70)

            for bench, comp in self.results["comparison"].items():
                report_lines.append(f"{bench:<15} {comp['base']:.4f}     {comp['lora']:.4f}     "
                                    f"{comp['delta']:+.4f}  {comp['improvement_percent']:+.1f}%")
            report_lines.append("")

        # 统计摘要
        if "summary" in self.results:
            summary = self.results["summary"]
            report_lines.append("统计摘要:")
            report_lines.append(f"  平均Δ值: {summary['average_delta']:.4f}")
            report_lines.append(f"  中位数Δ值: {summary.get('median_delta', 0):.4f}")
            report_lines.append(f"  正迁移基准数: {summary['positive_transfer']}/{summary['total_benchmarks']}")
            report_lines.append(f"  正迁移比例: {summary['positive_transfer_ratio']:.2%}")
            report_lines.append("")

        # 迁移效果分析
        if "comparison" in self.results:
            report_lines.append("迁移效果分析:")

            # 找出提升最大的基准
            max_improvement = max(self.results["comparison"].items(),
                                  key=lambda x: x[1]["delta"])
            report_lines.append(f"  最大提升: {max_improvement[0]} (Δ={max_improvement[1]['delta']:+.4f})")

            # 找出下降最多的基准
            min_improvement = min(self.results["comparison"].items(),
                                  key=lambda x: x[1]["delta"])
            report_lines.append(f"  最大下降: {min_improvement[0]} (Δ={min_improvement[1]['delta']:+.4f})")

            # 分析迁移模式
            positive_tasks = [b for b, c in self.results["comparison"].items() if c["delta"] > 0]
            negative_tasks = [b for b, c in self.results["comparison"].items() if c["delta"] < 0]

            if positive_tasks:
                report_lines.append(f"  正迁移任务: {', '.join(positive_tasks)}")
            if negative_tasks:
                report_lines.append(f"  负迁移任务: {', '.join(negative_tasks)}")
            report_lines.append("")

        # 结论与建议
        report_lines.append("结论与建议:")
        report_lines.append("-" * 70)

        if "summary" in self.results:
            summary = self.results["summary"]

            if summary['positive_transfer_ratio'] > 0.7:
                report_lines.append("1. 五子棋微调对通用推理能力产生了显著的正向迁移效果。")
                report_lines.append("2. 棋类训练可能增强了模型的逻辑推理和规划能力。")
                report_lines.append("3. 建议进一步探索结构化任务训练对通用能力的提升机制。")

            elif summary['positive_transfer_ratio'] > 0.5:
                report_lines.append("1. 五子棋微调对通用推理能力有一定的正向迁移效果。")
                report_lines.append("2. 棋类训练可能对某些类型的推理任务有特定帮助。")
                report_lines.append("3. 建议分析具体哪些推理能力得到了提升。")

            elif summary['positive_transfer_ratio'] > 0.3:
                report_lines.append("1. 五子棋微调的迁移效果有限。")
                report_lines.append("2. 棋类任务的特定性可能限制了泛化能力。")
                report_lines.append("3. 建议尝试更多样化的训练任务或调整训练策略。")

            else:
                report_lines.append("1. 五子棋微调未观察到明显的正向迁移效果。")
                report_lines.append("2. 可能的原因：任务领域差异过大或训练数据不足。")
                report_lines.append("3. 建议：增加训练数据多样性或尝试不同的迁移学习策略。")

        report_lines.append("")
        report_lines.append("后续研究方向:")
        report_lines.append("1. 增加样本量以提高统计显著性")
        report_lines.append("2. 进行错误分析以了解迁移的具体机制")
        report_lines.append("3. 尝试不同的微调策略（数据混合、多任务学习等）")
        report_lines.append("4. 探究棋类复杂度对迁移效果的影响")
        report_lines.append("5. 扩展到更多推理任务和模型架构")

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"分析报告已保存到: {output_path}")

        # 同时在控制台输出
        print('\n'.join(report_lines))

    def analyze_error_cases(self, output_path: str = "./error_analysis.txt"):
        """分析错误案例"""
        if "evaluations" not in self.results:
            print("没有详细的评估数据用于错误分析")
            return

        error_cases = []

        for model_type in ["base", "lora"]:
            if model_type in self.results["evaluations"]:
                for bench, bench_data in self.results["evaluations"][model_type].items():
                    for detail in bench_data.get("details", []):
                        if not detail.get("correct", True):
                            error_cases.append({
                                "model": model_type,
                                "benchmark": bench,
                                "question": detail.get("question", detail.get("context", "N/A")),
                                "response": detail.get("response", "N/A"),
                                "gold": detail.get("gold_answer", detail.get("gold_label", "N/A")),
                                "pred": detail.get("pred_answer", detail.get("pred_label", "N/A"))
                            })

        # 生成错误分析报告
        if error_cases:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("错误案例分析报告\n")
                f.write("=" * 70 + "\n\n")

                # 按模型和基准统计
                error_stats = {}
                for case in error_cases:
                    key = f"{case['model']}_{case['benchmark']}"
                    error_stats[key] = error_stats.get(key, 0) + 1

                f.write("错误分布统计:\n")
                for key, count in error_stats.items():
                    f.write(f"  {key}: {count}个错误\n")
                f.write("\n")

                # 详细错误案例
                f.write("典型错误案例:\n")
                f.write("-" * 70 + "\n")

                for i, case in enumerate(error_cases[:10]):  # 只显示前10个
                    f.write(f"\n案例 {i + 1}:\n")
                    f.write(f"  模型: {case['model']}\n")
                    f.write(f"  基准: {case['benchmark']}\n")
                    f.write(f"  问题: {case['question'][:200]}...\n")
                    f.write(f"  模型回答: {case['response'][:200]}...\n")
                    f.write(f"  预测: {case['pred']}\n")
                    f.write(f"  正确答案: {case['gold']}\n")

            print(f"错误分析报告已保存到: {output_path}")
        else:
            print("没有找到错误案例")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="迁移结果分析")
    parser.add_argument("--results_path", type=str, default="./migration_results.json",
                        help="结果文件路径")
    parser.add_argument("--output_dir", type=str, default="./analysis",
                        help="输出目录")

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("迁移结果分析")
    print("=" * 70)

    # 初始化分析器
    analyzer = MigrationResultsAnalyzer(args.results_path)

    # 生成摘要表格
    summary_csv = Path(args.output_dir) / "summary.csv"
    df = analyzer.create_summary_table(str(summary_csv))

    if not df.empty:
        print("\n性能对比摘要:")
        print("-" * 70)
        print(df.to_string(index=False))

    # 绘制迁移效果图
    migration_plot = Path(args.output_dir) / "migration_effects.png"
    analyzer.plot_migration_effects(str(migration_plot))

    # 绘制详细分析图
    detailed_plot = Path(args.output_dir) / "detailed_analysis.png"
    analyzer.plot_detailed_analysis(str(detailed_plot))

    # 生成分析报告
    report_path = Path(args.output_dir) / "analysis_report.txt"
    analyzer.generate_report(str(report_path))

    # 错误分析
    error_path = Path(args.output_dir) / "error_analysis.txt"
    analyzer.analyze_error_cases(str(error_path))

    print(f"\n所有分析结果已保存到: {args.output_dir}")
    print("分析完成!")


if __name__ == "__main__":
    main()