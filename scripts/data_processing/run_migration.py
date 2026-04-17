"""
推理迁移实验运行脚本
"""

import torch
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reasoning_transfer_experiment import ReasoningMigrationEvaluator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="推理迁移实验")

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="基座模型名称或路径"
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        default="./checkpoints/qwen7b-gomoku-lora/final_model",
        help="LoRA权重路径"
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["gsm8k", "logiqa", "math", "hellaswag"],
        help="要评估的基准列表"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="每个基准的样本数"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="运行设备"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="输出目录"
    )

    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="跳过基座模型评估，只评估LoRA模型"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("推理能力迁移实验")
    print("=" * 70)
    print(f"基座模型: {args.base_model}")
    print(f"LoRA路径: {args.lora_path}")
    print(f"评估基准: {args.benchmarks}")
    print(f"每个基准样本数: {args.num_samples}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 70)

    # 初始化评估器
    evaluator = ReasoningMigrationEvaluator(
        base_model_name=args.base_model,
        lora_path=args.lora_path if os.path.exists(args.lora_path) else None,
        device=args.device,
        torch_dtype=torch.float16
    )

    # 设置样本数
    num_samples_dict = {
        "gsm8k": args.num_samples,
        "logiqa": args.num_samples,
        "math": min(args.num_samples, 50),  # MATH较难，样本少些
        "bbh": min(args.num_samples, 30),  # BBH较难，样本少些
        "hellaswag": args.num_samples
    }

    # 设置要评估的模型类型
    model_types = ["lora"] if args.skip_base else ["base", "lora"]

    try:
        # 运行评估
        results = evaluator.run_comprehensive_evaluation(
            benchmarks=args.benchmarks,
            num_samples=num_samples_dict,
            model_types=model_types
        )

        # 输出最终结果
        print("\n" + "=" * 70)
        print("实验完成!")
        print("=" * 70)

        if "comparison" in results:
            print("\n迁移效果对比:")
            print("-" * 70)
            print(f"{'基准':<15} {'Base':<10} {'LoRA':<10} {'Δ':<10} {'变化':<10}")
            print("-" * 70)

            for bench, comp in results["comparison"].items():
                delta_sign = "+" if comp["delta"] >= 0 else ""
                change = "↑提升" if comp["improvement"] else "↓下降"
                print(f"{bench:<15} {comp['base']:.4f}     {comp['lora']:.4f}     "
                      f"{delta_sign}{comp['delta']:.4f}  {change}")

        if "summary" in results:
            summary = results["summary"]
            print("\n总体统计:")
            print(f"平均Δ值: {summary['average_delta']:.4f}")
            print(f"正迁移基准数: {summary['positive_transfer']}/{summary['total_benchmarks']}")
            print(f"正迁移比例: {summary['positive_transfer_ratio']:.2%}")

        # 保存最终报告
        report_path = os.path.join(args.output_dir, "migration_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("推理能力迁移实验报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"实验配置:\n")
            f.write(f"  基座模型: {args.base_model}\n")
            f.write(f"  LoRA路径: {args.lora_path}\n")
            f.write(f"  评估基准: {args.benchmarks}\n")
            f.write(f"  样本数量: {args.num_samples}\n\n")

            if "comparison" in results:
                f.write("迁移效果对比:\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'基准':<15} {'Base':<10} {'LoRA':<10} {'Δ':<10} {'变化':<10}\n")
                f.write("-" * 70 + "\n")

                for bench, comp in results["comparison"].items():
                    delta_sign = "+" if comp["delta"] >= 0 else ""
                    change = "↑提升" if comp["improvement"] else "↓下降"
                    f.write(f"{bench:<15} {comp['base']:.4f}     {comp['lora']:.4f}     "
                            f"{delta_sign}{comp['delta']:.4f}  {change}\n")

            if "summary" in results:
                f.write("\n总体统计:\n")
                f.write(f"  平均Δ值: {summary['average_delta']:.4f}\n")
                f.write(f"  正迁移基准数: {summary['positive_transfer']}/{summary['total_benchmarks']}\n")
                f.write(f"  正迁移比例: {summary['positive_transfer_ratio']:.2%}\n\n")

            f.write("\n结论:\n")
            if "summary" in results:
                if summary['positive_transfer_ratio'] > 0.5:
                    f.write("五子棋微调对通用推理能力产生了总体正向迁移。\n")
                elif summary['positive_transfer_ratio'] < 0.3:
                    f.write("五子棋微调对通用推理能力的迁移效果有限。\n")
                else:
                    f.write("五子棋微调对部分推理任务有正向迁移效果。\n")

            f.write("\n建议:\n")
            f.write("1. 增加样本量以获得更可靠结果\n")
            f.write("2. 进行错误分析以了解迁移机制\n")
            f.write("3. 尝试不同的微调策略（数据量、学习率等）\n")

        print(f"\n详细报告已保存到: {report_path}")

    except Exception as e:
        print(f"\n评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        evaluator.cleanup()

    print("\n实验结束!")


if __name__ == "__main__":
    main()