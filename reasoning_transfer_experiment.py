"""
基于围棋和五子棋的大语言模型推理能力迁移研究 - 实验代码
文件名: reasoning_transfer_experiment.py
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from peft import PeftModel


warnings.filterwarnings('ignore')


# 设置随机种子保证可复现性
def set_random_seed(seed: int = 42):
    """设置所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_random_seed(42)


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 模型配置
    base_model_path: str = "your_base_model_path"
    chess_finetuned_path: str = "your_chess_finetuned_model_path"
    model_name: str = "Qwen/Qwen2-7B-Instruct"  # 根据实际情况修改

    # 实验配置
    batch_size: int = 4
    max_length: int = 2048
    temperature: float = 0.1
    top_p: float = 0.95
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试配置
    n_samples_per_test: int = 100  # 每个测试集的样本数
    save_results: bool = True
    results_dir: str = "./experiment_results"

    # 评估基准
    math_benchmarks: List[str] = None
    logic_benchmarks: List[str] = None
    planning_benchmarks: List[str] = None

    def __post_init__(self):
        if self.math_benchmarks is None:
            self.math_benchmarks = ["gsm8k"]

        if self.logic_benchmarks is None:
            self.logic_benchmarks = []

        if self.planning_benchmarks is None:
            self.planning_benchmarks = []


class ModelManager:
    """模型管理器"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.base_model = None
        self.chess_model = None
        self.tokenizer = None

    def     load_models(self):
        """加载基础模型和棋类微调模型"""
        print("正在加载模型...")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # 加载基础模型
        print(f"加载基础模型: {self.config.base_model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            trust_remote_code=True
        )

        # 加载棋类微调模型（LoRA）
        print(f"加载棋类微调模型 (LoRA): {self.config.chess_finetuned_path}")
        self.chess_model = PeftModel.from_pretrained(
            self.base_model,
            self.config.chess_finetuned_path,

        )

        # 设置为评估模式
        self.base_model.eval()
        self.chess_model.eval()

        print("模型加载完成!")

    def generate_response(self, model, prompt: str, max_new_tokens: int = 512) -> str:
        model_device = next(model.parameters()).device  # ★ 真实 device

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,  # ★ 关键：禁止 padding
        )

        # ★ 统一送到 model.device（只做一次）
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].size(1)
        response = self.tokenizer.decode(
            outputs[0, input_len:],
            skip_special_tokens=True
        )
        return response.strip()


class ReasoningBenchmarkEvaluator:
    """推理基准评估器"""

    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.config = model_manager.config

    def evaluate_math_reasoning(self) -> Dict[str, Any]:
        """评估数学推理能力"""
        print("\n" + "=" * 50)
        print("数学推理能力评估")
        print("=" * 50)

        results = {}

        # 测试GSM8K
        if "gsm8k" in self.config.math_benchmarks:
            print("\n评估GSM8K...")
            gsm8k_results = self._evaluate_gsm8k()
            results["gsm8k"] = gsm8k_results

        # 测试MathQA
        if "mathqa" in self.config.math_benchmarks:
            print("\n评估MathQA...")
            mathqa_results = self._evaluate_mathqa()
            results["mathqa"] = mathqa_results

        return results

    def evaluate_logical_reasoning(self) -> Dict[str, Any]:
        """评估逻辑推理能力"""
        print("\n" + "=" * 50)
        print("逻辑推理能力评估")
        print("=" * 50)

        results = {}

        # 测试LogiQA
        if "logiqa" in self.config.logic_benchmarks:
            print("\n评估LogiQA...")
            logiqa_results = self._evaluate_logiqa()
            results["logiqa"] = logiqa_results

        # 测试ReClor
        if "reclor" in self.config.logic_benchmarks:
            print("\n评估ReClor...")
            reclor_results = self._evaluate_reclor()
            results["reclor"] = reclor_results

        return results

    def evaluate_planning_ability(self) -> Dict[str, Any]:
        """评估规划能力"""
        print("\n" + "=" * 50)
        print("规划能力评估")
        print("=" * 50)

        results = {}

        # 测试汉诺塔问题
        if "tower_of_hanoi" in self.config.planning_benchmarks:
            print("\n评估汉诺塔问题...")
            hanoi_results = self._evaluate_tower_of_hanoi()
            results["tower_of_hanoi"] = hanoi_results

        # 测试积木世界问题
        if "blocks_world" in self.config.planning_benchmarks:
            print("\n评估积木世界问题...")
            blocks_results = self._evaluate_blocks_world()
            results["blocks_world"] = blocks_results

        return results

    def _evaluate_gsm8k(self) -> Dict[str, Any]:
        try:
            from datasets import load_dataset
            import numpy as np
            from tqdm import tqdm

            dataset = load_dataset("gsm8k", "main")
            test_data = dataset["test"]

            n_samples = min(self.config.n_samples_per_test, len(test_data))
            indices = np.random.choice(len(test_data), n_samples, replace=False)
            sampled_data = test_data.select(indices)

            base_scores = []
            chess_scores = []
            cot_scores = []
            detailed_comparisons = []

            for example in tqdm(sampled_data, desc="GSM8K评估"):
                question = example["question"]
                ground_truth = example["answer"]

                base_prompt = f"""Solve the following problem and give the final numeric answer.

    Question: {question}

    Answer:"""

                cot_prompt = f"""Solve the following problem step by step.
    Show your reasoning clearly and then give the final numeric answer.

    Question: {question}

    Let's think step by step.
    """

                base_response = self.mm.generate_response(
                    self.mm.base_model, base_prompt, max_new_tokens=64
                )

                chess_response = self.mm.generate_response(
                    self.mm.chess_model, base_prompt, max_new_tokens=64
                )

                chess_response_cot = self.mm.generate_response(
                    self.mm.chess_model, cot_prompt, max_new_tokens=128
                )

                base_answer = self._extract_numeric_answer(base_response)
                chess_answer = self._extract_numeric_answer(chess_response)
                true_answer = self._extract_numeric_answer(ground_truth)

                base_correct = self._compare_answers(base_answer, true_answer)
                chess_correct = self._compare_answers(chess_answer, true_answer)

                base_scores.append(bool(base_correct))
                chess_scores.append(bool(chess_correct))

                cot_score = self._cot_structure_score(chess_response_cot)
                cot_scores.append(cot_score)

                if base_correct != chess_correct:
                    detailed_comparisons.append({
                        "question": question,
                        "ground_truth": ground_truth,
                        "base_response": base_response,
                        "chess_response": chess_response,
                        "chess_response_cot": chess_response_cot,
                        "cot_score": cot_score
                    })

            results = self._calculate_results(
                base_scores, chess_scores, detailed_comparisons
            )

            results["avg_chess_cot_score"] = float(np.mean(cot_scores))
            results["cot_score_distribution"] = {
                "min": int(np.min(cot_scores)),
                "max": int(np.max(cot_scores))
            }

            return results

        except Exception as e:
            print(f"GSM8K评估出错: {e}")
            return {"error": str(e)}

    def _evaluate_reclor(self) -> Dict[str, Any]:
        """评估ReClor逻辑推理"""
        try:

            dataset = load_dataset("edinburghnlp/reclor", split="validation")
            test_data = dataset

            n_samples = min(self.config.n_samples_per_test, len(test_data))
            indices = np.random.choice(len(test_data), n_samples, replace=False)
            sampled_data = test_data.select(indices)

            base_scores = []
            chess_scores = []

            for example in tqdm(sampled_data, desc="ReClor评估"):
                question = example["question"]
                options = example["answers"]
                label = example["label"]

                prompt = f"""请回答以下逻辑推理问题：

问题: {question}

选项:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

请选择正确选项（A, B, C, D）："""

                base_response = self.mm.generate_response(self.mm.base_model, prompt, max_new_tokens=10)
                chess_response = self.mm.generate_response(self.mm.chess_model, prompt, max_new_tokens=10)

                base_answer = self._extract_option_letter(base_response)
                chess_answer = self._extract_option_letter(chess_response)

                base_correct = (base_answer == label)
                chess_correct = (chess_answer == label)

                base_scores.append(base_correct)
                chess_scores.append(chess_correct)

            return self._calculate_results(base_scores, chess_scores)

        except Exception as e:
            print(f"ReClor评估出错: {e}")
            return {"error": str(e)}

    def _evaluate_tower_of_hanoi(self) -> Dict[str, Any]:
        """评估汉诺塔规划能力"""
        # 定义汉诺塔问题
        hanoi_problems = [
            {
                "name": "3 disks",
                "initial": "A: [3,2,1], B: [], C: []",
                "goal": "A: [], B: [], C: [3,2,1]",
                "optimal_steps": 7,
                "constraints": "每次只能移动一个盘子，大盘子不能放在小盘子上面"
            },
            {
                "name": "4 disks",
                "initial": "A: [4,3,2,1], B: [], C: []",
                "goal": "A: [], B: [], C: [4,3,2,1]",
                "optimal_steps": 15,
                "constraints": "每次只能移动一个盘子，大盘子不能放在小盘子上面"
            }
        ]

        base_scores = []
        chess_scores = []
        detailed_analyses = []

        for problem in hanoi_problems:
            prompt = f"""解决汉诺塔问题：

初始状态: {problem['initial']}
目标状态: {problem['goal']}
约束条件: {problem['constraints']}

请给出一步一步的解决方案："""

            base_response = self.mm.generate_response(self.mm.base_model, prompt)
            chess_response = self.mm.generate_response(self.mm.chess_model, prompt)

            # 评估解决方案质量
            base_score = self._evaluate_hanoi_solution(base_response, problem)
            chess_score = self._evaluate_hanoi_solution(chess_response, problem)

            base_scores.append(base_score)
            chess_scores.append(chess_score)

            detailed_analyses.append({
                "problem": problem["name"],
                "base_solution": base_response[:500],  # 截取前500字符
                "chess_solution": chess_response[:500],
                "base_score": base_score,
                "chess_score": chess_score
            })

        return {
            "base_scores": base_scores,
            "chess_scores": chess_scores,
            "base_avg": np.mean(base_scores),
            "chess_avg": np.mean(chess_scores),
            "improvement": np.mean(chess_scores) - np.mean(base_scores),
            "detailed_analyses": detailed_analyses[:3]  # 只保存前3个详细分析
        }

    def _evaluate_blocks_world(self) -> Dict[str, Any]:
        """评估积木世界规划能力"""
        blocks_problems = [
            {
                "name": "Simple stacking",
                "initial": "A在桌上，B在桌上，C在桌上",
                "goal": "A在桌上，B在A上，C在B上",
                "optimal_actions": 2
            },
            {
                "name": "Complex rearrangement",
                "initial": "A在桌上，B在A上，C在桌上，D在C上",
                "goal": "A在桌上，B在桌上，C在B上，D在C上",
                "optimal_actions": 3
            }
        ]

        base_scores = []
        chess_scores = []

        for problem in blocks_problems:
            prompt = f"""解决积木世界问题：

初始状态: {problem['initial']}
目标状态: {problem['goal']}

请给出达到目标状态所需的操作序列："""

            base_response = self.mm.generate_response(self.mm.base_model, prompt)
            chess_response = self.mm.generate_response(self.mm.chess_model, prompt)

            # 简单评估：检查是否提到了关键操作
            base_score = self._evaluate_blocks_solution(base_response, problem)
            chess_score = self._evaluate_blocks_solution(chess_response, problem)

            base_scores.append(base_score)
            chess_scores.append(chess_score)

        return {
            "base_avg": np.mean(base_scores),
            "chess_avg": np.mean(chess_scores),
            "improvement": np.mean(chess_scores) - np.mean(base_scores)
        }

    def _calculate_results(
            self,
            base_scores: List[bool],
            chess_scores: List[bool],
            detailed_comparisons: List[Dict] = None
    ) -> Dict[str, Any]:
        """计算评估结果（"""
        base_arr = np.array(base_scores, dtype=np.int32)
        chess_arr = np.array(chess_scores, dtype=np.int32)

        base_acc = float(base_arr.mean())
        chess_acc = float(chess_arr.mean())

        # 统计显著性（配对 t-test）
        if len(base_arr) > 1:
            t_stat, p_value = stats.ttest_rel(base_arr, chess_arr)
        else:
            p_value = None

        # 置信区间
        if len(base_arr) >= 30:
            base_ci = stats.t.interval(
                0.95, len(base_arr) - 1,
                loc=base_acc,
                scale=stats.sem(base_arr)
            )
            chess_ci = stats.t.interval(
                0.95, len(chess_arr) - 1,
                loc=chess_acc,
                scale=stats.sem(chess_arr)
            )
        else:
            base_ci, chess_ci = (None, None), (None, None)

        improvement = chess_acc - base_acc

        return {
            "base_accuracy": base_acc,
            "chess_accuracy": chess_acc,
            "improvement": improvement,
            "relative_improvement": improvement / base_acc if base_acc > 0 else 0.0,
            "p_value": float(p_value) if p_value is not None else None,
            "statistically_significant": bool(p_value < 0.05) if p_value is not None else None,
            "base_confidence_interval": base_ci,
            "chess_confidence_interval": chess_ci,
            "sample_size": int(len(base_arr)),
            "detailed_comparisons": detailed_comparisons[:5] if detailed_comparisons else []
        }

    def _extract_numeric_answer(self, text: str):
        import re
        match = re.search(r"####\s*(-?\d+\.?\d*)", text)
        if match:
            return float(match.group(1))
        return None

    def _extract_option_letter(self, text: str) -> str:
        """从文本中提取选项字母"""
        import re
        # 查找A-D的选项
        match = re.search(r'[A-D]', text.upper())
        return match.group(0) if match else ""

    def _compare_answers(self, answer1, answer2, tolerance=1e-3) -> bool:
        """比较两个答案是否相等（考虑数值误差）"""
        if answer1 is None or answer2 is None:
            return False
        return abs(answer1 - answer2) < tolerance

    def _evaluate_hanoi_solution(self, solution: str, problem: Dict) -> float:
        """评估汉诺塔解决方案质量"""
        score = 0.0

        # 检查是否提到了关键步骤
        keywords = ["移动", "盘子", "从", "到", "柱子"]
        keyword_count = sum(1 for keyword in keywords if keyword in solution)
        score += min(keyword_count / len(keywords), 1.0) * 0.5

        # 检查步骤数量
        import re
        steps = re.findall(r'第.*步|步骤\d+|step', solution, re.IGNORECASE)
        if steps:
            step_count = len(steps)
            # 粗略评估步骤合理性
            if step_count >= problem["optimal_steps"] - 2 and step_count <= problem["optimal_steps"] + 5:
                score += 0.5

        return score

    def _evaluate_blocks_solution(self, solution: str, problem: Dict) -> float:
        """评估积木世界解决方案质量"""
        # 简单关键词匹配
        required_keywords = ["放在", "移动", "拿起", "放下"]
        found_keywords = sum(1 for keyword in required_keywords if keyword in solution)
        return min(found_keywords / len(required_keywords), 1.0)

    def _cot_structure_score(self, text: str) -> int:
        score = 0
        t = text.lower()

        if "step" in t:
            score += 1
        if any(k in t for k in ["therefore", "thus", "so", "hence"]):
            score += 1
        if sum(c.isdigit() for c in text) >= 5:
            score += 1
        if "\n" in text:
            score += 1

        return score


class TransferEffectAnalyzer:
    """迁移效应分析器"""

    def __init__(self, all_results: Dict[str, Dict]):
        self.results = all_results

    def analyze_overall_transfer(self) -> Dict[str, Any]:
        """分析整体迁移效果"""
        categories = {
            "math_reasoning": [],
            "logical_reasoning": [],
            "planning_ability": []
        }

        # 分类整理结果
        for benchmark, result in self.results.items():
            if benchmark in ["gsm8k", "mathqa"]:
                categories["math_reasoning"].append(result)
            elif benchmark in ["logiqa", "reclor"]:
                categories["logical_reasoning"].append(result)
            elif benchmark in ["tower_of_hanoi", "blocks_world"]:
                categories["planning_ability"].append(result)

        # 计算各类别的平均提升
        analysis = {}
        for category, results_list in categories.items():
            if results_list:
                improvements = [r.get("improvement", 0) for r in results_list if "improvement" in r]
                if improvements:
                    analysis[category] = {
                        "mean_improvement": np.mean(improvements),
                        "std_improvement": np.std(improvements),
                        "max_improvement": np.max(improvements),
                        "min_improvement": np.min(improvements),
                        "num_benchmarks": len(improvements)
                    }

        # 计算综合迁移效果
        all_improvements = []
        for category_data in analysis.values():
            all_improvements.append(category_data["mean_improvement"])

        if all_improvements:
            analysis["overall"] = {
                "mean_transfer_effect": np.mean(all_improvements),
                "transfer_consistency": 1 - np.std(all_improvements) / np.mean(all_improvements) if np.mean(
                    all_improvements) != 0 else 0
            }

        return analysis

    def analyze_skill_correlation(self) -> Dict[str, Any]:
        """分析技能相关性"""
        # 这里可以添加更复杂的相关性分析
        # 例如：棋类训练中的特定技能与推理提升的关系
        return {
            "note": "需要手动分析棋类任务表现与推理提升的关系"
        }

    def generate_report(self) -> str:
        """生成实验报告"""
        overall_analysis = self.analyze_overall_transfer()

        report = f"""
# 大语言模型推理能力迁移实验报告

## 一、实验概述
- 实验目的: 验证棋类任务微调对大语言模型通用推理能力的迁移效果
- 评估基准: {len(self.results)}个推理任务
- 测试样本数: 约{sum(r.get('sample_size', 0) for r in self.results.values() if isinstance(r, dict))}个

## 二、实验结果摘要

### 1. 数学推理能力
"""

        # 添加数学推理结果
        math_benchmarks = [b for b in ["gsm8k", "mathqa"] if b in self.results]
        for benchmark in math_benchmarks:
            result = self.results[benchmark]
            report += f"- {benchmark.upper()}: 基础模型准确率={result.get('base_accuracy', 0):.3f}, "
            report += f"棋类模型准确率={result.get('chess_accuracy', 0):.3f}, "
            report += f"提升={result.get('improvement', 0):+.3f}\n"

        report += """
### 2. 逻辑推理能力
"""

        # 添加逻辑推理结果
        logic_benchmarks = [b for b in ["logiqa", "reclor"] if b in self.results]
        for benchmark in logic_benchmarks:
            result = self.results[benchmark]
            report += f"- {benchmark.upper()}: 基础模型准确率={result.get('base_accuracy', 0):.3f}, "
            report += f"棋类模型准确率={result.get('chess_accuracy', 0):.3f}, "
            report += f"提升={result.get('improvement', 0):+.3f}\n"

        report += """
### 3. 规划能力
"""

        # 添加规划能力结果
        planning_benchmarks = [b for b in ["tower_of_hanoi", "blocks_world"] if b in self.results]
        for benchmark in planning_benchmarks:
            result = self.results[benchmark]
            report += f"- {benchmark}: 基础模型得分={result.get('base_avg', 0):.3f}, "
            report += f"棋类模型得分={result.get('chess_avg', 0):.3f}, "
            report += f"提升={result.get('improvement', 0):+.3f}\n"

        report += f"""
## 三、迁移效果分析

### 整体迁移效果
- 平均提升: {overall_analysis.get('overall', {}).get('mean_transfer_effect', 0):.4f}
- 迁移一致性: {overall_analysis.get('overall', {}).get('transfer_consistency', 0):.3f}

### 分领域分析
"""

        for category, data in overall_analysis.items():
            if category != "overall":
                report += f"- {category}: 平均提升={data.get('mean_improvement', 0):.4f} (±{data.get('std_improvement', 0):.4f})\n"

        report += """
## 四、结论与建议

### 主要发现
1. 棋类任务微调对数学推理能力的影响: {}
2. 棋类任务微调对逻辑推理能力的影响: {}
3. 棋类任务微调对规划能力的影响: {}

### 后续建议
1. 扩展测试范围到更多推理任务
2. 分析不同棋类游戏（围棋vs五子棋）的影响差异
3. 研究迁移效果的机制和边界条件
""".format(
            "正向迁移" if overall_analysis.get('math_reasoning', {}).get('mean_improvement', 0) > 0 else "无显著迁移",
            "正向迁移" if overall_analysis.get('logical_reasoning', {}).get('mean_improvement',
                                                                            0) > 0 else "无显著迁移",
            "正向迁移" if overall_analysis.get('planning_ability', {}).get('mean_improvement', 0) > 0 else "无显著迁移"
        )

        return report


class VisualizationGenerator:
    """可视化生成器"""

    @staticmethod
    def plot_results(results: Dict[str, Dict], save_path: str = "./results_plot.png"):
        """绘制实验结果"""
        # 提取数据
        benchmarks = []
        base_accs = []
        chess_accs = []
        improvements = []

        for benchmark, data in results.items():
            if isinstance(data, dict) and "base_accuracy" in data:
                benchmarks.append(benchmark)
                base_accs.append(data["base_accuracy"])
                chess_accs.append(data["chess_accuracy"])
                improvements.append(data.get("improvement", 0))

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 子图1: 准确率对比
        x = np.arange(len(benchmarks))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, base_accs, width, label='基础模型', color='skyblue')
        bars2 = ax1.bar(x + width / 2, chess_accs, width, label='棋类模型', color='lightcoral')

        ax1.set_xlabel('评估基准')
        ax1.set_ylabel('准确率')
        ax1.set_title('推理能力对比：基础模型 vs 棋类微调模型')
        ax1.set_xticks(x)
        ax1.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 在柱子上添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3点垂直偏移
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)

        # 子图2: 提升幅度
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax2.bar(benchmarks, improvements, color=colors)

        ax2.set_xlabel('评估基准')
        ax2.set_ylabel('提升幅度')
        ax2.set_title('棋类微调带来的推理能力提升')
        ax2.set_xticks(x)
        ax2.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)

        # 在柱子上添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax2.annotate(f'{height:+.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3 if height >= 0 else -12),
                         textcoords="offset points",
                         ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"可视化图表已保存至: {save_path}")

    @staticmethod
    def plot_transfer_pattern(analysis: Dict[str, Any], save_path: str = "./transfer_pattern.png"):
        """绘制迁移模式图"""
        categories = ["math_reasoning", "logical_reasoning", "planning_ability"]
        means = []
        stds = []

        for category in categories:
            if category in analysis:
                means.append(analysis[category]["mean_improvement"])
                stds.append(analysis[category]["std_improvement"])
            else:
                means.append(0)
                stds.append(0)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(categories))
        bars = ax.bar(x, means, yerr=stds, capsize=10, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])

        ax.set_xlabel('推理能力类别')
        ax.set_ylabel('平均提升幅度')
        ax.set_title('棋类任务微调对不同推理能力的迁移效果')
        ax.set_xticks(x)
        ax.set_xticklabels(['数学推理', '逻辑推理', '规划能力'])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:+.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.evaluator = None
        self.results = {}

        # 创建结果目录
        if config.save_results and not os.path.exists(config.results_dir):
            os.makedirs(config.results_dir)

    def run_full_experiment(self):
        """运行完整实验"""
        print("=" * 60)
        print("大语言模型推理能力迁移实验")
        print("=" * 60)

        # 1. 加载模型
        self.model_manager.load_models()

        # 2. 初始化评估器
        self.evaluator = ReasoningBenchmarkEvaluator(self.model_manager)

        # 3. 运行各项测试
        print("\n开始运行推理能力迁移实验...")

        # 数学推理测试
        math_results = self.evaluator.evaluate_math_reasoning()
        self.results.update(math_results)

        # 逻辑推理测试
        logic_results = self.evaluator.evaluate_logical_reasoning()
        self.results.update(logic_results)

        # 规划能力测试
        planning_results = self.evaluator.evaluate_planning_ability()
        self.results.update(planning_results)

        # 4. 分析迁移效果
        print("\n分析迁移效果...")
        analyzer = TransferEffectAnalyzer(self.results)
        transfer_analysis = analyzer.analyze_overall_transfer()
        self.results["transfer_analysis"] = transfer_analysis

        # 5. 生成报告和可视化
        if self.config.save_results:
            self._save_results()
            self._generate_visualizations()

            report = analyzer.generate_report()
            report_path = os.path.join(self.config.results_dir, "experiment_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n实验报告已保存至: {report_path}")

        print("\n" + "=" * 60)
        print("实验完成!")
        print("=" * 60)

        return self.results

    def _save_results(self):
        """保存实验结果"""
        # 保存详细结果
        results_path = os.path.join(self.config.results_dir, "detailed_results.json")

        # 转换为可JSON序列化的格式
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (np.integer, np.floating)):
                        serializable_results[key][subkey] = float(subvalue)
                    elif isinstance(subvalue, np.ndarray):
                        serializable_results[key][subkey] = subvalue.tolist()
                    elif isinstance(subvalue, (tuple, list)):
                        # 处理元组（如置信区间）
                        if isinstance(subvalue, tuple):
                            serializable_results[key][subkey] = list(subvalue)
                        else:
                            serializable_results[key][subkey] = subvalue
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = value

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"详细结果已保存至: {results_path}")

        # 保存摘要结果
        summary = {}
        for benchmark, result in self.results.items():
            if isinstance(result, dict):
                if "base_accuracy" in result:
                    summary[benchmark] = {
                        "base_accuracy": result["base_accuracy"],
                        "chess_accuracy": result["chess_accuracy"],
                        "improvement": result.get("improvement", 0),
                        "p_value": result.get("p_value"),
                        "significant": result.get("statistically_significant")
                    }

        summary_path = os.path.join(self.config.results_dir, "summary_results.csv")
        df = pd.DataFrame.from_dict(summary, orient='index')
        df.to_csv(summary_path)
        print(f"摘要结果已保存至: {summary_path}")

    def _generate_visualizations(self):
        """生成可视化图表"""
        viz = VisualizationGenerator()

        # 过滤出有准确率数据的基准测试
        plot_data = {}
        for benchmark, result in self.results.items():
            if isinstance(result, dict) and "base_accuracy" in result:
                plot_data[benchmark] = result

        if plot_data:
            plot_path = os.path.join(self.config.results_dir, "accuracy_comparison.png")
            viz.plot_results(plot_data, plot_path)

            if "transfer_analysis" in self.results:
                pattern_path = os.path.join(self.config.results_dir, "transfer_pattern.png")
                viz.plot_transfer_pattern(self.results["transfer_analysis"], pattern_path)


def main():
    """主函数"""
    # reasoning_transfer_experiment.py 修改示例
    config = ExperimentConfig(
        # 基础模型：使用在线模型ID（Hugging Face将自动下载或使用缓存）
        base_model_path="Qwen/Qwen2.5-7B-Instruct",

        # 棋类微调模型：使用训练脚本生成的本地路径
        chess_finetuned_path="./checkpoints/qwen7b-gomoku-lora/final_model",

        # Tokenizer名称，通常与基础模型在线ID一致
        model_name="Qwen/Qwen2.5-7B-Instruct",

        # 其他参数保持不变
        n_samples_per_test=50,
        save_results=True,
        results_dir="./experiment_results"
    )

    # 运行实验
    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()

    # 打印关键结果
    print("\n关键结果摘要:")
    print("-" * 40)

    for benchmark, result in results.items():
        if isinstance(result, dict) and "base_accuracy" in result:
            print(f"{benchmark.upper()}:")
            print(f"  基础模型: {result['base_accuracy']:.3f}")
            print(f"  棋类模型: {result['chess_accuracy']:.3f}")
            print(f"  提升幅度: {result.get('improvement', 0):+.3f}")
            if result.get('p_value'):
                sig = "显著" if result.get('statistically_significant') else "不显著"
                print(f"  统计显著性: p={result['p_value']:.4f} ({sig})")
            print()

    # 打印迁移分析
    if "transfer_analysis" in results:
        analysis = results["transfer_analysis"]
        if "overall" in analysis:
            print("整体迁移效果:")
            print(f"  平均提升: {analysis['overall']['mean_transfer_effect']:.4f}")
            print(f"  迁移一致性: {analysis['overall']['transfer_consistency']:.3f}")


if __name__ == "__main__":
    main()