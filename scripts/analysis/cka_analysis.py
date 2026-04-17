"""
cka_analysis.py

CKA（Centered Kernel Alignment）表示相似度分析
比较基础模型与五子棋微调模型在数学推理任务上的内部表示差异

用法:
    python cka_analysis.py
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "finetuned_model_path": "./checkpoints/qwen-gomoku-real/final_model",
    "output_dir": "./cka_results_real",
    "n_samples": 20,           # 用于CKA计算的样本数（20个足够）
    "max_length": 256,
    "layers_to_analyze": list(range(0, 28, 2)),  # 每隔2层分析一次，共14层
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ==================== CKA计算 ====================
def center_kernel(K):
    """对kernel矩阵做中心化"""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_CKA(X, Y):
    """
    计算线性CKA
    X, Y: (n_samples, hidden_dim) 的激活矩阵
    返回: CKA相似度值 [0, 1]
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    # 中心化
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # 计算HSIC
    XXT = X @ X.T
    YYT = Y @ Y.T

    hsic_xy = np.sum(center_kernel(XXT) * center_kernel(YYT))
    hsic_xx = np.sum(center_kernel(XXT) * center_kernel(XXT))
    hsic_yy = np.sum(center_kernel(YYT) * center_kernel(YYT))

    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0

    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


# ==================== 激活值提取 ====================
class ActivationExtractor:
    """提取模型指定层的激活值"""

    def __init__(self, model, layer_indices):
        self.model = model
        self.layer_indices = layer_indices
        self.activations = {}
        self.hooks = []

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # last token pooling: 取最后一个token的表示
            # 比均值池化更敏感，能捕捉LoRA对特定位置的细粒度修改
            self.activations[layer_idx] = hidden[:, -1, :].detach().cpu().float().numpy()
        return hook

    def register_hooks(self):
        """注册钩子到transformer层"""
        # 兼容PEFT包装的模型
        try:
            layers = self.model.base_model.model.model.layers
        except AttributeError:
            try:
                layers = self.model.model.layers
            except AttributeError:
                layers = self.model.base_model.layers

        for idx in self.layer_indices:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._make_hook(idx))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear(self):
        self.activations.clear()


# ==================== 主分析流程 ====================
def load_math_prompts(n_samples):
    """加载GSM8K数学题作为探针"""
    print("加载GSM8K数学题...")
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        prompts = []
        for idx in indices:
            q = dataset[int(idx)]["question"]
            prompts.append(f"Solve this math problem step by step.\n\nQuestion: {q}\n\nAnswer:")
        print(f"✅ 加载了 {len(prompts)} 个数学题")
        return prompts
    except Exception as e:
        print(f"GSM8K加载失败: {e}，使用备用题目")
        return [
            f"Solve: If x + {i} = {i*2}, what is x? Answer step by step."
            for i in range(1, n_samples + 1)
        ]


def extract_activations_for_model(model, tokenizer, prompts, layer_indices, desc=""):
    """对一个模型提取所有层的激活值矩阵"""
    extractor = ActivationExtractor(model, layer_indices)
    extractor.register_hooks()

    # 每层的激活值列表：{layer_idx: [activation_per_sample]}
    all_activations = {idx: [] for idx in layer_indices}

    for prompt in tqdm(prompts, desc=desc):
        extractor.clear()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=False,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        for idx in layer_indices:
            if idx in extractor.activations:
                all_activations[idx].append(extractor.activations[idx][0])  # (hidden_dim,)

    extractor.remove_hooks()

    # 转成矩阵：{layer_idx: ndarray (n_samples, hidden_dim)}
    activation_matrices = {}
    for idx in layer_indices:
        if all_activations[idx]:
            activation_matrices[idx] = np.stack(all_activations[idx], axis=0)

    return activation_matrices


def compute_cka_per_layer(base_activations, ft_activations, layer_indices):
    """计算每层的CKA值"""
    cka_scores = {}
    for idx in layer_indices:
        if idx in base_activations and idx in ft_activations:
            score = linear_CKA(base_activations[idx], ft_activations[idx])
            cka_scores[idx] = score
            print(f"  Layer {idx:2d}: CKA = {score:.4f}")
    return cka_scores


def plot_cka_results(cka_scores_math, output_dir):
    """绘制CKA结果图"""
    layers = sorted(cka_scores_math.keys())
    math_values = [cka_scores_math[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, math_values, 'o-', color='#2196F3', linewidth=2,
            markersize=7, label='数学推理任务（GSM8K）')
    ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='CKA=0.99参考线')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='CKA=0.95参考线')

    ax.set_xlabel('Transformer层编号', fontsize=12)
    ax.set_ylabel('CKA相似度', fontsize=12)
    ax.set_title('基础模型 vs 五子棋微调模型\n数学推理任务上的逐层表示相似度（CKA）', fontsize=13)
    ax.set_ylim(0.8, 1.02)
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 标注均值
    mean_val = np.mean(math_values)
    ax.text(0.02, 0.05, f'平均CKA = {mean_val:.4f}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cka_similarity.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图表已保存至: {save_path}")


def main():
    print("=" * 60)
    print("CKA表示相似度分析")
    print("比较：基础模型 vs 五子棋微调模型")
    print("探针任务：数学推理（GSM8K）")
    print("=" * 60)

    np.random.seed(42)
    layer_indices = CONFIG["layers_to_analyze"]

    # 加载tokenizer
    print("\n加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数学题
    prompts = load_math_prompts(CONFIG["n_samples"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # ========== 第一步：提取基础模型激活值 ==========
    print("\n[1/4] 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model_path"],
        quantization_config=bnb_config,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=True,
    )
    base_model.eval()

    print("[2/4] 提取基础模型激活值...")
    base_activations = extract_activations_for_model(
        base_model, tokenizer, prompts, layer_indices, desc="基础模型"
    )

    del base_model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ========== 第二步：提取微调模型激活值 ==========
    print("\n[3/4] 加载微调模型...")
    base_for_ft = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model_path"],
        quantization_config=bnb_config,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(base_for_ft, CONFIG["finetuned_model_path"])
    ft_model.eval()

    print("[4/4] 提取微调模型激活值...")
    ft_activations = extract_activations_for_model(
        ft_model, tokenizer, prompts, layer_indices, desc="微调模型"
    )

    del ft_model
    torch.cuda.empty_cache()
    gc.collect()

    # ========== 计算CKA ==========
    print("\n计算逐层CKA相似度（数学推理任务）...")
    cka_scores = compute_cka_per_layer(base_activations, ft_activations, layer_indices)

    # ========== 输出结果 ==========
    mean_cka = np.mean(list(cka_scores.values()))
    min_cka = min(cka_scores.values())
    max_cka = max(cka_scores.values())

    print("\n" + "=" * 40)
    print(f"平均CKA: {mean_cka:.4f}")
    print(f"最小CKA: {min_cka:.4f} (层 {min(cka_scores, key=cka_scores.get)})")
    print(f"最大CKA: {max_cka:.4f} (层 {max(cka_scores, key=cka_scores.get)})")

    if mean_cka > 0.98:
        print("\n结论：CKA极高（>0.98），五子棋微调几乎没有改变模型处理数学题的内部表示。")
        print("这从机制上解释了为什么跨域迁移为零。")
    elif mean_cka > 0.95:
        print("\n结论：CKA较高（>0.95），微调对内部表示影响很小。")
    else:
        print("\n结论：CKA有明显变化，微调改变了部分层的表示。")

    # 保存数值结果
    results = {
        "layer_cka_scores": {str(k): v for k, v in cka_scores.items()},
        "mean_cka": mean_cka,
        "min_cka": min_cka,
        "max_cka": max_cka,
        "n_samples": CONFIG["n_samples"],
        "model": CONFIG["finetuned_model_path"],
    }
    results_path = os.path.join(CONFIG["output_dir"], "cka_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 数值结果已保存至: {results_path}")

    # 绘图
    plot_cka_results(cka_scores, CONFIG["output_dir"])

    print("\n✅ CKA分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()