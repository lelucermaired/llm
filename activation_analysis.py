"""
激活差异分析
用hook记录base和maxlora在BBH推理过程中每层的激活值
计算KL散度和余弦距离，比较正迁移vs负迁移任务的差异模式

指标：
  1. 余弦距离：base vs maxlora 每层隐状态的方向差异
  2. L2距离：激活幅度差异
  3. KL散度：激活分布差异（对token维度做softmax后计算）

任务：
  - object_counting（正迁移 +24%）
  - tracking_shuffled_objects_three_objects（负迁移 -22%）
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# ── 配置 ──────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH   = "/root/autodl-tmp/llm-project/checkpoints/qwen-gomoku-maxlora/final_model"
HF_CACHE    = "/root/autodl-tmp/hf_cache"
OUTPUT_DIR  = "/root/autodl-tmp/llm-project/results/activation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 20   # 每个任务取多少题做平均（够快，结果稳定）
MAX_NEW_TOKENS = 1  # 只看输入处理阶段，不需要生成

TARGET_TASKS = {
    "object_counting"                          : "正迁移 (+24%)",
    "tracking_shuffled_objects_three_objects"  : "负迁移 (-22%)",
}

PROMPT_TEMPLATE = """\
{input}

Think step by step and give your final answer after "the answer is".
"""

# ── Hook机制 ──────────────────────────────────────────────
class ActivationRecorder:
    """给每层的残差流出口注册forward hook，记录隐状态"""
    def __init__(self):
        self.activations = {}   # layer_idx -> list of tensors
        self.hooks = []

    def register(self, model):
        self.clear()
        # 兼容PeftModel和普通model
        layers = self._get_layers(model)
        for idx, layer in enumerate(layers):
            h = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(h)
        print(f"  [Hook] 注册了 {len(self.hooks)} 层")

    def _get_layers(self, model):
        # PeftModel包了一层，需要取base_model
        try:
            return model.base_model.model.model.layers
        except AttributeError:
            try:
                return model.model.layers
            except AttributeError:
                return model.layers

    def _make_hook(self, idx):
        def hook(module, input, output):
            # output可能是tuple（Transformer层返回(hidden, cache)等）
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # 取最后一个token的隐状态，shape: (batch, seq_len, hidden)
            # 我们取最后一个位置，代表"读完整道题之后"的表征
            last_token = hidden[:, -1, :].detach().float().cpu()
            if idx not in self.activations:
                self.activations[idx] = []
            self.activations[idx].append(last_token)
        return hook

    def clear(self):
        self.activations = {}
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_mean_activation(self):
        """返回 {layer_idx: mean_vector (hidden_dim,)}"""
        result = {}
        for idx, tensors in self.activations.items():
            stacked = torch.cat(tensors, dim=0)  # (N, hidden)
            result[idx] = stacked.mean(dim=0)    # (hidden,)
        return result

# ── 数据加载 ──────────────────────────────────────────────
def load_bbh_task(task_name):
    ds = load_dataset(
        "lukaemon/bbh", task_name,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
    return [{"input": row["input"], "target": row["target"]} for row in split]

# ── 收集激活值 ────────────────────────────────────────────
def collect_activations(model, tokenizer, examples, recorder, n=20):
    """
    对前n道题，做一次forward pass（不生成）
    recorder会自动记录每层激活
    """
    model.eval()
    for ex in tqdm(examples[:n], desc="  收集激活", leave=False):
        prompt = PROMPT_TEMPLATE.format(input=ex["input"])
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            _ = model(**inputs)

    return recorder.get_mean_activation()

# ── 计算差异指标 ──────────────────────────────────────────
def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    cos_sim = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()
    return 1.0 - cos_sim  # 距离 = 1 - 相似度

def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.norm(a - b, p=2).item()

def kl_divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    """对隐状态向量做softmax后计算KL散度"""
    p = torch.softmax(a, dim=0)
    q = torch.softmax(b, dim=0)
    # KL(p||q)
    kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum().item()
    return kl

def compute_layer_diffs(base_acts, lora_acts, num_layers):
    results = {
        "cosine_dist": [],
        "l2_dist":     [],
        "kl_div":      [],
    }
    for layer in range(num_layers):
        if layer not in base_acts or layer not in lora_acts:
            for k in results:
                results[k].append(float("nan"))
            continue
        a = base_acts[layer]
        b = lora_acts[layer]
        results["cosine_dist"].append(cosine_distance(a, b))
        results["l2_dist"].append(l2_distance(a, b))
        results["kl_div"].append(kl_divergence(a, b))
    return results

# ── 绘图 ──────────────────────────────────────────────────
def plot_activation_diff(all_diffs, num_layers):
    """
    all_diffs: {task_name: {"cosine_dist": [...], "l2_dist": [...], "kl_div": [...]}}
    """
    metrics   = ["cosine_dist", "l2_dist", "kl_div"]
    m_labels  = ["Cosine Distance", "L2 Distance", "KL Divergence"]
    m_desc    = [
        "(larger = more directional change)",
        "(larger = larger magnitude change)",
        "(larger = more distribution change)",
    ]

    task_colors = {
        "object_counting"                         : ("#2c7bb6", "o", "正迁移 object_counting"),
        "tracking_shuffled_objects_three_objects" : ("#d7191c", "s", "负迁移 tracking"),
    }

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "Activation Difference: Base vs MaxLoRA per Layer\n"
        "Qwen2.5-7B (r=64, Gomoku) — Positive vs Negative Transfer Tasks",
        fontsize=14, fontweight="bold",
    )

    x = np.arange(num_layers)

    for ax, metric, label, desc in zip(axes, metrics, m_labels, m_desc):
        for task, diffs in all_diffs.items():
            color, marker, legend_label = task_colors[task]
            vals = diffs[metric]
            ax.plot(
                x, vals,
                color=color, marker=marker, markersize=5,
                linewidth=2, label=legend_label, alpha=0.9,
            )
            ax.fill_between(x, vals, alpha=0.08, color=color)

        # 分区背景
        ax.axvspan(0,  14, alpha=0.04, color="blue")
        ax.axvspan(14, 23, alpha=0.04, color="red")
        ax.axvspan(23, 28, alpha=0.04, color="green")

        # 区域标注（只在第一个子图加）
        if metric == "cosine_dist":
            ymax = ax.get_ylim()[1]
            ax.text(7,  ymax * 0.92, "Early\n(0-13)",  ha="center", color="blue",  fontsize=9, alpha=0.7)
            ax.text(18, ymax * 0.92, "Middle\n(14-22)", ha="center", color="red",   fontsize=9, alpha=0.7)
            ax.text(25, ymax * 0.92, "Late\n(23-27)",   ha="center", color="green", fontsize=9, alpha=0.7)

        ax.set_ylabel(f"{label}\n{desc}", fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(range(0, num_layers, 2))

    axes[-1].set_xlabel("Layer Index", fontsize=11)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "activation_diff_per_layer.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

def plot_zone_comparison(all_diffs, num_layers):
    """
    按早/中/晚三个区域聚合，做柱状图对比
    正迁移 vs 负迁移任务在每个区域的激活差异
    """
    zones = {
        "Early\n(0-13)":  list(range(0, 14)),
        "Middle\n(14-22)": list(range(14, 23)),
        "Late\n(23-27)":  list(range(23, 28)),
    }
    metric = "cosine_dist"

    task_colors = {
        "object_counting"                         : "#2c7bb6",
        "tracking_shuffled_objects_three_objects" : "#d7191c",
    }
    task_labels = {
        "object_counting"                         : "正迁移 object_counting",
        "tracking_shuffled_objects_three_objects" : "负迁移 tracking",
    }

    zone_names = list(zones.keys())
    x = np.arange(len(zone_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Zone-wise Activation Difference (Cosine Distance)\n"
        "Base vs MaxLoRA — Positive vs Negative Transfer",
        fontsize=13, fontweight="bold",
    )

    for i, (task, diffs) in enumerate(all_diffs.items()):
        vals_by_zone = []
        for zone, layer_idxs in zones.items():
            zone_vals = [diffs[metric][l] for l in layer_idxs
                         if l < len(diffs[metric]) and not np.isnan(diffs[metric][l])]
            vals_by_zone.append(np.mean(zone_vals) if zone_vals else 0)

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, vals_by_zone, width,
            label=task_labels[task],
            color=task_colors[task],
            alpha=0.85, edgecolor="white",
        )
        for bar, val in zip(bars, vals_by_zone):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.00002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(zone_names, fontsize=11)
    ax.set_ylabel("Mean Cosine Distance (base vs maxlora)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "zone_activation_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[保存] {out}")
    plt.close()

# ── 打印汇总 ──────────────────────────────────────────────
def print_summary(all_diffs, num_layers):
    zones = {
        "early (0-13)":   list(range(0, 14)),
        "middle (14-22)": list(range(14, 23)),
        "late (23-27)":   list(range(23, 28)),
    }

    print("\n── 区域余弦距离均值 ──────────────────────────────────")
    print(f"{'区域':<18} {'object_counting':>20} {'tracking':>20}")
    print("-" * 60)

    summary = {}
    for zone, layer_idxs in zones.items():
        row = {}
        for task, diffs in all_diffs.items():
            vals = [diffs["cosine_dist"][l] for l in layer_idxs
                    if l < len(diffs["cosine_dist"]) and not np.isnan(diffs["cosine_dist"][l])]
            row[task] = np.mean(vals) if vals else float("nan")
        summary[zone] = row

        tasks = list(all_diffs.keys())
        print(f"{zone:<18} {row.get(tasks[0], float('nan')):>20.6f} "
              f"{row.get(tasks[1], float('nan')):>20.6f}")

    # 保存
    out = os.path.join(OUTPUT_DIR, "activation_summary.json")
    serializable = {
        zone: {task: float(v) for task, v in row.items()}
        for zone, row in summary.items()
    }
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[保存] {out}")

    # 关键对比
    print("\n── 关键发现 ──────────────────────────────────────────")
    tasks = list(all_diffs.keys())
    if len(tasks) >= 2:
        for zone in zones:
            v1 = summary[zone].get(tasks[0], float("nan"))
            v2 = summary[zone].get(tasks[1], float("nan"))
            diff = v1 - v2
            direction = "正迁移更大" if diff > 0 else "负迁移更大"
            print(f"  {zone}: 正迁移={v1:.6f}, 负迁移={v2:.6f} → {direction} (Δ={abs(diff):.6f})")

# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Activation Difference Analysis")
    print("  Base vs MaxLoRA, per layer, per task")
    print("=" * 60)

    # 加载数据
    print("\n[数据] 加载BBH任务...")
    task_data = {}
    for task in TARGET_TASKS:
        examples = load_bbh_task(task)
        task_data[task] = examples
        print(f"  {task}: {len(examples)} 条，取前 {NUM_SAMPLES} 条")

    # 加载base模型
    print("\n[加载] base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # 加载maxlora模型
    print("[加载] maxlora model...")
    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    lora_model.eval()

    # 检测层数
    try:
        num_layers = len(lora_model.base_model.model.model.layers)
    except Exception:
        num_layers = 28
    print(f"[信息] 检测到 {num_layers} 层")

    recorder_base = ActivationRecorder()
    recorder_lora = ActivationRecorder()

    all_diffs = {}

    for task, examples in task_data.items():
        print(f"\n── 任务: {task} ({TARGET_TASKS[task]}) ──")

        # 收集base激活
        print("  [Base] 收集激活...")
        recorder_base.register(base_model)
        base_acts = collect_activations(base_model, tokenizer, examples,
                                        recorder_base, n=NUM_SAMPLES)
        recorder_base.clear()

        # 收集lora激活
        print("  [LoRA] 收集激活...")
        recorder_lora.register(lora_model)
        lora_acts = collect_activations(lora_model, tokenizer, examples,
                                        recorder_lora, n=NUM_SAMPLES)
        recorder_lora.clear()

        # 计算差异
        print("  [计算] 逐层差异指标...")
        diffs = compute_layer_diffs(base_acts, lora_acts, num_layers)
        all_diffs[task] = diffs

        # 快速预览
        cos_vals = diffs["cosine_dist"]
        top3 = sorted(range(num_layers),
                      key=lambda i: cos_vals[i] if not np.isnan(cos_vals[i]) else -1,
                      reverse=True)[:3]
        print(f"  激活差异最大的3层: {top3} "
              f"(cosine_dist={[f'{cos_vals[i]:.4f}' for i in top3]})")

    # 保存原始数据
    raw_out = os.path.join(OUTPUT_DIR, "all_diffs.json")
    with open(raw_out, "w") as f:
        json.dump({task: {k: [float(x) for x in v]
                          for k, v in diffs.items()}
                   for task, diffs in all_diffs.items()}, f, indent=2)
    print(f"\n[保存] {raw_out}")

    # 绘图
    print("\n[绘图] 逐层差异折线图...")
    plot_activation_diff(all_diffs, num_layers)

    print("[绘图] 区域对比柱状图...")
    plot_zone_comparison(all_diffs, num_layers)

    print_summary(all_diffs, num_layers)

    print(f"\n✓ 完成！输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()