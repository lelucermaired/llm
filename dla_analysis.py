"""
dla_analysis.py

Direct Logit Attribution (DLA) 分析
量化每个Attention头和MLP层对最终token预测的直接贡献

研究问题：
  五子棋SFT微调改变了哪些组件对数学推理任务的预测贡献？
  base vs v2(CE) vs dft(DFT)

方法：
  对于每个transformer层l，将其输出投影到unembedding矩阵：
  DLA(l) = W_U @ (W_O @ attn_out[l])  ← attention贡献
  DLA(l) = W_U @ mlp_out[l]           ← MLP贡献

  关注目标token（正确答案的第一个token）的logit值变化。

参考：
  同门研究：大模型逻辑抑制电路的跨架构共性
  Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
"""

import os, json, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "models": {
        "v2":           "./archive/checkpoints/qwen-gomoku-real/final_model",
        "cot_detailed": "./checkpoints/qwen-gomoku-cot-detailed/final_model",
    },
    "output_dir": "./results/dla_cot_vs_v2",
    "n_layers": 28,
    "n_heads": 28,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# 数学推理测试题（只取答案首token明确的题目）
MATH_PROBES = [
    ("What is 15 + 27? Answer with just the number.", "42"),
    ("What is 7 multiplied by 8? Answer with just the number.", "56"),
    ("What is 100 divided by 4? Answer with just the number.", "25"),
    ("What is the square root of 144? Answer with just the number.", "12"),
    ("What is 3 to the power of 4? Answer with just the number.", "81"),
    ("What is 45 + 67? Answer with just the number.", "112"),
    ("What is 9 multiplied by 9? Answer with just the number.", "81"),
    ("What is 200 divided by 8? Answer with just the number.", "25"),
    ("What is 2 to the power of 8? Answer with just the number.", "256"),
    ("What is the next prime number after 13? Answer with just the number.", "17"),
]


def load_model(adapter_path, base_path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path, quantization_config=bnb,
        device_map="auto", local_files_only=True,
        trust_remote_code=True, low_cpu_mem_usage=True,
        output_hidden_states=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    return model


def get_unembedding(model):
    """获取unembedding矩阵 W_U"""
    # Qwen2.5的结构：model.base_model.model.lm_head.weight
    try:
        # PeftModel包装
        lm_head = model.base_model.model.lm_head
    except AttributeError:
        lm_head = model.lm_head
    return lm_head.weight.float()  # (vocab_size, hidden_dim)


def compute_dla_for_prompt(model, tokenizer, prompt, answer, device):
    """
    计算单个prompt的DLA：每层attention和MLP对answer首token的贡献

    Returns:
        attn_dla: (n_layers,) 各层attention对answer token的logit贡献
        mlp_dla:  (n_layers,) 各层MLP对answer token的logit贡献
        answer_token_id: 答案首token的id
    """
    # 构建输入
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 答案首token
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_tokens:
        return None, None, None
    answer_token_id = answer_tokens[0]

    # 获取unembedding矩阵
    W_U = get_unembedding(model)  # (vocab, hidden)
    answer_vec = W_U[answer_token_id]  # (hidden,)

    # 注册hook收集各层输出
    attn_outputs = {}
    mlp_outputs = {}

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            # output可能是tuple，取第一个元素
            out = output[0] if isinstance(output, tuple) else output
            attn_outputs[layer_idx] = out.detach().float()
        return hook

    def make_mlp_hook(layer_idx):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            mlp_outputs[layer_idx] = out.detach().float()
        return hook

    # 注册hooks
    hooks = []
    try:
        base_model = model.base_model.model
    except AttributeError:
        base_model = model

    for i, layer in enumerate(base_model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 移除hooks
    for h in hooks:
        h.remove()

    # 计算DLA：每层输出在answer token方向上的投影
    # DLA(l) = answer_vec · layer_output[last_token]
    last_pos = inputs["input_ids"].shape[1] - 1

    attn_dla = np.zeros(CONFIG["n_layers"])
    mlp_dla = np.zeros(CONFIG["n_layers"])

    for i in range(CONFIG["n_layers"]):
        if i in attn_outputs:
            attn_out = attn_outputs[i][0, last_pos, :]  # (hidden,)
            # 归一化后计算内积
            attn_dla[i] = torch.dot(
                answer_vec.to(attn_out.device),
                attn_out
            ).item()

        if i in mlp_outputs:
            mlp_out = mlp_outputs[i][0, last_pos, :]  # (hidden,)
            mlp_dla[i] = torch.dot(
                answer_vec.to(mlp_out.device),
                mlp_out
            ).item()

    return attn_dla, mlp_dla, answer_token_id


def analyze_model(model_name, adapter_path, tokenizer, device):
    """对一个模型在所有探针题上计算平均DLA"""
    print(f"\n分析模型：{model_name}")
    model = load_model(adapter_path, CONFIG["base_model_path"])

    all_attn_dla = []
    all_mlp_dla = []

    for prompt, answer in tqdm(MATH_PROBES, desc=f"DLA {model_name}"):
        attn_dla, mlp_dla, _ = compute_dla_for_prompt(
            model, tokenizer, prompt, answer, device)
        if attn_dla is not None:
            all_attn_dla.append(attn_dla)
            all_mlp_dla.append(mlp_dla)

    del model
    torch.cuda.empty_cache()

    mean_attn = np.mean(all_attn_dla, axis=0)  # (n_layers,)
    mean_mlp = np.mean(all_mlp_dla, axis=0)

    return mean_attn, mean_mlp


def plot_dla_comparison(results, output_dir):
    """绘制base vs v2的DLA对比图"""
    n_layers = CONFIG["n_layers"]
    layers = list(range(n_layers))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Direct Logit Attribution: v2(伪推理链) vs cot_detailed(详细推理链)\n数学推理任务，answer token方向投影",
                 fontsize=13)

    colors = {"v2": "#2196F3", "cot_detailed": "#FF5722"}

    # 上左：Attention DLA各层
    ax = axes[0, 0]
    for name, (attn, mlp) in results.items():
        ax.plot(layers, attn, label=name, color=colors[name], linewidth=1.5)
    ax.set_title("Attention层DLA（逐层）")
    ax.set_xlabel("Layer")
    ax.set_ylabel("DLA (logit贡献)")
    ax.legend()
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # 上右：MLP DLA各层
    ax = axes[0, 1]
    for name, (attn, mlp) in results.items():
        ax.plot(layers, mlp, label=name, color=colors[name], linewidth=1.5)
    ax.set_title("MLP层DLA（逐层）")
    ax.set_xlabel("Layer")
    ax.set_ylabel("DLA (logit贡献)")
    ax.legend()
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # 下左：DLA差值（v2 - base），Attention
    ax = axes[1, 0]
    base_attn, _ = results["v2"]
    v2_attn, _ = results["cot_detailed"]
    delta_attn = v2_attn - base_attn
    colors_bar = ['#E53935' if d < 0 else '#43A047' for d in delta_attn]
    ax.bar(layers, delta_attn, color=colors_bar, alpha=0.8)
    ax.set_title("Attention DLA差值（cot_detailed - v2）\n正=cot_detailed更强，负=v2更强")
    ax.set_xlabel("Layer")
    ax.set_ylabel("ΔDLA")
    ax.axhline(0, color='gray', linewidth=1)

    # 下右：DLA差值（v2 - base），MLP
    ax = axes[1, 1]
    _, base_mlp = results["v2"]
    _, v2_mlp = results["cot_detailed"]
    delta_mlp = v2_mlp - base_mlp
    colors_bar = ['#E53935' if d < 0 else '#43A047' for d in delta_mlp]
    ax.bar(layers, delta_mlp, color=colors_bar, alpha=0.8)
    ax.set_title("MLP DLA差值（cot_detailed - v2）\n正=cot_detailed更强，负=v2更强")
    ax.set_xlabel("Layer")
    ax.set_ylabel("ΔDLA")
    ax.axhline(0, color='gray', linewidth=1)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "dla_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 图表已保存: {save_path}")
    return save_path


def main():
    print("=" * 65)
    print("Direct Logit Attribution (DLA) 分析")
    print("量化五子棋SFT对数学推理任务预测贡献的影响")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for model_name, adapter_path in CONFIG["models"].items():
        attn_dla, mlp_dla = analyze_model(
            model_name, adapter_path, tokenizer, device)
        results[model_name] = (attn_dla, mlp_dla)

    # 打印关键数据
    print("\n" + "=" * 65)
    print("DLA分析结果（各层平均，数学推理任务）")
    print("=" * 65)

    base_attn, base_mlp = results["v2"]
    v2_attn, v2_mlp = results["cot_detailed"]
    delta_attn = v2_attn - base_attn
    delta_mlp = v2_mlp - base_mlp

    # 找变化最大的层
    top_attn_layers = np.argsort(np.abs(delta_attn))[-5:][::-1]
    top_mlp_layers = np.argsort(np.abs(delta_mlp))[-5:][::-1]

    print("\nAttention层变化最大的5层（v2 - base）：")
    for l in top_attn_layers:
        print(f"  Layer {l:2d}: Δ={delta_attn[l]:+.2f} "
              f"(base={base_attn[l]:.2f}, v2={v2_attn[l]:.2f})")

    print("\nMLP层变化最大的5层（v2 - base）：")
    for l in top_mlp_layers:
        print(f"  Layer {l:2d}: Δ={delta_mlp[l]:+.2f} "
              f"(base={base_mlp[l]:.2f}, v2={v2_mlp[l]:.2f})")

    # 统计方向
    attn_pos = (delta_attn > 0).sum()
    attn_neg = (delta_attn < 0).sum()
    mlp_pos = (delta_mlp > 0).sum()
    mlp_neg = (delta_mlp < 0).sum()

    print(f"\nAttention：{attn_pos}层正向变化，{attn_neg}层负向变化")
    print(f"MLP：      {mlp_pos}层正向变化，{mlp_neg}层负向变化")

    total_attn_change = np.abs(delta_attn).sum()
    total_mlp_change = np.abs(delta_mlp).sum()
    print(f"\n总变化量对比：")
    print(f"  Attention总变化：{total_attn_change:.2f}")
    print(f"  MLP总变化：      {total_mlp_change:.2f}")
    print(f"  Attention/MLP比值：{total_attn_change/total_mlp_change:.3f}")

    if total_attn_change > total_mlp_change:
        print(f"\n→ cot_detailed vs v2：Attention变化大于MLP（{total_attn_change/total_mlp_change:.2f}x）")
        print(f"  详细推理链主要影响attention查询方向，MLP贡献变化不显著")
        print(f"  与v2的零迁移机制相同，推理链质量未能触及MLP")
    else:
        print(f"\n→ cot_detailed vs v2：MLP变化大于Attention（{total_mlp_change/total_attn_change:.2f}x）")
        print(f"  详细推理链使MLP对数学推理的贡献发生更大变化，可能是正向迁移的机制来源")

    # 保存数据
    save_data = {
        "base_attn": base_attn.tolist(),
        "base_mlp": base_mlp.tolist(),
        "v2_attn": v2_attn.tolist(),
        "v2_mlp": v2_mlp.tolist(),
        "delta_attn": delta_attn.tolist(),
        "delta_mlp": delta_mlp.tolist(),
        "top_attn_layers": top_attn_layers.tolist(),
        "top_mlp_layers": top_mlp_layers.tolist(),
    }
    with open(os.path.join(CONFIG["output_dir"], "dla_data.json"),
              "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    # 绘图
    plot_dla_comparison(results, CONFIG["output_dir"])

    print("\n" + "=" * 65)
    print("✅ DLA分析完成")
    print(f"数据：{CONFIG['output_dir']}/dla_data.json")
    print(f"图表：{CONFIG['output_dir']}/dla_comparison.png")
    print("=" * 65)


if __name__ == "__main__":
    main()