import os, json, re, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset

CONFIG = {
    "base_model_path":  "Qwen/Qwen2.5-7B-Instruct",
    "v2_adapter":       "./archive/checkpoints/qwen-gomoku-real/final_model",
    "output_dir":       "./results/dla_reset_analysis",
    "n_layers":         28,
    "n_questions":      50,
    "min_steps":        4,
    "cache_dir":        "./cache",
}
os.makedirs(CONFIG["output_dir"], exist_ok=True)

def load_questions():
    cache = os.path.join(CONFIG["output_dir"], "gsm8k_questions.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            qs = json.load(f)
        print(f"[缓存] {len(qs)}道GSM8K难题")
        return qs[:CONFIG["n_questions"]]
    print("加载GSM8K...")
    ds = load_dataset("gsm8k", "main")
    hard = []
    for item in ds["test"]:
        m = re.search(r'####\s*([\d,]+)', item["answer"])
        if not m: continue
        ans = m.group(1).replace(',', '')
        steps = len([l for l in item["answer"].split('\n') if l.strip()])
        if steps >= CONFIG["min_steps"]:
            hard.append({"question": item["question"], "answer": ans, "steps": steps})
    hard.sort(key=lambda x: x["steps"], reverse=True)
    selected = hard[:CONFIG["n_questions"]]
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    return selected

def load_model(reset_layers=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model_path"], quantization_config=bnb,
        device_map="auto", local_files_only=True,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, CONFIG["v2_adapter"])
    if reset_layers:
        n = 0
        for name, param in model.named_parameters():
            if ('lora_A' in name or 'lora_B' in name) and ('q_proj' in name or 'v_proj' in name):
                m = re.search(r'\.(\d+)\.', name)
                if m and int(m.group(1)) < reset_layers:
                    param.data.zero_()
                    n += 1
        print(f"  重置{n}个参数（前{reset_layers}层q+v）")
    model.eval()
    return model

def get_ans_dir(model, ans_id):
    """获取answer token在lm_head中的方向向量"""
    # 遍历所有module找lm_head
    for name, mod in model.named_modules():
        if 'lm_head' in name and hasattr(mod, 'weight') and mod.weight is not None:
            w = mod.weight
            if w.shape[0] > 1000:  # vocab size > 1000，确认是lm_head
                vec = w[ans_id].detach().float()
                return vec / (vec.norm() + 1e-8)
    raise RuntimeError("找不到lm_head weight")

def compute_dla(model, tokenizer, question, answer, device):
    prompt = f"Solve step by step.\nProblem: {question}\nAnswer:"
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]
    last_pos = seq_len - 1

    ans_toks = tokenizer.encode(answer, add_special_tokens=False)
    if not ans_toks:
        return None, None
    ans_id = ans_toks[0]

    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][seq_len:], skip_special_tokens=True).strip()
            correct = answer in resp.replace(',', '')

            fwd = model(**inputs, output_hidden_states=True)
            hs = fwd.hidden_states  # len=n_layers+1

        ans_dir = get_ans_dir(model, ans_id).cpu()

        dla = np.zeros(CONFIG["n_layers"])
        for i in range(CONFIG["n_layers"]):
            h_in  = hs[i  ][0, last_pos, :].float().cpu()
            h_out = hs[i+1][0, last_pos, :].float().cpu()
            dla[i] = torch.dot(h_out - h_in, ans_dir).item()
        return dla, correct
    except Exception as e:
        print(f"\n  错误: {e}")
        return None, None

def analyze(name, model, tokenizer, questions, device):
    cache = os.path.join(CONFIG["output_dir"], f"dla_{name}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            d = json.load(f)
        print(f"[缓存] {name}: acc={d['accuracy']:.3f}")
        return np.array(d["dla"]), d["accuracy"]

    all_dla, n_ok = [], 0
    for q in tqdm(questions, desc=f"  {name}"):
        dla, ok = compute_dla(model, tokenizer, q["question"], q["answer"], device)
        if dla is None: continue
        all_dla.append(dla)
        if ok: n_ok += 1

    mean_dla = np.mean(all_dla, axis=0) if all_dla else np.zeros(CONFIG["n_layers"])
    acc = n_ok / len(questions)
    print(f"  ✅ {name}: acc={acc:.3f}, valid={len(all_dla)}/{len(questions)}")
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"dla": mean_dla.tolist(), "accuracy": acc}, f, indent=2)
    return mean_dla, acc

def plot(v2_dla, r_dla, output_dir):
    layers = list(range(CONFIG["n_layers"]))
    delta = r_dla - v2_dla
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DLA: v2 vs reset_10_qv  (GSM8K hard)")
    axes[0].plot(layers, v2_dla, label="v2", color="#FF5722", lw=1.5)
    axes[0].plot(layers, r_dla, label="reset_10_qv", color="#2196F3", lw=1.5)
    axes[0].axvline(10, color='green', ls='--', lw=1)
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].set_title("Layer DLA"); axes[0].legend()
    colors = ['#43A047' if d > 0 else '#E53935' for d in delta]
    axes[1].bar(layers, delta, color=colors, alpha=0.85)
    axes[1].axvline(10, color='green', ls='--', lw=1)
    axes[1].axhline(0, color='gray', lw=1)
    axes[1].set_title("Delta DLA (reset - v2)")
    plt.tight_layout()
    path = os.path.join(output_dir, "dla_reset_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表: {path}")

def main():
    print("=" * 60)
    print("DLA分析：v2 vs reset_10_qv")
    print("=" * 60)
    questions = load_questions()
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    v2_cache = os.path.join(CONFIG["output_dir"], "dla_v2.json")
    if not os.path.exists(v2_cache):
        m = load_model(None)
        v2_dla, v2_acc = analyze("v2", m, tokenizer, questions, device)
        del m; import gc; gc.collect(); torch.cuda.empty_cache()
    else:
        with open(v2_cache, encoding="utf-8") as f:
            d = json.load(f)
        v2_dla, v2_acc = np.array(d["dla"]), d["accuracy"]
        print(f"[缓存] v2: acc={v2_acc:.3f}")

    r_cache = os.path.join(CONFIG["output_dir"], "dla_reset_10_qv.json")
    if not os.path.exists(r_cache):
        m = load_model(10)
        r_dla, r_acc = analyze("reset_10_qv", m, tokenizer, questions, device)
        del m; import gc; gc.collect(); torch.cuda.empty_cache()
    else:
        with open(r_cache, encoding="utf-8") as f:
            d = json.load(f)
        r_dla, r_acc = np.array(d["dla"]), d["accuracy"]
        print(f"[缓存] reset_10_qv: acc={r_acc:.3f}")

    delta = r_dla - v2_dla
    top5 = np.argsort(np.abs(delta))[::-1][:5]
    print(f"\n准确率：v2={v2_acc:.3f}  reset={r_acc:.3f}  delta={r_acc-v2_acc:+.3f}")
    print(f"\nDLA变化最大5层（reset-v2）：")
    for li in top5:
        print(f"  L{li:2d}: {delta[li]:+.3f}  (v2={v2_dla[li]:.3f}, reset={r_dla[li]:.3f})")
    shallow = np.abs(delta[:10]).sum()
    deep = np.abs(delta[10:]).sum()
    print(f"\n浅层变化(0-9)：{shallow:.3f}  深层变化(10-27)：{deep:.3f}")
    if shallow > deep:
        print("→ 浅层DLA变化更大，重置直接影响浅层贡献方向")
    else:
        print("→ 深层DLA变化更大，浅层重置通过残差流级联影响深层")
    plot(v2_dla, r_dla, CONFIG["output_dir"])
    with open(os.path.join(CONFIG["output_dir"], "dla_reset_results.json"), "w") as f:
        json.dump({"v2_acc": v2_acc, "reset_acc": r_acc,
                   "v2_dla": v2_dla.tolist(), "reset_dla": r_dla.tolist(),
                   "delta": delta.tolist(), "shallow": float(shallow), "deep": float(deep)}, f, indent=2)
    print("✅ 完成")

if __name__ == "__main__":
    main()