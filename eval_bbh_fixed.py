import torch
import gc
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==================== 配置 ====================
base_model_name = 'Qwen/Qwen2.5-7B-Instruct'
adapter_path = './checkpoints/qwen-gomoku-maxlora/final_model'

# 需要评测的任务列表（可自由增删）
tasks = [
    'logical_deduction_five_objects',
    'object_counting'
]

# 每个任务最多评测的样本数（-1 表示全部）
limit_per_task = 100


# ==================== 答案提取函数 ====================
def extract_answer(resp, gold):
    """
    从模型输出中提取答案，支持：
    - Yes/No
    - 括号选项 (A)、(B) 等
    - 独立字母 A、B、C 等（转为 (A) 格式）
    - 数字（整数，支持负数）
    """
    resp = resp.strip()

    # 1. Yes/No 类型
    if gold in ['Yes', 'No']:
        if 'yes' in resp.lower()[:200]:
            return 'Yes'
        if 'no' in resp.lower()[:200]:
            return 'No'
        return resp[:10]

    # 2. 括号选项 (A) (B) ...
    matches = re.findall(r'\(([A-E])\)', resp)
    if matches:
        return '(' + matches[-1] + ')'

    # 3. 独立大写字母（转为括号格式）
    matches = re.findall(r'\b([A-E])\b', resp)
    if matches:
        return '(' + matches[-1] + ')'

    # 4. 数字（整数，支持负数）
    if gold.lstrip('-').isdigit():
        nums = re.findall(r'-?\d+', resp)
        if nums:
            return nums[-1]

    # 失败时返回原始响应的前20个字符
    return resp[:20]


# ==================== 评测函数 ====================
def evaluate_model(model, tokenizer, task_name, limit=100):
    ds = load_dataset('lukaemon/bbh', task_name, split='test', cache_dir='/root/autodl-tmp/hf_cache')
    correct = 0
    total = min(limit, len(ds)) if limit > 0 else len(ds)
    for i in range(total):
        item = ds[i]
        prompt = item['input'] + '\nAnswer:'
        msgs = [{'role': 'user', 'content': prompt}]
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(p, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        gold = item['target'].strip()
        pred = extract_answer(resp, gold)
        if pred == gold:
            correct += 1
    return correct, total


# ==================== 主程序 ====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# ----- Base 模型 -----
print("=== BASE ===")
model_base = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)
model_base.eval()

base_results = {}
for task in tasks:
    correct, total = evaluate_model(model_base, tokenizer, task, limit_per_task)
    acc = correct / total
    base_results[task] = acc
    print(f"{task:45s}: {correct:3d}/{total:3d} = {acc:.1%}")

del model_base
gc.collect()
torch.cuda.empty_cache()

# ----- MaxLoRA 模型 -----
print("\n=== MAXLORA (r=64) ===")
model_lora = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)
model_lora = PeftModel.from_pretrained(model_lora, adapter_path)
model_lora.eval()

lora_results = {}
for task in tasks:
    correct, total = evaluate_model(model_lora, tokenizer, task, limit_per_task)
    acc = correct / total
    lora_results[task] = acc
    print(f"{task:45s}: {correct:3d}/{total:3d} = {acc:.1%}")

del model_lora
gc.collect()
torch.cuda.empty_cache()

# ----- 汇总 -----
print("\n=== SUMMARY ===")
print(f"{'Task':45s} {'Base':>8} {'LoRA':>8} {'Delta':>8}")
for task in tasks:
    b = base_results[task]
    l = lora_results[task]
    d = l - b
    print(f"{task:45s} {b:8.1%} {l:8.1%} {d:+8.1%}")