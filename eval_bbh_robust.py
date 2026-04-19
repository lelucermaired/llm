import torch, gc, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_name = 'Qwen/Qwen2.5-7B-Instruct'
adapter = './checkpoints/qwen-gomoku-maxlora/final_model'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

# 定义任务列表及对应的答案类型
tasks = {
    'logical_deduction_three_objects': 'choice',
    'logical_deduction_five_objects': 'choice',
    'logical_deduction_seven_objects': 'choice',
    'temporal_sequences': 'choice',
    'geometric_shapes': 'choice',
    'reasoning_about_colored_objects': 'choice',
    'multistep_arithmetic_two': 'number',
    'object_counting': 'number'
}

def extract_answer(resp, task_type):
    resp = resp.strip()
    if task_type == 'choice':
        # 匹配 (A) 或 (B) 等
        m = re.search(r'\(([A-E])\)', resp)
        if m:
            return '(' + m.group(1) + ')'
        m = re.search(r'\b([A-E])\b', resp)
        if m:
            return '(' + m.group(1) + ')'
        return None
    elif task_type == 'number':
        # 提取最后一个数字（支持负数）
        numbers = re.findall(r'-?\d+', resp)
        if numbers:
            return numbers[-1]
        return None
    else:
        return None

def evaluate(model, tokenizer, task_name, task_type, split='test', limit=100, verbose=False):
    ds = load_dataset('lukaemon/bbh', task_name, split=split, cache_dir='/root/autodl-tmp/hf_cache')
    correct = 0
    total = min(limit, len(ds))
    for i in range(total):
        item = ds[i]
        prompt = item['input'] + '\nAnswer:'
        msgs = [{'role': 'user', 'content': prompt}]
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(p, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        gold = item['target'].strip()
        pred = extract_answer(resp, task_type)
        if verbose and i < 5:
            print(f"  [{i}] gold={gold}  pred={pred}  resp_preview={resp[:80]}")
        if pred is not None and pred == gold:
            correct += 1
    return correct, total

tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

print("=== BASE ===")
model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb, device_map={"":0}, trust_remote_code=True)
model.eval()
base_results = {}
for task_name, task_type in tasks.items():
    print(f"\nEvaluating {task_name}...")
    correct, total = evaluate(model, tokenizer, task_name, task_type, verbose=True)
    acc = correct / total
    base_results[task_name] = acc
    print(f"  {task_name:40s}: {correct:3d}/{total:3d} = {acc:.1%}")
del model; gc.collect(); torch.cuda.empty_cache()

print("\n=== MAXLORA (r=64) ===")
model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb, device_map={"":0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model.eval()
lora_results = {}
for task_name, task_type in tasks.items():
    print(f"\nEvaluating {task_name}...")
    correct, total = evaluate(model, tokenizer, task_name, task_type, verbose=True)
    acc = correct / total
    lora_results[task_name] = acc
    print(f"  {task_name:40s}: {correct:3d}/{total:3d} = {acc:.1%}")
del model; gc.collect(); torch.cuda.empty_cache()

print("\n=== SUMMARY ===")
print(f"{'Task':45s} {'Base':>8} {'LoRA':>8} {'Delta':>8}")
for task_name in tasks:
    b = base_results[task_name]
    l = lora_results[task_name]
    d = l - b
    print(f"{task_name:45s} {b:8.1%} {l:8.1%} {d:+8.1%}")