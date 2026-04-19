import torch, gc, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_name = 'Qwen/Qwen2.5-7B-Instruct'
adapter = './checkpoints/qwen-gomoku-maxlora/final_model'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

tasks = [
    'logical_deduction_three_objects',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'temporal_sequences',
    'geometric_shapes',
    'reasoning_about_colored_objects',
    'multistep_arithmetic_two',
    'object_counting'
]

def extract_answer(resp, gold):
    resp = resp.strip()
    # 提取括号中的选项
    m = re.search(r'\(([A-E])\)', resp)
    if m:
        return '(' + m.group(1) + ')'
    # 提取独立字母
    m = re.search(r'\b([A-E])\b', resp)
    if m:
        return '(' + m.group(1) + ')'
    # 对于多步算术，提取数字
    if gold.replace('.','').isdigit():
        nums = re.findall(r'-?\d+', resp)
        if nums:
            return nums[-1]
    return resp[:20]

def evaluate(model, tokenizer, task_name, split='test', limit=100):
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
        pred = extract_answer(resp, gold)
        if pred == gold:
            correct += 1
    return correct, total

tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

print("=== BASE ===")
model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb, device_map={"":0}, trust_remote_code=True)
model.eval()
base_results = {}
for t in tasks:
    correct, total = evaluate(model, tokenizer, t)
    base_results[t] = correct / total
    print(f"{t:40s}: {correct:3d}/{total:3d} = {correct/total:.1%}")
del model; gc.collect(); torch.cuda.empty_cache()

print("\n=== MAXLORA (r=64) ===")
model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb, device_map={"":0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model.eval()
lora_results = {}
for t in tasks:
    correct, total = evaluate(model, tokenizer, t)
    lora_results[t] = correct / total
    print(f"{t:40s}: {correct:3d}/{total:3d} = {correct/total:.1%}")
del model; gc.collect(); torch.cuda.empty_cache()

print("\n=== SUMMARY ===")
print(f"{'Task':40s} {'Base':>8} {'LoRA':>8} {'Delta':>8}")
for t in tasks:
    b = base_results[t]
    l = lora_results[t]
    d = l - b
    print(f"{t:40s} {b:8.1%} {l:8.1%} {d:+8.1%}")