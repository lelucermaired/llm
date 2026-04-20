import torch, gc, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
adapter_path = "./checkpoints/llama8b-gomoku-maxlora/final_model"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

tasks = ['logical_deduction_five_objects', 'object_counting']
limit_per_task = 100

def extract_answer(resp, gold):
    resp = resp.strip()
    if gold in ['Yes', 'No']:
        if 'yes' in resp.lower()[:200]: return 'Yes'
        if 'no' in resp.lower()[:200]: return 'No'
        return resp[:10]
    matches = re.findall(r'\(([A-E])\)', resp)
    if matches: return '(' + matches[-1] + ')'
    matches = re.findall(r'\b([A-E])\b', resp)
    if matches: return '(' + matches[-1] + ')'
    if gold.lstrip('-').isdigit():
        nums = re.findall(r'-?\d+', resp)
        if nums: return nums[-1]
    return resp[:20]

def evaluate(model, tokenizer, task_name, limit=100):
    ds = load_dataset('lukaemon/bbh', task_name, split='test', cache_dir='/root/autodl-tmp/hf_cache')
    correct = 0
    total = min(limit, len(ds))
    for i in range(total):
        item = ds[i]
        prompt = item['input'] + '\nAnswer:'
        msgs = [{'role':'user','content':prompt}]
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(p, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        gold = item['target'].strip()
        pred = extract_answer(resp, gold)
        if pred == gold: correct += 1
    return correct, total

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

print("=== BASE ===")
model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb, device_map={"":0}, trust_remote_code=True)
model.eval()
base_res = {}
for t in tasks:
    c, total = evaluate(model, tokenizer, t, limit_per_task)
    base_res[t] = c / total
    print(f"{t:45s}: {c:3d}/{total:3d} = {c/total:.1%}")
del model; gc.collect(); torch.cuda.empty_cache()

print("\n=== MAXLORA (r=64) ===")
model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb, device_map={"":0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
lora_res = {}
for t in tasks:
    c, total = evaluate(model, tokenizer, t, limit_per_task)
    lora_res[t] = c / total
    print(f"{t:45s}: {c:3d}/{total:3d} = {c/total:.1%}")
del model; gc.collect(); torch.cuda.empty_cache()

print("\n=== SUMMARY ===")
print(f"{'Task':45s} {'Base':>8} {'LoRA':>8} {'Delta':>8}")
for t in tasks:
    b = base_res[t]
    l = lora_res[t]
    d = l - b
    print(f"{t:45s} {b:8.1%} {l:8.1%} {d:+8.1%}")