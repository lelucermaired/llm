"""
eval_yi_all.py
Yi-1.5-9B: 验证五子棋 + BBH评测，一次跑完
"""
import json, torch, gc, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_name = '/root/autodl-tmp/models/yi15-9b'
adapter = './checkpoints/yi9b-gomoku-maxlora/final_model'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== 1. 验证五子棋 =====
print("=" * 60)
print("STEP 1: 验证五子棋学习质量")
print("=" * 60)

with open('datasets/real_games_v2/train.json', 'r') as f:
    samples = json.load(f)[:5]

model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb,
    device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

exact = 0
for i, s in enumerate(samples):
    msgs = [{'role':'user','content':s['instruction']}]
    p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(p, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    gold = s['output'].strip()
    if resp[:30] == gold[:30]:
        exact += 1
    print(f'[{i}] gold: {gold[:80]}')
    print(f'     pred: {resp[:80]}')
    print()
print(f'五子棋复现: exact={exact}/5')
del model; gc.collect(); torch.cuda.empty_cache()

# ===== 2. BBH评测 =====
print()
print("=" * 60)
print("STEP 2: BBH评测")
print("=" * 60)

tasks = ['navigate', 'tracking_shuffled_objects_three_objects', 'logical_deduction_five_objects']
task_data = {}
for t in tasks:
    ds = load_dataset('lukaemon/bbh', t, cache_dir='/root/autodl-tmp/hf_cache')
    task_data[t] = list(ds['test'])[:100]

def extract_answer(resp, gold):
    resp = resp.strip()
    if gold in ['Yes', 'No']:
        if 'yes' in resp.lower()[:200]:
            return 'Yes'
        if 'no' in resp.lower()[:200]:
            return 'No'
        return resp[:10]
    matches = re.findall(r'\(([A-E])\)', resp)
    if matches:
        return '(' + matches[-1] + ')'
    matches = re.findall(r'\b([A-E])\b', resp)
    if matches:
        return '(' + matches[-1] + ')'
    return resp[:10]

def eval_bbh(model, tag):
    print(f'\n=== {tag} ===')
    for t in tasks:
        correct = 0
        total = len(task_data[t])
        for item in task_data[t]:
            prompt = item['input'] + '\nAnswer:'
            msgs = [{'role':'user','content':prompt}]
            p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(p, return_tensors='pt').to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            resp = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            gold = item['target'].strip()
            pred = extract_answer(resp, gold)
            if pred == gold:
                correct += 1
        short = t[:25]
        print(f'  {short:25s}: {correct}/{total} = {correct/total:.1%}')

# Base
model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb,
    device_map={'':0}, trust_remote_code=True)
model.eval()
eval_bbh(model, 'Yi-1.5-9B BASE')
del model; gc.collect(); torch.cuda.empty_cache()

# Maxlora
model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb,
    device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model.eval()
eval_bbh(model, 'Yi-1.5-9B MAXLORA')
del model; gc.collect(); torch.cuda.empty_cache()

print('\nDone.')
