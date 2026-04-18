import torch, gc, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_name = 'Qwen/Qwen2.5-7B-Instruct'
adapter = './checkpoints/qwen-gomoku-maxlora/final_model'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

tasks = ['navigate', 'tracking_shuffled_objects_three_objects', 'logical_deduction_five_objects']
task_data = {}
for t in tasks:
    ds = load_dataset('lukaemon/bbh', t, cache_dir='/root/autodl-tmp/hf_cache')
    task_data[t] = list(ds['test'])[:100]

tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

def extract_answer(resp, gold):
    resp = resp.strip()
    # Yes/No
    if gold in ['Yes', 'No']:
        if 'yes' in resp.lower()[:200]:
            return 'Yes'
        if 'no' in resp.lower()[:200]:
            return 'No'
        return resp[:10]
    # (A)/(B)/(C) - find last option mentioned
    matches = re.findall(r'\(([A-E])\)', resp)
    if matches:
        return '(' + matches[-1] + ')'
    matches = re.findall(r'\b([A-E])\b', resp)
    if matches:
        return '(' + matches[-1] + ')'
    return resp[:10]

def eval_model(model, tag):
    print(f'=== {tag} ===')
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
model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb, device_map={'':0}, trust_remote_code=True)
model.eval()
eval_model(model, 'BASE')
del model; gc.collect(); torch.cuda.empty_cache()

# Maxlora
model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb, device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model.eval()
eval_model(model, 'MAXLORA (r=64 gomoku)')
del model; gc.collect(); torch.cuda.empty_cache()

print('\nDone.')