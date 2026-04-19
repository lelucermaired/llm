"""
eval_strategyqa.py
StrategyQA评测: 多步推理Yes/No问答
对所有模型跑 base vs maxlora
"""
import torch, gc, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

# 加载数据
print("Loading StrategyQA...")
ds = load_dataset('ChilleD/StrategyQA', cache_dir='/root/autodl-tmp/hf_cache')
# 取test或train的前100条
if 'test' in ds:
    samples = list(ds['test'])[:100]
elif 'validation' in ds:
    samples = list(ds['validation'])[:100]
else:
    samples = list(ds['train'])[:100]

# 查看数据格式
print(f"Loaded {len(samples)} samples")
print(f"Sample keys: {list(samples[0].keys())}")
print(f"Sample: {samples[0]}")

def extract_yesno(resp):
    resp_lower = resp.lower().strip()[:200]
    # 先找明确的yes/no
    if resp_lower.startswith('yes'):
        return True
    if resp_lower.startswith('no'):
        return False
    if 'the answer is yes' in resp_lower:
        return True
    if 'the answer is no' in resp_lower:
        return False
    if 'yes' in resp_lower and 'no' not in resp_lower:
        return True
    if 'no' in resp_lower and 'yes' not in resp_lower:
        return False
    return None

def eval_model(model, tokenizer, tag):
    correct = 0
    invalid = 0
    total = len(samples)
    for i, item in enumerate(samples):
        # StrategyQA格式可能不同，适配
        if 'question' in item:
            question = item['question']
        elif 'input' in item:
            question = item['input']
        else:
            question = str(item)

        if 'answer' in item:
            gold = item['answer']
        elif 'target' in item:
            gold = item['target']
        else:
            gold = None

        # gold可能是bool或str
        if isinstance(gold, bool):
            gold_bool = gold
        elif isinstance(gold, str):
            gold_bool = gold.lower().strip() in ['yes', 'true', '1']
        else:
            continue

        prompt = f"{question}\nAnswer Yes or No:"
        msgs = [{'role': 'user', 'content': prompt}]
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(p, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=100, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_yesno(resp)

        if pred is None:
            invalid += 1
        elif pred == gold_bool:
            correct += 1

    valid = total - invalid
    acc = correct / valid if valid > 0 else 0
    print(f"  {tag}: {correct}/{valid} = {acc:.1%} (invalid: {invalid})")

# 模型列表
models = [
    ("Qwen-7B", "Qwen/Qwen2.5-7B-Instruct", "./checkpoints/qwen-gomoku-maxlora/final_model"),
    ("Qwen-3B", "/root/autodl-tmp/models/qwen25-3b", "./checkpoints/qwen3b-gomoku-maxlora/final_model"),
    ("Yi-9B", "/root/autodl-tmp/models/yi15-9b", "./checkpoints/yi9b-gomoku-maxlora/final_model"),
]

for model_name, base_path, adapter_path in models:
    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base
    model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb,
        device_map={'':0}, trust_remote_code=True)
    model.eval()
    eval_model(model, tokenizer, f"{model_name} base")
    del model; gc.collect(); torch.cuda.empty_cache()

    # Maxlora
    import os
    if os.path.exists(adapter_path):
        model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb,
            device_map={'':0}, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        eval_model(model, tokenizer, f"{model_name} maxlora")
        del model; gc.collect(); torch.cuda.empty_cache()
    else:
        print(f"  [skip] adapter not found: {adapter_path}")

print("\nDone.")