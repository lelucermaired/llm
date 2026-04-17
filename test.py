# debug_move.py
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "./checkpoints/qwen-gomoku-cot-detailed/final_model"

with open('./datasets/real_games_v2/train.json', encoding='utf-8') as f:
    data = json.load(f)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                            device_map="auto", local_files_only=True, trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True)

# 只看前5个样本的原始输出
for i in range(5):
    prompt = data[i]['instruction']
    optimal = re.search(r"最佳落子[：:]\s*([A-O]\d{1,2})", data[i]['output'])
    optimal_move = optimal.group(1) if optimal else "未知"

    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print(f"\n=== 样本{i + 1} ===")
    print(f"引擎最优解: {optimal_move}")
    print(f"模型输出:\n{resp}")
    print(f"---")