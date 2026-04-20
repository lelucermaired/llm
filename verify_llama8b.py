import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base = "meta-llama/Llama-3.1-8B-Instruct"
adapter = "./checkpoints/llama8b-gomoku-maxlora/final_model"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

with open("datasets/real_games_v2/train.json", "r") as f:
    samples = json.load(f)[:10]

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map={"":0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

exact = 0
for i, s in enumerate(samples):
    msgs = [{"role": "user", "content": s["instruction"]}]
    p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    gold = s["output"].strip()
    if resp[:30] == gold[:30]:
        exact += 1
    print(f"[{i}] gold: {gold[:80]}\n     pred: {resp[:80]}\n")
print(f"exact={exact}/10")