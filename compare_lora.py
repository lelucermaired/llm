import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "lora_gomoku_cpu"   # 你刚才保存的目录
max_new_tokens=128

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype = torch.float16
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

prompt = """你是一个五子棋分析助手。

当前棋盘如下（黑子●，白子○，空位·）：

  A B C D E
1 · · · · ·
2 · ● ● · ·
3 · ○ ○ · ·
4 · · · · ·
5 · · · · ·

现在轮到黑棋（●）。

问题：
最优落子位置是什么？请简要说明理由。

"""

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

print("\n=== LoRA Model Output ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
