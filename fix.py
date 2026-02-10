import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "lora_gomoku_cpu"

print("=== 测试LoRA模型加载 ===")

# 1. 先加载基础模型
print("1. 加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True
)

# 2. 查看LoRA文件夹内容
print(f"\n2. 检查LoRA路径: {LORA_PATH}")
import os

if os.path.exists(LORA_PATH):
    files = os.listdir(LORA_PATH)
    print(f"LoRA文件夹中的文件: {files}")
else:
    print("❌ LoRA路径不存在！")
    exit()

# 3. 尝试加载LoRA
print("\n3. 尝试加载LoRA适配器...")
try:
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print("✅ LoRA加载成功！")

    # 测试一个简单的问题
    print("\n4. 测试模型推理...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 合并权重（为了简单推理）
    print("合并LoRA权重...")
    model = model.merge_and_unload()

    # 简单测试
    prompt = "五子棋的基本规则是什么？"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=1.0
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"问题: {prompt}")
    print(f"回答: {response}")

except Exception as e:
    print(f"❌ LoRA加载失败: {e}")
    print("\n可能的解决方案：")
    print("1. 检查LoRA路径是否正确")
    print("2. 确保LoRA模型与基座模型兼容")
    print("3. 尝试重新训练或下载LoRA模型")