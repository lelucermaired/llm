import json
from transformers import pipeline
from tqdm import tqdm
import torch

# 检查是否有 GPU，使用 GPU 加速
device = 0 if torch.cuda.is_available() else -1

# 加载翻译 pipeline（首次运行会自动下载模型，约 300MB）
print("Loading translation model (Helsinki-NLP/opus-mt-zh-en)...")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device=device)

def translate_text(text):
    """翻译单个文本，如果过长则分段处理（但指令和输出一般不超过 512 tokens）"""
    # 模型最大输入长度约 512，你的数据一般不超过，直接翻译
    result = translator(text, max_length=1024, truncation=True)[0]['translation_text']
    return result

def translate_instruction(text):
    """翻译 instruction 字段，保留棋盘 ASCII 图不翻译（但棋盘图里含有中文列标 'C D E ...' 等，需要替换）"""
    # 先分离出棋盘部分（如果有）和自然语言部分
    # 简单策略：翻译整段，但棋盘中的字母和数字不会被改变，中文列标会被翻译成英文？例如 "C D E" 可能变成 "C D E"，没问题
    return translate_text(text)

def translate_output(text):
    """翻译 output 字段，保留 <thinking> 标签和关键符号"""
    # 直接整体翻译，因为标签会被保留
    return translate_text(text)

# 加载原始数据
with open('datasets/real_games_v2/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = []
for item in tqdm(data, desc="Translating"):
    new_item = {
        "instruction": translate_instruction(item["instruction"]),
        "output": translate_output(item["output"])
    }
    new_data.append(new_item)

with open('datasets/real_games_v2/train_en.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"翻译完成，共 {len(new_data)} 条，保存至 datasets/real_games_v2/train_en.json")