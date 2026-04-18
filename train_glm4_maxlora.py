"""
train_glm4_maxlora.py
GLM-4-9B-Chat 五子棋 maxlora 训练
"""
import os, json, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

base = '/root/autodl-tmp/models/glm4-9b'
output_dir = './checkpoints/glm4-gomoku-maxlora'

# GLM-4 config兼容性修复
config = AutoConfig.from_pretrained(base, trust_remote_code=True)
if not hasattr(config, 'max_length') or config.max_length is None:
    config.max_length = getattr(config, 'seq_length', 131072)

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    base, config=config, quantization_config=bnb, device_map={'':0},
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

model = prepare_model_for_kbit_training(model)

# 查看模型结构找到正确的module名
print("Checking model structure...")
target_modules = []
for name, _ in model.named_modules():
    for key in ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        if name.endswith(key) and key not in target_modules:
            target_modules.append(key)
if not target_modules:
    target_modules = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']
print(f"Target modules: {target_modules}")

lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=target_modules,
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

ds = load_dataset('json', data_files='datasets/real_games_v2/train.json')

def tokenize(ex):
    texts = []
    for i in range(len(ex['instruction'])):
        msgs = [{'role': 'user', 'content': ex['instruction'][i]},
                {'role': 'assistant', 'content': ex['output'][i]}]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = f"<|user|>\n{ex['instruction'][i]}<|assistant|>\n{ex['output'][i]}"
        texts.append(text)
    tok = tokenizer(texts, truncation=True, padding=False, max_length=1024)
    tok['labels'] = tok['input_ids'].copy()
    return tok

tokenized = ds.map(tokenize, batched=True, remove_columns=ds['train'].column_names)

args = TrainingArguments(
    output_dir=output_dir, num_train_epochs=3,
    per_device_train_batch_size=1, gradient_accumulation_steps=8,
    learning_rate=2e-4, warmup_steps=50, lr_scheduler_type='cosine',
    logging_steps=10, save_strategy='no',
    bf16=True, report_to='none',
    gradient_checkpointing=True, dataloader_num_workers=2,
)
trainer = Trainer(
    model=model, args=args, train_dataset=tokenized['train'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)
trainer.train()
model.save_pretrained(f'{output_dir}/final_model')
tokenizer.save_pretrained(f'{output_dir}/final_model')
print('DONE')