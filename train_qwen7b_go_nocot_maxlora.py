import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0+PTX;8.6+PTX;8.9+PTX"

BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
DATA_PATH = "datasets/go_nocot_train.json"
OUTPUT_DIR = "./checkpoints/qwen7b_go_nocot_maxlora"
CACHE_DIR = "./cache"
MAX_LENGTH = 1024

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def main():
    print("=" * 70)
    print("Qwen2.5-7B GO_NOCOT MaxLoRA 训练")
    print(f"base_model = {BASE_MODEL}")
    print(f"data_path  = {DATA_PATH}")
    print(f"output_dir = {OUTPUT_DIR}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print("Loading dataset...")
    ds = load_dataset("json", data_files=DATA_PATH)

    sample = ds["train"][0]
    print("Sample fields:", sample.keys())
    print("Sample output:", sample["output"][:80])

    def tokenize(examples):
        texts = []
        for inst, out in zip(examples["instruction"], examples["output"]):
            msgs = [
                {"role": "user", "content": inst},
                {"role": "assistant", "content": out},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                text = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
            texts.append(text)

        tok = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing dataset",
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        fp16=False,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Training...")
    trainer.train()

    final_dir = f"{OUTPUT_DIR}/final_model"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(f"{OUTPUT_DIR}/train_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": BASE_MODEL,
                "data_path": DATA_PATH,
                "output_dir": OUTPUT_DIR,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": TARGET_MODULES,
                "epochs": 3,
                "lr": 2e-4,
                "grad_accum": 8,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("DONE")
    print(f"saved to: {final_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
