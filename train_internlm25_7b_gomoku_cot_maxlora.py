import argparse
import json
import os
import sys
from datetime import datetime

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0+PTX;8.6+PTX;8.9+PTX"

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


DEFAULT_CONFIG = {
    "base_model": "/root/autodl-tmp/models/internlm2_5-7b-chat",
    "dataset_path": "/root/autodl-tmp/llm-project/datasets/real_games_v2/train.json",
    "output_dir": "./checkpoints/internlm25-7b-gomoku-cot-maxlora",
    "cache_dir": "./cache",
    "max_length": 1024,
    "seed": 42,
    "lora": {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": [
            "wqkv",
            "wo",
            "w1",
            "w2",
            "w3",
        ],
    },
    "train": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "warmup_steps": 50,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "bf16": True,
        "fp16": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train InternLM2.5-7B-Chat on Gomoku CoT with QLoRA MaxLoRA config.")
    parser.add_argument("--base_model", default=DEFAULT_CONFIG["base_model"])
    parser.add_argument("--dataset", default=DEFAULT_CONFIG["dataset_path"])
    parser.add_argument("--output", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--cache_dir", default=DEFAULT_CONFIG["cache_dir"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--epochs", type=float, default=DEFAULT_CONFIG["train"]["num_train_epochs"])
    return parser.parse_args()


def setup_environment(args):
    print("=" * 72)
    print("InternLM2.5-7B Gomoku CoT MaxLoRA training")
    print(f"Start:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {args.base_model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Output:     {args.output}")
    print("=" * 72)

    set_seed(DEFAULT_CONFIG["seed"])
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        print("Warning: CUDA is not available. Training will be very slow.")

    config_to_save = {
        "base_model": args.base_model,
        "dataset_path": args.dataset,
        "output_dir": args.output,
        "cache_dir": args.cache_dir,
        "max_length": args.max_length,
        "seed": DEFAULT_CONFIG["seed"],
        "lora": DEFAULT_CONFIG["lora"],
        "train": {**DEFAULT_CONFIG["train"], "num_train_epochs": args.epochs},
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)


def load_data(dataset_path, cache_dir):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_dataset("json", data_files=dataset_path, cache_dir=cache_dir)
    sample = dataset["train"][0]
    if "instruction" not in sample or "output" not in sample:
        raise ValueError(f"Dataset must contain instruction/output fields, got: {list(sample.keys())}")

    print(f"Loaded {len(dataset['train'])} samples.")
    print(f"Sample instruction: {sample['instruction'][:80]}")
    print(f"Sample output:      {sample['output'][:80]}")
    return dataset


def build_model_and_tokenizer(base_model, cache_dir):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=DEFAULT_CONFIG["lora"]["r"],
        lora_alpha=DEFAULT_CONFIG["lora"]["lora_alpha"],
        lora_dropout=DEFAULT_CONFIG["lora"]["lora_dropout"],
        bias=DEFAULT_CONFIG["lora"]["bias"],
        target_modules=DEFAULT_CONFIG["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def format_chat(tokenizer, instruction, output):
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"


def preprocess_dataset(dataset, tokenizer, max_length):
    def tokenize_batch(examples):
        texts = [
            format_chat(tokenizer, inst, out)
            for inst, out in zip(examples["instruction"], examples["output"])
        ]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    print(f"Tokenized samples: {len(tokenized['train'])}")
    return tokenized


def train_model(model, tokenizer, dataset, output_dir, epochs):
    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=DEFAULT_CONFIG["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=DEFAULT_CONFIG["train"]["gradient_accumulation_steps"],
        learning_rate=DEFAULT_CONFIG["train"]["learning_rate"],
        warmup_steps=DEFAULT_CONFIG["train"]["warmup_steps"],
        max_grad_norm=DEFAULT_CONFIG["train"]["max_grad_norm"],
        optim=DEFAULT_CONFIG["train"]["optim"],
        lr_scheduler_type=DEFAULT_CONFIG["train"]["lr_scheduler_type"],
        weight_decay=DEFAULT_CONFIG["train"]["weight_decay"],
        logging_strategy="steps",
        logging_steps=DEFAULT_CONFIG["train"]["logging_steps"],
        save_strategy=DEFAULT_CONFIG["train"]["save_strategy"],
        save_total_limit=DEFAULT_CONFIG["train"]["save_total_limit"],
        gradient_checkpointing=DEFAULT_CONFIG["train"]["gradient_checkpointing"],
        bf16=DEFAULT_CONFIG["train"]["bf16"] and supports_bf16,
        fp16=DEFAULT_CONFIG["train"]["fp16"] and not supports_bf16,
        report_to="none",
        seed=DEFAULT_CONFIG["seed"],
        data_seed=DEFAULT_CONFIG["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        ),
    )

    print("Training...")
    result = trainer.train()
    print(f"Training finished. Loss: {result.training_loss:.4f}")

    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, ensure_ascii=False, indent=2)

    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final adapter to: {final_dir}")


def main():
    args = parse_args()
    try:
        setup_environment(args)
        dataset = load_data(args.dataset, args.cache_dir)
        model, tokenizer = build_model_and_tokenizer(args.base_model, args.cache_dir)
        tokenized = preprocess_dataset(dataset, tokenizer, args.max_length)
        train_model(model, tokenizer, tokenized, args.output, args.epochs)
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
