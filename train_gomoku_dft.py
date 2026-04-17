"""
train_gomoku_dft.py

Dynamic Fine-Tuning（DFT）版五子棋训练
严格按照 Wu et al., ICLR 2026 (arXiv:2508.05629) 公式(9)实现

DFT loss（token级别）：
  L_DFT = -sum_t [ sg(P(y_t | y<t, x)) * log P(y_t | y<t, x) ]

其中 sg() 是stop-gradient，P(y_t)在前向传播中计算但不参与梯度。
等价于 -P * log P，即用token概率本身作为权重缩放CE loss。

与标准CE的区别（Appendix A.3）：
  CE梯度：  -1/P(y) * dP/dθ  （低概率token梯度被放大）
  DFT梯度：       -1 * dP/dθ  （均匀缩放，避免低概率token梯度爆炸）
"""

import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0+PTX;8.6+PTX;8.9+PTX'
import sys, json
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
import torch.nn.functional as F

_orig = torch.load
def _patched(*a, **kw): kw['weights_only'] = False; return _orig(*a, **kw)
torch.load = _patched

import warnings; warnings.filterwarnings("ignore")

CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "local_model_path": None,
    "dataset_path": "./datasets/real_games_v2/train.json",
    "output_dir": "./checkpoints/qwen-gomoku-dft",
    "cache_dir": "./cache",

    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
    },
    "lora": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj"],
    },
    "training": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 3e-6,
        "warmup_steps": 100,
        "max_grad_norm": 2.0,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": 0.1,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "seed": 42,
        "gradient_checkpointing": True,
        "fp16": False,
        "bf16": True,
    },
    "generation": {
        "max_length": 1024,
    }
}


class DFTTrainer(Trainer):
    """
    严格按照公式(9)实现DFT loss：
    L_DFT = -sum_t [ sg(P(y_t)) * log P(y_t) ]
           = sum_t [ sg(P(y_t)) * CE(y_t) ]

    注意：
    1. sg()用.detach()实现，权重不参与反向传播
    2. token级别加权（论文Table 5验证token级优于sentence级）
    3. loss数值约为CE的0.x倍（因为P<1，权重会压缩loss）
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, L, V)

        # shift：位置i预测位置i+1
        shift_logits = logits[..., :-1, :].contiguous()   # (B, L-1, V)
        shift_labels = labels[..., 1:].contiguous()        # (B, L-1)

        B, L, V = shift_logits.shape

        # 计算每个token的概率 P(y_t)
        probs = F.softmax(shift_logits.float(), dim=-1)    # (B, L-1, V) float32

        # 取label位置的概率
        labels_for_gather = shift_labels.clone()
        labels_for_gather[labels_for_gather == -100] = 0
        label_probs = probs.gather(
            dim=-1, index=labels_for_gather.unsqueeze(-1)
        ).squeeze(-1)                                       # (B, L-1)

        # DFT权重 = sg(P(y_t))，stop-gradient，不参与反向传播
        dft_weights = label_probs.detach()                  # (B, L-1)

        # 标准CE loss（per token，reduction=none）
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, L)                                        # (B, L-1)

        # DFT loss = P(y_t) * CE(y_t)，仅对有效token计算
        mask = (shift_labels != -100).float()
        dft_loss = (ce_loss * dft_weights * mask).sum() / mask.sum().clamp(min=1)

        return (dft_loss, outputs) if return_outputs else dft_loss


def setup():
    print("=" * 65)
    print("五子棋 Dynamic Fine-Tuning（DFT）")
    print("严格按照 Wu et al., ICLR 2026 公式(9)")
    print("L_DFT = -sum_t [ sg(P(y_t)) * log P(y_t) ]")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print("\nDFT vs CE：")
    print("  CE梯度  = -1/P(y) * dP/dθ  ← 低概率token梯度被放大，不稳定")
    print("  DFT梯度 = -1 * dP/dθ       ← 均匀缩放，等效于均一奖励")
    print("\n预期loss数值：CE约2.1，DFT约0.x（P<1故权重压缩loss）")
    set_seed(CONFIG["training"]["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)} | "
              f"显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")


def load_data():
    print("\n加载数据集...")
    ds = load_dataset("json", data_files=CONFIG["dataset_path"],
                      cache_dir=CONFIG["cache_dir"])
    print(f"✅ 样本数: {len(ds['train'])}")
    return ds


def build_model():
    print("\n加载模型，应用QLoRA + DFT...")
    model_name = CONFIG["local_model_path"] or CONFIG["model_name"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=CONFIG["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=CONFIG["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=CONFIG["quantization"]["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=CONFIG["quantization"]["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, cache_dir=CONFIG["cache_dir"], use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=CONFIG["cache_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora"]["r"],
        lora_alpha=CONFIG["lora"]["lora_alpha"],
        lora_dropout=CONFIG["lora"]["lora_dropout"],
        bias=CONFIG["lora"]["bias"],
        target_modules=CONFIG["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def preprocess(dataset, tokenizer):
    def tokenize(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            text = (f"<|im_start|>user\n{examples['instruction'][i]}<|im_end|>\n"
                    f"<|im_start|>assistant\n{examples['output'][i]}<|im_end|>")
            texts.append(text)
        tok = tokenizer(texts, truncation=True, padding=False,
                        max_length=CONFIG["generation"]["max_length"])
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = dataset.map(tokenize, batched=True,
                            remove_columns=dataset["train"].column_names,
                            desc="Tokenizing")
    print(f"✅ 预处理完成，样本数: {len(tokenized['train'])}")
    return tokenized


def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    weight_files = ["adapter_model.safetensors", "adapter_model.bin"]
    has_cfg = os.path.exists(os.path.join(output_dir, "adapter_config.json"))
    has_w = any(os.path.exists(os.path.join(output_dir, f)) and
                os.path.getsize(os.path.join(output_dir, f)) > 1024*1024
                for f in weight_files)
    if not has_cfg or not has_w:
        raise RuntimeError("模型保存验证失败")
    print(f"✅ 模型保存至: {output_dir}")


def train(model, tokenizer, dataset):
    print("\n开始DFT训练（1 epoch）...")
    gpu_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
    use_bf16 = CONFIG["training"]["bf16"] and gpu_cap[0] >= 8

    args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["training"]["num_train_epochs"],
        per_device_train_batch_size=CONFIG["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["training"]["gradient_accumulation_steps"],
        learning_rate=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
        warmup_steps=CONFIG["training"]["warmup_steps"],
        max_grad_norm=CONFIG["training"]["max_grad_norm"],
        optim=CONFIG["training"]["optim"],
        lr_scheduler_type=CONFIG["training"]["lr_scheduler_type"],
        logging_strategy="steps",
        logging_steps=CONFIG["training"]["logging_steps"],
        save_strategy=CONFIG["training"]["save_strategy"],
        save_total_limit=CONFIG["training"]["save_total_limit"],
        bf16=use_bf16, fp16=False,
        report_to="none",
        seed=CONFIG["training"]["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=CONFIG["training"]["gradient_checkpointing"],
        dataloader_num_workers=2,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    trainer = DFTTrainer(
        model=model, args=args,
        train_dataset=dataset["train"],
        data_collator=collator,
    )

    result = trainer.train()
    print(f"\n✅ DFT训练完成！Loss: {result.training_loss:.4f}")
    print(f"  （DFT loss = P*CE，数值比CE小，正常现象）")

    final_dir = os.path.join(CONFIG["output_dir"], "final_model")
    save_model(model, tokenizer, final_dir)

    with open(os.path.join(CONFIG["output_dir"], "training_metrics.json"), "w") as f:
        json.dump(result.metrics, f, indent=2)


def main():
    try:
        setup()
        dataset = load_data()
        model, tokenizer = build_model()
        tokenized = preprocess(dataset, tokenizer)
        train(model, tokenizer, tokenized)

        print("\n" + "=" * 65)
        print("🎉 DFT训练完成！")
        print(f"模型: {os.path.join(CONFIG['output_dir'], 'final_model')}")
        print("\n下一步：用shallow_eval.py加入dft模型评测")
        print("对比：base / v2(CE) / dft(DFT)")
        print("=" * 65)

    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback; traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())