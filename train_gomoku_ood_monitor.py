"""
train_gomoku_ood_monitor.py

带OOD监控的五子棋LoRA训练
每隔eval_steps步在OOD题库上评测，记录完整OOD曲线
自动保存OOD峰值检查点

理论依据：Jin et al. (2025) "RL Fine-Tuning Heals OOD Forgetting"
  SFT训练过程中OOD性能先升后降（SFT Generalization Paradox）
  存在SFTMaxOOD检查点，其OOD性能可能显著高于训练终点
  该检查点无法从训练loss判断，必须在OOD验证集上监控

用法：
  python train_gomoku_ood_monitor.py
  python train_gomoku_ood_monitor.py --eval_steps 20 --epochs 2
"""

import os, sys, json, re, copy, argparse
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0+PTX;8.6+PTX;8.9+PTX'

import torch
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, set_seed, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

_orig = torch.load
def _patched(*a, **kw): kw['weights_only'] = False; return _orig(*a, **kw)
torch.load = _patched

import warnings; warnings.filterwarnings("ignore")

# ==================== 配置 ====================

BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = "./datasets/real_games_v2/train.json"
OUTPUT_DIR   = "./checkpoints/qwen-gomoku-ood-monitor"
CACHE_DIR    = "./cache"

LORA_CONFIG = {
    "r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
    "bias": "none", "target_modules": ["q_proj", "v_proj"],
}

TRAIN_CONFIG = {
    "num_train_epochs":            2,       # 训练2 epoch，确保能观察到完整的先升后降曲线
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate":               3e-6,
    "warmup_steps":                50,
    "max_grad_norm":               2.0,
    "optim":                       "adamw_torch",
    "lr_scheduler_type":           "linear",
    "weight_decay":                0.1,
    "seed":                        42,
    "gradient_checkpointing":      True,
    "bf16":                        True,
    "max_length":                  1024,
}

OOD_CONFIG = {
    "eval_steps":    25,    # 每25步评测一次OOD（约每4%的epoch）
    "max_new_tokens": 80,
    "n_ood_samples":  40,   # 每次评测40道题（速度和精度的平衡）
}

# ==================== OOD题库（精简版，40题）====================

OOD_SAMPLES = [
    # 数学（15题）
    ("What is 15 + 27?", "42"),
    ("What is 7 multiplied by 8?", "56"),
    ("What is 100 divided by 4?", "25"),
    ("What is the square root of 144?", "12"),
    ("What is 3 to the power of 4?", "81"),
    ("If x + 3 = 10, what is x?", "7"),
    ("What is 25% of 80?", "20"),
    ("What is the average of 10, 20, 30, 40, 50?", "30"),
    ("A rectangle has length 8 and width 5. Area?", "40"),
    ("What is 15% of 200?", "30"),
    ("If 2x - 4 = 10, what is x?", "7"),
    ("What is the next prime after 13?", "17"),
    ("What is 9 multiplied by 9?", "81"),
    ("What is 2 to the power of 8?", "256"),
    ("A shirt costs $40, 20% off. Sale price?", "32"),
    # 空间推理（10题）
    ("A is to the left of B. B is above C. What is A relative to C?", "upper-left"),
    ("A is above B. B is to the right of C. What is A relative to C?", "upper-right"),
    ("A is to the left of B. B is to the left of C. What is A relative to C?", "left"),
    ("A is above B. B is above C. What is A relative to C?", "above"),
    ("A is to the right of B. B is below C. What is A relative to C?", "lower-right"),
    ("Start facing East. Turn left. Turn left again. Which direction?", "west"),
    ("Start facing North. Turn right three times. Which direction?", "west"),
    ("Start facing South. Turn right. Which direction now?", "west"),
    ("X is north of Y. Y is east of Z. Where is X relative to Z?", "upper-right"),
    ("A above B. B right of C. C above D. What is A relative to D?", "upper-right"),
    # 规划推理（10题）
    ("Tower of Hanoi: 2 disks. Minimum moves?", "3"),
    ("Tower of Hanoi: 3 disks. Minimum moves?", "7"),
    ("Tower of Hanoi 1 disk. Minimum moves?", "1"),
    ("Tower of Hanoi 4 disks. Minimum moves?", "15"),
    ("Blocks: A on B, B on table. Goal: B on A. Minimum moves?", "2"),
    ("Blocks: C on B, B on A. Goal: all separate. First move?", "move c"),
    ("Grid start (1,1) goal (3,3). Minimum moves?", "4"),
    ("Pancake sort [2,1]. Minimum flips?", "1"),
    ("Pancake sort [3,2,1]. Minimum flips?", "2"),
    ("Tower of Hanoi 5 disks. Minimum moves?", "31"),
    # 逻辑推理（5题）
    ("All cats are mammals. Whiskers is a cat. Is Whiskers a mammal? yes or no.", "yes"),
    ("No fish can walk. Nemo is a fish. Can Nemo walk? yes or no.", "no"),
    ("All A are B. All B are C. Is every A a C? yes or no.", "yes"),
    ("Either P or Q. Not P. Is Q true? yes or no.", "yes"),
    ("If P then Q. Q then R. P is true. Is R true? yes or no.", "yes"),
]


# ==================== OOD评测函数 ====================

def evaluate_ood(model, tokenizer, device, samples=None):
    """快速OOD评测，返回准确率"""
    if samples is None:
        samples = OOD_SAMPLES
    model.eval()
    correct = 0
    with torch.no_grad():
        for prompt, answer in samples:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.generate(
                **inputs,
                max_new_tokens=OOD_CONFIG["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            resp = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()
            correct += int(answer.lower() in resp.lower())
    model.train()
    return correct / len(samples)


# ==================== OOD监控Callback ====================

class OODMonitorCallback(TrainerCallback):
    """
    在训练过程中周期性评测OOD性能
    自动保存OOD峰值检查点
    """
    def __init__(self, tokenizer, output_dir, eval_steps):
        self.tokenizer  = tokenizer
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.history    = []   # [{step, epoch, ood_acc, train_loss}]
        self.best_acc   = 0.0
        self.best_step  = 0
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_dir   = os.path.join(output_dir, "best_ood_checkpoint")
        os.makedirs(self.best_dir, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return

        ood_acc = evaluate_ood(model, self.tokenizer, self.device)
        train_loss = state.log_history[-1].get("loss", 0) if state.log_history else 0
        epoch = state.epoch or 0

        record = {
            "step":       state.global_step,
            "epoch":      round(epoch, 3),
            "ood_acc":    round(ood_acc, 4),
            "train_loss": round(train_loss, 4),
        }
        self.history.append(record)

        # 判断是否是OOD峰值
        is_best = ood_acc > self.best_acc
        if is_best:
            self.best_acc  = ood_acc
            self.best_step = state.global_step
            # 保存峰值检查点
            model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)
            print(f"\n  🏆 新OOD峰值！step={state.global_step} "
                  f"epoch={epoch:.3f} ood={ood_acc:.3f} "
                  f"(loss={train_loss:.4f}) → 已保存至 {self.best_dir}")
        else:
            print(f"\n  📊 OOD评测：step={state.global_step} "
                  f"epoch={epoch:.3f} ood={ood_acc:.3f} "
                  f"(best={self.best_acc:.3f}@step{self.best_step})")

        # 保存历史曲线
        history_path = os.path.join(self.output_dir, "ood_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "history":    self.history,
                "best_acc":   self.best_acc,
                "best_step":  self.best_step,
                "n_samples":  len(OOD_SAMPLES),
            }, f, indent=2)

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*60}")
        print(f"OOD监控总结")
        print(f"{'='*60}")
        print(f"最优OOD准确率：{self.best_acc:.3f}（step={self.best_step}）")
        if self.history:
            final_acc = self.history[-1]["ood_acc"]
            print(f"训练终点OOD：   {final_acc:.3f}")
            print(f"峰值-终点差值： {self.best_acc - final_acc:+.3f}")
            if self.best_acc > final_acc + 0.02:
                print(f"✅ 发现SFT Generalization Paradox：OOD在训练中途达到峰值后下降")
                print(f"   推荐使用峰值检查点：{self.best_dir}")
            else:
                print(f"= OOD曲线较平稳，未观察到明显的先升后降")

        print(f"\nOOD曲线：")
        for r in self.history:
            marker = " 🏆" if r["step"] == self.best_step else ""
            print(f"  step={r['step']:4d} epoch={r['epoch']:.2f} "
                  f"ood={r['ood_acc']:.3f} loss={r['train_loss']:.4f}{marker}")


# ==================== 主流程 ====================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_steps", type=int, default=OOD_CONFIG["eval_steps"])
    p.add_argument("--epochs",     type=int, default=TRAIN_CONFIG["num_train_epochs"])
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    OOD_CONFIG["eval_steps"] = args.eval_steps
    TRAIN_CONFIG["num_train_epochs"] = args.epochs

    print("=" * 65)
    print("五子棋LoRA训练 + OOD实时监控")
    print(f"理论：SFT Generalization Paradox（Jin et al. 2025）")
    print(f"配置：r=8, q+v, {args.epochs} epoch, eval每{args.eval_steps}步")
    print(f"OOD题库：{len(OOD_SAMPLES)}题（数学/空间/规划/逻辑）")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    set_seed(TRAIN_CONFIG["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    print("\n加载数据集...")
    ds = load_dataset("json", data_files=DATASET_PATH, cache_dir=CACHE_DIR)
    print(f"✅ 样本数: {len(ds['train'])}")

    # 加载模型
    print("\n加载模型 + QLoRA...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, cache_dir=CACHE_DIR, use_cache=False,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, cache_dir=CACHE_DIR,
        local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_CONFIG["r"], lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"], bias=LORA_CONFIG["bias"],
        target_modules=LORA_CONFIG["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 预处理
    def tokenize(examples):
        texts = [
            f"<|im_start|>user\n{examples['instruction'][i]}<|im_end|>\n"
            f"<|im_start|>assistant\n{examples['output'][i]}<|im_end|>"
            for i in range(len(examples["instruction"]))
        ]
        tok = tokenizer(texts, truncation=True, padding=False,
                        max_length=TRAIN_CONFIG["max_length"])
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = ds.map(tokenize, batched=True,
                       remove_columns=ds["train"].column_names,
                       desc="Tokenizing")

    # 使用已知base结果作为基准（math=0.720, spatial=0.360, planning=0.520, logic=1.000）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_ood = 0.600  # 40题混合任务的base近似值
    print(f"  使用已知base OOD基准：{base_ood:.3f}")

    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({
            "base_ood": base_ood,
            "eval_steps": args.eval_steps,
            "epochs": args.epochs,
            "n_ood_samples": len(OOD_SAMPLES),
            "start_time": datetime.now().isoformat(),
        }, f, indent=2)

    # OOD监控callback
    ood_callback = OODMonitorCallback(
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
    )

    # 训练参数
    gpu_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
    use_bf16 = TRAIN_CONFIG["bf16"] and gpu_cap[0] >= 8

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        max_grad_norm=TRAIN_CONFIG["max_grad_norm"],
        optim=TRAIN_CONFIG["optim"],
        lr_scheduler_type=TRAIN_CONFIG["lr_scheduler_type"],
        logging_strategy="steps", logging_steps=5,
        save_strategy="no",   # 不保存常规checkpoint，只保存OOD最优
        bf16=use_bf16, fp16=False,
        report_to="none",
        seed=TRAIN_CONFIG["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=TRAIN_CONFIG["gradient_checkpointing"],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
        callbacks=[ood_callback],
    )

    print(f"\n开始训练（{args.epochs} epoch，每{args.eval_steps}步评测OOD）...")
    result = trainer.train()
    print(f"\n✅ 训练完成！Loss: {result.training_loss:.4f}")

    # 保存最终模型
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # 最终OOD评测
    final_ood = evaluate_ood(model, tokenizer, device)
    print(f"\n最终OOD准确率：{final_ood:.3f}")
    print(f"OOD峰值：      {ood_callback.best_acc:.3f}（step={ood_callback.best_step}）")
    print(f"初始OOD：      {base_ood:.3f}")
    print(f"\n关键对比：")
    print(f"  peak - final = {ood_callback.best_acc - final_ood:+.3f}")
    print(f"  peak - base  = {ood_callback.best_acc - base_ood:+.3f}")
    print(f"  final - base = {final_ood - base_ood:+.3f}")

    print(f"\n✅ OOD历史：{args.output_dir}/ood_history.json")
    print(f"✅ 最优检查点：{args.output_dir}/best_ood_checkpoint")
    print(f"✅ 最终模型：{final_dir}")

    # 追加结论
    with open(os.path.join(args.output_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg.update({
        "final_ood": final_ood,
        "best_ood": ood_callback.best_acc,
        "best_step": ood_callback.best_step,
        "base_ood": base_ood,
        "peak_minus_final": ood_callback.best_acc - final_ood,
        "peak_minus_base": ood_callback.best_acc - base_ood,
    })
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()