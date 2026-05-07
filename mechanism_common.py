import contextlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
DEFAULT_CACHE_DIR = "/root/autodl-tmp/hf_cache"
DEFAULT_OUTPUT_DIR = "./results/mechanism"

MODEL_SPECS = {
    "BASE": None,
    "GOMOKU_COT": "./checkpoints/qwen-gomoku-maxlora/final_model",
    "GOMOKU_NOCOT": "./checkpoints/qwen7b-gomoku-nocot/final_model",
    "GO_COT": "./checkpoints/qwen7b-go-cot-maxlora-v2/final_model",
    "GO_NOCOT": "./checkpoints/qwen7b_go_nocot_maxlora/final_model",
}

TASKS = {
    "object_counting": "object_counting",
    "logical_deduction_3": "logical_deduction_three_objects",
    "logical_deduction_5": "logical_deduction_five_objects",
    "logical_deduction_7": "logical_deduction_seven_objects",
    "geometric_shapes": "geometric_shapes",
    "temporal_sequences": "temporal_sequences",
    "multistep_arithmetic": "multistep_arithmetic_two",
    "navigate": "navigate",
    "colored_objects": "reasoning_about_colored_objects",
    "tracking_shuffled": "tracking_shuffled_objects_three_objects",
    "causal_judgement": "causal_judgement",
    "web_of_lies": "web_of_lies",
    "ruin_names": "ruin_names",
    "movie_recommendation": "movie_recommendation",
    "word_sorting": "word_sorting",
    "dyck_languages": "dyck_languages",
}

REPRESENTATIVE_TASKS = [
    "object_counting",
    "logical_deduction_3",
    "dyck_languages",
    "tracking_shuffled",
    "colored_objects",
]

ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}
TASK_MAX_NEW = {"web_of_lies": 1024}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(base_model: str = DEFAULT_BASE_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    model_name: str,
    base_model: str = DEFAULT_BASE_MODEL,
    use_eval: bool = True,
):
    adapter_path = MODEL_SPECS[model_name]
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=make_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if use_eval:
        model.eval()
    return model


def load_task(task_label: str, cache_dir: str = DEFAULT_CACHE_DIR):
    task_name = TASKS[task_label]
    return load_dataset("lukaemon/bbh", task_name, split="test", cache_dir=cache_dir)


def load_train_examples(task_label: str, n_shot: int = 3, cache_dir: str = DEFAULT_CACHE_DIR):
    task_name = TASKS[task_label]
    try:
        train_ds = load_dataset("lukaemon/bbh", task_name, split="train", cache_dir=cache_dir)
        if len(train_ds) >= n_shot:
            return train_ds.select(range(n_shot))
    except Exception:
        pass
    test_ds = load_task(task_label, cache_dir=cache_dir)
    return test_ds.select(range(min(n_shot, len(test_ds))))


def build_fewshot_messages(task_label: str, n_shot: int, cache_dir: str = DEFAULT_CACHE_DIR):
    ds = load_train_examples(task_label, n_shot=n_shot, cache_dir=cache_dir)
    messages = []
    for ex in ds:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": str(ex["target"])})
    return messages


def extract_web_of_lies(resp: str) -> str:
    matches = re.findall(r"\b(Yes|No)\b", resp, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
    match = re.search(
        r"Therefore.*?(does not tell the truth|tells the truth)",
        resp,
        re.IGNORECASE,
    )
    if match:
        phrase = match.group(0).lower()
        return "No" if "does not" in phrase else "Yes"
    if "does not tell the truth" in resp.lower():
        return "No"
    if "tells the truth" in resp.lower():
        return "Yes"
    return "NONE"


def extract_answer(resp: str, task_label: str) -> str:
    if task_label == "web_of_lies":
        return extract_web_of_lies(resp)

    if task_label in {"navigate", "causal_judgement"}:
        matches = re.findall(r"\b(Yes|No)\b", resp, re.IGNORECASE)
        return matches[-1].capitalize() if matches else "NONE"

    match = re.search(r"\(([A-Z])\)", resp)
    if match:
        return match.group(1)

    match = re.search(r"\b(Valid|Invalid)\b", resp, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    numbers = re.findall(r"\b\d+\b", resp)
    if numbers:
        return numbers[-1]

    return resp.strip()[:24]


def gold_answer(example, task_label: str) -> str:
    gold = str(example["target"]).strip()
    if task_label.startswith("logical_deduction"):
        return re.sub(r"[()]", "", gold)
    return re.sub(r"[()]", "", gold)


def generate_response(
    model,
    tokenizer,
    messages,
    max_new_tokens: int,
):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return prompt, resp


def evaluate_tasks(
    model,
    tokenizer,
    task_labels: List[str],
    n_shot: int = 3,
    limit: int = 50,
    cache_dir: str = DEFAULT_CACHE_DIR,
    default_max_new: int = 512,
):
    results = {}
    for task_label in task_labels:
        fewshot = build_fewshot_messages(task_label, n_shot=n_shot, cache_dir=cache_dir)
        ds = load_task(task_label, cache_dir=cache_dir)
        start = n_shot
        end = min(start + limit, len(ds))
        subset = ds.select(range(start, end))
        correct = 0
        for ex in subset:
            messages = fewshot + [{"role": "user", "content": ex["input"]}]
            _, resp = generate_response(
                model,
                tokenizer,
                messages,
                max_new_tokens=TASK_MAX_NEW.get(task_label, default_max_new),
            )
            pred = re.sub(r"[()]", "", extract_answer(resp, task_label)).strip()
            gold = gold_answer(ex, task_label)
            if pred == gold:
                correct += 1
        results[task_label] = {
            "correct": correct,
            "total": len(subset),
            "acc": correct / max(len(subset), 1),
        }
    return results


def model_num_layers(model) -> int:
    candidates = [
        getattr(getattr(model, "config", None), "num_hidden_layers", None),
        getattr(getattr(model, "config", None), "n_layer", None),
    ]
    for value in candidates:
        if isinstance(value, int):
            return value
    max_layer = -1
    for name, _ in model.named_modules():
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match:
            max_layer = max(max_layer, int(match.group(1)))
    return max_layer + 1


def layer_band(layer_idx: int, n_layers: int) -> str:
    if n_layers <= 0:
        return "unknown"
    band = n_layers // 3
    if layer_idx < band:
        return "shallow"
    if layer_idx < 2 * band:
        return "middle"
    return "deep"


def parse_layer_index(module_name: str) -> Optional[int]:
    match = re.search(r"\.layers\.(\d+)\.", module_name)
    return int(match.group(1)) if match else None


def parse_leaf_name(module_name: str) -> str:
    return module_name.split(".")[-1]


def module_kind(leaf_name: str) -> str:
    if leaf_name in ATTENTION_MODULES:
        return "attention"
    if leaf_name in MLP_MODULES:
        return "mlp"
    return "other"


def iter_lora_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield name, module


def active_adapter_name(module) -> Optional[str]:
    keys = list(module.lora_A.keys())
    if not keys:
        return None
    if "default" in keys:
        return "default"
    return keys[0]


@dataclass
class LoraRecord:
    module_name: str
    layer_idx: int
    band: str
    leaf_name: str
    kind: str
    delta_norm: float
    base_norm: float
    ratio: float


def collect_lora_records(model) -> List[LoraRecord]:
    n_layers = model_num_layers(model)
    records: List[LoraRecord] = []
    for name, module in iter_lora_modules(model):
        adapter = active_adapter_name(module)
        if adapter is None:
            continue
        layer_idx = parse_layer_index(name)
        if layer_idx is None:
            continue
        leaf_name = parse_leaf_name(name)
        kind = module_kind(leaf_name)
        scaling = module.scaling[adapter]
        lora_a = module.lora_A[adapter].weight.detach().float()
        lora_b = module.lora_B[adapter].weight.detach().float()
        delta = torch.matmul(lora_b, lora_a) * float(scaling)
        if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
            base_weight = module.base_layer.weight.detach().float()
        elif hasattr(module, "weight"):
            base_weight = module.weight.detach().float()
        else:
            continue
        delta_norm = torch.norm(delta, p="fro").item()
        base_norm = torch.norm(base_weight, p="fro").item()
        ratio = delta_norm / base_norm if base_norm > 0 else 0.0
        records.append(
            LoraRecord(
                module_name=name,
                layer_idx=layer_idx,
                band=layer_band(layer_idx, n_layers),
                leaf_name=leaf_name,
                kind=kind,
                delta_norm=delta_norm,
                base_norm=base_norm,
                ratio=ratio,
            )
        )
    return records


@contextlib.contextmanager
def zero_selected_lora_scaling(model, predicate):
    saved: List[Tuple[object, str, float]] = []
    for name, module in iter_lora_modules(model):
        adapter = active_adapter_name(module)
        if adapter is None:
            continue
        if predicate(name, module):
            value = float(module.scaling[adapter])
            saved.append((module, adapter, value))
            module.scaling[adapter] = 0.0
    try:
        yield
    finally:
        for module, adapter, value in saved:
            module.scaling[adapter] = value


def write_json(path: str, data) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: str, header: List[str], rows: Iterable[Iterable]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            values = []
            for cell in row:
                text = str(cell)
                if "," in text or '"' in text:
                    text = '"' + text.replace('"', '""') + '"'
                values.append(text)
            f.write(",".join(values) + "\n")

