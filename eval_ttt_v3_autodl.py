import json
import re
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_PATH = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "datasets/tic_tac_toe_benchmark.json"

MODELS = [
    ("BASE", None),
    ("GOMOKU", "./checkpoints/qwen-gomoku-maxlora/final_model"),
    ("GO_COT", "./checkpoints/qwen7b-go-cot-maxlora-v2/final_model"),
]

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def make_prompt_original(instruction: str) -> str:
    return instruction


def make_prompt_neutral(instruction: str) -> str:
    return instruction.replace('例如 "2 3"', '例如 "行号 列号"')


def make_prompt_noexample(instruction: str) -> str:
    return re.sub(r"（格式：行号 列号.*?）", "（格式：行号 列号）", instruction)


PROMPT_VARIANTS = [
    ("original", make_prompt_original),
    ("neutral", make_prompt_neutral),
    ("noexample", make_prompt_noexample),
]


def parse_board(instruction: str):
    board = {}
    for line in instruction.splitlines():
        m = re.match(r"^(\d)\s*\|\s*([XO.])\s*([XO.])\s*([XO.])", line.strip())
        if not m:
            continue
        row = int(m.group(1))
        vals = [m.group(2), m.group(3), m.group(4)]
        for col, val in enumerate(vals, start=1):
            board[(row, col)] = val
    return board


def get_current_player(instruction: str) -> str:
    m = re.search(r"轮到\s*([XO])\s*走", instruction)
    return m.group(1) if m else "O"


def get_empty_cells(board):
    return [(r, c) for (r, c), v in board.items() if v == "."]


def check_win(board, player: str) -> bool:
    lines = [
        [(1, 1), (1, 2), (1, 3)],
        [(2, 1), (2, 2), (2, 3)],
        [(3, 1), (3, 2), (3, 3)],
        [(1, 1), (2, 1), (3, 1)],
        [(1, 2), (2, 2), (3, 2)],
        [(1, 3), (2, 3), (3, 3)],
        [(1, 1), (2, 2), (3, 3)],
        [(1, 3), (2, 2), (3, 1)],
    ]
    for line in lines:
        if all(board.get(cell) == player for cell in line):
            return True
    return False


def find_winning_move(board, player: str):
    wins = []
    for r, c in get_empty_cells(board):
        board[(r, c)] = player
        if check_win(board, player):
            wins.append((r, c))
        board[(r, c)] = "."
    return wins


def classify_position(board, player: str) -> str:
    opponent = "X" if player == "O" else "O"
    if find_winning_move(board, player):
        return "win"
    if find_winning_move(board, opponent):
        return "block"
    return "normal"


def extract_move(resp: str):
    m = re.search(r"(\d+)\s+(\d+)", resp)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def is_legal_move(board, pred):
    if pred is None:
        return False
    return board.get(pred) == "."


def load_model(adapter_path):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def evaluate(model, tokenizer, dataset, prompt_fn, limit=108):
    total = min(limit, len(dataset))
    exact_correct = 0
    legal_count = 0
    win_correct = 0
    win_total = 0
    block_correct = 0
    block_total = 0
    normal_correct = 0
    normal_total = 0
    pred_counter = Counter()

    for item in dataset[:total]:
        instruction = prompt_fn(item["instruction"])
        gold = item["output"].strip()
        board = parse_board(item["instruction"])
        player = get_current_player(item["instruction"])
        pos_type = classify_position(board, player)

        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        pred = extract_move(resp)
        pred_str = f"{pred[0]} {pred[1]}" if pred else "NONE"
        pred_counter[pred_str] += 1

        legal = is_legal_move(board, pred)
        exact = pred_str == gold
        if legal:
            legal_count += 1
        if exact:
            exact_correct += 1

        if pos_type == "win":
            win_total += 1
            winning_moves = find_winning_move(board, player)
            if pred in winning_moves:
                win_correct += 1
        elif pos_type == "block":
            block_total += 1
            opponent = "X" if player == "O" else "O"
            blocking_moves = find_winning_move(board, opponent)
            if pred in blocking_moves:
                block_correct += 1
        else:
            normal_total += 1
            if exact:
                normal_correct += 1

    return {
        "exact": exact_correct / total,
        "legal": legal_count / total,
        "win_rate": win_correct / win_total if win_total else None,
        "block_rate": block_correct / block_total if block_total else None,
        "normal_exact": normal_correct / normal_total if normal_total else None,
        "win_total": win_total,
        "block_total": block_total,
        "normal_total": normal_total,
        "top3": pred_counter.most_common(3),
    }


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    boards = [parse_board(item["instruction"]) for item in dataset]
    players = [get_current_player(item["instruction"]) for item in dataset]
    types = [classify_position(b, p) for b, p in zip(boards, players)]
    print("Benchmark局面分布:", dict(Counter(types)))
    print("Gold答案分布:", Counter(item["output"] for item in dataset).most_common(10))
    print()

    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    all_results = defaultdict(dict)

    for model_name, adapter_path in MODELS:
        print("=" * 60)
        print("Model:", model_name)
        print("=" * 60)
        model = load_model(adapter_path)

        for variant_name, prompt_fn in PROMPT_VARIANTS:
            result = evaluate(model, tokenizer, dataset, prompt_fn)
            all_results[model_name][variant_name] = result

            win_str = f"{result['win_rate']*100:.1f}%" if result["win_rate"] is not None else "N/A"
            block_str = f"{result['block_rate']*100:.1f}%" if result["block_rate"] is not None else "N/A"
            normal_str = f"{result['normal_exact']*100:.1f}%" if result["normal_exact"] is not None else "N/A"
            print(f"[{variant_name}]")
            print(
                f"  exact={result['exact']*100:.1f}%"
                f"  legal={result['legal']*100:.1f}%"
                f"  win={win_str}(n={result['win_total']})"
                f"  block={block_str}(n={result['block_total']})"
                f"  normal={normal_str}(n={result['normal_total']})"
            )
            print("  top3=", result["top3"])
        print()

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print("SUMMARY: exact match by prompt variant")
    print("=" * 96)
    print(
        "Model".ljust(12)
        + "original".rjust(12)
        + "neutral".rjust(12)
        + "noexample".rjust(12)
        + "legal(neu)".rjust(12)
        + "win(neu)".rjust(12)
        + "block(neu)".rjust(12)
    )
    print("-" * 96)
    for model_name, _ in MODELS:
        ro = all_results[model_name]["original"]
        rn = all_results[model_name]["neutral"]
        rx = all_results[model_name]["noexample"]
        win_str = f"{rn['win_rate']*100:.1f}%" if rn["win_rate"] is not None else "N/A"
        block_str = f"{rn['block_rate']*100:.1f}%" if rn["block_rate"] is not None else "N/A"
        print(
            model_name.ljust(12)
            + f"{ro['exact']*100:.1f}%".rjust(12)
            + f"{rn['exact']*100:.1f}%".rjust(12)
            + f"{rx['exact']*100:.1f}%".rjust(12)
            + f"{rn['legal']*100:.1f}%".rjust(12)
            + win_str.rjust(12)
            + block_str.rjust(12)
        )


if __name__ == "__main__":
    main()
