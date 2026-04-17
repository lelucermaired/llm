"""
new_target_eval.py

评测已有五子棋微调模型（v2）在三类新目标任务上的表现：
1. 空间推理（StepGame风格：多步位置关系推断）
2. 逻辑推理（LogiQA风格：条件推断）
3. 序列规划（积木/迷宫：给定初始状态输出操作序列）

对比基础模型 vs 微调模型，检验是否存在正向迁移。

用法：
    python new_target_eval.py
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "finetuned_model_path": "./checkpoints/qwen-gomoku-r64/final_model",
    "output_dir": "./new_target_results_r64",
    "max_new_tokens": 80,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 1. 空间推理（StepGame风格）====================
# 给定一系列两两位置关系，推断最终两物体的相对位置
# 答案为方向词：left / right / above / below / upper-left / upper-right / lower-left / lower-right / overlap

SPATIAL_SAMPLES = [
    # 2步推理
    {
        "question": "A is to the left of B. B is to the left of C. What is the relation of A to C?",
        "answer": "left",
        "steps": 2
    },
    {
        "question": "A is above B. B is above C. What is the relation of A to C?",
        "answer": "above",
        "steps": 2
    },
    {
        "question": "A is to the right of B. B is above C. What is the relation of A to C?",
        "answer": "upper-right",
        "steps": 2
    },
    {
        "question": "A is below B. B is to the left of C. What is the relation of A to C?",
        "answer": "lower-left",
        "steps": 2
    },
    {
        "question": "A is to the left of B. B is below C. What is the relation of A to C?",
        "answer": "lower-left",
        "steps": 2
    },
    # 3步推理
    {
        "question": "A is to the left of B. B is to the left of C. C is to the left of D. What is the relation of A to D?",
        "answer": "left",
        "steps": 3
    },
    {
        "question": "A is above B. B is to the right of C. C is above D. What is the relation of A to D?",
        "answer": "upper-right",
        "steps": 3
    },
    {
        "question": "A is to the left of B. B is above C. C is to the right of D. What is the relation of A to D?",
        "answer": "above",
        "steps": 3
    },
    {
        "question": "A is below B. B is to the left of C. C is below D. What is the relation of A to D?",
        "answer": "lower-left",
        "steps": 3
    },
    {
        "question": "A is to the right of B. B is below C. C is to the left of D. What is the relation of A to D?",
        "answer": "below",
        "steps": 3
    },
    # 4步推理
    {
        "question": "A is to the left of B. B is above C. C is to the right of D. D is above E. What is the relation of A to E?",
        "answer": "upper-right",
        "steps": 4
    },
    {
        "question": "A is above B. B is to the left of C. C is below D. D is to the right of E. What is the relation of A to E?",
        "answer": "upper-left",
        "steps": 4
    },
    {
        "question": "A is to the right of B. B is to the right of C. C is below D. D is to the left of E. What is the relation of A to E?",
        "answer": "below",
        "steps": 4
    },
    {
        "question": "A is below B. B is below C. C is to the left of D. D is below E. What is the relation of A to E?",
        "answer": "lower-left",
        "steps": 4
    },
    {
        "question": "A is above B. B is above C. C is to the right of D. D is above E. What is the relation of A to E?",
        "answer": "upper-right",
        "steps": 4
    },
    # 5步推理
    {
        "question": "A is to the left of B. B is above C. C is to the left of D. D is below E. E is to the left of F. What is the relation of A to F?",
        "answer": "left",
        "steps": 5
    },
    {
        "question": "A is above B. B is to the right of C. C is above D. D is to the right of E. E is above F. What is the relation of A to F?",
        "answer": "upper-right",
        "steps": 5
    },
    {
        "question": "A is to the right of B. B is below C. C is to the right of D. D is above E. E is to the right of F. What is the relation of A to F?",
        "answer": "right",
        "steps": 5
    },
    {
        "question": "A is below B. B is to the left of C. C is below D. D is to the left of E. E is below F. What is the relation of A to F?",
        "answer": "lower-left",
        "steps": 5
    },
    {
        "question": "A is to the left of B. B is to the left of C. C is above D. D is to the right of E. E is above F. What is the relation of A to F?",
        "answer": "above",
        "steps": 5
    },
]

# ==================== 2. 逻辑推理（LogiQA风格）====================
# 给定条件，选择正确推论（A/B/C/D四选一）

LOGIC_SAMPLES = [
    {
        "question": """All doctors are scientists. Some scientists are artists. Which of the following must be true?
A. All doctors are artists.
B. Some doctors are artists.
C. Some artists are doctors.
D. No conclusion about doctors and artists can be drawn.""",
        "answer": "D",
    },
    {
        "question": """If it rains, the ground is wet. The ground is wet. Which of the following is correct?
A. It must have rained.
B. It may or may not have rained.
C. It did not rain.
D. We cannot determine whether it rained.""",
        "answer": "D",
    },
    {
        "question": """All cats are mammals. All mammals are animals. Whiskers is a cat. Which must be true?
A. Whiskers is an animal.
B. All animals are cats.
C. Some mammals are not animals.
D. Whiskers is not a mammal.""",
        "answer": "A",
    },
    {
        "question": """No fish can walk. Some pets are fish. Which must be true?
A. All pets can walk.
B. Some pets cannot walk.
C. No pets can walk.
D. All fish are pets.""",
        "answer": "B",
    },
    {
        "question": """Either John or Mary will attend the meeting, but not both. John did not attend. Which must be true?
A. Neither attended.
B. Both attended.
C. Mary attended.
D. Mary did not attend.""",
        "answer": "C",
    },
    {
        "question": """All students who passed the exam studied hard. Tom did not study hard. Which must be true?
A. Tom passed the exam.
B. Tom did not pass the exam.
C. Some students who studied hard failed.
D. All students who studied hard passed.""",
        "answer": "B",
    },
    {
        "question": """If a number is divisible by 6, it is divisible by both 2 and 3. 18 is divisible by 6. Which must be true?
A. 18 is divisible by 2 only.
B. 18 is divisible by 3 only.
C. 18 is divisible by both 2 and 3.
D. 18 is not divisible by 2.""",
        "answer": "C",
    },
    {
        "question": """All red balls are heavy. Some heavy objects are metal. Which must be true?
A. All red balls are metal.
B. Some red balls are metal.
C. No red balls are metal.
D. None of the above must be true.""",
        "answer": "D",
    },
    {
        "question": """In a group, everyone who likes chess also likes puzzles. Sarah likes puzzles but not chess. Which must be true?
A. Sarah likes chess.
B. Everyone who likes puzzles likes chess.
C. Not everyone who likes puzzles likes chess.
D. Sarah does not like puzzles.""",
        "answer": "C",
    },
    {
        "question": """If the alarm rings, either there is a fire or it is a drill. There is no fire and it is not a drill. Which must be true?
A. The alarm rang.
B. The alarm did not ring.
C. There is a fire.
D. It is a drill.""",
        "answer": "B",
    },
    {
        "question": """All managers have at least 5 years of experience. Alice has 3 years of experience. Which must be true?
A. Alice is a manager.
B. Alice is not a manager.
C. Some managers have fewer than 5 years of experience.
D. Alice will become a manager.""",
        "answer": "B",
    },
    {
        "question": """Some birds cannot fly. All penguins are birds. Which must be true?
A. All penguins can fly.
B. No penguins can fly.
C. Penguins are birds.
D. Some birds are penguins.""",
        "answer": "C",
    },
    {
        "question": """If it is Monday, the store is closed. The store is open. Which must be true?
A. It is Monday.
B. It is not Monday.
C. The store is always open.
D. It might be Monday.""",
        "answer": "B",
    },
    {
        "question": """All squares are rectangles. Figure X is not a rectangle. Which must be true?
A. Figure X is a square.
B. Figure X is not a square.
C. All rectangles are squares.
D. Figure X is a triangle.""",
        "answer": "B",
    },
    {
        "question": """Some employees are managers. All managers receive bonuses. Which must be true?
A. All employees receive bonuses.
B. Some employees receive bonuses.
C. No employees receive bonuses.
D. All bonus recipients are managers.""",
        "answer": "B",
    },
    {
        "question": """If P then Q. If Q then R. P is true. Which must be true?
A. R is true.
B. P is false.
C. Q is false.
D. R is false.""",
        "answer": "A",
    },
    {
        "question": """All lawyers passed the bar exam. Some people who passed the bar exam work in finance. Which must be true?
A. All lawyers work in finance.
B. Some lawyers work in finance.
C. No lawyers work in finance.
D. None of the above must be true.""",
        "answer": "D",
    },
    {
        "question": """Nobody who is tired can concentrate. Lisa cannot concentrate. Which must be true?
A. Lisa is tired.
B. Lisa may or may not be tired.
C. Lisa is not tired.
D. Lisa can concentrate.""",
        "answer": "B",
    },
    {
        "question": """All prime numbers greater than 2 are odd. 7 is a prime number greater than 2. Which must be true?
A. 7 is even.
B. 7 is odd.
C. 7 is not prime.
D. All odd numbers are prime.""",
        "answer": "B",
    },
    {
        "question": """Either the project is on time or it is over budget, but not both. The project is not over budget. Which must be true?
A. The project is on time.
B. The project is over budget.
C. The project is both on time and over budget.
D. The project is neither on time nor over budget.""",
        "answer": "A",
    },
]

# ==================== 3. 序列规划（积木移动）====================
# 给定初始状态和目标状态，输出最少操作步骤

PLANNING_SAMPLES = [
    {
        "question": """You have 3 blocks: A, B, C.
Initial state: A is on the table. B is on A. C is on B.
Goal state: C is on the table. B is on C. A is on B.
What is the minimum sequence of moves? (Each move: pick up one block and place it on table or on another block. You can only move the top block of a stack.)
Answer with the move sequence like: Move X from Y to Z.""",
        "answer": "Move C from B to table. Move B from A to table. Move A from table to somewhere... ",
        "answer_key": "Move C",  # 第一步必须是Move C
        "steps": 3
    },
    {
        "question": """You have 3 blocks: A, B, C.
Initial state: B is on the table. A is on B. C is on the table (separate stack).
Goal state: A is on the table. B is on A. C is on B.
What is the minimum sequence of moves?""",
        "answer_key": "Move A",
        "steps": 3
    },
    {
        "question": """You have 2 blocks: A, B.
Initial state: A is on the table. B is on A.
Goal state: B is on the table. A is on B.
What is the minimum sequence of moves?""",
        "answer_key": "Move B",
        "steps": 2
    },
    {
        "question": """Maze navigation: You are at position (1,1) in a 3x3 grid. The exit is at (3,3). You can move Up, Down, Left, Right. No obstacles.
What is the shortest path?""",
        "answer_key": "Right",
        "steps": 4
    },
    {
        "question": """Tower of Hanoi with 2 disks. Disks are on peg A (disk 1 on top of disk 2). Move all disks to peg C using peg B as auxiliary. You cannot place a larger disk on a smaller one.
What is the sequence of moves?""",
        "answer_key": "Move disk 1",
        "steps": 3
    },
    {
        "question": """You have 4 blocks: A, B, C, D.
Initial state: All blocks on the table separately.
Goal state: D on table, C on D, B on C, A on B.
What is the minimum sequence of moves?""",
        "answer_key": "Move C onto D",
        "steps": 3
    },
    {
        "question": """Sliding puzzle: You have a 2x2 grid with tiles 1, 2, 3 and one empty space.
Initial: 1 2 / 3 _  (where _ is empty)
Goal:    1 2 / _ 3
What is the minimum move sequence? (Move tile into empty space)""",
        "answer_key": "Move 3",
        "steps": 1
    },
    {
        "question": """Robot navigation: A robot is at (0,0) facing North. It needs to reach (2,1).
Available commands: Move Forward, Turn Left, Turn Right.
What is the shortest command sequence?""",
        "answer_key": "Move Forward",
        "steps": 4
    },
    {
        "question": """You have 3 blocks: X, Y, Z.
Initial state: Z on table, Y on Z, X on Y.
Goal state: X on table, Y on X, Z on Y.
What is the minimum sequence of moves?""",
        "answer_key": "Move X",
        "steps": 5
    },
    {
        "question": """Pancake sorting: You have pancakes of sizes 3, 1, 2 (top to bottom).
Goal: Sort them so smallest is on top: 1, 2, 3.
You can flip the top N pancakes at once.
What is the minimum sequence of flips?""",
        "answer_key": "Flip",
        "steps": 3
    },
]


# ==================== 评测函数 ====================

def load_model(path, base_path, is_base=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if is_base:
        base.eval()
        return base
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens=80, device="cuda"):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def check_spatial(response, answer):
    return answer.lower() in response.lower()


def check_logic(response, answer):
    # 检查回复中是否包含正确选项
    resp = response.strip().upper()
    # 优先匹配开头的选项字母
    for pattern in [f"{answer}.", f"{answer}:", f"answer is {answer}", f"answer: {answer}", answer]:
        if pattern.upper() in resp:
            return True
    return False


def check_planning(response, answer_key):
    return answer_key.lower() in response.lower()


def evaluate(model, tokenizer, samples, task_type, desc, device):
    correct = 0
    results = []
    for sample in tqdm(samples, desc=desc):
        response = generate(model, tokenizer, sample["question"],
                          max_new_tokens=CONFIG["max_new_tokens"], device=device)
        if task_type == "spatial":
            is_correct = check_spatial(response, sample["answer"])
        elif task_type == "logic":
            is_correct = check_logic(response, sample["answer"])
        else:  # planning
            is_correct = check_planning(response, sample["answer_key"])

        correct += int(is_correct)
        results.append({
            "question": sample["question"][:100],
            "expected": sample.get("answer", sample.get("answer_key")),
            "response": response[:150],
            "correct": is_correct,
        })

    acc = correct / len(samples)
    return acc, results


# ==================== 主流程 ====================

def main():
    print("=" * 65)
    print("新目标任务评测：空间推理 / 逻辑推理 / 序列规划")
    print("基础模型 vs 五子棋微调模型（v2）")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}

    # ========== 基础模型 ==========
    print("\n[1/2] 基础模型评测...")
    base_model = load_model(None, CONFIG["base_model_path"], is_base=True)

    base_spatial_acc, base_spatial_res = evaluate(
        base_model, tokenizer, SPATIAL_SAMPLES, "spatial", "基础×空间推理", device)
    base_logic_acc, base_logic_res = evaluate(
        base_model, tokenizer, LOGIC_SAMPLES, "logic", "基础×逻辑推理", device)
    base_plan_acc, base_plan_res = evaluate(
        base_model, tokenizer, PLANNING_SAMPLES, "planning", "基础×序列规划", device)

    print(f"\n  空间推理: {base_spatial_acc:.3f}")
    print(f"  逻辑推理: {base_logic_acc:.3f}")
    print(f"  序列规划: {base_plan_acc:.3f}")

    del base_model
    torch.cuda.empty_cache()

    # ========== 微调模型 ==========
    print("\n[2/2] 微调模型评测...")
    ft_model = load_model(CONFIG["finetuned_model_path"], CONFIG["base_model_path"])

    ft_spatial_acc, ft_spatial_res = evaluate(
        ft_model, tokenizer, SPATIAL_SAMPLES, "spatial", "微调×空间推理", device)
    ft_logic_acc, ft_logic_res = evaluate(
        ft_model, tokenizer, LOGIC_SAMPLES, "logic", "微调×逻辑推理", device)
    ft_plan_acc, ft_plan_res = evaluate(
        ft_model, tokenizer, PLANNING_SAMPLES, "planning", "微调×序列规划", device)

    print(f"\n  空间推理: {ft_spatial_acc:.3f}")
    print(f"  逻辑推理: {ft_logic_acc:.3f}")
    print(f"  序列规划: {ft_plan_acc:.3f}")

    del ft_model
    torch.cuda.empty_cache()

    # ========== 汇总 ==========
    print("\n" + "=" * 65)
    print("汇总：基础模型 vs 微调模型")
    print("=" * 65)
    print(f"{'任务':<15} {'基础模型':>10} {'微调模型':>10} {'变化':>10} {'结论'}")
    print("-" * 60)

    tasks = [
        ("空间推理", base_spatial_acc, ft_spatial_acc),
        ("逻辑推理", base_logic_acc, ft_logic_acc),
        ("序列规划", base_plan_acc, ft_plan_acc),
    ]

    for name, base_acc, ft_acc in tasks:
        delta = ft_acc - base_acc
        if delta > 0.05:
            label = "✅ 正向迁移"
        elif delta < -0.05:
            label = "❌ 负向迁移"
        else:
            label = "— 零迁移"
        print(f"{name:<15} {base_acc:>10.3f} {ft_acc:>10.3f} {delta:>+10.3f}  {label}")

    # 和GSM8K对比
    print("\n参考（已有数据）：")
    print(f"{'GSM8K数学推理':<15} {'~0.480':>10} {'~0.460':>10} {'~-0.020':>10}  — 零迁移")

    # ========== 核心结论 ==========
    print("\n" + "=" * 65)
    print("核心结论")
    print("=" * 65)

    positive = [n for n, b, f in tasks if f - b > 0.05]
    negative = [n for n, b, f in tasks if f - b < -0.05]
    zero = [n for n, b, f in tasks if abs(f - b) <= 0.05]

    if positive:
        print(f"\n→ ✅ 在以下任务出现正向迁移：{positive}")
        print("  说明五子棋推理能力向这些任务存在迁移路径")
        print("  与GSM8K零迁移形成对比，迁移具有任务选择性")
    if zero:
        print(f"\n→ — 以下任务仍为零迁移：{zero}")
    if negative:
        print(f"\n→ ❌ 以下任务出现负向迁移：{negative}")

    if not positive:
        print("\n→ 三类任务均为零迁移")
        print("  结论推广：五子棋LoRA微调对空间推理、逻辑推理、序列规划均无显著影响")
        print("  零迁移结论不局限于数学任务，具有跨任务普遍性")

    # ========== 保存 ==========
    save_data = {
        "spatial": {"base": base_spatial_acc, "ft": ft_spatial_acc,
                    "delta": ft_spatial_acc - base_spatial_acc,
                    "base_results": base_spatial_res, "ft_results": ft_spatial_res},
        "logic":   {"base": base_logic_acc, "ft": ft_logic_acc,
                    "delta": ft_logic_acc - base_logic_acc,
                    "base_results": base_logic_res, "ft_results": ft_logic_res},
        "planning":{"base": base_plan_acc, "ft": ft_plan_acc,
                    "delta": ft_plan_acc - base_plan_acc,
                    "base_results": base_plan_res, "ft_results": ft_plan_res},
    }
    save_path = os.path.join(CONFIG["output_dir"], "new_target_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")


if __name__ == "__main__":
    main()