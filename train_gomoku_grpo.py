"""
train_gomoku_grpo.py

五子棋GRPO强化学习训练
理论依据：Chu et al. (2025) "SFT Memorizes, RL Generalizes"
          ViGaL (2025) 游戏RL→通用推理正向迁移

奖励函数设计：
  1. 合法性奖励：落子坐标合法 +1.0，不合法 -1.0
  2. 格式奖励：输出包含"最佳落子：XX"格式 +0.2
  3. 威胁奖励：落子位置有防守/进攻价值 +0.3（可选）

显存优化（12GB）：
  - 4bit量化base model
  - LoRA adapter（r=8）
  - 每次只生成2个rollout
  - gradient_checkpointing
"""

import os, re, json, copy
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0+PTX;8.6+PTX;8.9+PTX'

import torch
import numpy as np
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import GRPOConfig, GRPOTrainer

import warnings; warnings.filterwarnings("ignore")

_orig = torch.load
def _patched(*a, **kw): kw['weights_only'] = False; return _orig(*a, **kw)
torch.load = _patched

# ==================== 配置 ====================

CONFIG = {
    "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
    "dataset_path":    "./datasets/real_games_v2/train.json",
    "output_dir":      "./checkpoints/qwen-gomoku-grpo",
    "cache_dir":       "./cache",
}

BOARD_SIZE = 15
COL_LETTERS = "ABCDEFGHIJKLMNO"
BLACK, WHITE, EMPTY = 1, 2, 0


# ==================== 五子棋工具 ====================

def parse_board(instruction):
    """从instruction解析棋盘状态"""
    board = [[EMPTY]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    col_header_match = re.search(r'([A-O](?:\s+[A-O])+)', instruction)
    if not col_header_match:
        return None, None
    cols = col_header_match.group(1).split()
    col_offset = COL_LETTERS.index(cols[0])
    row_pattern = re.compile(r'^\s*(\d{1,2})([\s·●○]+)', re.MULTILINE)
    for m in row_pattern.finditer(instruction):
        row_num = int(m.group(1))
        if row_num < 1 or row_num > BOARD_SIZE:
            continue
        r = row_num - 1
        cells = re.findall(r'[·●○]', m.group(2))
        for ci, cell in enumerate(cells):
            c = col_offset + ci
            if c >= BOARD_SIZE:
                break
            if cell == '●':
                board[r][c] = BLACK
            elif cell == '○':
                board[r][c] = WHITE
    black_count = sum(board[r][c]==BLACK for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    white_count = sum(board[r][c]==WHITE for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    current = BLACK if black_count <= white_count else WHITE
    return board, current


def parse_move(response):
    """从回答中提取落子坐标"""
    patterns = [
        r'最佳落子[：:]\s*([A-O]\d{1,2})',
        r'落在\s*([A-O]\d{1,2})',
        r'([A-O]\d{1,2})',
    ]
    for pat in patterns:
        m = re.search(pat, response)
        if m:
            move_str = m.group(1).strip()
            col = COL_LETTERS.index(move_str[0])
            row = int(move_str[1:]) - 1
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                return row, col
    return None


def in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def evaluate_move_value(board, r, c, color):
    """简单评估落子价值（用于威胁奖励）"""
    DIRECTIONS = [(1,0),(0,1),(1,1),(1,-1)]
    opp = WHITE if color == BLACK else BLACK
    score = 0

    temp = copy.deepcopy(board)
    temp[r][c] = color

    def count_line(b, sr, sc, dr, dc, col):
        cnt = 0
        nr, nc = sr, sc
        while in_bounds(nr, nc) and b[nr][nc] == col:
            cnt += 1; nr += dr; nc += dc
        return cnt

    for dr, dc in DIRECTIONS:
        fwd = count_line(temp, r, c, dr, dc, color)
        bwd = count_line(temp, r-dr, c-dc, -dr, -dc, color)
        total = fwd + bwd - 1
        if total >= 5: score += 10.0
        elif total == 4: score += 3.0
        elif total == 3: score += 1.0

    temp2 = copy.deepcopy(board)
    temp2[r][c] = opp
    for dr, dc in DIRECTIONS:
        fwd = count_line(temp2, r, c, dr, dc, opp)
        bwd = count_line(temp2, r-dr, c-dc, -dr, -dc, opp)
        total = fwd + bwd - 1
        if total >= 5: score += 9.0
        elif total == 4: score += 2.5
        elif total == 3: score += 0.8

    return score


# ==================== 奖励函数 ====================

def compute_reward(prompt, response):
    """
    计算GRPO奖励
    返回 [-1.0, 1.5] 范围的奖励值

    奖励组成：
    - 格式奖励：输出包含合法坐标格式 +0.2
    - 合法性奖励：落子位置是空位 +0.8
    - 价值奖励：落子有实际威胁价值 +0.5（归一化）
    - 非法惩罚：落子位置非空 -1.0
    """
    # 解析棋盘
    board, color = parse_board(prompt)
    if board is None:
        return 0.0

    # 解析落子
    move = parse_move(response)

    # 格式奖励：能解析出坐标就给
    if move is None:
        return -0.5  # 没有给出落子坐标

    r, c = move
    format_reward = 0.2

    # 合法性检查：落点必须在棋盘内且为空位
    if not in_bounds(r, c):
        return format_reward - 1.0

    if board[r][c] != EMPTY:
        return format_reward - 1.0  # 落在已有子的位置

    # 合法性奖励
    legality_reward = 0.8

    # 价值奖励：评估落子的进攻/防守价值
    value = evaluate_move_value(board, r, c, color)
    # 归一化：value通常在0-15之间，映射到0-0.5
    value_reward = min(value / 20.0, 0.5)

    total = format_reward + legality_reward + value_reward
    return float(total)


def reward_fn(prompts, completions, **kwargs):
    """GRPO奖励函数接口"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # prompt可能是list of messages或string
        if isinstance(prompt, list):
            # chat format
            prompt_text = " ".join([m.get("content","") for m in prompt])
        else:
            prompt_text = str(prompt)

        if isinstance(completion, list):
            completion_text = " ".join([m.get("content","") for m in completion
                                         if m.get("role") == "assistant"])
        else:
            completion_text = str(completion)

        reward = compute_reward(prompt_text, completion_text)
        rewards.append(reward)
    return rewards


# ==================== 数据准备 ====================

def prepare_dataset(dataset_path, tokenizer, max_samples=500):
    """准备GRPO训练数据（只需要prompt，不需要标准答案）"""
    print(f"加载数据集：{dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    # 只取前max_samples条，GRPO不需要太多
    data = data[:max_samples]

    prompts = []
    for item in data:
        # 转为chat格式
        # 截断过长的instruction，只保留棋盘状态部分
        instr = item["instruction"][:800]  # 限制800字符
        messages = [{"role": "user", "content": instr}]
        prompts.append(messages)

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"✅ 数据集大小：{len(dataset)}")
    return dataset


# ==================== 主流程 ====================

def main():
    print("=" * 65)
    print("五子棋 GRPO 强化学习训练")
    print("奖励：落子合法性 + 棋形价值")
    print(f"理论：RL generalizes, SFT memorizes (Chu et al. 2025)")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"], trust_remote_code=True,
        cache_dir=CONFIG["cache_dir"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO需要left padding

    # 加载模型（4bit量化）
    print("\n加载模型（4bit量化+LoRA）...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model_path"],
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CONFIG["cache_dir"],
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 准备数据
    dataset = prepare_dataset(CONFIG["dataset_path"], tokenizer)

    # GRPO训练配置（针对12GB显存优化）
    grpo_config = GRPOConfig(
        output_dir=CONFIG["output_dir"],
        # 生成参数（trl 1.0.0参数名）
        max_completion_length=256,  # 最大生成长度
        num_generations=2,          # 每个prompt生成2个rollout（显存限制）
        # 训练参数
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        # GRPO特有参数
        beta=0.01,                  # KL惩罚系数
        # 优化
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,   # Windows兼容
        # 日志
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        # 生成配置
        temperature=0.9,
        top_p=0.95,
        remove_unused_columns=False,
    )

    print("\n开始GRPO训练...")
    print(f"  num_generations=2（每prompt采样2个rollout）")
    print(f"  奖励函数：合法性+棋形价值")
    print(f"  beta={grpo_config.beta}（KL约束）")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    result = trainer.train()

    print(f"\n✅ GRPO训练完成！")
    print(f"   最终loss：{result.training_loss:.4f}")

    # 保存模型
    final_dir = os.path.join(CONFIG["output_dir"], "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"   模型保存至：{final_dir}")

    print("\n下一步：运行eval_all_50.py加入grpo模型，对比SFT vs RL的OOD迁移差异")


if __name__ == "__main__":
    main()