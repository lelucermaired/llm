"""
Connect4 (四子棋) Benchmark Generator

生成带可证明最优解的Connect4评测集，用于测试五子棋LoRA微调的近域迁移能力。

棋盘: 7列 x 6行 (标准Connect4)
坐标: 列(col) 0-6, 行(row) 0-5, row=0为底部
落子规则: 重力落子,只能放在该列最底空位
胜利条件: 横/竖/斜四连

设计要点:
1. 使用minimax+alpha-beta搜索到终局,保证"正确答案"可证
2. 允许多个最优列并列(评测时任一匹配即正确)
3. 难度分级: easy(3-6步) / medium(7-12步) / hard(有明确威胁)
4. 输出格式与五子棋benchmark保持一致
"""

import json
import random
import os
from copy import deepcopy
from typing import List, Tuple, Optional

# ========== 配置 ==========
CONFIG = {
    "n_easy": 20,
    "n_medium": 20,
    "n_hard": 10,
    "search_depth": 8,  # minimax搜索深度,8层对50题生成量很快
    "output_path": "./datasets/connect4_benchmark/test.json",
    "seed": 42,
}

ROWS = 6
COLS = 7
EMPTY = 0
P1 = 1  # 先手 (X)
P2 = 2  # 后手 (O)


# ========== 棋盘核心逻辑 ==========
def new_board():
    return [[EMPTY] * COLS for _ in range(ROWS)]


def valid_moves(board):
    """返回所有可下的列号 (顶行空的列)"""
    return [c for c in range(COLS) if board[ROWS - 1][c] == EMPTY]


def drop(board, col, player):
    """在col列落子, 返回新board和落点row。无效则返回(None, -1)"""
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            new_b = deepcopy(board)
            new_b[r][col] = player
            return new_b, r
    return None, -1


def check_win(board, player):
    """检查player是否四连"""
    # 横
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == player for i in range(4)):
                return True
    # 竖
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == player for i in range(4)):
                return True
    # 正斜 /
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == player for i in range(4)):
                return True
    # 反斜 \
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == player for i in range(4)):
                return True
    return False


def is_full(board):
    return all(board[ROWS - 1][c] != EMPTY for c in range(COLS))


# ========== 局面评估(用于minimax非终局剪枝) ==========
def count_windows(board, player):
    """计算player的潜在连线数,用于非终局评分"""
    score = 0
    opp = P2 if player == P1 else P1

    def window_score(window):
        p_cnt = window.count(player)
        o_cnt = window.count(opp)
        e_cnt = window.count(EMPTY)
        if p_cnt == 4:
            return 10000
        if p_cnt == 3 and e_cnt == 1:
            return 50
        if p_cnt == 2 and e_cnt == 2:
            return 5
        if o_cnt == 3 and e_cnt == 1:
            return -80  # 对手威胁惩罚更大,强迫堵
        return 0

    # 横
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += window_score([board[r][c + i] for i in range(4)])
    # 竖
    for c in range(COLS):
        for r in range(ROWS - 3):
            score += window_score([board[r + i][c] for i in range(4)])
    # 正斜
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += window_score([board[r + i][c + i] for i in range(4)])
    # 反斜
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            score += window_score([board[r - i][c + i] for i in range(4)])

    # 中心控制加分
    center_count = sum(1 for r in range(ROWS) if board[r][3] == player)
    score += center_count * 3
    return score


# ========== Minimax + Alpha-Beta ==========
def minimax(board, depth, alpha, beta, maximizing_player, current_player):
    """
    maximizing_player: 我们要帮其决策的玩家(根节点的player)
    current_player: 当前轮到谁下
    返回: (最佳分数, 最佳列号)
    """
    valid = valid_moves(board)
    opp = P2 if current_player == P1 else P1

    # 终局检测: 先检查对手上一步是否赢了
    if check_win(board, opp):
        return (-100000 if current_player == maximizing_player else 100000), None
    if is_full(board):
        return 0, None
    if depth == 0:
        return count_windows(board, maximizing_player), None

    if current_player == maximizing_player:
        best_score = -float('inf')
        best_col = valid[0]
        # 优先搜索中心列(剪枝更高效)
        ordered = sorted(valid, key=lambda c: abs(c - 3))
        for col in ordered:
            new_b, _ = drop(board, col, current_player)
            score, _ = minimax(new_b, depth - 1, alpha, beta, maximizing_player, opp)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_col
    else:
        best_score = float('inf')
        best_col = valid[0]
        ordered = sorted(valid, key=lambda c: abs(c - 3))
        for col in ordered:
            new_b, _ = drop(board, col, current_player)
            score, _ = minimax(new_b, depth - 1, alpha, beta, maximizing_player, opp)
            if score < best_score:
                best_score = score
                best_col = col
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best_score, best_col


def find_optimal_columns(board, player, depth, tolerance=0):
    """
    找出所有最优列(允许tolerance分数内的并列最优)。
    返回 (最优分数, [最优列列表])
    """
    valid = valid_moves(board)
    scores = {}
    opp = P2 if player == P1 else P1

    for col in valid:
        new_b, _ = drop(board, col, player)
        # 立即赢的列分数最高
        if check_win(new_b, player):
            scores[col] = 99999
            continue
        # 递归搜索
        score, _ = minimax(new_b, depth - 1, -float('inf'), float('inf'), player, opp)
        scores[col] = score

    best_score = max(scores.values())
    # 允许tolerance内的列都算最优(处理评估函数误差)
    optimal_cols = [c for c, s in scores.items() if s >= best_score - tolerance]
    return best_score, sorted(optimal_cols), scores


# ========== 棋盘文本化 ==========
def board_to_string(board):
    """把棋盘渲染成人类可读的文本 (从顶行往底行打印)"""
    lines = []
    lines.append("列号: 0 1 2 3 4 5 6")
    lines.append("     -------------")
    for r in range(ROWS - 1, -1, -1):
        row_str = f"行{r}:  "
        for c in range(COLS):
            cell = board[r][c]
            if cell == EMPTY:
                row_str += ". "
            elif cell == P1:
                row_str += "X "
            else:
                row_str += "O "
        lines.append(row_str)
    return "\n".join(lines)


def build_instruction(board, player_to_move):
    """构造给模型的prompt"""
    player_mark = "X" if player_to_move == P1 else "O"
    opp_mark = "O" if player_to_move == P1 else "X"
    board_str = board_to_string(board)

    instruction = (
        "你正在玩四子棋(Connect4)。棋盘为7列x6行,采用重力落子规则(棋子会落到该列最底空位)。\n"
        "胜利条件: 横向、纵向或斜向连成4子即获胜。\n\n"
        f"当前棋盘(X先手, O后手):\n{board_str}\n\n"
        f"你执 {player_mark} 子,对手执 {opp_mark} 子。现在轮到你下。\n"
        "请分析局势,选择最佳落子列。\n\n"
        "输出格式: 只输出一个0到6之间的整数,表示你选择的列号,不要输出其他内容。"
    )
    return instruction


# ========== 题目生成 ==========
def random_playout(n_plies, rng):
    """随机走n_plies步,返回局面和下一手玩家"""
    board = new_board()
    player = P1
    for _ in range(n_plies):
        valid = valid_moves(board)
        if not valid:
            break
        # 80%随机,20%偏好中心列(让局面更真实)
        if rng.random() < 0.8:
            col = rng.choice(valid)
        else:
            col = min(valid, key=lambda c: abs(c - 3))
        new_b, _ = drop(board, col, player)
        if check_win(new_b, player):
            # 走到胜局就重来
            return random_playout(n_plies, rng)
        board = new_b
        player = P2 if player == P1 else P1
    if is_full(board):
        return random_playout(n_plies, rng)
    return board, player


def has_immediate_threat(board, player):
    """判断是否存在'必须应对'的威胁: 对手下一手能赢或自己下一手能赢"""
    opp = P2 if player == P1 else P1
    # 自己能立即赢
    for col in valid_moves(board):
        new_b, _ = drop(board, col, player)
        if check_win(new_b, player):
            return True
    # 对手下一手能赢
    for col in valid_moves(board):
        new_b, _ = drop(board, col, opp)
        if check_win(new_b, opp):
            return True
    return False


def generate_question(difficulty, rng, search_depth):
    """生成单道题,带最大尝试次数防止死循环"""
    max_attempts = 100
    for _ in range(max_attempts):
        if difficulty == "easy":
            n_plies = rng.randint(3, 6)
            board, player = random_playout(n_plies, rng)
            if has_immediate_threat(board, player):
                continue  # easy不要威胁局面
        elif difficulty == "medium":
            n_plies = rng.randint(7, 12)
            board, player = random_playout(n_plies, rng)
        elif difficulty == "hard":
            n_plies = rng.randint(8, 14)
            board, player = random_playout(n_plies, rng)
            if not has_immediate_threat(board, player):
                continue  # hard必须有威胁

        # 求解
        best_score, optimal_cols, all_scores = find_optimal_columns(
            board, player, search_depth, tolerance=0
        )

        if len(optimal_cols) == 0:
            continue

        # easy: 最优列越多越容易,不苛求单解
        # hard: 希望最优列少(强制正确决策),但允许1-2列
        if difficulty == "hard" and len(optimal_cols) > 2:
            continue

        return {
            "board": board,
            "player": player,
            "optimal_cols": optimal_cols,
            "all_scores": all_scores,
            "best_score": best_score,
            "difficulty": difficulty,
            "n_plies": n_plies,
        }
    return None


def build_dataset(config):
    rng = random.Random(config["seed"])
    questions = []

    spec = [
        ("easy", config["n_easy"]),
        ("medium", config["n_medium"]),
        ("hard", config["n_hard"]),
    ]

    qid = 0
    for difficulty, n in spec:
        print(f"\n生成 {difficulty} 难度题目 ({n}题)...")
        generated = 0
        while generated < n:
            q = generate_question(difficulty, rng, config["search_depth"])
            if q is None:
                print(f"  [警告] 第{generated+1}题生成失败,跳过")
                continue

            entry = {
                "id": f"c4_{qid:03d}",
                "difficulty": q["difficulty"],
                "instruction": build_instruction(q["board"], q["player"]),
                "output": str(q["optimal_cols"][0]),  # 训练/参考答案用第一个最优列
                "optimal_columns": q["optimal_cols"],  # 评测时任一匹配即可
                "meta": {
                    "n_plies": q["n_plies"],
                    "player_to_move": q["player"],
                    "best_score": q["best_score"],
                    "all_scores": {str(k): v for k, v in q["all_scores"].items()},
                    "board": q["board"],
                },
            }
            questions.append(entry)
            qid += 1
            generated += 1
            if generated % 5 == 0:
                print(f"  已完成 {generated}/{n}")

    return questions


# ========== 主流程 ==========
def main():
    os.makedirs(os.path.dirname(CONFIG["output_path"]), exist_ok=True)
    print(f"开始生成Connect4 benchmark,种子={CONFIG['seed']},搜索深度={CONFIG['search_depth']}")

    dataset = build_dataset(CONFIG)

    with open(CONFIG["output_path"], "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 已生成 {len(dataset)} 道题,保存至: {CONFIG['output_path']}")

    # 打印样例
    print("\n" + "=" * 60)
    print("样例题目预览:")
    print("=" * 60)
    for d in ["easy", "medium", "hard"]:
        sample = next((q for q in dataset if q["difficulty"] == d), None)
        if sample:
            print(f"\n【{d.upper()}】 id={sample['id']}")
            print(sample["instruction"])
            print(f"最优列: {sample['optimal_columns']}")
            print(f"参考答案: {sample['output']}")
            print("-" * 60)


if __name__ == "__main__":
    main()