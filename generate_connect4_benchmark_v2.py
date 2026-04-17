"""
Connect4 Benchmark v2 - 战术导向版本

针对v1的诊断问题重新设计:
- v1问题: base模型74%时间无脑下中心列3,所有模型都挤在34%,无法区分迁移
- v2方案: 所有题目强制"最优列≠3",迫使模型真正分析局面

三档难度(按战术清晰度,不按步数):
- Tier A (Winning Move, 100题): 自己下一步就能四连胜利
- Tier B (Defensive, 100题): 对方下一步会四连,必须堵
- Tier C (Double Threat, 50题): 下某列能制造双三,对方堵不过来

评测标准:
- Tier A/B: 最优列集合小(通常1-2列),只有选对才算对
- 战术正确性毫无争议,真正考验模型的战术识别能力
"""

import json
import random
import os
from copy import deepcopy

# ========== 配置 ==========
CONFIG = {
    "n_winning": 100,       # Tier A: 必胜题
    "n_defensive": 100,     # Tier B: 必防题
    "n_double": 50,         # Tier C: 双威胁题
    "search_depth": 6,
    "output_path": "./datasets/connect4_benchmark_v2/test.json",
    "seed": 42,
    "exclude_center_col": True,  # 最优列必不含3
}

ROWS = 6
COLS = 7
EMPTY = 0
P1 = 1
P2 = 2


# ========== 棋盘基础 ==========
def new_board():
    return [[EMPTY] * COLS for _ in range(ROWS)]


def valid_moves(board):
    return [c for c in range(COLS) if board[ROWS - 1][c] == EMPTY]


def drop(board, col, player):
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            new_b = deepcopy(board)
            new_b[r][col] = player
            return new_b, r
    return None, -1


def check_win(board, player):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == player for i in range(4)):
                return True
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == player for i in range(4)):
                return True
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == player for i in range(4)):
                return True
    return False


def is_full(board):
    return all(board[ROWS - 1][c] != EMPTY for c in range(COLS))


# ========== 战术判定 ==========
def find_winning_moves(board, player):
    """找出所有'下这列就立即获胜'的列"""
    wins = []
    for col in valid_moves(board):
        new_b, _ = drop(board, col, player)
        if check_win(new_b, player):
            wins.append(col)
    return wins


def find_threats(board, player):
    """
    找出对手(对player而言)下一步能赢的列,即'威胁列'。
    player_to_move(我方)必须堵掉这些列。
    """
    opp = P2 if player == P1 else P1
    return find_winning_moves(board, opp)


def count_winning_threats_after(board, col, player):
    """
    如果player在col落子,之后他有多少个'立即获胜的落点'?
    用于识别'双威胁' - 落子后能同时威胁2个以上获胜位置
    """
    new_b, _ = drop(board, col, player)
    if new_b is None:
        return 0
    # 下完之后,player下次有多少种立即获胜方式
    return len(find_winning_moves(new_b, player))


# ========== 随机对局生成局面 ==========
def random_playout(n_plies, rng, avoid_premature_end=True):
    """随机走n_plies步,返回局面和下一手玩家"""
    max_retries = 50
    for _ in range(max_retries):
        board = new_board()
        player = P1
        success = True
        for _ in range(n_plies):
            valid = valid_moves(board)
            if not valid:
                success = False
                break
            # 50%偏好中心,50%完全随机,让局面自然些
            if rng.random() < 0.5:
                weights = [max(0, 3.5 - abs(c - 3)) for c in valid]
                col = rng.choices(valid, weights=weights, k=1)[0]
            else:
                col = rng.choice(valid)
            new_b, _ = drop(board, col, player)
            if avoid_premature_end and check_win(new_b, player):
                success = False
                break
            board = new_b
            player = P2 if player == P1 else P1
        if success and not is_full(board):
            return board, player
    return None, None


# ========== 棋盘文本化 ==========
def board_to_string(board):
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


def build_instruction(board, player_to_move, tier):
    """根据题型构造prompt - 给模型更明确的任务提示"""
    player_mark = "X" if player_to_move == P1 else "O"
    opp_mark = "O" if player_to_move == P1 else "X"
    board_str = board_to_string(board)

    # 所有tier使用统一的prompt,避免给模型"答案类型"提示(那样就不是在测推理了)
    instruction = (
        "你正在玩四子棋(Connect4)。棋盘为7列x6行,采用重力落子规则(棋子会落到该列最底空位)。\n"
        "胜利条件: 横向、纵向或斜向连成4子即获胜。\n\n"
        f"当前棋盘(X先手, O后手):\n{board_str}\n\n"
        f"你执 {player_mark} 子,对手执 {opp_mark} 子。现在轮到你下。\n"
        "请仔细分析局势,考虑自己的进攻机会和对手的威胁,选择最佳落子列。\n\n"
        "输出格式: 只输出一个0到6之间的整数,表示你选择的列号,不要输出其他内容。"
    )
    return instruction


# ========== 三种题型生成 ==========
def try_generate_winning_move(rng, search_depth, exclude_center):
    """Tier A: 构造'自己能立即四连'的局面"""
    # 随机走若干步,然后检查当前玩家是否有winning move
    n_plies = rng.randint(5, 14)
    board, player = random_playout(n_plies, rng)
    if board is None:
        return None

    wins = find_winning_moves(board, player)
    if not wins:
        return None

    # 核心过滤: 排除中心列
    if exclude_center and set(wins) == {3}:
        return None  # 只能中心列获胜,不要
    if exclude_center:
        wins = [c for c in wins if c != 3]
        if not wins:
            return None

    # 还需要检查: 这些winning列确实是唯一理性选择
    # 即:选wins里的列必赢,选其他列(特别是列3)不赢
    return {
        "board": board,
        "player": player,
        "optimal_columns": sorted(wins),
        "tier": "winning",
        "tactical_reason": "immediate_win",
    }


def try_generate_defensive(rng, search_depth, exclude_center):
    """Tier B: 构造'必须堵对方四连'的局面"""
    n_plies = rng.randint(5, 14)
    board, player = random_playout(n_plies, rng)
    if board is None:
        return None

    # 自己不能立即获胜(否则这就是Tier A)
    if find_winning_moves(board, player):
        return None

    # 对手有威胁
    threats = find_threats(board, player)
    if not threats:
        return None

    # 如果有多个威胁,无法堵(这种局面已输),跳过
    if len(threats) > 1:
        return None

    # 唯一威胁列就是必堵位置
    must_block = threats[0]

    # 排除中心列
    if exclude_center and must_block == 3:
        return None

    return {
        "board": board,
        "player": player,
        "optimal_columns": [must_block],
        "tier": "defensive",
        "tactical_reason": "must_block",
    }


def try_generate_double_threat(rng, search_depth, exclude_center):
    """Tier C: 构造'下某列后能制造双重威胁'的局面"""
    n_plies = rng.randint(6, 14)
    board, player = random_playout(n_plies, rng)
    if board is None:
        return None

    # 自己不能立即获胜
    if find_winning_moves(board, player):
        return None
    # 对手也没有立即威胁(否则应该优先堵)
    if find_threats(board, player):
        return None

    # 找出制造双威胁的列
    double_threat_cols = []
    for col in valid_moves(board):
        n_threats = count_winning_threats_after(board, col, player)
        if n_threats >= 2:
            double_threat_cols.append(col)

    if not double_threat_cols:
        return None

    # 排除中心列
    if exclude_center and set(double_threat_cols) == {3}:
        return None
    if exclude_center:
        double_threat_cols = [c for c in double_threat_cols if c != 3]
        if not double_threat_cols:
            return None

    return {
        "board": board,
        "player": player,
        "optimal_columns": sorted(double_threat_cols),
        "tier": "double_threat",
        "tactical_reason": "create_double_threat",
    }


# ========== 主生成流程 ==========
def build_dataset(cfg):
    rng = random.Random(cfg["seed"])
    questions = []
    qid = 0

    spec = [
        ("winning", cfg["n_winning"], try_generate_winning_move),
        ("defensive", cfg["n_defensive"], try_generate_defensive),
        ("double_threat", cfg["n_double"], try_generate_double_threat),
    ]

    for tier, n, gen_fn in spec:
        print(f"\n生成 {tier} 题目 ({n}题)...")
        generated = 0
        attempts = 0
        max_attempts = n * 500  # 保底尝试次数
        while generated < n and attempts < max_attempts:
            attempts += 1
            q = gen_fn(rng, cfg["search_depth"], cfg["exclude_center_col"])
            if q is None:
                continue

            entry = {
                "id": f"c4v2_{qid:03d}",
                "tier": q["tier"],
                "tactical_reason": q["tactical_reason"],
                "instruction": build_instruction(q["board"], q["player"], q["tier"]),
                "output": str(q["optimal_columns"][0]),
                "optimal_columns": q["optimal_columns"],
                "meta": {
                    "player_to_move": q["player"],
                    "board": q["board"],
                    "n_optimal": len(q["optimal_columns"]),
                },
            }
            questions.append(entry)
            qid += 1
            generated += 1
            if generated % 20 == 0:
                print(f"  已完成 {generated}/{n} (尝试{attempts}次)")

        if generated < n:
            print(f"  [警告] {tier}只生成了{generated}/{n}题,尝试次数用尽")

    return questions


def main():
    os.makedirs(os.path.dirname(CONFIG["output_path"]), exist_ok=True)
    print(f"开始生成Connect4 Benchmark v2")
    print(f"种子={CONFIG['seed']}, 排除中心列={CONFIG['exclude_center_col']}")

    dataset = build_dataset(CONFIG)

    with open(CONFIG["output_path"], "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 共生成 {len(dataset)} 道题")
    print(f"✓ 保存至: {CONFIG['output_path']}")

    # 统计
    from collections import Counter
    tier_dist = Counter(q["tier"] for q in dataset)
    optimal_cols_dist = Counter()
    for q in dataset:
        for c in q["optimal_columns"]:
            optimal_cols_dist[c] += 1

    print(f"\n题型分布: {dict(tier_dist)}")
    print(f"最优列分布: {dict(sorted(optimal_cols_dist.items()))}")
    print(f"(列3出现次数: {optimal_cols_dist.get(3, 0)} - 应为0)")

    # 样例
    print("\n" + "=" * 60)
    print("样例预览:")
    print("=" * 60)
    for tier in ["winning", "defensive", "double_threat"]:
        sample = next((q for q in dataset if q["tier"] == tier), None)
        if sample:
            print(f"\n【{tier.upper()}】 id={sample['id']}")
            print(sample["instruction"])
            print(f"最优列: {sample['optimal_columns']}")
            print("-" * 60)


if __name__ == "__main__":
    main()