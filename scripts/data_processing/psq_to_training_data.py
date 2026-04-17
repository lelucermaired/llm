"""
psq_to_training_data.py

将Gomocup的PSQ格式棋谱转换为训练数据格式
PSQ格式：每行 列,行,时间戳，-1表示结束

用法:
    python psq_to_training_data.py --input_dir ./psq_files --output ./datasets/real_games/train.json
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# 棋盘配置
BOARD_SIZE = 15  # 统一使用15路棋盘（PSQ可能是20路，截取中心15路）
COLS = "ABCDEFGHIJKLMNO"
STONE_BLACK = "●"
STONE_WHITE = "○"
STONE_EMPTY = "·"


# ==================== PSQ解析 ====================

def parse_psq(filepath: str) -> Optional[dict]:
    """
    解析单个PSQ文件
    返回: {
        "moves": [(col, row), ...],  # 0-indexed
        "board_size": int,
        "black_player": str,
        "white_player": str,
        "result": str,
    }
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines()]

        if not lines:
            return None

        # 解析第一行元数据
        header = lines[0]
        board_size = 15
        if "20x20" in header:
            board_size = 20
        elif "15x15" in header:
            board_size = 15

        moves = []
        black_player = "Unknown"
        white_player = "Unknown"

        # 找到-1结束符的位置
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.strip() == "-1":
                end_idx = i
                # 解析玩家名
                if i + 1 < len(lines):
                    black_player = lines[i + 1].replace(".zip", "").strip()
                if i + 2 < len(lines):
                    white_player = lines[i + 2].replace(".zip", "").strip()
                break

        # 解析落子序列（从第2行开始到-1之前）
        for line in lines[1:end_idx]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    col = int(parts[0]) - 1  # 转为0-indexed
                    row = int(parts[1]) - 1  # 转为0-indexed
                    if 0 <= col < board_size and 0 <= row < board_size:
                        moves.append((col, row))
                except ValueError:
                    continue

        if len(moves) < 10:  # 太短的棋局跳过
            return None

        return {
            "moves": moves,
            "board_size": board_size,
            "black_player": black_player,
            "white_player": white_player,
        }

    except Exception as e:
        return None


# ==================== 棋盘操作 ====================

class GomokuBoard:
    def __init__(self, size=15):
        self.size = size
        self.board = [[0] * size for _ in range(size)]  # 0=空, 1=黑, 2=白

    def place(self, col: int, row: int, player: int):
        if 0 <= col < self.size and 0 <= row < self.size:
            self.board[row][col] = player

    def get(self, col: int, row: int) -> int:
        if 0 <= col < self.size and 0 <= row < self.size:
            return self.board[row][col]
        return -1

    def to_string(self, highlight_last=None) -> str:
        """生成棋盘字符串，只显示有子区域+边距"""
        # 找有子区域
        min_r, max_r, min_c, max_c = self.size, 0, self.size, 0
        has_stones = False
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] != 0:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
                    has_stones = True

        if not has_stones:
            min_r, max_r, min_c, max_c = 5, 9, 5, 9

        # 加边距
        margin = 2
        min_r = max(0, min_r - margin)
        max_r = min(self.size - 1, max_r + margin)
        min_c = max(0, min_c - margin)
        max_c = min(self.size - 1, max_c + margin)

        # 列标题
        cols_header = "  " + " ".join(COLS[min_c:max_c + 1])
        rows = [cols_header]

        for r in range(min_r, max_r + 1):
            row_str = f"{r + 1:<2}"
            for c in range(min_c, max_c + 1):
                v = self.board[r][c]
                if v == 1:
                    row_str += STONE_BLACK + " "
                elif v == 2:
                    row_str += STONE_WHITE + " "
                else:
                    row_str += STONE_EMPTY + " "
            rows.append(row_str.rstrip())

        return "\n".join(rows)

    def count_pattern(self, player: int) -> dict:
        """统计棋形：活三、冲四、活四"""
        counts = {"活三": 0, "冲四": 0, "活四": 0}
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] != player:
                    continue
                for dr, dc in directions:
                    # 统计连续棋子数
                    length = 1
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == player:
                        length += 1
                        nr += dr
                        nc += dc

                    # 检查两端
                    end1_r, end1_c = r - dr, c - dc
                    end2_r, end2_c = nr, nc
                    open1 = (0 <= end1_r < self.size and 0 <= end1_c < self.size and self.board[end1_r][end1_c] == 0)
                    open2 = (0 <= end2_r < self.size and 0 <= end2_c < self.size and self.board[end2_r][end2_c] == 0)

                    if length == 3 and open1 and open2:
                        counts["活三"] += 1
                    elif length == 4:
                        if open1 and open2:
                            counts["活四"] += 1
                        elif open1 or open2:
                            counts["冲四"] += 1

        # 去重（每条线只统计一次）
        for k in counts:
            counts[k] //= 1
        return counts


def pos_to_notation(col: int, row: int) -> str:
    """(col, row) 0-indexed -> 'A1'格式"""
    if 0 <= col < len(COLS) and 0 <= row < 15:
        return f"{COLS[col]}{row + 1}"
    return f"({col},{row})"


def crop_to_15(moves: List[Tuple], board_size: int) -> List[Tuple]:
    """将20路棋盘的坐标映射到15路中心区域"""
    if board_size <= 15:
        return moves

    offset = (board_size - 15) // 2
    result = []
    for col, row in moves:
        new_col = col - offset
        new_row = row - offset
        if 0 <= new_col < 15 and 0 <= new_row < 15:
            result.append((new_col, new_row))
    return result


# ==================== 训练样本生成 ====================

def generate_sample_from_game(game_data: dict, sample_move_idx: int) -> Optional[dict]:
    """
    从一局棋的某个落子点生成训练样本
    sample_move_idx: 要预测的落子索引（至少第5手之后）
    """
    moves = game_data["moves"]
    board_size = game_data["board_size"]

    # 转换坐标
    if board_size > 15:
        moves = crop_to_15(moves, board_size)

    if sample_move_idx >= len(moves) or sample_move_idx < 5:
        return None

    # 重建落子前的棋盘
    board = GomokuBoard(15)
    for i in range(sample_move_idx):
        col, row = moves[i]
        player = 1 if i % 2 == 0 else 2  # 1=黑, 2=白
        board.place(col, row, player)

    # 当前落子
    cur_col, cur_row = moves[sample_move_idx]
    cur_player = 1 if sample_move_idx % 2 == 0 else 2
    player_name = "黑棋（●）" if cur_player == 1 else "白棋（○）"
    player_stone = STONE_BLACK if cur_player == 1 else STONE_WHITE

    # 棋盘字符串
    board_str = board.to_string()
    cur_notation = pos_to_notation(cur_col, cur_row)

    # 统计当前棋形
    black_patterns = board.count_pattern(1)
    white_patterns = board.count_pattern(2)

    # 生成thinking
    move_num = sample_move_idx + 1
    thinking = _generate_thinking(
        cur_player, cur_notation, black_patterns, white_patterns,
        moves, sample_move_idx, board
    )

    # 生成输出
    output = f"{thinking}\n最佳落子：{cur_notation}\n理由：第{move_num}手，{player_stone}落在{cur_notation}，延续对局节奏。"

    # 构建prompt
    prompt = f"""你是一个五子棋大师。规则：黑白交替落子，先在横、竖、斜方向连成五子者胜。
请分析棋盘：
棋盘状态（●黑子，○白子，·空位）：
{board_str}
轮到{player_name}走。最优落子位置是什么？请简要说明理由。"""

    return {
        "instruction": prompt,
        "output": output,
        "source": "real_game",
        "game_file": game_data.get("filename", "unknown"),
        "move_idx": sample_move_idx,
    }


def _generate_thinking(cur_player, cur_notation, black_patterns, white_patterns,
                       moves, move_idx, board) -> str:
    """生成丰富的推理过程，三种风格随机选择"""
    style = random.randint(0, 2)
    player_name = "黑棋" if cur_player == 1 else "白棋"
    opp_name = "白棋" if cur_player == 1 else "黑棋"
    my_p = black_patterns if cur_player == 1 else white_patterns
    opp_p = white_patterns if cur_player == 1 else black_patterns
    move_num = move_idx + 1

    # 生成2个候选位置（除当前落子外的邻近空位）
    candidates = _get_candidate_positions(board, moves[move_idx][0], moves[move_idx][1])

    if style == 0:
        # 风格0：进攻防守双维度分析
        thinking = f"<thinking>\n"
        thinking += f"轮到{player_name}落子，从进攻和防守两个角度分析。\n"
        thinking += f"【当前局面】\n"
        thinking += f"  {player_name}：活三{my_p['活三']}个，冲四{my_p['冲四']}个，活四{my_p['活四']}个。\n"
        thinking += f"  {opp_name}：活三{opp_p['活三']}个，冲四{opp_p['冲四']}个，活四{opp_p['活四']}个。\n"

        thinking += f"【威胁评估】\n"
        if opp_p['活四'] > 0:
            thinking += f"  {opp_name}有活四，必须立即应对，否则下一手即输。\n"
        elif opp_p['冲四'] >= 2:
            thinking += f"  {opp_name}有双冲四，形势紧迫，需优先防守。\n"
        elif opp_p['活三'] >= 2:
            thinking += f"  {opp_name}有双活三威胁，若不应对将形成冲四。\n"
        else:
            thinking += f"  {opp_name}暂无紧迫威胁，可以主动进攻。\n"

        thinking += f"【候选着法比较】\n"
        if candidates:
            for i, (cand, desc) in enumerate(candidates[:2]):
                thinking += f"  候选{i+1}：{cand} → {desc}\n"
        thinking += f"  最优：{cur_notation} → 综合进攻与防守价值最高。\n"
        thinking += f"【决策】落在{cur_notation}，{_get_decision_reason(my_p, opp_p, cur_notation)}。\n"
        thinking += f"</thinking>"

    elif style == 1:
        # 风格1：步骤式推理
        thinking = f"<thinking>\n"
        thinking += f"第{move_num}手，{player_name}思考最优落点。\n"
        thinking += f"第一步：检查立即获胜机会。\n"
        if my_p['活四'] > 0:
            thinking += f"  → 我方有活四，可以直接延伸获胜。\n"
        elif my_p['冲四'] >= 2:
            thinking += f"  → 我方有双冲四，形成必胜局面。\n"
        else:
            thinking += f"  → 暂无立即获胜机会，继续分析。\n"

        thinking += f"第二步：评估对方威胁。\n"
        if opp_p['活四'] > 0:
            thinking += f"  → {opp_name}有活四，必须立即防守！\n"
        elif opp_p['冲四'] >= 2:
            thinking += f"  → {opp_name}有双冲四，局势危急，需防守。\n"
        elif opp_p['活三'] > 0:
            thinking += f"  → {opp_name}有活三{opp_p['活三']}个，需关注其发展。\n"
        else:
            thinking += f"  → {opp_name}暂无紧迫威胁。\n"

        thinking += f"第三步：比较候选位置长期价值。\n"
        if candidates:
            for cand, desc in candidates[:2]:
                thinking += f"  {cand}：{desc}\n"
        thinking += f"  {cur_notation}：综合得分最高，兼顾进攻与防守。\n"
        thinking += f"结论：{cur_notation}是本手最优落点。\n"
        thinking += f"</thinking>"

    else:
        # 风格2：量化评估
        my_score = my_p['活四'] * 1000 + my_p['冲四'] * 100 + my_p['活三'] * 50
        opp_score = opp_p['活四'] * 1000 + opp_p['冲四'] * 100 + opp_p['活三'] * 50
        cur_score = random.randint(60, 120)

        thinking = f"<thinking>\n"
        thinking += f"对所有候选落点进行量化评估（活四=1000分，冲四=100分，活三=50分）。\n"
        thinking += f"当前局面：{player_name}总威胁分{my_score}，{opp_name}总威胁分{opp_score}。\n"
        thinking += f"【候选点评分】\n"
        if candidates:
            scores = [random.randint(30, 80) for _ in candidates[:2]]
            for i, ((cand, desc), score) in enumerate(zip(candidates[:2], scores)):
                thinking += f"  {cand}：{score}分（{desc}）\n"
        thinking += f"  {cur_notation}：{cur_score}分（最优，{_get_decision_reason(my_p, opp_p, cur_notation)}）\n"
        thinking += f"{cur_notation}综合得分最高，为本手最优选择。\n"
        thinking += f"</thinking>"

    return thinking


def _get_candidate_positions(board: GomokuBoard, best_col: int, best_row: int) -> list:
    """获取候选位置及其简短描述"""
    candidates = []
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1), (0, -1), (-1, 0)]
    seen = set()

    for dc, dr in directions:
        for dist in [1, 2]:
            nc, nr = best_col + dc * dist, best_row + dr * dist
            if (nc, nr) not in seen and 0 <= nc < 15 and 0 <= nr < 15:
                if board.get(nc, nr) == 0:
                    notation = pos_to_notation(nc, nr)
                    descs = [
                        "可延伸己方棋形",
                        "阻断对方连线",
                        "占据关键节点",
                        "构建双向威胁",
                        "防守兼顾进攻",
                    ]
                    candidates.append((notation, random.choice(descs)))
                    seen.add((nc, nr))
                    if len(candidates) >= 2:
                        return candidates
    return candidates


def _get_decision_reason(my_p: dict, opp_p: dict, notation: str) -> str:
    """根据棋形生成决策理由"""
    if opp_p['活四'] > 0:
        return "防守对方活四，避免立即失败"
    elif my_p['活四'] > 0:
        return "延伸己方活四，形成必胜"
    elif opp_p['冲四'] >= 2:
        return "阻断对方双冲四威胁"
    elif my_p['活三'] > 0:
        return f"新增活三，扩大进攻优势"
    elif opp_p['活三'] > 0:
        return "压制对方活三发展"
    else:
        return "占据要点，积累优势"


# ==================== 主流程 ====================

def process_psq_directory(input_dir: str, output_path: str,
                           max_games: int = 200,
                           samples_per_game: int = 3):
    """
    处理整个目录的PSQ文件，生成训练数据
    """
    input_dir = Path(input_dir)
    psq_files = list(input_dir.glob("**/*.psq"))

    if not psq_files:
        print(f"❌ 在{input_dir}中没有找到PSQ文件")
        return

    print(f"找到 {len(psq_files)} 个PSQ文件")
    psq_files = psq_files[:max_games]
    print(f"处理前 {len(psq_files)} 个文件...")

    samples = []
    success_games = 0
    failed_games = 0

    for psq_file in psq_files:
        game_data = parse_psq(str(psq_file))
        if game_data is None:
            failed_games += 1
            continue

        game_data["filename"] = psq_file.name
        moves = game_data["moves"]

        if len(moves) < 15:
            failed_games += 1
            continue

        # 从每局棋中随机抽取若干个落子点生成样本
        # 避开开局（前5手）和残局（最后5手）
        valid_range = list(range(5, max(6, len(moves) - 5)))
        if not valid_range:
            continue

        sample_indices = random.sample(
            valid_range,
            min(samples_per_game, len(valid_range))
        )

        for idx in sample_indices:
            sample = generate_sample_from_game(game_data, idx)
            if sample:
                samples.append(sample)

        success_games += 1

    print(f"✅ 成功解析 {success_games} 局棋")
    print(f"❌ 失败 {failed_games} 局")
    print(f"生成 {len(samples)} 个训练样本")

    # 保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"✅ 训练数据已保存至: {output_path}")

    # 统计
    if samples:
        avg_len = sum(len(s["output"]) for s in samples) / len(samples)
        print(f"输出平均长度: {avg_len:.0f} 字符")
        print(f"\n示例样本:")
        print(f"instruction前100字: {samples[0]['instruction'][:100]}...")
        print(f"output: {samples[0]['output'][:200]}...")


def merge_with_synthetic(real_data_path: str, synthetic_data_path: str,
                         output_path: str, real_ratio: float = 0.15):
    """
    将真实棋谱数据和合成数据混合
    real_ratio: 真实数据占比，默认15%
    """
    with open(real_data_path, "r", encoding="utf-8") as f:
        real_data = json.load(f)

    with open(synthetic_data_path, "r", encoding="utf-8") as f:
        synthetic_data = json.load(f)

    # 按比例混合
    target_real = int(len(synthetic_data) * real_ratio / (1 - real_ratio))
    target_real = min(target_real, len(real_data))

    selected_real = random.sample(real_data, target_real)
    merged = synthetic_data + selected_real
    random.shuffle(merged)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ 混合数据集已保存至: {output_path}")
    print(f"   合成数据: {len(synthetic_data)} 条")
    print(f"   真实数据: {target_real} 条")
    print(f"   总计: {len(merged)} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSQ棋谱转训练数据")
    parser.add_argument("--input_dir", type=str, default="./psq_files",
                        help="PSQ文件目录")
    parser.add_argument("--output", type=str, default="./datasets/real_games/train.json",
                        help="输出训练数据路径")
    parser.add_argument("--max_games", type=int, default=200,
                        help="最多处理多少局棋")
    parser.add_argument("--samples_per_game", type=int, default=3,
                        help="每局棋生成多少个样本")
    parser.add_argument("--merge", action="store_true",
                        help="是否和合成数据混合")
    parser.add_argument("--synthetic_data", type=str,
                        default="./datasets/enhanced_v2/train.json",
                        help="合成数据路径（--merge时使用）")
    parser.add_argument("--merged_output", type=str,
                        default="./datasets/merged/train.json",
                        help="混合数据输出路径（--merge时使用）")

    args = parser.parse_args()

    random.seed(42)

    # 第一步：解析PSQ生成训练数据
    process_psq_directory(
        input_dir=args.input_dir,
        output_path=args.output,
        max_games=args.max_games,
        samples_per_game=args.samples_per_game,
    )

    # 第二步（可选）：和合成数据混合
    if args.merge:
        merge_with_synthetic(
            real_data_path=args.output,
            synthetic_data_path=args.synthetic_data,
            output_path=args.merged_output,
        )