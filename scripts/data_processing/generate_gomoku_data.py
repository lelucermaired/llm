"""
enhanced_data_generator.py

增强版五子棋训练数据生成器 - 强调通用推理能力
"""

import json
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import os
from datetime import datetime
from datasets import load_dataset
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 配置 ====================
CONFIG = {
    "board_size": 15,
    "num_diagnostic": 0,
    "num_rule": 100,
    "num_decision": 1500,
    "num_planning": 100,
    "num_general_reasoning": 300,
    "min_stones": 8,
    "max_stones": 40,
    "output_dir": "./datasets/enhanced_v2",
    "train_test_split": 0.9,
    "seed": 42,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


class Stone(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class GomokuBoard:
    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1),
                  (0, -1), (-1, 0), (-1, -1), (-1, 1)]

    PATTERNS = {
        "活二": [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]],
        "活三": [[0, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
        "冲四": [[1, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 1, 0, 1, 1]],
        "活四": [[1, 1, 1, 1, 0]],
        "长连": [[1, 1, 1, 1, 1]],
    }

    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.history = []

    def reset(self):
        self.board.fill(Stone.EMPTY.value)
        self.history.clear()

    def place_stone(self, x: int, y: int, player: Stone) -> bool:
        if 0 <= x < self.size and 0 <= y < self.size:
            if self.board[x, y] == Stone.EMPTY.value:
                self.board[x, y] = player.value
                self.history.append((x, y, player))
                return True
        return False

    def get_board_text(self, use_coordinates=True) -> str:
        board_text = ""
        if use_coordinates:
            board_text += "   " + " ".join([chr(65 + i) for i in range(min(15, self.size))]) + "\n"
        for i in range(self.size):
            if use_coordinates:
                board_text += f"{i + 1:2d} "
            for j in range(self.size):
                stone = self.board[i, j]
                if stone == Stone.BLACK.value:
                    board_text += "● "
                elif stone == Stone.WHITE.value:
                    board_text += "○ "
                else:
                    board_text += "· "
            board_text += "\n"
        return board_text.rstrip()

    def count_patterns(self, player: Stone) -> Dict[str, List[Tuple]]:
        patterns_found = {name: [] for name in self.PATTERNS.keys()}
        player_val = player.value
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == player_val:
                    for dx, dy in self.DIRECTIONS:
                        line = []
                        for k in range(5):
                            x, y = i + dx * k, j + dy * k
                            if 0 <= x < self.size and 0 <= y < self.size:
                                if self.board[x, y] == player_val:
                                    line.append(1)
                                elif self.board[x, y] == Stone.EMPTY.value:
                                    line.append(0)
                                else:
                                    line.append(-1)
                            else:
                                line.append(-1)
                        for pattern_name, patterns in self.PATTERNS.items():
                            for pattern in patterns:
                                if len(line) >= len(pattern) and line[:len(pattern)] == pattern:
                                    coords = []
                                    for k in range(len(pattern)):
                                        x, y = i + dx * k, j + dy * k
                                        coords.append((x, y))
                                    patterns_found[pattern_name].append(coords)
                                    break
        for pattern_name in patterns_found:
            unique = []
            seen = set()
            for coords in patterns_found[pattern_name]:
                start = coords[0]
                if start not in seen:
                    seen.add(start)
                    unique.append(coords)
            patterns_found[pattern_name] = unique
        return patterns_found

    def has_winner(self) -> Tuple[bool, Stone]:
        for i in range(self.size):
            for j in range(self.size):
                stone = self.board[i, j]
                if stone != Stone.EMPTY.value:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = 1
                        for k in range(1, 5):
                            x, y = i + dx * k, j + dy * k
                            if 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == stone:
                                count += 1
                            else:
                                break
                        if count >= 5:
                            return True, Stone(stone)
        return False, Stone.EMPTY

    def get_winner_coords(self, winner: Stone) -> List[Tuple[int, int]]:
        player_val = winner.value
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == player_val:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        coords = [(i + dx * k, j + dy * k) for k in range(5)]
                        if all(0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == player_val for x, y in coords):
                            return coords
        return []

    def get_empty_positions(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == Stone.EMPTY.value]


class EnhancedGomokuDataGenerator:
    def __init__(self, config):
        self.config = config
        self.board = GomokuBoard(config["board_size"])

    def _coord_to_str(self, x, y):
        col = chr(65 + y)
        row = x + 1
        return f"{col}{row}"

    def _format_pattern_examples(self, patterns, max_examples=3):
        examples = []
        for coords in patterns[:max_examples]:
            coord_strs = [self._coord_to_str(x, y) for x, y in coords]
            examples.append("→".join(coord_strs))
        return "; ".join(examples)

    def generate_random_board(self, ensure_no_winner=True, max_attempts=50):
        for attempt in range(max_attempts):
            self.board.reset()
            num_stones = random.randint(self.config["min_stones"], self.config["max_stones"])
            stones_placed = 0
            attempts = 0
            while stones_placed < num_stones and attempts < num_stones * 3:
                x = random.randint(0, self.board.size - 1)
                y = random.randint(0, self.board.size - 1)
                player = random.choice([Stone.BLACK, Stone.WHITE])
                if self.board.place_stone(x, y, player):
                    stones_placed += 1
                attempts += 1

            has_winner, _ = self.board.has_winner()
            if ensure_no_winner and has_winner:
                continue
            if not ensure_no_winner or not has_winner:
                return True
        print("警告：无法生成满足条件的棋盘")
        return False

    def evaluate_position(self, board: GomokuBoard, player: Stone) -> Dict[Tuple[int, int], float]:
        empty = board.get_empty_positions()
        scores = {}
        for x, y in empty:
            board.board[x, y] = player.value
            patterns = board.count_patterns(player)
            score = 0
            if len(patterns.get("活四", [])) > 0:
                score += 1000
            if len(patterns.get("冲四", [])) > 0:
                score += 100
            if len(patterns.get("活三", [])) > 0:
                score += 50
            if len(patterns.get("活二", [])) > 0:
                score += 10
            center = board.size // 2
            center_dist = abs(x - center) + abs(y - center)
            score += (board.size - center_dist) * 0.5
            board.board[x, y] = Stone.EMPTY.value
            scores[(x, y)] = score
        return scores

    def find_best_move(self, board: GomokuBoard, player: Stone) -> Tuple[Optional[Tuple[int, int]], float]:
        scores = self.evaluate_position(board, player)
        if not scores:
            return None, 0.0
        best = max(scores.items(), key=lambda x: x[1])
        return best[0], best[1]

    def can_win_in_steps(self, board: GomokuBoard, player: Stone, steps: int) -> bool:
        if steps == 0:
            return False
        has_winner, winner = board.has_winner()
        if has_winner and winner == player:
            return True
        empty = board.get_empty_positions()
        for x, y in empty:
            board.place_stone(x, y, player)
            has_winner, winner = board.has_winner()
            if has_winner and winner == player:
                board.board[x, y] = Stone.EMPTY.value
                return True
            if steps > 1:
                opponent = Stone.WHITE if player == Stone.BLACK else Stone.BLACK
                opp_empty = board.get_empty_positions()
                if opp_empty:
                    ox, oy = random.choice(opp_empty)
                    board.place_stone(ox, oy, opponent)
                    if self.can_win_in_steps(board, player, steps - 1):
                        board.board[ox, oy] = Stone.EMPTY.value
                        board.board[x, y] = Stone.EMPTY.value
                        return True
                    board.board[ox, oy] = Stone.EMPTY.value
            board.board[x, y] = Stone.EMPTY.value
        return False

    # ========== 决策样本（改进版） ==========
    def create_decision_sample(self, sample_id: int) -> Dict:
        if not self.generate_random_board(ensure_no_winner=True):
            return {
                "id": f"decision_{sample_id:04d}",
                "instruction": "棋盘生成失败，跳过。",
                "input": "",
                "output": "无法生成样本。",
                "metadata": {"type": "decision", "error": True}
            }

        board_text = self.board.get_board_text()
        current_player = random.choice(["黑棋（●）", "白棋（○）"])
        player_stone = Stone.BLACK if "黑" in current_player else Stone.WHITE
        opponent_stone = Stone.WHITE if player_stone == Stone.BLACK else Stone.BLACK

        empty_positions = self.board.get_empty_positions()
        if len(empty_positions) == 0:
            return {
                "id": f"decision_{sample_id:04d}",
                "instruction": f"轮到{current_player}走，但棋盘已满。",
                "input": "", "output": "棋盘已满，和棋。",
                "metadata": {"type": "decision"}
            }

        scores = self.evaluate_position(self.board, player_stone)
        if not scores:
            return {
                "id": f"decision_{sample_id:04d}",
                "instruction": f"轮到{current_player}走，但无合法着法。",
                "input": "", "output": "无合法着法，和棋。",
                "metadata": {"type": "decision"}
            }

        sorted_moves = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_moves[:3]
        best_move, best_score = top3[0]
        best_coord = self._coord_to_str(best_move[0], best_move[1])

        my_patterns = self.board.count_patterns(player_stone)
        opp_patterns = self.board.count_patterns(opponent_stone)
        my_live3 = len(my_patterns.get("活三", []))
        my_rush4 = len(my_patterns.get("冲四", []))
        my_live4 = len(my_patterns.get("活四", []))
        opp_live3 = len(opp_patterns.get("活三", []))
        opp_rush4 = len(opp_patterns.get("冲四", []))
        opp_live4 = len(opp_patterns.get("活四", []))
        player_name = "黑棋" if player_stone == Stone.BLACK else "白棋"
        opponent_name = "白棋" if player_stone == Stone.BLACK else "黑棋"

        thinking_style = random.randint(0, 2)

        if thinking_style == 0:
            thinking = "<thinking>\n"
            thinking += f"现在轮到{player_name}落子，我需要从进攻和防守两个角度分析。\n\n"
            thinking += f"【当前局面分析】\n"
            if my_live4 > 0 or my_rush4 > 0:
                thinking += f"  {player_name}已有威胁性棋形：活四{my_live4}个、冲四{my_rush4}个，处于进攻态势。\n"
            elif my_live3 > 0:
                thinking += f"  {player_name}有活三{my_live3}个，可以考虑进一步扩展。\n"
            else:
                thinking += f"  {player_name}暂无明显棋形优势。\n"
            if opp_live4 > 0 or opp_rush4 > 0:
                thinking += f"  ⚠️ {opponent_name}有活四{opp_live4}个、冲四{opp_rush4}个，必须立即防守！\n"
            elif opp_live3 > 0:
                thinking += f"  {opponent_name}有活三{opp_live3}个，需要注意防守。\n"
            else:
                thinking += f"  {opponent_name}暂无直接威胁。\n"
            thinking += f"\n【候选着法评估】\n"
            for i, (move, score) in enumerate(top3):
                coord = self._coord_to_str(move[0], move[1])
                self.board.board[move[0], move[1]] = player_stone.value
                new_patterns = self.board.count_patterns(player_stone)
                self.board.board[move[0], move[1]] = Stone.EMPTY.value
                new_live3 = len(new_patterns.get("活三", []))
                new_rush4 = len(new_patterns.get("冲四", []))
                new_live4 = len(new_patterns.get("活四", []))
                thinking += f"  候选{i+1}：{coord}（得分{score:.1f}）\n"
                if new_live4 > 0:
                    thinking += f"    → 可形成活四{new_live4}个，直接威胁获胜\n"
                elif new_rush4 > my_rush4:
                    thinking += f"    → 新增冲四{new_rush4 - my_rush4}个，形成连续威胁\n"
                elif new_live3 > my_live3:
                    thinking += f"    → 新增活三{new_live3 - my_live3}个，扩大进攻空间\n"
                else:
                    center = self.board.size // 2
                    dist = abs(move[0] - center) + abs(move[1] - center)
                    thinking += f"    → 控制关键位置，距中心{dist}步\n"
            thinking += f"\n【决策】\n"
            if opp_live4 > 0 or opp_rush4 > 0:
                thinking += f"对方有紧急威胁，{best_coord}既能防守又有进攻价值，得分最高（{best_score:.1f}）。\n"
            else:
                thinking += f"综合比较，{best_coord}得分最高（{best_score:.1f}），能最大化己方棋形优势。\n"
            thinking += "</thinking>\n\n"

        elif thinking_style == 1:
            thinking = "<thinking>\n"
            thinking += f"第一步：扫描是否有立即获胜的机会。\n"
            if my_live4 > 0 or my_rush4 > 0:
                thinking += f"  → {player_name}有冲四/活四，{best_coord}可以直接威胁。\n"
            else:
                thinking += f"  → 暂无立即获胜机会，继续分析。\n"
            thinking += f"\n第二步：检查是否需要立即防守。\n"
            if opp_live4 > 0:
                thinking += f"  → {opponent_name}有活四，必须在{best_coord}附近防守，否则对方下一步获胜！\n"
            elif opp_rush4 > 0:
                thinking += f"  → {opponent_name}有冲四{opp_rush4}个，需要封堵关键点。\n"
            else:
                thinking += f"  → {opponent_name}暂无紧迫威胁，可以主动进攻。\n"
            thinking += f"\n第三步：比较候选点的长期价值。\n"
            for i, (move, score) in enumerate(top3[:2]):
                coord = self._coord_to_str(move[0], move[1])
                thinking += f"  {coord}：综合得分{score:.1f}"
                if i == 0:
                    thinking += "（最优）"
                thinking += "\n"
            thinking += f"\n结论：{best_coord}在进攻价值、防守价值、位置价值三个维度上表现最佳。\n"
            thinking += "</thinking>\n\n"

        else:
            thinking = "<thinking>\n"
            thinking += f"对{player_name}所有可能落子点进行评分：活四=1000分，冲四=100分，活三=50分，活二=10分，位置加成。\n\n"
            thinking += f"当前局面：{player_name}活三{my_live3}个、冲四{my_rush4}个；{opponent_name}活三{opp_live3}个、冲四{opp_rush4}个。\n\n"
            thinking += f"得分最高的候选点：\n"
            for i, (move, score) in enumerate(top3):
                coord = self._coord_to_str(move[0], move[1])
                thinking += f"  {i+1}. {coord} → {score:.1f}分\n"
            thinking += f"\n{best_coord}得分{best_score:.1f}分，"
            if best_score > 100:
                thinking += "远高于其他候选点，优先选择。\n"
            elif best_score > 50:
                thinking += "明显优于其他点，具有活三或防守价值。\n"
            else:
                thinking += "略高于其他点，主要体现在位置控制上。\n"
            thinking += "</thinking>\n\n"

        # 落子后棋形变化
        self.board.board[best_move[0], best_move[1]] = player_stone.value
        new_patterns = self.board.count_patterns(player_stone)
        self.board.board[best_move[0], best_move[1]] = Stone.EMPTY.value
        new_live3 = len(new_patterns.get("活三", []))
        new_rush4 = len(new_patterns.get("冲四", []))
        new_live4 = len(new_patterns.get("活四", []))

        answer = thinking + f"最佳落子：{best_coord}\n"
        if new_live4 > 0:
            answer += f"理由：落在{best_coord}可形成活四，下一步直接获胜。"
        elif new_rush4 > my_rush4:
            answer += f"理由：落在{best_coord}新增冲四{new_rush4 - my_rush4}个，形成双重威胁，对方难以同时防守。"
        elif new_live3 > my_live3:
            answer += f"理由：落在{best_coord}新增活三{new_live3 - my_live3}个，扩大进攻态势。"
        elif opp_rush4 > 0 or opp_live4 > 0:
            answer += f"理由：封堵对方{opponent_name}的威胁，同时保持己方棋形。"
        else:
            center = self.board.size // 2
            dist = abs(best_move[0] - center) + abs(best_move[1] - center)
            answer += f"理由：{best_coord}是当前局面价值最高的位置（得分{best_score:.1f}），控制关键区域，距棋盘中心{dist}步。"

        question = f"""【五子棋单步决策】规则：连五获胜。

当前棋盘（●黑子，○白子，·空位）：
{board_text}

轮到{current_player}走。请详细分析当前局面，评估候选着法，给出最佳落子位置及理由。"""

        return {
            "id": f"decision_{sample_id:04d}",
            "instruction": question,
            "input": "",
            "output": answer,
            "metadata": {
                "type": "decision",
                "current_player": current_player,
                "best_move": best_coord,
                "best_score": float(best_score),
                "thinking_style": thinking_style,
            }
        }

    # ========== 规划样本 ==========
    def create_planning_sample(self, sample_id: int) -> Dict:
        found = False
        for _ in range(20):
            if self.generate_random_board(ensure_no_winner=True):
                black_patterns = self.board.count_patterns(Stone.BLACK)
                if len(black_patterns.get("活三", [])) >= 1 or len(black_patterns.get("冲四", [])) >= 1:
                    found = True
                    break
        if not found:
            self.generate_random_board(ensure_no_winner=True)

        board_text = self.board.get_board_text()
        player = random.choice(["黑棋", "白棋"])
        player_stone = Stone.BLACK if player == "黑棋" else Stone.WHITE
        steps = random.randint(2, 3)

        can_win = self.can_win_in_steps(self.board, player_stone, steps)

        if can_win:
            moves = []
            temp_board = GomokuBoard()
            temp_board.board = self.board.board.copy()
            for step in range(steps):
                best, _ = self.find_best_move(temp_board, player_stone)
                if best:
                    moves.append(best)
                    temp_board.place_stone(best[0], best[1], player_stone)
                    opponent = Stone.WHITE if player_stone == Stone.BLACK else Stone.BLACK
                    opp_empty = temp_board.get_empty_positions()
                    if opp_empty and step < steps - 1:
                        ox, oy = random.choice(opp_empty)
                        temp_board.place_stone(ox, oy, opponent)
            move_sequence = " → ".join([self._coord_to_str(x, y) for x, y in moves])
            answer = f"<thinking>\n分析：当前局面{player}有优势，经过计算可以在{steps}步内获胜。\n"
            answer += f"可行的进攻路线：{move_sequence}\n</thinking>\n\n"
            answer += f"可以获胜。着法序列：{move_sequence}"
        else:
            answer = f"<thinking>\n分析：当前局面{player}没有直接获胜的连续手段，对方有足够的防守空间。\n</thinking>\n\n无法在{steps}步内获胜。"

        question = f"""【五子棋多步规划】棋盘如下：

{board_text}

假设轮到{player}走，请问{player}能否在{steps}步内获胜？如果可以，请给出具体的着法序列；如果不能，请说明原因。请详细推理。"""

        return {
            "id": f"planning_{sample_id:04d}",
            "instruction": question,
            "input": "",
            "output": answer,
            "metadata": {
                "type": "planning",
                "player": player,
                "steps": steps,
                "can_win": can_win
            }
        }

    # ========== 通用推理样本 ==========
    def create_general_reasoning_samples(self, num_samples: int) -> List[Dict]:
        print(f"正在从GSM8K加载通用推理样本...")
        try:
            dataset = load_dataset("gsm8k", "main")
            train_data = dataset["train"]
        except Exception as e:
            print(f"加载GSM8K失败: {e}，将使用备用简单数学题")
            return self._create_fallback_math_samples(num_samples)

        samples = []
        indices = random.sample(range(len(train_data)), min(num_samples, len(train_data)))
        for i, idx in enumerate(indices):
            item = train_data[idx]
            question = item["question"]
            answer = item["answer"]
            final_ans = re.findall(r'####\s*(\-?\d+\.?\d*)', answer)
            final_ans = final_ans[-1] if final_ans else ""
            instruction = f"""请解答以下数学题，并给出详细的推理步骤。

题目：{question}

请先写出推理过程，再给出最终答案。"""
            samples.append({
                "id": f"gsm8k_{i:04d}",
                "instruction": instruction,
                "input": "",
                "output": answer.strip(),
                "metadata": {
                    "type": "general_reasoning",
                    "source": "gsm8k",
                    "final_answer": final_ans
                }
            })
        return samples

    def _create_fallback_math_samples(self, num_samples):
        samples = []
        simple_problems = [
            {"q": "小明有5个苹果，小红给了他3个，然后又吃掉了2个，还剩几个？", "a": "小明原来有5个，得到3个后共有8个，吃掉2个剩6个。所以答案是6。", "ans": "6"},
            {"q": "一辆车每小时行驶60公里，2.5小时行驶多少公里？", "a": "距离 = 速度 × 时间 = 60 × 2.5 = 150公里。", "ans": "150"},
            {"q": "一个长方形的长是8厘米，宽是5厘米，面积是多少？", "a": "面积 = 长 × 宽 = 8 × 5 = 40平方厘米。", "ans": "40"},
        ]
        for i in range(num_samples):
            prob = random.choice(simple_problems)
            instruction = f"请解答以下数学题：\n\n{prob['q']}\n\n请写出推理过程。"
            samples.append({
                "id": f"simplemath_{i:04d}",
                "instruction": instruction,
                "input": "",
                "output": prob['a'],
                "metadata": {"type": "general_reasoning", "source": "simple", "final_answer": prob['ans']}
            })
        return samples

    def _generate_rule_samples(self, num_samples):
        rule_qa_pairs = [
            {"q": "五子棋的获胜条件是什么？", "a": "五子棋的获胜条件是：对局双方轮流落子，任一方首先在横、竖或斜方向上形成连续的五个己方棋子即获胜。"},
            {"q": "五子棋中有'吃子'规则吗？", "a": "没有。五子棋没有吃子规则，棋子落子后不能移动。"},
            {"q": "请解释什么是'活三'？", "a": "'活三'指一方形成的连续三个棋子，且两端都没有被对方棋子或边界阻挡，存在两种方式可以形成活四。"},
        ]
        samples = []
        for i in range(num_samples):
            qa = random.choice(rule_qa_pairs)
            instruction = f"规则问题：{qa['q']}"
            samples.append({
                "id": f"rule_{i:04d}",
                "instruction": instruction,
                "input": "",
                "output": qa['a'],
                "metadata": {"type": "rule"}
            })
        return samples

    def generate_all_samples(self):
        all_samples = []

        print(f"生成 {CONFIG['num_rule']} 个规则样本...")
        rule_samples = self._generate_rule_samples(CONFIG['num_rule'])
        all_samples.extend(rule_samples)

        print(f"生成 {CONFIG['num_decision']} 个决策样本...")
        for i in range(CONFIG['num_decision']):
            if (i + 1) % 100 == 0:
                print(f"  决策样本 {i+1}/{CONFIG['num_decision']}")
            sample = self.create_decision_sample(i + 1)
            if sample.get("metadata", {}).get("error"):
                continue
            all_samples.append(sample)

        print(f"生成 {CONFIG['num_planning']} 个规划样本...")
        for i in range(CONFIG['num_planning']):
            if (i + 1) % 50 == 0:
                print(f"  规划样本 {i+1}/{CONFIG['num_planning']}")
            sample = self.create_planning_sample(i + 1)
            all_samples.append(sample)

        if CONFIG['num_general_reasoning'] > 0:
            print(f"生成 {CONFIG['num_general_reasoning']} 个通用推理样本...")
            general_samples = self.create_general_reasoning_samples(CONFIG['num_general_reasoning'])
            all_samples.extend(general_samples)

        random.shuffle(all_samples)
        print(f"总计生成 {len(all_samples)} 个样本。")
        return all_samples


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def save_dataset(data, config):
    os.makedirs(config["output_dir"], exist_ok=True)
    split_idx = int(len(data) * config["train_test_split"])
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_file = os.path.join(config["output_dir"], "train.json")
    val_file = os.path.join(config["output_dir"], "val.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(train_data), f, ensure_ascii=False, indent=2)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(val_data), f, ensure_ascii=False, indent=2)

    print(f"训练集保存至: {train_file} ({len(train_data)} 样本)")
    print(f"验证集保存至: {val_file} ({len(val_data)} 样本)")

    type_counts = {}
    for sample in data:
        t = sample["metadata"].get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    stats = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "总样本数": len(data),
        "训练样本数": len(train_data),
        "验证样本数": len(val_data),
        "各类别数量": type_counts
    }
    with open(os.path.join(config["output_dir"], "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计信息: {type_counts}")


def main():
    print("=" * 60)
    print("增强版五子棋训练数据生成器 - 通用推理版 v2")
    print("=" * 60)

    generator = EnhancedGomokuDataGenerator(CONFIG)
    all_data = generator.generate_all_samples()
    save_dataset(all_data, CONFIG)

    print("\n✅ 数据生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()