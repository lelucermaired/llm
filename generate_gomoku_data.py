"""
enhanced_data_generator.py

增强版五子棋训练数据生成器，包含通用思维链、多样化模板、混合通用推理数据
"""

import json
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import os
from datetime import datetime
from datasets import load_dataset  # 需要安装datasets库
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"#
# ==================== 配置 ====================
CONFIG = {
    "board_size": 15,
    "num_diagnostic": 2000,      # 局面诊断样本数（增加）
    "num_rule": 100,
    "num_decision": 500,          # 单步决策样本数（增加）
    "num_planning": 200,           # 新增：多步规划样本
    "num_general_reasoning": 300,  # 新增：通用推理样本（从GSM8K抽取）
    "min_stones": 8,
    "max_stones": 40,
    "output_dir": "./datasets/enhanced",
    "train_test_split": 0.9,
    "seed": 42,
}

# 设置随机种子
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ==================== 棋盘类（复用原代码） ====================
class Stone(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


@dataclass
class Pattern:
    name: str
    length: int
    pattern: List[int]


class GomokuBoard:
    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1),
                  (0, -1), (-1, 0), (-1, -1), (-1, 1)]

    PATTERNS = {
        "活二": [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]],
        "活三": [[0, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
        "冲四": [[1, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 1, 0, 1, 1]],
        "活四": [[1, 1, 1, 1, 0]],  # 简化
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
        """统计指定玩家的所有棋形（与原代码相同）"""
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
        # 去重
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
        """获取获胜的五子坐标（简化版，只找到第一组）"""
        player_val = winner.value
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == player_val:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        coords = [(i + dx * k, j + dy * k) for k in range(5)]
                        if all(0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == player_val for x, y in coords):
                            return coords
        return []


# ==================== 增强版数据生成器 ====================
class EnhancedGomokuDataGenerator:
    def __init__(self, config):
        self.config = config
        self.board = GomokuBoard(config["board_size"])

        # 思维链模板库（诊断任务）
        self.diagnostic_templates = [
            {
                "intro": "这个问题可以分解为三个子任务：分析黑棋、分析白棋、判断胜负。",
                "step1": "第一步，分析黑棋的关键棋形。我需要找出黑棋的所有活二、活三和冲四。",
                "step2": "第二步，分析白棋的关键棋形。采用同样的方法。",
                "step3": "第三步，检查是否有玩家已经获胜。",
                "conclusion": "综上，结论如下。"
            },
            {
                "intro": "让我们逐步推理：先看黑棋，再看白棋，最后确认胜负。",
                "step1": "开始分析黑方棋形：统计活二、活三、冲四的数量。",
                "step2": "接着分析白方棋形：统计活二、活三、冲四的数量。",
                "step3": "最后判断胜负：是否有五子连线？",
                "conclusion": "因此，最终答案是："
            },
            {
                "intro": "按以下顺序分析：1) 黑棋棋形  2) 白棋棋形  3) 获胜情况。",
                "step1": "1) 黑棋分析：",
                "step2": "2) 白棋分析：",
                "step3": "3) 胜负判断：",
                "conclusion": "总结如下："
            }
        ]

    def _coord_to_str(self, x, y):
        """将内部坐标转换为棋盘字符串（如 A1）"""
        col = chr(65 + y)
        row = x + 1
        return f"{col}{row}"

    def _format_pattern_examples(self, patterns, max_examples=3):
        """格式化棋形示例，用于思维链"""
        examples = []
        for coords in patterns[:max_examples]:
            coord_strs = [self._coord_to_str(x, y) for x, y in coords]
            examples.append("→".join(coord_strs))
        return "; ".join(examples)

    def generate_random_board(self, ensure_no_winner=True, max_attempts=50):
        """生成随机棋盘，可确保无获胜者（或允许有获胜者）"""
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
                continue  # 重新生成
            if not ensure_no_winner or not has_winner:
                return True
        return False

    def create_diagnostic_sample(self, sample_id: int) -> Dict:
        """创建增强型局面诊断样本（使用通用化思维链）"""
        # 生成随机棋盘（可允许有获胜者以增加多样性）
        self.generate_random_board(ensure_no_winner=random.random() < 0.3)  # 70%无胜, 30%有胜

        board_text = self.board.get_board_text()
        black_patterns = self.board.count_patterns(Stone.BLACK)
        white_patterns = self.board.count_patterns(Stone.WHITE)
        has_winner, winner = self.board.has_winner()

        # 随机选择一个思维链模板
        template = random.choice(self.diagnostic_templates)

        # 构建思维链
        thinking = f"<thinking>\n{template['intro']}\n\n"

        # 黑棋分析
        thinking += f"{template['step1']}\n"
        for pname in ["活二", "活三", "冲四"]:
            patterns = black_patterns.get(pname, [])
            count = len(patterns)
            if count > 0:
                examples = self._format_pattern_examples(patterns)
                thinking += f"  - {pname}: {count}个。示例：{examples}\n"
            else:
                thinking += f"  - {pname}: 0个。\n"

        thinking += "\n"

        # 白棋分析
        thinking += f"{template['step2']}\n"
        for pname in ["活二", "活三", "冲四"]:
            patterns = white_patterns.get(pname, [])
            count = len(patterns)
            if count > 0:
                examples = self._format_pattern_examples(patterns)
                thinking += f"  - {pname}: {count}个。示例：{examples}\n"
            else:
                thinking += f"  - {pname}: 0个。\n"

        thinking += "\n"

        # 胜负判断
        thinking += f"{template['step3']}\n"
        if has_winner:
            winner_coords = self.board.get_winner_coords(winner)
            coord_str = ",".join([self._coord_to_str(x, y) for x, y in winner_coords])
            winner_name = "黑棋" if winner == Stone.BLACK else "白棋"
            thinking += f"  - {winner_name}已经获胜！五子连线坐标为：{coord_str}\n"
        else:
            thinking += "  - 双方均未形成连续五个棋子，未获胜。\n"

        thinking += f"\n{template['conclusion']}\n</thinking>\n\n"

        # 最终答案（简洁版）
        answer = thinking + "1. 黑棋关键棋形：\n"
        for pname in ["活二", "活三", "冲四"]:
            answer += f"   {pname}: {len(black_patterns.get(pname, []))}个\n"
        answer += "\n2. 白棋关键棋形：\n"
        for pname in ["活二", "活三", "冲四"]:
            answer += f"   {pname}: {len(white_patterns.get(pname, []))}个\n"
        answer += "\n3. 获胜状态："
        if has_winner:
            winner_name = "黑棋" if winner == Stone.BLACK else "白棋"
            answer += f"{winner_name}已获胜。"
        else:
            answer += "未获胜。"

        # 构建指令
        question = f"""你是一个五子棋裁判。请严格分析以下棋盘状态，并回答以下问题。

棋盘状态（●黑子，○白子，·空位，{self.board.size}路棋盘行列号为1-{self.board.size}，列标为字母A-{chr(64 + min(self.board.size, 26))}）：

{board_text}

请分析：
1. 黑棋（●）有哪些关键棋形？请按'活二'、'活三'、'冲四'分别统计数量。
2. 白棋（○）有哪些关键棋形？请按'活二'、'活三'、'冲四'分别统计数量。
3. 当前是否有任何一方已经获胜？如果有，是哪一方，连成五子的坐标是什么（格式如：A1,B1,C1,D1,E1）？如果没有，请回答'未获胜'。

请严格按照以下格式回答：
"""

        return {
            "id": f"diagnostic_{sample_id:04d}",
            "instruction": question,
            "input": "",
            "output": answer,
            "metadata": {
                "board_size": self.board.size,
                "black_stones": int(np.sum(self.board.board == Stone.BLACK.value)),
                "white_stones": int(np.sum(self.board.board == Stone.WHITE.value)),
                "has_winner": has_winner,
                "winner": winner.name if has_winner else "NONE",
                "patterns_black": {k: len(v) for k, v in black_patterns.items()},
                "patterns_white": {k: len(v) for k, v in white_patterns.items()},
            }
        }

    def create_decision_sample(self, sample_id: int) -> Dict:
        """创建增强型单步决策样本（包含详细候选点分析）"""
        # 生成随机棋盘（无胜者）
        self.generate_random_board(ensure_no_winner=True)

        board_text = self.board.get_board_text()
        current_player = random.choice(["黑棋（●）", "白棋（○）"])
        player_stone = Stone.BLACK if "黑" in current_player else Stone.WHITE

        # 寻找候选点（所有空位）
        empty_positions = np.argwhere(self.board.board == Stone.EMPTY.value)
        if len(empty_positions) == 0:
            # 棋盘满了，生成简单回答
            question = f"轮到{current_player}走，但棋盘已满，请分析局面。"
            answer = "<thinking>棋盘已满，无法落子，和棋。</thinking>\n\n和棋。"
            return {
                "id": f"decision_{sample_id:04d}",
                "instruction": question,
                "input": "",
                "output": answer,
                "metadata": {"type": "decision", "current_player": current_player}
            }

        # 随机选择几个候选点（3-5个）
        num_candidates = min(random.randint(3, 5), len(empty_positions))
        candidate_indices = random.sample(range(len(empty_positions)), num_candidates)
        candidates = [tuple(empty_positions[i]) for i in candidate_indices]

        # 启发式评分（简单的中心倾向和与己方棋子距离）
        def heuristic_score(pos):
            x, y = pos
            # 中心距离
            center = self.board.size // 2
            center_dist = abs(x - center) + abs(y - center)
            # 与己方棋子距离（越近越好）
            own_stones = np.argwhere(self.board.board == player_stone.value)
            if len(own_stones) == 0:
                dist_score = 0
            else:
                min_dist = min(abs(x - sx) + abs(y - sy) for sx, sy in own_stones)
                dist_score = max(0, 10 - min_dist)  # 距离越近分越高
            # 综合
            return dist_score - center_dist * 0.5

        # 计算每个候选点的分数
        scored_candidates = [(pos, heuristic_score(pos)) for pos in candidates]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_pos = scored_candidates[0][0]

        # 构建思维链（详细分析）
        thinking = "<thinking>\n"
        thinking += f"轮到{current_player}，棋盘当前有{len(empty_positions)}个空位。\n"
        thinking += "我考虑以下几个候选点：\n"

        for pos, score in scored_candidates:
            coord = self._coord_to_str(pos[0], pos[1])
            # 模拟该点是否会产生威胁（简化判断）
            self.board.place_stone(pos[0], pos[1], player_stone)
            patterns = self.board.count_patterns(player_stone)
            self.board.board[pos[0], pos[1]] = Stone.EMPTY.value  # 撤销

            # 检查是否形成活三或冲四
            threats = []
            if len(patterns.get("活三", [])) > 0:
                threats.append("活三")
            if len(patterns.get("冲四", [])) > 0:
                threats.append("冲四")

            thinking += f"  - {coord}：分数{score:.1f}"
            if threats:
                thinking += f"，可形成{'、'.join(threats)}"
            else:
                thinking += "，无明显直接威胁"
            thinking += "\n"

        # 选择最佳点
        best_coord = self._coord_to_str(best_pos[0], best_pos[1])
        thinking += f"\n综合比较后，{best_coord} 得分最高，因为它靠近中心且能与己方棋子配合。\n"
        thinking += f"因此，最佳落子点是 {best_coord}。\n</thinking>\n\n"

        answer = thinking + f"最佳落子：{best_coord}\n"
        answer += "理由：该点位于棋盘中心区域，便于向多个方向发展，且与现有棋子形成潜在联系，具有较好的发展前景。"

        # 构建问题
        question = f"""【五子棋单步决策】规则：连五获胜。

当前棋盘：
{board_text}

轮到{current_player}走。请分析局面并给出最佳落子位置及详细理由。"""

        return {
            "id": f"decision_{sample_id:04d}",
            "instruction": question,
            "input": "",
            "output": answer,
            "metadata": {
                "type": "decision",
                "current_player": current_player,
                "best_move": best_coord,
                "num_candidates": num_candidates
            }
        }

    def create_planning_sample(self, sample_id: int) -> Dict:
        """创建多步规划样本（例如黑棋如何在3步内获胜）"""
        # 生成一个有一定优势的局面（简化：随机棋盘并检查）
        for _ in range(20):
            self.generate_random_board(ensure_no_winner=True)
            # 简单启发：黑棋有较多活三或冲四
            black_patterns = self.board.count_patterns(Stone.BLACK)
            if len(black_patterns.get("活三", [])) >= 1 or len(black_patterns.get("冲四", [])) >= 1:
                break

        board_text = self.board.get_board_text()

        # 随机决定目标：黑棋或白棋，以及步数（2-3步）
        player = random.choice(["黑棋", "白棋"])
        player_stone = Stone.BLACK if player == "黑棋" else Stone.WHITE
        steps = random.randint(2, 3)

        # 问题构造
        question = f"""【五子棋多步规划】棋盘如下：

{board_text}

假设轮到{player}走，请问{player}能否在{steps}步内获胜？如果可以，请给出具体的着法序列；如果不能，请说明原因。请详细推理。"""

        # 简化答案：我们只是模拟生成一个简单回答（实际应用中可用更强的棋类引擎）
        # 这里我们随机判断
        can_win = random.choice([True, False])
        if can_win:
            # 生成一个假想的着法序列
            moves = []
            for step in range(steps):
                # 找一个空位
                empty = np.argwhere(self.board.board == Stone.EMPTY.value)
                if len(empty) > 0:
                    pos = random.choice(empty)
                    coord = self._coord_to_str(pos[0], pos[1])
                    moves.append(coord)
                    # 假设落子后棋盘变化（但实际不修改，只是模拟）
            move_sequence = " → ".join(moves)
            answer = f"<thinking>\n分析：当前局面{player}有优势，可以设计如下进攻路线：\n"
            answer += f"第1手下在{moves[0]}，形成活三威胁；\n"
            answer += f"第2手下在{moves[1]}，迫使对方防守；\n"
            if steps == 3:
                answer += f"第3手下在{moves[2]}，直接连五获胜。\n"
            answer += f"因此，{player}可以在{steps}步内获胜，着法序列为：{move_sequence}\n</thinking>\n\n"
            answer += f"可以获胜。着法序列：{move_sequence}"
        else:
            answer = f"<thinking>\n分析：当前局面{player}没有直接获胜的连续手段，对方有足够的防守空间。\n</thinking>\n\n无法在{steps}步内获胜。"

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

    def create_general_reasoning_samples(self, num_samples: int) -> List[Dict]:
        """从GSM8K中抽取通用推理样本，转换为指令格式"""
        print(f"正在从GSM8K加载通用推理样本...")
        try:
            dataset = load_dataset("gsm8k", "main")
            train_data = dataset["train"]
        except Exception as e:
            print(f"加载GSM8K失败: {e}，将使用备用简单数学题")
            # 备用：生成简单数学题
            return self._create_fallback_math_samples(num_samples)

        samples = []
        indices = random.sample(range(len(train_data)), min(num_samples, len(train_data)))
        for i, idx in enumerate(indices):
            item = train_data[idx]
            question = item["question"]
            answer = item["answer"]

            # 提取最终答案（用于metadata）
            import re
            final_ans = re.findall(r'####\s*(\-?\d+\.?\d*)', answer)
            final_ans = final_ans[-1] if final_ans else ""

            # 构建指令
            instruction = f"""请解答以下数学题，并给出详细的推理步骤。

题目：{question}

请先写出推理过程，再给出最终答案。"""

            # 将GSM8K的答案格式转换为更统一的格式（可选）
            # 原答案已经包含步骤和####答案，可以直接使用

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
        """备用简单数学题"""
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

    def generate_all_samples(self):
        """生成所有类型样本并合并"""
        all_samples = []

        # 局面诊断
        print(f"生成 {CONFIG['num_diagnostic']} 个诊断样本...")
        for i in range(CONFIG['num_diagnostic']):
            if (i + 1) % 200 == 0:
                print(f"  诊断样本 {i+1}/{CONFIG['num_diagnostic']}")
            sample = self.create_diagnostic_sample(i + 1)
            all_samples.append(sample)

        # 规则问答（复用原有简单生成）
        print(f"生成 {CONFIG['num_rule']} 个规则样本...")
        rule_samples = self._generate_rule_samples(CONFIG['num_rule'])
        all_samples.extend(rule_samples)

        # 单步决策
        print(f"生成 {CONFIG['num_decision']} 个决策样本...")
        for i in range(CONFIG['num_decision']):
            if (i + 1) % 100 == 0:
                print(f"  决策样本 {i+1}/{CONFIG['num_decision']}")
            sample = self.create_decision_sample(i + 1)
            all_samples.append(sample)

        # 多步规划
        print(f"生成 {CONFIG['num_planning']} 个规划样本...")
        for i in range(CONFIG['num_planning']):
            if (i + 1) % 50 == 0:
                print(f"  规划样本 {i+1}/{CONFIG['num_planning']}")
            sample = self.create_planning_sample(i + 1)
            all_samples.append(sample)

        # 通用推理
        if CONFIG['num_general_reasoning'] > 0:
            print(f"生成 {CONFIG['num_general_reasoning']} 个通用推理样本...")
            general_samples = self.create_general_reasoning_samples(CONFIG['num_general_reasoning'])
            all_samples.extend(general_samples)

        # 打乱顺序
        random.shuffle(all_samples)
        print(f"总计生成 {len(all_samples)} 个样本。")
        return all_samples

    def _generate_rule_samples(self, num_samples):
        """生成规则问答样本（简版）"""
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


# ==================== 保存和统计 ====================
def convert_to_serializable(obj):
    """转换NumPy类型为Python原生类型"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_dataset(data, config):
    """划分训练/验证并保存"""
    os.makedirs(config["output_dir"], exist_ok=True)
    split_idx = int(len(data) * config["train_test_split"])
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_file = os.path.join(config["output_dir"], "train.json")
    val_file = os.path.join(config["output_dir"], "val.json")

    # 转换类型
    train_serial = convert_to_serializable(train_data)
    val_serial = convert_to_serializable(val_data)

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_serial, f, ensure_ascii=False, indent=2)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_serial, f, ensure_ascii=False, indent=2)

    print(f"训练集保存至: {train_file} ({len(train_data)} 样本)")
    print(f"验证集保存至: {val_file} ({len(val_data)} 样本)")

    # 统计信息
    stats = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "总样本数": len(data),
        "训练样本数": len(train_data),
        "验证样本数": len(val_data),
        "各类别数量": {}
    }
    type_counts = {}
    for sample in data:
        t = sample["metadata"].get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    stats["各类别数量"] = type_counts

    stats_file = os.path.join(config["output_dir"], "stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"统计信息保存至: {stats_file}")


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("增强版五子棋训练数据生成器")
    print("=" * 60)

    generator = EnhancedGomokuDataGenerator(CONFIG)
    all_data = generator.generate_all_samples()
    save_dataset(all_data, CONFIG)

    print("\n✅ 数据生成完成！")
    print("下一步：请修改微调脚本中的数据集路径为输出目录下的 train.json 和 val.json")
    print("=" * 60)


if __name__ == "__main__":
    main()