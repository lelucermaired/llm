import json
import random
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from enum import Enum
import os
from datetime import datetime

# ==================== 配置 ====================
CONFIG = {
    "board_size": 15,  # 棋盘大小 (15x15)
    "num_samples": 1000,  # 要生成的样本总数
    "min_stones": 8,  # 每个局面的最小棋子数
    "max_stones": 40,  # 每个局面的最大棋子数
    "output_file": "./datasets/gomoku_diagnostic_dataset.json",
    "train_test_split": 0.9,  # 90%训练，10%验证
}


# ==================== 数据结构 ====================
class Stone(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


@dataclass
class Pattern:
    """棋形定义"""
    name: str
    length: int
    pattern: List[int]  # 1表示己方，0表示空，-1表示对方或边界


class GomokuBoard:
    """五子棋棋盘类"""

    # 8个方向向量: 右, 下, 右下, 左下, 左, 上, 左上, 右上
    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1),
                  (0, -1), (-1, 0), (-1, -1), (-1, 1)]

    # 常见棋形模式
    PATTERNS = {
        # 格式: [己方, 己方, 己方, 空位] 等
        "活二": [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]],
        "活三": [[0, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
        "冲四": [[1, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 1, 0, 1, 1]],
        "活四": [[1, 1, 1, 1, 0]],  # 实际是冲四的一种，这里简化
        "长连": [[1, 1, 1, 1, 1]],
    }

    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.history = []  # 落子历史

    def reset(self):
        """重置棋盘"""
        self.board.fill(Stone.EMPTY.value)
        self.history.clear()

    def place_stone(self, x: int, y: int, player: Stone) -> bool:
        """在位置(x,y)放置棋子"""
        if 0 <= x < self.size and 0 <= y < self.size:
            if self.board[x, y] == Stone.EMPTY.value:
                self.board[x, y] = player.value
                self.history.append((x, y, player))
                return True
        return False

    def get_board_text(self, use_coordinates=True) -> str:
        """将棋盘转换为文本表示"""
        board_text = ""
        if use_coordinates:
            # 添加列标
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
        """统计指定玩家的所有棋形"""
        patterns_found = {name: [] for name in self.PATTERNS.keys()}
        player_val = player.value

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == player_val:
                    # 检查8个方向
                    for dx, dy in self.DIRECTIONS:
                        # 获取5个连续位置的状态
                        line = []
                        for k in range(5):
                            x, y = i + dx * k, j + dy * k
                            if 0 <= x < self.size and 0 <= y < self.size:
                                if self.board[x, y] == player_val:
                                    line.append(1)
                                elif self.board[x, y] == Stone.EMPTY.value:
                                    line.append(0)
                                else:
                                    line.append(-1)  # 对方棋子
                            else:
                                line.append(-1)  # 边界

                        # 检查是否匹配任何模式
                        for pattern_name, patterns in self.PATTERNS.items():
                            for pattern in patterns:
                                if len(line) >= len(pattern) and line[:len(pattern)] == pattern:
                                    # 记录位置信息
                                    coords = []
                                    for k in range(len(pattern)):
                                        x, y = i + dx * k, j + dy * k
                                        coords.append((x, y))
                                    patterns_found[pattern_name].append(coords)
                                    break

        # 去重（简单的基于起点去重）
        for pattern_name in patterns_found:
            unique_patterns = []
            seen_starts = set()
            for coords in patterns_found[pattern_name]:
                start = coords[0]
                if start not in seen_starts:
                    seen_starts.add(start)
                    unique_patterns.append(coords)
            patterns_found[pattern_name] = unique_patterns

        return patterns_found

    def has_winner(self) -> Tuple[bool, Stone]:
        """检查是否有获胜者"""
        for i in range(self.size):
            for j in range(self.size):
                stone = self.board[i, j]
                if stone != Stone.EMPTY.value:
                    # 检查4个方向: 水平, 垂直, 对角线, 反对角线
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


# ==================== 数据生成器 ====================
class GomokuDataGenerator:
    """五子棋数据生成器"""

    def __init__(self, config):
        self.config = config
        self.board = GomokuBoard(config["board_size"])

    def generate_random_board(self) -> None:
        """生成一个随机的棋盘局面"""
        self.board.reset()

        # 随机决定棋子数量
        num_stones = random.randint(
            self.config["min_stones"],
            self.config["max_stones"]
        )

        # 随机放置棋子
        stones_placed = 0
        attempts = 0
        max_attempts = num_stones * 3

        while stones_placed < num_stones and attempts < max_attempts:
            x = random.randint(0, self.board.size - 1)
            y = random.randint(0, self.board.size - 1)
            player = random.choice([Stone.BLACK, Stone.WHITE])

            if self.board.place_stone(x, y, player):
                stones_placed += 1
            attempts += 1

        # 确保没有一方已经获胜（如果有，移除最后一些棋子）
        has_winner, _ = self.board.has_winner()
        while has_winner and len(self.board.history) > self.config["min_stones"]:
            # 移除最后3步
            for _ in range(3):
                if self.board.history:
                    x, y, _ = self.board.history.pop()
                    self.board.board[x, y] = Stone.EMPTY.value
            has_winner, _ = self.board.has_winner()

    def create_diagnostic_sample(self, sample_id: int) -> Dict:
        """创建一个局面诊断样本"""
        # 生成随机棋盘
        self.generate_random_board()

        # 分析棋盘
        board_text = self.board.get_board_text()
        black_patterns = self.board.count_patterns(Stone.BLACK)
        white_patterns = self.board.count_patterns(Stone.WHITE)
        has_winner, winner = self.board.has_winner()

        # 构建问题
        question = f"""你是一个五子棋裁判。请严格分析以下棋盘状态，并回答以下问题。

棋盘状态（●黑子，○白子，·空位，{self.board.size}路棋盘行列号为1-{self.board.size}，列标为字母A-{chr(64 + min(self.board.size, 26))}）：

{board_text}

请分析：
1. 黑棋（●）有哪些关键棋形？请按'活二'、'活三'、'冲四'分别统计数量。
2. 白棋（○）有哪些关键棋形？请按'活二'、'活三'、'冲四'分别统计数量。
3. 当前是否有任何一方已经获胜？如果有，是哪一方，连成五子的坐标是什么（格式如：A1,B1,C1,D1,E1）？如果没有，请回答'未获胜'。

请严格按照以下格式回答：
"""

        # 构建答案
        answer = "<thinking>\n"

        # 1. 分析黑棋
        answer += "首先分析黑棋（●）的棋形：\n"
        for pattern_name in ["活二", "活三", "冲四"]:
            patterns = black_patterns.get(pattern_name, [])
            if patterns:
                # 将内部坐标转换为棋盘坐标
                coord_strs = []
                for coords in patterns[:3]:  # 只显示前3个
                    board_coords = []
                    for x, y in coords:
                        col = chr(65 + y)
                        row = x + 1
                        board_coords.append(f"{col}{row}")
                    coord_strs.append("→".join(board_coords))

                answer += f"  - {pattern_name}: 共{len(patterns)}个"
                if coord_strs:
                    answer += f"。示例：{'; '.join(coord_strs)}"
                answer += "\n"
            else:
                answer += f"  - {pattern_name}: 0个\n"

        # 2. 分析白棋
        answer += "\n然后分析白棋（○）的棋形：\n"
        for pattern_name in ["活二", "活三", "冲四"]:
            patterns = white_patterns.get(pattern_name, [])
            if patterns:
                # 将内部坐标转换为棋盘坐标
                coord_strs = []
                for coords in patterns[:3]:
                    board_coords = []
                    for x, y in coords:
                        col = chr(65 + y)
                        row = x + 1
                        board_coords.append(f"{col}{row}")
                    coord_strs.append("→".join(board_coords))

                answer += f"  - {pattern_name}: 共{len(patterns)}个"
                if coord_strs:
                    answer += f"。示例：{'; '.join(coord_strs)}"
                answer += "\n"
            else:
                answer += f"  - {pattern_name}: 0个\n"

        # 3. 检查胜负
        answer += "\n最后检查是否有获胜方：\n"
        if has_winner:
            # 找到具体的五连位置（简化：从已有棋子中找）
            winner_color = "黑棋（●）" if winner == Stone.BLACK else "白棋（○）"
            answer += f"  - {winner_color}已经获胜。\n"
            # 这里简化处理，实际应该找到具体的五连坐标
            answer += "  - 连成五子的具体坐标需要进一步扫描棋盘确定。"
        else:
            answer += "  - 双方均未形成连续五个或以上棋子，未获胜。"

        answer += "\n</thinking>\n\n"

        # 最终答案
        answer += "1. 黑棋关键棋形：\n"
        for pattern_name in ["活二", "活三", "冲四"]:
            count = len(black_patterns.get(pattern_name, []))
            answer += f"   {pattern_name}: {count}个\n"

        answer += "\n2. 白棋关键棋形：\n"
        for pattern_name in ["活二", "活三", "冲四"]:
            count = len(white_patterns.get(pattern_name, []))
            answer += f"   {pattern_name}: {count}个\n"

        answer += "\n3. 获胜状态："
        if has_winner:
            winner_text = "黑棋（●）" if winner == Stone.BLACK else "白棋（○）"
            answer += f"{winner_text}已获胜。"
            # 简化的坐标（实际应计算）
            answer += "连五坐标示例：需具体分析棋盘。"
        else:
            answer += "未获胜。"

        return {
            "id": f"diagnostic_{sample_id:04d}",
            "instruction": question,
            "input": "",
            "output": answer,
            "metadata": {
                "board_size": self.board.size,
                "black_stones": np.sum(self.board.board == Stone.BLACK.value),
                "white_stones": np.sum(self.board.board == Stone.WHITE.value),
                "has_winner": has_winner,
                "winner": winner.name if has_winner else "NONE",
                "patterns_black": {k: len(v) for k, v in black_patterns.items()},
                "patterns_white": {k: len(v) for k, v in white_patterns.items()},
            }
        }

    def generate_dataset(self) -> List[Dict]:
        """生成完整数据集"""
        print(f"开始生成 {self.config['num_samples']} 个局面诊断样本...")

        dataset = []
        for i in range(self.config["num_samples"]):
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{self.config['num_samples']} 个样本")

            sample = self.create_diagnostic_sample(i + 1)
            dataset.append(sample)

        print(f"✅ 样本生成完成！共 {len(dataset)} 个样本")
        return dataset

    def save_dataset(self, dataset: List[Dict]) -> None:
        """保存数据集到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.config["output_file"]), exist_ok=True)

        # 划分训练/验证集
        split_idx = int(len(dataset) * self.config["train_test_split"])
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]

        # 保存训练集
        train_file = self.config["output_file"].replace(".json", "_train.json")
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(train_data), f, ensure_ascii=False, indent=2)  # 修改这里

        # 保存验证集
        val_file = self.config["output_file"].replace(".json", "_val.json")
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(val_data), f, ensure_ascii=False, indent=2)  # 修改这里

        print(f"✅ 训练集已保存: {train_file} ({len(train_data)} 个样本)")
        print(f"✅ 验证集已保存: {val_file} ({len(val_data)} 个样本)")

        # 生成统计信息
        self.generate_stats(dataset, train_data, val_data)

    def generate_stats(self, full_data, train_data, val_data):
        """生成数据统计信息"""
        stats = {
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "总样本数": len(full_data),
            "训练样本数": len(train_data),
            "验证样本数": len(val_data),
            "棋盘大小": self.config["board_size"],
            "样本类型": "局面诊断",
            "统计": {
                "有胜负的局面": sum(1 for d in full_data if d["metadata"]["has_winner"]),
                "平均黑子数": np.mean([d["metadata"]["black_stones"] for d in full_data]),
                "平均白子数": np.mean([d["metadata"]["white_stones"] for d in full_data]),
            }
        }

        stats_file = self.config["output_file"].replace(".json", "_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"✅ 统计信息已保存: {stats_file}")


# ==================== 其他数据类型的生成函数 ====================
def generate_rule_samples(num_samples=100) -> List[Dict]:
    """生成规则与概念样本"""
    print(f"生成 {num_samples} 个规则样本...")

    rule_qa_pairs = [
        {
            "q": "五子棋的获胜条件是什么？",
            "a": "五子棋的获胜条件是：对局双方轮流在棋盘上落子（通常黑先白后），任一方首先在横线、竖线或斜线方向上形成连续的五个或以上己方棋子，则立即获胜，游戏结束。"
        },
        {
            "q": "五子棋中有'吃子'的规则吗？棋子可以移动吗？",
            "a": "没有。五子棋没有吃子规则，棋子落在交叉点上后就不能移动，仅通过连成五子来判定胜负。"
        },
        {
            "q": "请解释什么是'活三'、'冲四'和'活四'？",
            "a": "'活三'指一方形成的连续三个棋子，且两端都没有被对方棋子阻挡，存在两种方式可以形成活四。'冲四'指形成的连续四个棋子，但一端已被阻挡，只剩一个点可以形成五连。'活四'指形成的连续四个棋子且两端都未被阻挡，已经必胜。"
        },
        {
            "q": "如果棋盘下满了棋子还没人连成五子怎么办？",
            "a": "这种情况称为'和棋'或'平局'，在正式规则中视为和棋。"
        },
        {
            "q": "黑棋第一步有特殊规定吗？",
            "a": "在基础无禁手规则中，黑棋第一步可以下在棋盘任何空点，通常下在天元（中心点）是为了获得最大优势，但不是强制规定。"
        },
    ]

    samples = []
    for i in range(num_samples):
        qa = random.choice(rule_qa_pairs)
        variations = [
            f"规则问题：{qa['q']}",
            f"请回答：{qa['q']}",
            f"关于五子棋规则：{qa['q']}",
            f"问题：{qa['q']}",
        ]

        question = random.choice(variations)

        samples.append({
            "id": f"rule_{i:04d}",
            "instruction": question,
            "input": "",
            "output": qa["a"],
            "metadata": {"type": "rule", "category": "basic"}
        })

    return samples


def generate_decision_samples(generator, num_samples=200) -> List[Dict]:
    """生成单步决策样本"""
    print(f"生成 {num_samples} 个单步决策样本...")

    samples = []
    for i in range(num_samples):
        # 生成随机棋盘
        generator.generate_random_board()
        board_text = generator.board.get_board_text()

        # 随机选择当前玩家
        current_player = random.choice(["黑棋（●）", "白棋（○）"])

        # 构建不同难度的问题
        problem_types = [
            f"轮到{current_player}走。请分析局面并给出最佳落子位置及详细理由。",
            f"现在{current_player}行动。请找出当前局面的最佳着法，并解释为什么这个点优于其他候选点。",
            f"{current_player}的回合。请完成：1. 局面评估 2. 候选点分析 3. 最终决策",
        ]

        question = f"""【五子棋单步决策】规则：连五获胜。

当前棋盘：
{board_text}

{random.choice(problem_types)}"""

        # 简化答案生成（实际应用中应该用棋类引擎生成正确答案）
        # 这里使用启发式：选择棋盘中心附近的空点
        center = generator.board.size // 2
        candidates = [(center, center), (center - 1, center), (center, center - 1),
                      (center + 1, center), (center, center + 1)]

        # 找到第一个空点作为"最佳点"
        best_move = None
        for x, y in candidates:
            if 0 <= x < generator.board.size and 0 <= y < generator.board.size:
                if generator.board.board[x, y] == Stone.EMPTY.value:
                    best_move = (x, y)
                    break

        if best_move is None:
            # 随机找一个空点
            empty_spots = np.argwhere(generator.board.board == Stone.EMPTY.value)
            if len(empty_spots) > 0:
                best_move = tuple(empty_spots[0])

        if best_move:
            col = chr(65 + best_move[1])
            row = best_move[0] + 1
            move_coord = f"{col}{row}"

            answer = f"""<thinking>
1. 局面分析：棋盘较为开放，双方棋子数量大致相当。
2. 候选点：考虑到发展潜力和中心控制，{move_coord}是一个好点。
3. 决策：选择{current_player}可以扩展势力范围的位置。
</thinking>

最佳落子：{move_coord}
理由：此点位于棋盘相对中心区域，有利于向多个方向发展，同时与现有棋子形成潜在联系，具有较好的发展前景。"""
        else:
            answer = "无合理落子点（棋盘已满或生成错误）。"

        samples.append({
            "id": f"decision_{i:04d}",
            "instruction": question,
            "input": "",
            "output": answer,
            "metadata": {
                "type": "decision",
                "current_player": current_player,
                "best_move": move_coord if best_move else "NONE"
            }
        })

    return samples


def convert_to_serializable(obj):
    """将NumPy类型转换为JSON可序列化的Python原生类型"""
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

# ==================== 主函数 ====================
def main():
    """生成完整的数据集"""
    print("=" * 60)
    print("五子棋训练数据生成器")
    print("=" * 60)

    # 1. 生成局面诊断样本
    generator = GomokuDataGenerator(CONFIG)
    diagnostic_data = generator.generate_dataset()

    # 2. 生成规则样本
    rule_data = generate_rule_samples(100)

    # 3. 生成决策样本
    decision_data = generate_decision_samples(generator, 200)

    # 4. 合并所有数据
    all_data = diagnostic_data + rule_data + decision_data
    random.shuffle(all_data)  # 打乱顺序

    print(f"\n✅ 所有数据生成完成！")
    print(f"   局面诊断: {len(diagnostic_data)} 个样本")
    print(f"   规则问答: {len(rule_data)} 个样本")
    print(f"   单步决策: {len(decision_data)} 个样本")
    print(f"   总计: {len(all_data)} 个样本")

    # 5. 保存完整数据集
    complete_file = "./datasets/complete_gomoku_dataset.json"
    os.makedirs(os.path.dirname(complete_file), exist_ok=True)

    # 清洗数据，转换所有NumPy类型为Python原生类型
    print("清洗数据，转换NumPy类型...")
    serializable_data = convert_to_serializable(all_data)

    with open(complete_file, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
    # 6. 保存各类型数据（供单独使用）
    generator.save_dataset(diagnostic_data)

    print("\n" + "=" * 60)
    print("下一步：")
    print(f"1. 检查生成的数据: {complete_file}")
    print("2. 运行微调脚本: python finetune_qwen7b_qlora.py")
    print("3. 调整数据集路径到微调脚本的CONFIG中")
    print("=" * 60)


if __name__ == "__main__":
    main()