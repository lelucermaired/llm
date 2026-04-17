"""
regen_cot.py

读取real_games_v2/train.json，重新生成高质量推理链
- 保留instruction、source、game_file、move_idx不变
- 从instruction中解析棋盘状态
- 从原始output中提取落子坐标
- 用五子棋引擎重新生成推理链替换output

用法：
  python regen_cot.py
  python regen_cot.py --input ./datasets/real_games_v2/train.json \
                      --output ./datasets/real_games_cot/train.json \
                      --n 500
"""

import json, re, copy, argparse
from tqdm import tqdm

BOARD_SIZE = 15
DIRECTIONS = [(1,0),(0,1),(1,1),(1,-1)]
DIR_NAMES  = {(1,0):"横向",(0,1):"纵向",(1,1):"正斜",(1,-1):"反斜"}
BLACK, WHITE, EMPTY = 1, 2, 0

COL_LETTERS = "ABCDEFGHIJKLMNO"

# ==================== 棋盘解析 ====================

def parse_board(instruction):
    """
    从instruction中解析棋盘状态，返回board矩阵和当前轮到谁
    board[r][c]: 0=空, 1=黑, 2=白
    """
    board = [[EMPTY]*BOARD_SIZE for _ in range(BOARD_SIZE)]

    # 提取列标题行，确定列偏移
    col_header_match = re.search(r'([A-O](?:\s+[A-O])+)', instruction)
    if not col_header_match:
        return None, None

    col_str = col_header_match.group(1)
    cols = col_str.split()
    col_offset = COL_LETTERS.index(cols[0])  # 第一列字母对应的index

    # 提取每行棋盘数据
    # 格式: " 1 · · ○ ..." 或 "10· · ● ..."
    row_pattern = re.compile(r'^\s*(\d{1,2})([\s·●○]+)', re.MULTILINE)
    for m in row_pattern.finditer(instruction):
        row_num = int(m.group(1))
        if row_num < 1 or row_num > BOARD_SIZE:
            continue
        r = row_num - 1

        # 提取这一行的棋子
        row_content = m.group(2)
        cells = re.findall(r'[·●○]', row_content)
        for ci, cell in enumerate(cells):
            c = col_offset + ci
            if c >= BOARD_SIZE:
                break
            if cell == '●':
                board[r][c] = BLACK
            elif cell == '○':
                board[r][c] = WHITE

    # 判断当前轮到谁
    black_count = sum(board[r][c]==BLACK for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    white_count = sum(board[r][c]==WHITE for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    current = BLACK if black_count <= white_count else WHITE

    return board, current


def parse_best_move(output):
    """
    从原始output中提取落子坐标
    支持格式：'最佳落子：G4'、'最优：G4'、'落在G4'、'G4'
    返回 (row_idx, col_idx) 或 None
    """
    patterns = [
        r'最佳落子[：:]\s*([A-O]\d{1,2})',
        r'最优[：:]\s*([A-O]\d{1,2})',
        r'落在\s*([A-O]\d{1,2})',
        r'([A-O]\d{1,2})',
    ]
    for pat in patterns:
        m = re.search(pat, output)
        if m:
            move_str = m.group(1)
            col = COL_LETTERS.index(move_str[0])
            row = int(move_str[1:]) - 1
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                return row, col
    return None


def coord_to_str(r, c):
    return f"{COL_LETTERS[c]}{r+1}"


# ==================== 五子棋引擎 ====================

def in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def analyze_pattern(board, r, c, dr, dc, color):
    """分析board[r][c]==color时在(dr,dc)方向的棋形"""
    fwd = 0
    nr, nc = r, c
    while in_bounds(nr, nc) and board[nr][nc] == color:
        fwd += 1
        nr += dr
        nc += dc

    bwd = 0
    nr, nc = r-dr, c-dc
    while in_bounds(nr, nc) and board[nr][nc] == color:
        bwd += 1
        nr -= dr
        nc -= dc

    total = fwd + bwd

    if total >= 5:
        return 'five'

    end1_r, end1_c = r + fwd*dr, c + fwd*dc
    end2_r, end2_c = r - (bwd+1)*dr, c - (bwd+1)*dc
    open1 = in_bounds(end1_r, end1_c) and board[end1_r][end1_c] == EMPTY
    open2 = in_bounds(end2_r, end2_c) and board[end2_r][end2_c] == EMPTY

    if total == 4:
        return 'live_four' if (open1 and open2) else ('rush_four' if (open1 or open2) else 'other')
    elif total == 3:
        return 'live_three' if (open1 and open2) else ('rush_three' if (open1 or open2) else 'other')
    return 'other'


def get_all_patterns(board, color):
    patterns = {t:[] for t in ['five','live_four','rush_four','live_three','rush_three']}
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == color:
                for dr, dc in DIRECTIONS:
                    prev_r, prev_c = r-dr, c-dc
                    if in_bounds(prev_r, prev_c) and board[prev_r][prev_c] == color:
                        continue
                    pt = analyze_pattern(board, r, c, dr, dc, color)
                    if pt in patterns:
                        patterns[pt].append((r, c, (dr, dc)))
    return patterns


def evaluate_move(board, r, c, color):
    opp = WHITE if color == BLACK else BLACK
    temp = copy.deepcopy(board)
    temp[r][c] = color

    score = 0
    descriptions = []

    for dr, dc in DIRECTIONS:
        pt = analyze_pattern(temp, r, c, dr, dc, color)
        dn = DIR_NAMES[(dr,dc)]
        if pt == 'five':
            score += 100000; descriptions.append(f"{dn}连五，直接获胜")
        elif pt == 'live_four':
            score += 10000;  descriptions.append(f"{dn}活四，对手必须应对")
        elif pt == 'rush_four':
            score += 1000;   descriptions.append(f"{dn}冲四")
        elif pt == 'live_three':
            score += 200;    descriptions.append(f"{dn}活三，威胁成型")
        elif pt == 'rush_three':
            score += 50;     descriptions.append(f"{dn}冲三")

    temp2 = copy.deepcopy(board)
    temp2[r][c] = opp
    for dr, dc in DIRECTIONS:
        pt = analyze_pattern(temp2, r, c, dr, dc, opp)
        dn = DIR_NAMES[(dr,dc)]
        if pt == 'five':
            score += 90000;  descriptions.append(f"封堵对手{dn}连五（必须防守）")
        elif pt == 'live_four':
            score += 9000;   descriptions.append(f"封堵对手{dn}活四")
        elif pt == 'rush_four':
            score += 900;    descriptions.append(f"封堵对手{dn}冲四")
        elif pt == 'live_three':
            score += 180;    descriptions.append(f"封堵对手{dn}活三")

    return score, descriptions


def get_nearby_moves(board, radius=2):
    """获取有子周围radius格内的空位"""
    candidates = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        nr, nc = r+dr, c+dc
                        if in_bounds(nr, nc) and board[nr][nc] == EMPTY:
                            candidates.add((nr, nc))
    return candidates


# ==================== 推理链生成 ====================

def generate_cot(board, color, best_r, best_c):
    """
    生成与棋盘真实对应的推理链
    board: 落子前的棋盘
    color: 当前落子方
    best_r, best_c: 落子坐标
    """
    opp = WHITE if color == BLACK else BLACK
    color_sym  = "●" if color == BLACK else "○"
    color_name = "黑棋" if color == BLACK else "白棋"
    opp_name   = "白棋" if color == BLACK else "黑棋"
    best_str   = coord_to_str(best_r, best_c)

    # 扫描当前棋形（落子前）
    my_pats  = get_all_patterns(board, color)
    opp_pats = get_all_patterns(board, opp)

    # 评估候选点（包含best_move在内的附近点）
    candidates_set = get_nearby_moves(board, radius=2)
    candidates_set.add((best_r, best_c))  # 确保best_move在候选中

    scored = []
    for r, c in candidates_set:
        sc, descs = evaluate_move(board, r, c, color)
        scored.append((sc, r, c, descs))
    scored.sort(reverse=True)

    # 找best_move在评分列表中的位置
    top5 = scored[:5]
    best_in_top = any(r == best_r and c == best_c for _, r, c, _ in top5)
    if not best_in_top:
        # 把best_move加入显示列表
        best_score, best_descs = evaluate_move(board, best_r, best_c, color)
        top5 = top5[:4] + [(best_score, best_r, best_c, best_descs)]

    lines = ["<thinking>"]
    lines.append(f"轮到{color_name}（{color_sym}）落子，系统分析当前局面。")
    lines.append("")

    # 1. 棋形统计
    lines.append("【当前棋形统计】")
    my_desc = []
    if my_pats['live_four']:  my_desc.append(f"活四{len(my_pats['live_four'])}个")
    if my_pats['rush_four']:  my_desc.append(f"冲四{len(my_pats['rush_four'])}个")
    if my_pats['live_three']: my_desc.append(f"活三{len(my_pats['live_three'])}个")
    if my_pats['rush_three']: my_desc.append(f"冲三{len(my_pats['rush_three'])}个")
    lines.append(f"  {color_name}：{'、'.join(my_desc) if my_desc else '暂无显著棋形'}")

    opp_desc = []
    if opp_pats['five']:       opp_desc.append(f"连五{len(opp_pats['five'])}个")
    if opp_pats['live_four']:  opp_desc.append(f"活四{len(opp_pats['live_four'])}个（必须封堵）")
    if opp_pats['rush_four']:  opp_desc.append(f"冲四{len(opp_pats['rush_four'])}个")
    if opp_pats['live_three']: opp_desc.append(f"活三{len(opp_pats['live_three'])}个")
    lines.append(f"  {opp_name}：{'、'.join(opp_desc) if opp_desc else '暂无显著棋形'}")
    lines.append("")

    # 2. 威胁评估
    lines.append("【威胁评估】")
    if opp_pats['live_four']:
        r0, c0, _ = opp_pats['live_four'][0]
        lines.append(f"  ⚠️  紧急！{opp_name}在{coord_to_str(r0,c0)}附近已有活四，"
                     f"必须立即封堵，否则下一手对手获胜。")
    elif opp_pats['rush_four']:
        r0, c0, _ = opp_pats['rush_four'][0]
        lines.append(f"  ⚠️  {opp_name}在{coord_to_str(r0,c0)}附近有冲四威胁，需优先考虑防守。")
    elif opp_pats['live_three']:
        lines.append(f"  注意：{opp_name}有{len(opp_pats['live_three'])}个活三，"
                     f"若不防守将在2步内形成活四。")
    else:
        lines.append(f"  {opp_name}暂无紧迫威胁，{color_name}可主动进攻。")
    lines.append("")

    # 3. 候选点分析
    lines.append("【候选着法分析】")
    shown = 0
    for sc, r, c, descs in top5:
        if shown >= 3 and not (r == best_r and c == best_c):
            continue
        move_str = coord_to_str(r, c)
        is_best = (r == best_r and c == best_c)
        marker = "★ 最优" if is_best else f"候选{shown+1}"
        lines.append(f"  {marker}：{move_str}（评分{sc}）")
        if descs:
            for d in descs[:2]:
                lines.append(f"    - {d}")
        else:
            lines.append(f"    - 占据要点，积累优势")
        if not is_best:
            shown += 1
    lines.append("")

    # 4. 决策
    best_sc, best_descs = evaluate_move(board, best_r, best_c, color)
    main_reason = best_descs[0] if best_descs else "综合进攻防守价值最高"

    lines.append("【决策】")
    lines.append(f"  落在{best_str}，{main_reason}。")
    lines.append("</thinking>")
    lines.append(f"最佳落子：{best_str}")
    lines.append(f"理由：{main_reason}，{color_name}落{best_str}。")

    return "\n".join(lines)


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="./datasets/real_games_v2/train.json")
    parser.add_argument("--output", default="./datasets/real_games_cot/train.json")
    parser.add_argument("--n",      type=int, default=None, help="只处理前N条（默认全部）")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"读取数据：{args.input}")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    if args.n:
        data = data[:args.n]

    print(f"处理 {len(data)} 条样本...")

    results = []
    skipped = 0
    parse_errors = 0

    for item in tqdm(data):
        # 解析棋盘
        board, color = parse_board(item["instruction"])
        if board is None:
            parse_errors += 1
            skipped += 1
            continue

        # 方案B：直接用引擎最优解，推理链内部完全自洽
        candidates_set = get_nearby_moves(board, radius=2)
        if not candidates_set:
            skipped += 1
            continue

        scored = []
        for r, c in candidates_set:
            sc, descs = evaluate_move(board, r, c, color)
            scored.append((sc, r, c, descs))
        scored.sort(reverse=True)

        if not scored:
            skipped += 1
            continue

        best_sc, best_r, best_c, _ = scored[0]

        # 重新生成推理链
        try:
            new_output = generate_cot(board, color, best_r, best_c)
        except Exception as e:
            skipped += 1
            continue

        new_item = {
            "instruction": item["instruction"],
            "output":      new_output,
        }
        # 保留原始元数据字段
        for k in ["source", "game_file", "move_idx"]:
            if k in item:
                new_item[k] = item[k]

        results.append(new_item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成：{len(results)} 条成功，{skipped} 条跳过（解析失败{parse_errors}条）")
    print(f"输出：{args.output}（{os.path.getsize(args.output)/1024:.0f} KB）")

    # 打印2条样本对比
    print("\n=== 推理链样本预览 ===")
    for i, item in enumerate(results[:2]):
        print(f"\n--- 样本{i+1} ---")
        print(item["output"][:700])
        print("...")


if __name__ == "__main__":
    main()