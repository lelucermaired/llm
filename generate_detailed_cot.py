"""
generate_detailed_cot_qwen.py

调用Qwen API生成详细因果推理链
API文档：https://help.aliyun.com/zh/model-studio/

用法：
  python generate_detailed_cot_qwen.py --n 200
  python generate_detailed_cot_qwen.py --n 200 --resume
"""

import json, re, copy, argparse, os, time, urllib.request, urllib.error
from tqdm import tqdm

# ==================== 配置 ====================

API_KEY    = "sk-d32a252f0a3b4dc1bb91b29423639e7f"
API_URL    = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL_NAME = "qwen-max"   # 可换 qwen-turbo（更快更便宜）或 qwen-max（质量更好）

BOARD_SIZE = 15
DIRECTIONS = [(1,0),(0,1),(1,1),(1,-1)]
DIR_NAMES  = {(1,0):"横向",(0,1):"纵向",(1,1):"正斜",(1,-1):"反斜"}
BLACK, WHITE, EMPTY = 1, 2, 0
COL_LETTERS = "ABCDEFGHIJKLMNO"

SYSTEM_PROMPT = """你是一个五子棋分析专家，负责生成详细的推理链用于训练数据。

生成要求（必须全部满足）：
1. 引用具体坐标说明威胁，如"黑棋在K3-K4-K5已有3子连线，两端K2和K6均为空位，构成活三"
2. 每步结论必须依赖前一步分析，形成因果链条
3. 必须包含反事实推理，如"若不在F9封堵，对手下一手在G10即可形成活四，届时黑棋将无法同时防守两个方向"
4. 最终决策说明为何选择该点而非其他候选
5. 推理链用<thinking>...</thinking>包裹
6. 最后给出"最佳落子：XX"和一句话理由
7. 总长度300-500字，简明但有实质内容"""


# ==================== 棋盘解析工具 ====================

def in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

def coord_to_str(r, c):
    return f"{COL_LETTERS[c]}{r+1}"

def parse_board(instruction):
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

def analyze_pattern(board, r, c, dr, dc, color):
    fwd = 0
    nr, nc = r, c
    while in_bounds(nr, nc) and board[nr][nc] == color:
        fwd += 1; nr += dr; nc += dc
    bwd = 0
    nr, nc = r-dr, c-dc
    while in_bounds(nr, nc) and board[nr][nc] == color:
        bwd += 1; nr -= dr; nc -= dc
    total = fwd + bwd
    if total >= 5: return 'five'
    end1_r, end1_c = r+fwd*dr, c+fwd*dc
    end2_r, end2_c = r-(bwd+1)*dr, c-(bwd+1)*dc
    open1 = in_bounds(end1_r, end1_c) and board[end1_r][end1_c] == EMPTY
    open2 = in_bounds(end2_r, end2_c) and board[end2_r][end2_c] == EMPTY
    if total == 4:
        return 'live_four' if (open1 and open2) else ('rush_four' if (open1 or open2) else 'other')
    elif total == 3:
        return 'live_three' if (open1 and open2) else ('rush_three' if (open1 or open2) else 'other')
    return 'other'

def get_stone_line(board, r, c, dr, dc, color):
    """返回从(r,c)起该方向的连续同色子坐标列表"""
    stones = []
    nr, nc = r, c
    while in_bounds(nr, nc) and board[nr][nc] == color:
        stones.append(coord_to_str(nr, nc))
        nr += dr; nc += dc
    nr, nc = r-dr, c-dc
    while in_bounds(nr, nc) and board[nr][nc] == color:
        stones.insert(0, coord_to_str(nr, nc))
        nr -= dr; nc -= dc
    return stones

def get_threats(board, color):
    """获取所有威胁，含具体坐标"""
    threats = []
    seen = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == color:
                for dr, dc in DIRECTIONS:
                    prev_r, prev_c = r-dr, c-dc
                    if in_bounds(prev_r, prev_c) and board[prev_r][prev_c] == color:
                        continue
                    pt = analyze_pattern(board, r, c, dr, dc, color)
                    if pt in ('five','live_four','rush_four','live_three'):
                        stones = get_stone_line(board, r, c, dr, dc, color)
                        key = (tuple(sorted(stones)), pt)
                        if key not in seen:
                            seen.add(key)
                            # 两端空位
                            fwd = len([s for s in stones])
                            end1_r = r + fwd*dr
                            end1_c = c + fwd*dc
                            bwd_start_r = r - (len(stones)-fwd+1)*dr if fwd < len(stones) else r-dr
                            # 简化：只记录连子和类型
                            threats.append({
                                "type": pt,
                                "dir": DIR_NAMES[(dr,dc)],
                                "stones": stones,
                            })
    return threats

def evaluate_move(board, r, c, color):
    opp = WHITE if color == BLACK else BLACK
    temp = copy.deepcopy(board)
    temp[r][c] = color
    score = 0
    attack, defend = [], []
    for dr, dc in DIRECTIONS:
        pt = analyze_pattern(temp, r, c, dr, dc, color)
        dn = DIR_NAMES[(dr,dc)]
        stones = get_stone_line(temp, r, c, dr, dc, color)
        ss = "-".join(stones)
        if pt == 'five':     score += 100000; attack.append(f"{dn}{ss}连五")
        elif pt == 'live_four': score += 10000; attack.append(f"{dn}{ss}活四")
        elif pt == 'rush_four': score += 1000;  attack.append(f"{dn}{ss}冲四")
        elif pt == 'live_three':score += 200;   attack.append(f"{dn}{ss}活三")
        elif pt == 'rush_three':score += 50;    attack.append(f"{dn}{ss}冲三")
    temp2 = copy.deepcopy(board)
    temp2[r][c] = opp
    for dr, dc in DIRECTIONS:
        pt = analyze_pattern(temp2, r, c, dr, dc, opp)
        dn = DIR_NAMES[(dr,dc)]
        stones = get_stone_line(temp2, r, c, dr, dc, opp)
        ss = "-".join(stones)
        if pt == 'five':     score += 90000; defend.append(f"封堵对手{dn}{ss}连五")
        elif pt == 'live_four': score += 9000; defend.append(f"封堵对手{dn}{ss}活四")
        elif pt == 'rush_four': score += 900;  defend.append(f"封堵对手{dn}{ss}冲四")
        elif pt == 'live_three':score += 180;  defend.append(f"封堵对手{dn}{ss}活三")
    return score, attack, defend

def get_top_candidates(board, color, top_n=5):
    candidates = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                for dr in range(-2,3):
                    for dc in range(-2,3):
                        nr, nc = r+dr, c+dc
                        if in_bounds(nr,nc) and board[nr][nc] == EMPTY:
                            candidates.add((nr,nc))
    scored = []
    for r, c in candidates:
        sc, atk, dfn = evaluate_move(board, r, c, color)
        scored.append((sc, r, c, atk, dfn))
    scored.sort(reverse=True)
    return scored[:top_n]


# ==================== Prompt构建 ====================

def build_prompt(instruction, board, color):
    color_name = "黑棋（●）" if color == BLACK else "白棋（○）"
    opp = WHITE if color == BLACK else BLACK
    opp_name = "白棋（○）" if color == BLACK else "黑棋（●）"

    my_threats  = get_threats(board, color)
    opp_threats = get_threats(board, opp)
    top5 = get_top_candidates(board, color, top_n=5)

    if not top5:
        return None, None
    best_move = coord_to_str(top5[0][1], top5[0][2])

    # 威胁描述
    def fmt_threats(ts):
        if not ts:
            return "  暂无显著威胁"
        lines = []
        for t in ts[:4]:
            stones_str = "→".join(t["stones"])
            lines.append(f"  {t['dir']} {stones_str}（{t['type']}）")
        return "\n".join(lines)

    # 候选点描述
    cand_lines = []
    for sc, r, c, atk, dfn in top5[:4]:
        mv = coord_to_str(r, c)
        details = []
        if atk: details.append("进攻：" + "、".join(atk[:2]))
        if dfn: details.append("防守：" + "、".join(dfn[:2]))
        detail_str = "；".join(details) if details else "占据要点"
        cand_lines.append(f"  {mv}（评分{sc}）：{detail_str}")
    cand_str = "\n".join(cand_lines)

    prompt = f"""以下是一个五子棋局面，请为训练数据生成详细推理链。

棋盘：
{instruction}

引擎分析结果：
当前落子方：{color_name}
对手：{opp_name}

{color_name}现有威胁：
{fmt_threats(my_threats)}

{opp_name}现有威胁：
{fmt_threats(opp_threats)}

候选落点（引擎评分从高到低）：
{cand_str}

引擎最优落点：{best_move}

请生成详细推理链，最终选择{best_move}落子。
必须包含：具体坐标引用、因果推理链、反事实分析（若不选{best_move}会发生什么）。"""

    return prompt, best_move


# ==================== API调用 ====================

def call_qwen_api(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            payload = json.dumps({
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 1000,
                "temperature": 0.7,
            }).encode("utf-8")

            req = urllib.request.Request(
                API_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            print(f"\n  HTTP {e.code}：{body[:200]}")
            if e.code == 429:
                wait = 2 ** (attempt + 2)
                print(f"  限速，等待{wait}秒...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"\n  错误（第{attempt+1}次）：{e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="./datasets/real_games_v2/train.json")
    parser.add_argument("--output", default="./datasets/real_games_detailed_cot/train.json")
    parser.add_argument("--n",      type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"读取数据：{args.input}")
    with open(args.input, encoding="utf-8") as f:
        source = json.load(f)

    # 断点续传
    results = []
    start_idx = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"断点续传：已有{start_idx}条")

    target = min(args.n, len(source))
    todo = source[start_idx:target]
    print(f"需要生成：{len(todo)} 条  |  模型：{MODEL_NAME}")
    print()

    errors = 0
    for i, item in enumerate(todo):
        idx = start_idx + i + 1
        print(f"[{idx}/{target}] ", end="", flush=True)

        board, color = parse_board(item["instruction"])
        if board is None:
            print("棋盘解析失败，跳过")
            errors += 1
            continue

        prompt, best_move = build_prompt(item["instruction"], board, color)
        if prompt is None:
            print("无候选点，跳过")
            errors += 1
            continue

        print(f"最优={best_move} ... ", end="", flush=True)

        response = call_qwen_api(prompt)
        if response is None:
            print("API失败，跳过")
            errors += 1
            continue

        print("✅")

        new_item = {
            "instruction": item["instruction"],
            "output": response,
        }
        for k in ["source", "game_file", "move_idx"]:
            if k in item:
                new_item[k] = item[k]
        results.append(new_item)

        # 每10条保存
        if len(results) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  → 已保存{len(results)}条")

        time.sleep(0.3)  # 避免限速

    # 最终保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成：{len(results)} 条，{errors} 条失败")
    print(f"输出：{args.output}（{os.path.getsize(args.output)/1024:.0f} KB）")

    if results:
        print("\n=== 样本预览 ===")
        print(results[0]["output"][:600])
        print("...")


if __name__ == "__main__":
    main()