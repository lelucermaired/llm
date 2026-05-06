import json
import os
import re

INPUT_PATH = "datasets/go_cot_train.json"
OUTPUT_PATH = "datasets/go_nocot_train.json"
OUTPUT_PREFIX = "最佳落子："

COORD_PATTERN = re.compile(r"\b([A-T](?:1[0-9]|[1-9]))\b", re.IGNORECASE)


def extract_final_move(text: str):
    if not isinstance(text, str):
        return None

    pos = text.rfind("</thinking>")
    if pos != -1:
        tail = text[pos:]
        matches = COORD_PATTERN.findall(tail)
        if matches:
            return matches[-1].upper()

    matches = COORD_PATTERN.findall(text)
    if matches:
        return matches[-1].upper()

    return None


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    failed = []

    for i, item in enumerate(data):
        if "instruction" not in item or "output" not in item:
            failed.append((i, "missing fields"))
            continue

        move = extract_final_move(item["output"])
        if move is None:
            failed.append((i, "cannot extract final move"))
            continue

        new_item = dict(item)
        new_item["output"] = f"{OUTPUT_PREFIX}{move}"
        new_data.append(new_item)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"输入样本数: {len(data)}")
    print(f"转换成功:   {len(new_data)}")
    print(f"转换失败:   {len(failed)}")
    print(f"输出文件:   {OUTPUT_PATH}")
    if failed:
        print("前10个失败样本:")
        for x in failed[:10]:
            print(x)
    print("=" * 60)


if __name__ == "__main__":
    main()
