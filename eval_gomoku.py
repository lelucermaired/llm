import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ==================== 配置 ====================
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "lora_gomoku_cpu"  # 你的LoRA适配器路径
TEST_CASE_FILE = "gomoku_test_cases.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================

def load_model_and_tokenizer(is_lora):
    """加载模型：基座模型 or 微调模型"""
    print(f"\n=== 正在加载 {'LoRA微调' if is_lora else '原始基座'}模型 ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if is_lora:
        # 方法1: 先加载基础模型到CPU，再加载LoRA权重
        try:
            print("尝试方法1: 使用device_map='auto'加载...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            # 加载LoRA权重
            model = PeftModel.from_pretrained(
                base_model,
                LORA_PATH,
                device_map="auto" if DEVICE == "cuda" else None
            )
        except Exception as e1:
            print(f"方法1失败: {e1}")

            # 方法2: 如果方法1失败，尝试完全加载到CPU
            try:
                print("尝试方法2: 完全加载到CPU...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )

                # 加载LoRA权重到CPU
                model = PeftModel.from_pretrained(
                    base_model,
                    LORA_PATH,
                    device_map=None
                )

                # 如果需要GPU，移动模型到GPU
                if DEVICE == "cuda":
                    model = model.to(DEVICE)

            except Exception as e2:
                print(f"方法2失败: {e2}")

                # 方法3: 使用PeftConfig
                try:
                    print("尝试方法3: 使用PeftConfig...")
                    # 先检查LoRA配置
                    config = PeftConfig.from_pretrained(LORA_PATH)

                    # 加载基础模型
                    base_model = AutoModelForCausalLM.from_pretrained(
                        config.base_model_name_or_path,
                        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                        device_map="auto" if DEVICE == "cuda" else None,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )

                    # 加载LoRA模型
                    model = PeftModel.from_pretrained(base_model, LORA_PATH)

                except Exception as e3:
                    print(f"方法3失败: {e3}")
                    raise RuntimeError("所有加载LoRA的方法都失败了")
    else:
        # 加载原始基座模型
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True
        )

    model.eval()

    # 检查模型是否在正确的设备上
    if DEVICE == "cuda":
        try:
            model = model.to(DEVICE)
        except:
            print("模型已经在正确的设备上")

    return model, tokenizer


def ask_model(model, tokenizer, prompt):
    """向模型提问并获取回答"""
    # 确保模型在正确的设备上
    if DEVICE == "cuda":
        model = model.to(DEVICE)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # 贪婪解码保证可复现
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取模型实际回答的部分（移除提示词）
    response = response[len(prompt):].strip() if len(response) > len(prompt) else response
    return response


def evaluate_answer(test_case, answer, model_name):
    """根据测试类型评估答案是否正确（简单关键词匹配）"""
    expected = test_case.get("expected_answer", "")
    keywords = test_case.get("expected_answer_keywords", [])

    print(f"\n【{test_case['id']}】{test_case['type']} - {model_name}")
    print(f"问题：{test_case['question'][:50]}...")
    print(f"模型回答：{answer[:100]}...")

    # 规则类问题：检查关键词
    if test_case["type"] == "rule":
        correct = any(keyword in answer for keyword in keywords)
        if correct:
            print("✅ 回答正确（包含关键规则描述）")
        else:
            print(f"❌ 回答可能不准确。期望关键词：{keywords}")
        return correct

    # 识别/决策类问题：检查精确答案或关键词
    if expected and expected in answer:
        print(f"✅ 回答正确（匹配预期答案：{expected}）")
        return True
    elif keywords and any(keyword in answer for keyword in keywords):
        print(f"✅ 回答合理（包含关键信息：{keywords}）")
        return True
    else:
        print(f"❌ 回答未达预期。期望：{expected or keywords}")
        return False


def main():
    # 1. 加载测试用例
    print("加载五子棋测试用例...")
    try:
        with open(TEST_CASE_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到测试用例文件 {TEST_CASE_FILE}")
        # 创建示例测试用例
        test_cases = [
            {
                "id": "001",
                "type": "rule",
                "question": "五子棋的基本胜利条件是什么？",
                "expected_answer_keywords": ["五个", "连成一线", "横竖斜"]
            }
        ]
        print("使用示例测试用例继续执行...")

    # 2. 分别测试两个模型
    results = {"base": [], "lora": []}

    for model_name, is_lora in [("Base", False), ("LoRA", True)]:
        print(f"\n{'=' * 50}")
        print(f"开始评估 {model_name} 模型")
        print('=' * 50)

        try:
            model, tokenizer = load_model_and_tokenizer(is_lora)
            correct_count = 0

            for i, test_case in enumerate(test_cases):
                # 构造提示词
                prompt = f"""你是一个五子棋专家。请根据以下信息精确回答问题。
{test_case.get('board', '')}
问题：{test_case['question']}
请直接给出答案，无需额外解释。"""

                answer = ask_model(model, tokenizer, prompt)
                is_correct = evaluate_answer(test_case, answer, model_name)

                if is_correct:
                    correct_count += 1

                results["lora" if is_lora else "base"].append({
                    "id": test_case["id"],
                    "question": test_case["question"],
                    "answer": answer,
                    "correct": is_correct
                })

                # 清理GPU缓存（如果有）
                if DEVICE == "cuda" and i % 2 == 0:
                    torch.cuda.empty_cache()

            # 3. 打印当前模型总体表现
            accuracy = correct_count / len(test_cases) if test_cases else 0
            print(f"\n📊 {model_name} 模型总体准确率：{correct_count}/{len(test_cases)} = {accuracy:.1%}")

        except Exception as e:
            print(f"加载{model_name}模型失败: {e}")
            print(f"跳过{model_name}模型的评估")
            continue

        finally:
            # 清理模型释放显存
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # 4. 对比分析
    print(f"\n{'=' * 60}")
    print("最终对比分析")
    print('=' * 60)

    if results["base"] and results["lora"]:
        base_correct = sum(1 for item in results["base"] if item["correct"])
        lora_correct = sum(1 for item in results["lora"] if item["correct"])
        total = len(test_cases)

        print(f"原始基座模型 正确数：{base_correct}/{total}")
        print(f"LoRA微调模型 正确数：{lora_correct}/{total}")
        print(f"绝对提升：{lora_correct - base_correct} 题")
    else:
        print("至少有一个模型加载失败，无法进行对比分析")

    # 5. 保存详细结果
    output_file = "gomoku_eval_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_cases": test_cases,
                "results": results,
                "summary": {
                    "base_accuracy": base_correct / total if results["base"] else 0,
                    "lora_accuracy": lora_correct / total if results["lora"] else 0,
                    "improvement": lora_correct - base_correct if results["base"] and results["lora"] else 0
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存至：{output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")


if __name__ == "__main__":
    # 设置更大的分页文件以处理大模型
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    main()