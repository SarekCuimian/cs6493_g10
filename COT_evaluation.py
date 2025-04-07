# %%
import os
import re
import time
import pandas as pd
import csv
from collections import Counter
import dashscope
from dotenv import load_dotenv

# 加载环境变量中的 API 密钥
load_dotenv("dashscope_api_key.env")
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("❌ DASHSCOPE_API_KEY not found!")


def call_model(
        prompt: str,
        model_name: str = "qwen2.5-math-1.5b-instruct",
        system_prompt: str = (
                "You are a helpful assistant. "
                "Please show your reasoning step by step (Chain of Thought). "
                "Then, on a new line at the end, write:\n"
                "'Final Answer: <the numeric result>'."
        ),
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_retries: int = 3,
        retry_delay: float = 0.3
):
    """
    调用 DashScope 模型，带有简单的错误处理和明确的提示，要求模型输出最终答案行。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        response = dashscope.Generation.call(
            api_key=api_key,
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            result_format="message"
        )

        status_code = response.get("status_code", None)
        if status_code == 429:
            print(f"⏳ [Attempt {attempt + 1}/{max_retries}] Rate limit! Wait {retry_delay}s...")
            time.sleep(retry_delay)
            continue

        try:
            # 从响应中提取内容和令牌使用情况
            content = response["output"]["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            return content, usage
        except (TypeError, KeyError, IndexError):
            print(f"⚠️ [Attempt {attempt + 1}/{max_retries}] Unexpected structure:\n{response}")
            time.sleep(retry_delay)

    print(f"❌ All {max_retries} attempts failed, returning empty.")
    return "", {}


def extract_final_answer(response_text: str):
    """
    从模型响应中提取最终的数字答案。
    """
    # 尝试匹配 \boxed{...} 格式
    box_match = re.search(r"\\boxed\{([^}]*)\}", response_text, flags=re.DOTALL)
    if box_match:
        box_content = box_match.group(1)
        numbers_in_box = re.findall(r"\d+(?:\.\d+)?", box_content)
        if numbers_in_box:
            return numbers_in_box[0].strip()
        return box_content.strip()

    # 尝试匹配 "Final Answer: ..." 格式
    fa_match = re.search(r"(?i)final answer\s*:\s*([^\n]*)", response_text)
    if fa_match:
        return fa_match.group(1).strip()

    # 回退到匹配文本中的最后一个数字
    numbers = re.findall(r"\d+(?:\.\d+)?", response_text)
    if numbers:
        return numbers[-1]

    # 最终回退：返回整个字符串
    return response_text.strip()


def extract_numeric(ans: str):
    """
    清理提取的答案，去除 $、单位、逗号和额外的标记。
    """
    ans = ans.strip()
    ans = re.sub(r"[^0-9.\-]", "", ans)
    return ans


def extract_gold_answer(gold_text: str):
    """
    从 GSM8K 风格的 '#### <number>' 行中提取数字答案。
    """
    match = re.search(r"####\s*(\d+(?:\.\d+)?)", str(gold_text))
    if match:
        return match.group(1)
    numbers = re.findall(r"\d+(?:\.\d+)?", str(gold_text))
    if numbers:
        return numbers[-1]
    return str(gold_text).strip()


def is_correct_with_tolerance(pred: str, gold: str, epsilon: float = 1e-5):
    """
    在可容忍的误差范围内比较预测答案和真实答案。
    """
    try:
        return abs(float(pred) - float(gold)) < epsilon
    except ValueError:
        return pred.strip() == gold.strip()


def is_abnormal_answer(ans: str, max_repeat: int = 10, max_len: int = 100):
    ans = ans.strip()

    # 空字符串或过长的数字字符串
    if len(ans) > max_len:
        return True

    # 检查是否有重复字符，如 777777...
    if re.match(r"^(\d)\1{" + str(max_repeat) + r",}$", ans):
        return True

    return False


def evaluate_model_on_dataset(df, model_name, dataset_name, model_short_names):
    short_name = model_short_names.get(model_name, model_name.replace("/", "_"))
    save_path = f"results/COT_{dataset_name}_{short_name}.csv"

    # 恢复：加载现有的结果文件
    done_indices = set()
    if os.path.exists(save_path):
        try:
            existing_df = pd.read_csv(save_path)
            done_indices = set(existing_df["index"].tolist())
            print(f"🔁 Resuming from existing result file: {save_path}")
        except Exception as e:
            print(f"⚠️ Failed to read existing result file: {e}")

    for idx, row in df.iterrows():
        if idx in done_indices:
            if idx == len(done_indices) - 1:
                print(f"⏩ Skipping already completed index {idx}")
            continue

        if dataset_name == "gsm8k":
            question = row["question"]
            gold_answer = row["answer"]
        elif dataset_name == "aime":
            question = row["Question"]
            gold_answer = row["Answer"]

        print(f"\n=== Question {idx} ===")
        print("Question:", question)

        try:
            response_text, usage = call_model(
                prompt=question,
                model_name=model_name,
                temperature=0.7,
                top_p=0.9
            )
            pred = extract_final_answer(response_text)
        except Exception as e:
            print(f"[Error] {e}")
            pred = ""
            response_text = ""
            usage = {}

        gold_clean = extract_gold_answer(gold_answer)
        pred = extract_numeric(pred)

        if is_abnormal_answer(pred):
            print("⚠️ Abnormal pattern detected! Marking as invalid.")
            pred = "INVALID"
            correct = False
        else:
            correct = is_correct_with_tolerance(pred, gold_clean)

        response_length = len(response_text)

        print("Gold Extracted:", repr(gold_clean))
        print("Predicted Final Answer:", repr(pred))
        print("Matched Correctly?", correct)

        completion_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        total_tokens = usage.get("total_tokens", 0)

        if dataset_name == "gsm8k":
            result_row = {
                "index": idx,
                "gold_clean": gold_clean,
                "predicted_answer": pred,
                "correct": correct,
                "response_length": response_length,
                "confidence": 0,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            }
        elif dataset_name == "aime":
            result_row = {
                "index": idx,
                "id": row["ID"],
                "year": row["Year"],
                "problem_number": row["Problem Number"],
                "gold_clean": gold_clean,
                "predicted_answer": pred,
                "correct": correct,
                "response_length": response_length,
                "confidence": 0,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            }

        # 立即追加到 CSV 文件
        write_header = not os.path.exists(save_path)
        with open(save_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(result_row)

        print(f"✅ Saved result for index {idx}")


if __name__ == "__main__":
    # 加载 GSM8K 数据集
    df_gsm8k = pd.read_csv("dataset/GSM8K/main_test.csv")
    num_samples_gsm8k = -1  # -1 表示使用完整数据集
    subset_gsm8k = df_gsm8k if num_samples_gsm8k == -1 else df_gsm8k.head(num_samples_gsm8k)

    # 加载 AIME 数据集
    df_aime = pd.read_csv("dataset/AIME_Dataset_1983_2024.csv")
    num_samples_aime = -1
    subset_aime = df_aime if num_samples_aime == -1 else df_aime.head(num_samples_aime)

    # 要评估的模型
    models_to_test = [
        "qwen2.5-math-1.5b-instruct"
    ]

    model_short_names = {
        "qwen2.5-math-1.5b-instruct": "qwen",
    }

    # 在 GSM8K 数据集上评估模型
    for model_name in models_to_test:
        print(f"\n\n==================== Evaluating Model: {model_name} on GSM8K ====================")
        evaluate_model_on_dataset(subset_gsm8k, model_name, "gsm8k", model_short_names)

    # 在 AIME 数据集上评估模型
    for model_name in models_to_test:
        print(f"\n\n==================== Evaluating Model: {model_name} on AIME ====================")
        evaluate_model_on_dataset(subset_aime, model_name, "aime", model_short_names)

    # GSM8K 单题测试
    target_index_gsm8k = 154
    sample_gsm8k = df_gsm8k.iloc[target_index_gsm8k]
    question_text_gsm8k = sample_gsm8k["question"]
    gold_answer_raw_gsm8k = sample_gsm8k["answer"]

    response_text_gsm8k, usage_gsm8k = call_model(
        prompt=question_text_gsm8k,
        model_name="qwen2.5-math-1.5b-instruct",
        temperature=0.7,
        top_p=0.9
    )
    predicted_gsm8k = extract_final_answer(response_text_gsm8k)

    gold_clean_gsm8k = extract_gold_answer(gold_answer_raw_gsm8k)
    pred_clean_gsm8k = extract_numeric(predicted_gsm8k)
    is_correct_gsm8k = (pred_clean_gsm8k == gold_clean_gsm8k)
    response_length_gsm8k = len(response_text_gsm8k)

    print("\n=== GSM8K Single Question Test ===")
    print("Index:", target_index_gsm8k)
    print("Gold Raw:", repr(gold_answer_raw_gsm8k))
    print("Question:", question_text_gsm8k)
    print("Predicted Final Answer:", repr(pred_clean_gsm8k))
    print("Gold Answer:", repr(gold_clean_gsm8k))
    print("Correct?:", is_correct_gsm8k)
    print(f"Response Text: {response_text_gsm8k}")
    print(f"Response Length: {response_length_gsm8k} chars")
    print(f"Confidence: 0")
    print("Token Usage:", usage_gsm8k)

    # AIME 单题测试
    target_index_aime = 154
    sample_aime = df_aime.iloc[target_index_aime]
    question_text_aime = sample_aime["Question"]
    gold_answer_raw_aime = sample_aime["Answer"]

    response_text_aime, usage_aime = call_model(
        prompt=question_text_aime,
        model_name="qwen2.5-math-1.5b-instruct",
        temperature=0.7,
        top_p=0.9
    )
    predicted_aime = extract_final_answer(response_text_aime)

    gold_clean_aime = extract_gold_answer(gold_answer_raw_aime)
    pred_clean_aime = extract_numeric(predicted_aime)
    is_correct_aime = is_correct_with_tolerance(pred_clean_aime, gold_clean_aime)
    response_length_aime = len(response_text_aime)

    print("\n=== AIME Single Question Test ===")
    print("Index:", target_index_aime)
    print("Year:", sample_aime.get("Year", "N/A"))
    print("Problem Number:", sample_aime.get("Problem Number", "N/A"))
    print("Question:", question_text_aime)
    print("Gold Raw:", repr(gold_answer_raw_aime))
    print("Gold Cleaned:", repr(gold_clean_aime))
    print("Predicted Answer:", repr(pred_clean_aime))
    print("Correct?:", is_correct_aime)
    print("Confidence:", f"{0:.2f}")
    print("Response Length:", response_length_aime)
    print("Token Usage:", usage_aime)
    print("\n=== Full Model Response ===\n")
    print(response_text_aime)
