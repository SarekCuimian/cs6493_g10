import os
import time
import pandas as pd
from dotenv import load_dotenv
import dashscope
import re

# Load environment variable (DASHSCOPE_API_KEY)
load_dotenv("dashscope_api_key.env")
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("❌ DASHSCOPE_API_KEY not found!")


def call_model(
        prompt: str,
        model_name: str,
        system_prompt: str = (
                "You are a helpful assistant. "
                "Please show your reasoning step by step (Chain of Thought). "
                "Then, on a new line at the end, write: 'Final Answer: <the result>'."
        ),
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_retries: int = 3,
        retry_delay: float = 0.3
) -> tuple[str, dict]:
    """
    Call the first model and return the full text response along with usage info.
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
            print(f"⏳ [Attempt {attempt + 1}/{max_retries}] Rate limit! Waiting {retry_delay}s...")
            time.sleep(retry_delay)
            continue
        try:
            content = response["output"]["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            return content, usage
        except (TypeError, KeyError, IndexError):
            print(f"⚠️ [Attempt {attempt + 1}/{max_retries}] Unexpected structure:\n{response}")
            time.sleep(retry_delay)
    print("❌ All attempts failed, returning empty result.")
    return "", {}


def judge_answers(
        responses_texts: list[str],
        reference_answer: str,
        judge_model: str,
        system_prompt: str = (
                "You are a strict judge. Given multiple model responses and a reference answer,\n"
                "1. Find the majority final answer (most frequent; tie-breaker: first occurrence).\n"
                "2. Check if it matches the reference answer.\n"
                "3. Respond EXACTLY with:\n"
                "Correct: True or False\n"
                "SelectedIndices: comma-separated indices (1-indexed)\n"
                "Do NOT include any reasoning, code, or LaTeX."
        ),
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 0.3
) -> dict:
    """
    Return both correctness, a confidence score, and the indices (1 - indexed) of the responses
    that are considered as the majority answer.
    """
    user_prompt = f"Reference answer (LaTeX): {reference_answer}\n\n"
    user_prompt += f"Below are {len(responses_texts)} responses from the model:\n"
    for i, text in enumerate(responses_texts, start=1):
        user_prompt += f"\n--- Response {i} ---\n{text}\n"
    user_prompt += (
        "\nQuestion: Are the majority of these responses correct?\n"
        "Respond in the format:\n"
        "Correct: True or False\n"
        "SelectedIndices: <comma separated indices (starting from 1)>\n"
    )

    for attempt in range(max_retries):
        response = dashscope.Generation.call(
            api_key=api_key,
            model=judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            result_format="message"
        )
        try:
            text = response["output"]["choices"][0]["message"]["content"]
            print(f"Judge Model Response (Attempt {attempt + 1}): {text}")
            status_code = response.get("status_code", None)
            print(f"Judge Model Status Code (Attempt {attempt + 1}): {status_code}")
        except (TypeError, KeyError, IndexError):
            print(f"Unexpected response structure (Attempt {attempt + 1}): {response}")
            time.sleep(retry_delay)
            continue

        # 调整正则表达式以匹配可能的关键词变化
        match = re.search(r"(?i)correct:\s*(true|false)", text, flags=re.IGNORECASE | re.MULTILINE)
        sel_match = re.search(r"(?i)(selectedindices|selected integers):\s*([\d\s,]+)", text,
                              flags=re.IGNORECASE | re.MULTILINE)

        if match and sel_match:
            is_correct = match.group(1).strip().lower() == "true"
            selected_indices_str = sel_match.group(2).strip()
            print(f"Selected indices string: {selected_indices_str}")

            # 增强索引解析逻辑：支持逗号/空格分隔，并去除每个索引的前后空格
            delimiters = [',', ' ']
            indices = []
            current = []
            for char in selected_indices_str:
                if char in delimiters and current:
                    indices.append(''.join(current).strip())
                    current = []
                else:
                    current.append(char)
            if current:
                indices.append(''.join(current).strip())

            try:
                selected_indices = [int(idx) for idx in indices if idx.isdigit()]
            except ValueError:
                selected_indices = []
            print(f"Parsed selected indices: {selected_indices}")

            return {"correct": is_correct, "selected_indices": selected_indices}
    print("❌ judge_answers: failed all retries.")
    return {"correct": False, "selected_indices": []}


def test_call_model():
    model_name = "qwen2.5-math-1.5b-instruct"
    prompt = "If $f(x) = \frac{3x - 2}{x - 2}$, what is the value of $f(-2)$?"
    output, usage = call_model(prompt, model_name)
    print("Call Model Test:")
    if output:
        print("Call model succeeded.")
        print("Output:", output)
        print("Usage:", usage)
    else:
        print("Call model failed.")


def test_judge_answers_simple():
    judge_model = "qwen2.5-math-1.5b-instruct"
    responses_texts = [
        "Some reasoning... Final Answer: 1",
        "Another reasoning... Final Answer: 1",
        "More reasoning... Final Answer: 1",
        "Even more reasoning... Final Answer: 1",
        "Last reasoning... Final Answer: 1"
    ]
    reference_answer = "1"
    print("\nTest Judge Answers (Simple Case):")
    judgement = judge_answers(responses_texts, reference_answer, judge_model)
    if judgement["selected_indices"]:
        print("Judge answers succeeded.")
        print("Judgement:", judgement)
    else:
        print("Judge answers failed.")


def test_judge_answers_complex():
    judge_model = "qwen2.5-math-1.5b-instruct"
    responses_texts = [
        "Some reasoning... Final Answer: 1",
        "Another reasoning... Final Answer: 2",
        "More reasoning... Final Answer: 1",
        "Even more reasoning... Final Answer: 3",
        "Last reasoning... Final Answer: 1"
    ]
    reference_answer = "1"
    print("\nTest Judge Answers (Complex Case):")
    judgement = judge_answers(responses_texts, reference_answer, judge_model)
    if judgement["selected_indices"]:
        print("Judge answers succeeded.")
        print("Judgement:", judgement)
    else:
        print("Judge answers failed.")


if __name__ == "__main__":
    test_call_model()
    test_judge_answers_simple()
    test_judge_answers_complex()

