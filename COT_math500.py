import os
import csv
import time
import pandas as pd
import dashscope
import re
from dotenv import load_dotenv

# ========== 加载API密钥 ==========
load_dotenv("dashscope_api_key.env")
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("❌ DASHSCOPE_API_KEY not found!")


def clean_latex_expression(expr: str) -> str:
    """规范化LaTeX表达式到可比较的格式"""
    # 第一步：移除所有排版命令和多余字符（修复正则表达式）
    expr = re.sub(
        r'\$?(left|right|big[gl]?|Big[gl]?|\,|:|;|mathrm|text|mathbf|boxed|displaystyle)'
        r'|[{}()\$]|\s+',
        '',
        expr
    )

    # 统一数学符号表示
    replacements = [
        (r'\\pi|π', 'π'),  # 统一希腊字母
        (r'\\times|\*|⋅', '×'),  # 统一乘法符号
        (r'\\frac{([^}]+)}{([^}]+)}', r'\1/\2'),  # 分数转换
        (r'^\\', ''),  # 移除残留命令符
    ]

    for pattern, repl in replacements:
        expr = re.sub(pattern, repl, expr)

    # 最终清理（转为小写去除非匹配差异）
    return expr.lower().strip()


# ========== 增强型答案提取 ==========
def extract_final_answer(text: str) -> str:
    """从模型响应中提取并规范化最终答案"""
    try:
        # 优先级1：提取boxed内容
        boxed_match = re.search(r'\\boxed{([^{}]*)}', text, re.DOTALL)
        if boxed_match:
            return clean_latex_expression(boxed_match.group(1))

        # 优先级2：Final Answer模式
        final_match = re.search(
            r'Final Answer\s*:\s*([^\n$]+)(?:\$|\\boxed|$)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if final_match:
            return clean_latex_expression(final_match.group(1))

        # 优先级3：查找最后一个数学表达式
        math_matches = re.findall(r'\$([^$]+)\$', text)
        if math_matches:
            return clean_latex_expression(math_matches[-1])

        # 回退策略：最后非空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return clean_latex_expression(lines[-1]) if lines else ""

    except Exception as e:
        print(f"⚠️ Error extracting answer: {str(e)}")
        return ""


# ========== 增强型判断逻辑 ==========
def judge_answers(
        responses_texts: list[str],
        reference_answer: str,
        judge_model: str,
        system_prompt: str = "比较回答与参考答案的数学等价性。最后用'Correct: True/False'和'SelectedIndices: [数字列表]'分行输出",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 0.3
) -> dict:
    # 标准化参考答案
    ref_clean = clean_latex_expression(reference_answer)
    print(f"\n🔍 参考答案标准化结果: {ref_clean}")

    # 收集所有标准化后的回答
    final_answers = [extract_final_answer(text) for text in responses_texts]
    print("📝 模型回答标准化结果:")
    for i, ans in enumerate(final_answers, 1):
        print(f"  回答{i}: {ans}")

    # 阶段1：精确匹配检查
    exact_matches = [i + 1 for i, ans in enumerate(final_answers) if ans == ref_clean]
    if exact_matches:
        print(f"✅ 精确匹配到回答: {exact_matches}")
        return {
            "correct": True,
            "selected_indices": exact_matches,
            "judge_response": f"Exact matches found: {exact_matches}"
        }

    # 阶段2：数值等价检查
    try:
        ref_num = float(ref_clean)
        num_matches = []
        for i, ans in enumerate(final_answers):
            try:
                if abs(float(ans) - ref_num) < 1e-6:
                    num_matches.append(i + 1)
            except ValueError:
                continue
        if num_matches:
            print(f"🔢 数值等价匹配: {num_matches}")
            return {
                "correct": True,
                "selected_indices": num_matches,
                "judge_response": f"Numerical matches: {num_matches}"
            }
    except ValueError:
        pass

    # 阶段3：调用大模型判断
    user_prompt = f"""\n参考答案：{ref_clean}\n\n请判断以下回答是否与参考答案数学等价：
    """ + '\n'.join([f"{i + 1}. {ans}" for i, ans in enumerate(final_answers)])

    for attempt in range(max_retries):
        try:
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

            if response.status_code != 200:
                raise Exception(f"API响应异常: {response}")

            judge_text = response.output.choices[0].message.content
            print(f"\n🧠 大模型判断原始响应:\n{judge_text}")

            # 解析判断结果
            correct = "correct: true" in judge_text.lower()
            indices = [int(i) for i in re.findall(r'\b\d+\b', judge_text) if i.isdigit()]
            valid_indices = [i for i in indices if 1 <= i <= len(responses_texts)]

            return {
                "correct": correct or bool(valid_indices),
                "selected_indices": valid_indices or [],
                "judge_response": judge_text
            }

        except Exception as e:
            print(f"⚠️ 第{attempt + 1}次判断请求失败: {str(e)}")
            time.sleep(retry_delay)

    # 阶段4：多数投票回退
    answer_counts = {}
    for ans in final_answers:
        answer_counts[ans] = answer_counts.get(ans, 0) + 1
    majority_answer, max_count = max(answer_counts.items(), key=lambda x: x[1])

    selected_indices = [i + 1 for i, ans in enumerate(final_answers) if ans == majority_answer]
    correct_status = (majority_answer == ref_clean) or (max_count / len(final_answers) > 0.5)

    print(f"⚖️ 多数投票结果：{selected_indices} (置信度 {max_count / len(final_answers):.0%})")
    return {
        "correct": correct_status,
        "selected_indices": selected_indices,
        "judge_response": "Fallback to majority voting"
    }


# ========== 模型调用模块 ==========
def call_model(
        prompt: str,
        model_name: str,
        system_prompt: str = (
                "你是一位数学问题解决专家。请使用逐步推理（思维链）展示您的解题过程，"
                "并在最后以格式'Final Answer: <答案>'给出最终结论。"
        ),
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_retries: int = 3,
        retry_delay: float = 0.3
) -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = dashscope.Generation.call(
                api_key=api_key,
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                result_format="message"
            )

            if response.get("status_code") == 429:
                print(f"⏳ 第{attempt + 1}次重试（频率限制）...")
                time.sleep(retry_delay)
                continue

            if response.status_code != 200:
                raise Exception(f"API错误代码 {response.status_code}")

            content = response.output.choices[0].message.content
            usage = response.usage or {}
            return content, usage

        except Exception as e:
            print(f"⚠️ 第{attempt + 1}次调用失败: {str(e)}")
            time.sleep(retry_delay)

    print("❌ 模型调用全部失败")
    return "", {}


# ========== 主执行流程 ==========
if __name__ == "__main__":
    # 配置参数
    data_path = "dataset/MATH-500/math500_processed.csv"
    models_to_test = ["qwen2.5-math-1.5b-instruct"]
    samples_per_question = 3
    temperature = 0.7
    top_p = 0.9

    # 加载数据集
    df = pd.read_csv(data_path)
    print(f"📂 已加载数据集，共{len(df)}条问题")

    for model_name in models_to_test:
        print(f"\n{'=' * 30} 开始评估模型: {model_name} {'=' * 30}")

        # 初始化结果文件
        save_path = f"results/COT_math500_{model_name.split('/')[-1]}.csv"
        done_indices = set()

        if os.path.exists(save_path):
            try:
                existing_df = pd.read_csv(save_path)
                done_indices = set(existing_df["index"].tolist())
                print(f"⏩ 从{save_path}恢复进度，已处理{len(done_indices)}个问题")
            except Exception as e:
                print(f"⚠️ 恢复进度失败: {str(e)}")

        # 遍历数据集
        for idx, row in df.iterrows():
            if idx in done_indices:
                continue

            problem = row["problem"]
            answer = row["answer"]

            print(f"\n{'=' * 20} 问题 #{idx} {'=' * 20}")
            print(f"❓ 问题内容:\n{problem}\n")
            print(f"🔑 参考答案: {answer}")

            # 收集模型回答
            responses = []
            usages = []
            total_completion_tokens = 0
            total_prompt_tokens = 0
            total_total_tokens = 0
            for i in range(samples_per_question):
                print(f"\n🔄 生成回答 {i + 1}/{samples_per_question}...")
                response, usage = call_model(problem, model_name)
                responses.append(response)
                usages.append(usage)
                total_completion_tokens += usage.get('completion_tokens', 0)
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_total_tokens += usage.get('total_tokens', 0)
                print(f"生成的回答:\n{'-' * 20}\n{response}\n{'-' * 20}")

            # 计算总响应长度
            response_length = sum(len(response) for response in responses)

            # 判断答案正确性
            print("\n🔎 开始判断回答正确性...")
            judgement = judge_answers(
                responses_texts=responses,
                reference_answer=answer,
                judge_model=model_name
            )

            # 记录结果
            result_row = {
                "index": idx,
                "gold_answer": answer,
                "correct": judgement["correct"],
                "response_length": response_length,
                "confidence": len(judgement["selected_indices"]) / samples_per_question,
                "completion_tokens": total_completion_tokens,
                "prompt_tokens": total_prompt_tokens,
                "total_tokens": total_total_tokens
            }

            # 保存到CSV
            write_header = not os.path.exists(save_path)
            with open(save_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=result_row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(result_row)

            print(f"\n✔️ 保存结果: {'正确' if judgement['correct'] else '错误'} " +
                  f"(置信度 {result_row['confidence']:.0%})")
