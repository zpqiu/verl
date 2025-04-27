import os
import random
import re

from openai import OpenAI

PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''

URL = os.environ.get("XVERIFY_URL", "")


def is_format_correct(completion):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    # pattern = r"^<think>.*?</think>"
    if not re.match(pattern, completion, re.DOTALL | re.MULTILINE):
        return False
    # check if all tags only appear once
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    # tags = ["<think>", "</think>"]
    for tag in tags:
        if completion.count(tag) != 1:
            return False
    
    # check if <think>...</think> is empty
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, completion, re.DOTALL | re.MULTILINE)
    if think_match and think_match.group(1).strip() == "":
        return False
    
    return True

def calculate_format_reward(completion):
    """Reward function that checks if the completion has a specific format."""
    return 1.0 if is_format_correct(completion) else -1.0


def extract_qwen_output(prompt):
    model_output = prompt.split("Assistant: <think>")[-1]

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    model_output = model_output.strip()
    return model_output


def extract_answer_part(response):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    return ""


def compute_score(data_source, solution_str, ground_truth, extra_info):
    if extra_info is None or "question" not in extra_info:
        raise ValueError("Extra info is required and must contain 'question'")
    
    do_print = False
    if random.randint(0, 512) == 1:  
        do_print = True
    if do_print:
        print(f"Response Case: {solution_str}, Question: {extra_info['question']}, GT: {ground_truth}")

    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    response = extract_qwen_output(solution_str)
    # # extract <answer>...</answer>
    final_answer = extract_answer_part(response)

    model = OpenAI(
        base_url=URL,
        api_key="sk-proj-1234567890"
    )

    prompt = PROMPT.format(question=extra_info["question"], output=final_answer, answer=ground_truth[0])

    try:
        response_obj = model.chat.completions.create(
            model="IAAR-Shanghai/xVerify-3B-Ia",
            messages=[
                {
                    'role': 'user', 
                    'content': prompt
                }
            ],
            temperature=0.1,
            max_tokens=2048,
            top_p=0.7,
            timeout=60
        )
        if response_obj.choices[0].message.content == "Correct":
            return {
                "score": 1.0,
                "acc": 1.0,
                "pred": final_answer,
            }
    except Exception as e:
        print(f"[xVerify] Error: {e}")
        return {
            "score": -1.0,
            "acc": 0.0,
            "pred": "",
        }
    
    # Very unlikely to be correct after the above matches
    return {
        "score": -1.0,
        "acc": 0.0,
        "pred": "",
    }
