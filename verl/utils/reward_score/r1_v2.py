import torch
import re
import random
from symeval import EvaluatorMathBatch

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def calculate_accuracy_reward(completion, solution):
    """Reward function that checks if the completion is the same as the ground truth."""
    if completion.strip() == "":
        return -1.0

    last_boxed_str = last_boxed_only_string(completion)
    if last_boxed_str is None:
        return -1.0

    # remove \boxed
    if last_boxed_str[7:-1].strip() == solution.strip():
        return 1.0

    # check if the completion is a valid latex expression
    try:
        evaluator = EvaluatorMathBatch()
        if evaluator.eq(solution, last_boxed_str[7:-1]):
            return 1.0
        else:
            return -1.0
    except Exception as e:
        print(f"Error checking if the completion is a valid latex expression: {e}")
        return -1.0


# case = '<think>123</think><think>123</think><answer>456</answer>'
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
    return model_output


def extract_answer_part(response):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    return ""

def compute_score(solution_str, ground_truth) -> float:
    response = extract_qwen_output(solution_str)
    # # extract <answer>...</answer>
    # final_answers = [extract_answer_part(response) for response in responses]
    final_answer = extract_answer_part(response)

    do_print = False
    if random.randint(0, 16) == 1:  
        do_print = True
        
    # format_reward = calculate_format_reward(response)
    try:
        accuracy_reward = calculate_accuracy_reward(final_answer, ground_truth)
    except Exception as e:
        print(f"Error calculating accuracy reward: {e}")
        return 0.0

    if do_print:
        print(f"Response Case: {response}")
        print(f"[Score: {accuracy_reward}] Answer Case: {final_answer} <====> GT: {ground_truth}")

    if accuracy_reward == 1.0:
        return 1.0
    else:
        return 0.0
