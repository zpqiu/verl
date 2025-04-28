import random

from math_verify import parse, verify


def compute_score(data_source, solution_str, ground_truth, extra_info):
    should_log = random.randint(0, 512) == 1
    result = None
    
    if should_log:
        print("\n=== Skywork Scoring Debug Log ===")
        print(f"Response: {solution_str}")
        print(f"Ground Truth: {ground_truth}")

    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        result = {
            "score": -1.0,
            "acc": 0.0,
            "pred": "",
        }
        if should_log:
            print("Parsing failed")
            print(f"Final Result: {result}")
            print("================================\n")
        return result
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        result = {
            "score": -1.0,
            "acc": 0.0,
            "pred": "",
        }
        if should_log:
            print("Invalid parse result length")
            print(f"Final Result: {result}")
            print("================================\n")
        return result
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        result = {
            "score": 1.0,
            "acc": 1.0,
            "pred": math_verify_parsed[1],
        }
        if should_log:
            print("Exact string match found")
            print(f"Final Result: {result}")
            print("================================\n")
        return result
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if len(math_verify_parsed[1]) > 10 and len(math_verify_parsed[1]) > len(gt) * 5:
                print(f"[WARNNING] Skip verification for {math_verify_parsed[1]}, gt: {gt}")
                continue

            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                result = {
                    "score": 1.0,
                    "acc": 1.0,
                    "pred": math_verify_parsed[1],
                }
                if should_log:
                    print("Semantic verification succeeded")
                    print(f"Final Result: {result}")
                    print("================================\n")
                return result
        except Exception:
            if should_log:
                print(f"Verification failed for ground truth: {gt}")
            continue
    
    # Very unlikely to be correct after the above matches
    result = {
        "score": -1.0,
        "acc": 0.0,
        "pred": "",
    }
    if should_log:
        print("No matches found")
        print(f"Final Result: {result}")
        print("================================\n")
    return result
