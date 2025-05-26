import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/orz')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'orz'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    train_dataset = datasets.load_dataset('Skywork/Skywork-OR1-RL-Data', split='math')

    sys_msg = [
        {
            "content": "Solve the following math problem step by step. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.",
            "role": "system"
        }
    ]

    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    # 把难度值从 0-16 映射到 0-4，0-4 分别代表 1-5 级难度
    def normalize_difficulty(raw_value: int) -> int:
        return raw_value // 4

    # add a row to each data item that represents a unique id
    def make_map_fn():

        def process_fn(example, idx):
            extra_info = example.pop('extra_info')
            prompt = example.pop('prompt')
            extra_info['question'] = prompt[0]['content']
            gt = eval(example.pop('reward_model')['ground_truth'])[0]

            data = {
                "data_source": example.pop('data_source'),
                "prompt": sys_msg + prompt,
                "difficulty": normalize_difficulty(extra_info['model_difficulty']['DeepSeek-R1-Distill-Qwen-1.5B']),
                "ability": "math",
                "reward_model": {"ground_truth": gt, "style": "rule"},
                "extra_info": extra_info
            }
            return data

        return process_fn
    
    def filter_invalid_data(example):
        diff1 = example['extra_info']['model_difficulty']['DeepSeek-R1-Distill-Qwen-1.5B']
        diff2 = example['extra_info']['model_difficulty']['DeepSeek-R1-Distill-Qwen-7B']
        diff3 = example['extra_info']['model_difficulty']['DeepSeek-R1-Distill-Qwen-32B']

        if diff1 == 16 and diff2 == 16 and diff3 == 16 and "aime" not in example['data_source']:
            return False

        return True

    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset = train_dataset.filter(filter_invalid_data)
    print(f"After filtering, the dataset size is {len(train_dataset)}")
    train_dataset = train_dataset.map(function=make_map_fn(), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)