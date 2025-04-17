# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/orz')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'orz'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    train_dataset = datasets.load_dataset('Skywork/Skywork-OR1-RL-Data', split='math')

    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn():

        def process_fn(example, idx):
            extra_info = example.pop('extra_info')

            data = {
                "data_source": example.pop('data_source'),
                "prompt": example.pop('prompt'),
                "difficulty": extra_info['model_difficulty']['DeepSeek-R1-Distill-Qwen-7B'],
                "ability": "math",
                "reward_model": example.pop('reward_model'),
                "extra_info": extra_info
            }
            return data

        return process_fn

    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset = train_dataset.map(function=make_map_fn(), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
