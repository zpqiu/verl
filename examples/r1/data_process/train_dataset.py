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

from verl.utils.hdfs_io import copy, makedirs
import argparse


OPEN_R1_SYS = ("You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
               "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
               "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>")


DATA_SOURCES = [
    'pe-nlp/math-level3to5-Filtered',
    'pe-nlp/DeepScaleR-40k-Prompt-Filtered',
]

SOURCE_EPISODES = {
    'pe-nlp/math-level3to5-Filtered': 2,
    'pe-nlp/DeepScaleR-40k-Prompt-Filtered': 10,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source, idx_offset):

        def process_fn(example, idx):
            if "question" in example:
                question = example.pop('question')
            elif "problem" in example:
                question = example.pop('problem')
            else:
                raise ValueError(f"Unknown data source: {data_source}")

            question = question + ' ' + instruction_following
            solution = example.pop('ground_truth_answer')
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": OPEN_R1_SYS
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx + idx_offset
                }
            }
            return data

        return process_fn

    dataset_list = []
    global_idx_offset = 0
    for data_source in DATA_SOURCES:
        print(f"Loading the {data_source} dataset from huggingface...", flush=True)
        raw_datset = datasets.load_dataset(data_source, split='train', trust_remote_code=True)
        for round in range(SOURCE_EPISODES[data_source]):
            raw_datset = raw_datset.shuffle()
            dataset_list.append(raw_datset.map(function=make_map_fn('train', data_source, global_idx_offset), with_indices=True))
            global_idx_offset += len(raw_datset)

    all_train_dataset = datasets.concatenate_datasets(dataset_list)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
