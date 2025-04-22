import datasets

train_dataset = datasets.load_dataset('Skywork/Skywork-OR1-RL-Data', split='math')

def extract_difficulty(example):
    return {**example, 'difficulty_1.5b': example['extra_info']['model_difficulty']['DeepSeek-R1-Distill-Qwen-1.5B'], 
            "difficulty_7b": example['extra_info']['model_difficulty']['DeepSeek-R1-Distill-Qwen-7B'],
            "difficulty_32b": example['extra_info']['model_difficulty']['DeepSeek-R1-Distill-Qwen-32B'],
            }

train_dataset = train_dataset.map(extract_difficulty)

train_dataset = train_dataset.to_pandas()

# stats difficulty distribution
difficulty_1_5b = train_dataset['difficulty_1.5b']
difficulty_7b = train_dataset['difficulty_7b']
difficulty_32b = train_dataset['difficulty_32b']

print(difficulty_1_5b.value_counts())
print(difficulty_7b.value_counts())
print(difficulty_32b.value_counts())
