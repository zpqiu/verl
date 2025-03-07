import numpy as np
import torch
from torch.utils.data import Sampler, SubsetRandomSampler
import random

class DynamicSampler(Sampler[int]):
    def __init__(self, data_source, difficulties, total_steps, batch_size, init_level=1):
        self.data_source = data_source
        self.difficulties = np.array(difficulties)
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.difficulty_levels = None
        self.current_level = init_level
        self.accuracy_history = []
        self.split_into_levels(difficulties)
    
    def set_current_level(self, level):
        self.current_level = level

    def split_into_levels(self, difficulties):
        # 根据难度将样本分成10个级别
        # 难度值在0-1之间，0表示最难，1表示最简单
        # difficulties 是一个长度为 len(data_source) 的数组
        self.difficulty_levels = np.array([max(min(int(x * 10) + 1, 10), 1) for x in difficulties])
        self.current_level = min(self.difficulty_levels)
        print('[Sampler] Current level: {}'.format(self.current_level))

        # 按难度 level group by
        self.difficulty_levels_grouped = {}
        for i, level in enumerate(self.difficulty_levels):
            if level not in self.difficulty_levels_grouped:
                self.difficulty_levels_grouped[level] = []
            self.difficulty_levels_grouped[level].append(i)

        # 每个难度 level 建立一个 SubsetRandomSampler
        # self.samplers = {}
        # for level, indices in self.difficulty_levels_grouped.items():
        #     self.samplers[level] = SubsetRandomSampler(indices)
        
    def sample_from_level(self, level, batch_size):
        # 根据难度级别采样样本
        # 这里假设每个级别内部样本均匀分布
        # 可以根据具体问题调整采样方式
        level = max(min(level, 10), 1)
        # 使用 sampels[level] 采样
        # return self.samplers[level].sample(batch_size)
    
        indices = self.difficulty_levels_grouped[level]
        if len(indices) <= batch_size:
            return random.choices(indices, k=batch_size)
        else:
            return random.sample(indices, k=batch_size)

    def __iter__(self):
        # 混合采样核心逻辑
        for _ in range(self.total_steps):
            main_samples = self.sample_from_level(self.current_level, int(0.8*self.batch_size))
            next_level_samples = self.sample_from_level(self.current_level+1, int(0.2*self.batch_size))
            
            # 使用random.shuffle来打乱样本
            combined_samples = main_samples + next_level_samples
            # print(combined_samples[:20])
            random.shuffle(combined_samples)
            yield from combined_samples
    
    def update_policy(self, latest_accuracy):
        self.accuracy_history.append(latest_accuracy)
        # 使用动量更新策略防止震荡
        # if len(self.accuracy_history) > 3:
        #     momentum = np.polyfit(range(3), self.accuracy_history[-3:], 1)[0]
        #     if momentum < -0.05:
        #         self.current_level = max(1, self.current_level-1)
        # 调用调整函数
        self.current_level = self.adjust_difficulty(
            self.current_level, 
            latest_accuracy,
            self.accuracy_history
        )

    def adjust_difficulty(self, current_level, accuracy, history):
        # 平滑处理：使用滑动窗口平均减少波动影响
        smoothed_acc = np.mean(history[-2:])
        next_level = current_level

        if smoothed_acc > 0.75:
            print('[Sampler] Performance is too good, jump to level {}'.format(current_level + 1))
            # 性能优异：跳跃式提升难度（防局部最优）
            next_level = min(current_level + 1, 10)
        # elif smoothed_acc > 0.65:
        #     # 稳定提升：线性增加难度
        #     next_level = min(current_level + 1, 10)
        elif smoothed_acc < 0.3:
            # 性能下降：回退到安全区并增加混合采样
            print('[Sampler] Warning: performance is too low, back to level {}'.format(current_level))
            next_level = max(current_level - 1, 1)
        else:
            # 保持当前难度但调整采样分布
            next_level = current_level
        
        # next level may have no samples, so we need to find the nearest level that has samples
        while len(self.sample_from_level(next_level, 1)) == 0 and next_level <= 10:
            next_level = next_level + 1
        if next_level > 10:
            next_level = current_level
            print("[Sampler] Warning: No samples for next level, using current level")

        return next_level
        
    def __len__(self):
        return self.total_steps * self.batch_size

class CurriculumSampler(Sampler[int]):
    def __init__(self, data_source, difficulties, total_steps, batch_size, initial_lambda=-10.0, final_lambda=10.0):
        """
        data_source: the dataset (or list-like object)
        difficulties: numpy array or list of float difficulty labels in
        (0 = easiest, 1 = hardest)
        initial_lambda: starting lambda value (should be negative to favor easy questions)
        final_lambda: final lambda value (should be positive to favor hard questions)
        total_steps: the total number of training steps over which to update lambda
        """
        self.data_source = data_source
        self.difficulties = np.array(difficulties)
        self.initial_lambda = initial_lambda # e.g., -5.0
        self.final_lambda = final_lambda # e.g., 5.0
        self.num_samples = len(data_source)
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.current_step = 0
        self.sampled_indices = set()
        # self.update_step(0)
        print(max(self.difficulties), '----', min(self.difficulties))

    def update_step(self, step):
        """
        Update the current training step and compute a new lambda based on a curriculum schedule.
        We use a simple linear interpolation between initial_lambda and final_lambda.
        """
        self.current_step = step

    def calculate_sigmoid_weights(self):
        progress = min(self.current_step / self.total_steps, 1.0)
        batch_lambda = self.initial_lambda * (1 - progress) + self.final_lambda * progress
        weights = np.exp(batch_lambda * (self.difficulties - 1))
        # Normalize the weights to form a probability distribution.
        probabilities = weights / np.sum(weights)
        return probabilities
    
    def calculate_linear_weights(self):
        target_difficulty = min(self.current_step / self.total_steps, 1.0)
        # 计算每个样本与目标难度的接近程度
        # 距离越近，权重越高
        distance = np.abs(self.difficulties - target_difficulty)
        weights = 1.0 / (distance + 0.01)  # 添加小值避免除零错误
        
        # 归一化权重形成概率分布
        probabilities = weights / np.sum(weights)
        return probabilities
    
    def calculate_nonlinear_weights(self):
        """
        实现非线性难度增长策略：
        - 简单问题快速通过
        - 中等难度放缓
        - 困难问题更缓慢增长
        """
        # 使用S形曲线来控制难度增长
        progress = min(self.current_step / self.total_steps, 1.0)
        min_difficulty = min(self.difficulties)
        max_difficulty = max(self.difficulties)
        difficulty_range = max_difficulty - min_difficulty
        
        # 使用平滑的S形曲线代替分段函数
        # 这将避免在边界处出现突然的跳跃
        
        # 使用sigmoid函数创建S形曲线
        # 调整参数使曲线在简单问题区域快速上升，中等区域平缓，困难区域缓慢上升
        scale = 5
        x = progress * scale  # 将[0,1]映射到[-5,5]，使sigmoid函数覆盖更广的范围
        max_sigmoid_value = 1 / (1 + np.exp(-scale)) - 0.5
        sigmoid_value = (1 / (1 + np.exp(-x)) - 0.5) * (1 / max_sigmoid_value)
        adjusted_progress = sigmoid_value * difficulty_range + min_difficulty
        print('adjusted_progress', adjusted_progress)
        
        # 计算每个样本与目标难度的接近程度
        distance = np.abs(self.difficulties - adjusted_progress)
        weights = 1.0 / (distance + 0.01)  # 添加小值避免除零错误
        
        # 如果样本已经被采样过，则权重为0
        # weights[list(self.sampled_indices)] = 0

        # 归一化权重形成概率分布
        probabilities = weights / (np.sum(weights) + 1e-10)
        return probabilities

    def __iter__(self):
        # Compute weights using the revised exponential weighting function.
        # weight = exp(lambda_val * (difficulty - 1))
        # For difficulty=1 (hard), weight = exp(0) = 1.
        # For difficulty=0 (easy), weight = exp(-lambda_val). 
        # Thus, when lambda_val is negative (early training), easy samples get high weight.
        # When lambda_val is positive (late training), easy samples get near zero weight.
        batches = len(self.data_source) // self.batch_size

        for i in range(batches):
            probabilities = self.calculate_nonlinear_weights()
            # Sample indices according to the computed probabilities.
            sampled_indices = np.random.choice(
                self.num_samples, size=self.batch_size, replace=False, p=probabilities
            )
            self.current_step += 1
            yield from sampled_indices.tolist()
            self.sampled_indices.update(sampled_indices.tolist())
        
        last_batch_size = self.num_samples % self.batch_size
        if last_batch_size > 0:
            probabilities = self.calculate_nonlinear_weights()
            sampled_indices = np.random.choice(
                self.num_samples, size=last_batch_size, replace=False, p=probabilities
            )
            self.current_step += 1
            yield from sampled_indices.tolist()
            self.sampled_indices.update(sampled_indices.tolist())

        self.sampled_indices = set()

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    # build a test dataset
    # my_dataset = np.arange(1000)
    # my_difficulties = np.arange(1000)/1000.0
    # total_steps = 2 * len(my_dataset) // 8 # total training iterations

    # sampler = CurriculumSampler(
    #     data_source=my_dataset,
    #     difficulties=my_difficulties,
    #     total_steps=total_steps,
    #     batch_size=8
    # )
    # data_loader = torch.utils.data.DataLoader(my_dataset, sampler=sampler, batch_size=8)


    # i = 1
    # for epoch in range(2):
    #     for batch in data_loader:
    #         i += 1
    #         if i % 10 == 0:
    #             print(data_loader.sampler.current_step)
    #             print(f"batch {i}: {batch}, mean difficulty: {np.mean(my_difficulties[batch])}")

    import datasets
    train_dataset = datasets.load_dataset('pe-nlp/Big-Math-RL-Verified-Filtered2', split='train', trust_remote_code=True)
    train_dataset = train_dataset.filter(lambda x: x['llama8b_solve_rate'] is not None and x['llama8b_solve_rate'] < 0.95)
    train_dataset = train_dataset.map(lambda x: {**x, 'difficulty': 1.0 - x['llama8b_solve_rate']})
    # train_dataset = train_dataset.shuffle(seed=42).select(range(len(train_dataset) // 2))

    dataset = np.arange(len(train_dataset))

    difficulties = train_dataset.to_pandas()['difficulty'].values

    sampler = DynamicSampler(
        data_source=dataset,
        difficulties=difficulties,
        total_steps=len(train_dataset) // 512,
        batch_size=512,
        init_level=1
    )
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=512)

    i = 1
    data = []
    for batch in data_loader:
        # print(batch)
        i += 1
        if i % 1 == 0:
            # print(data_loader.sampler.current_step)
            print(f"batch {i}: mean difficulty: {np.mean(difficulties[batch])}")
            data.append([i, np.mean(difficulties[batch])])

    # import pandas as pd
    # df = pd.DataFrame(data, columns=['step', 'mean_difficulty'])
    # df.to_csv('nonlinear_difficulty_growth.csv', index=False)
