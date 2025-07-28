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
import asyncio
import heapq
import logging
import os
import random
import threading
from collections import defaultdict
from typing import Any
from uuid import uuid4

import datasets
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from pydantic import BaseModel
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.profiler import simple_timer
from verl.utils.reward_score.math_dapo import compute_score
from verl.utils.rollout_trace import (RolloutTraceConfig, rollout_trace_attr,
                                      rollout_trace_op)
from verl.workers.rollout.async_server import async_server_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class EarlyStoppingCoordinator:
    """协调早停机制的全局状态管理器"""
    
    def __init__(self, expected_prompt_num: int):
        self.expected_prompt_num = expected_prompt_num
        self.completed_prompts = set()  # 已完成的prompt sample_index集合
        self.invalid_prompt_count = 0  # 无效的prompt数量
        self.should_stop = False
        self.lock = threading.Lock()
        print(f"[EarlyStoppingCoordinator] 初始化: expected_prompt_num={expected_prompt_num}")
    
    def report_completion(self, sample_index: int, is_valid: bool) -> bool:
        """报告某个prompt的完成状态
        
        Args:
            sample_index: 完成的prompt的sample_index
            is_valid: 是否有效
        Returns:
            bool: 是否应该触发早停
        """
        with self.lock:
            if self.should_stop:
                return True
            
            if is_valid:
                self.completed_prompts.add(sample_index)
            else:
                self.invalid_prompt_count += 1
            
            completed_count = len(self.completed_prompts)
            
            if completed_count >= self.expected_prompt_num:
                self.should_stop = True
                print(f"[EarlyStoppingCoordinator] 触发早停: {completed_count}/{self.expected_prompt_num} prompts 已完成, 无效的prompt数量: {self.invalid_prompt_count}")
                return True
            else:
                print(f"[EarlyStoppingCoordinator] 进度更新: {completed_count}/{self.expected_prompt_num} prompts 已完成, 无效的prompt数量: {self.invalid_prompt_count}")
                return False
    
    def should_stop_generation(self) -> bool:
        """检查是否应该停止生成"""
        with self.lock:
            return self.should_stop
    
    def get_completed_prompts(self) -> set:
        """获取已完成的prompt集合"""
        with self.lock:
            return self.completed_prompts.copy()


@ray.remote(concurrency_groups={"acquire": 1, "release": 10, "reset": 1})
class GlobalLoadBalancer:
    """
    全局负载均衡器，只负责分配服务器索引，不处理实际的generate调用
    使用 threading.Semaphore 而不是 asyncio.Queue 来避免 Ray Actor 中的并发问题
    """

    def __init__(self, config: DictConfig, num_servers: int, max_cache_size: int = 10000):
        """Initialize the GlobalLoadBalancer.

        Args:
            config (DictConfig): YAML config.
            num_servers (int): 服务器数量
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.num_servers = num_servers

        # 使用 threading.Semaphore 替代 asyncio.Queue
        self.max_loads_per_server = 300
        self.total_capacity = self.max_loads_per_server * num_servers
        self._semaphore = threading.Semaphore(self.total_capacity)
        self._current_loads = [0] * num_servers  # 跟踪每个服务器的当前负载
        self._lock = threading.Lock()  # 保护 _current_loads 的并发访问
        
        print(f"[GlobalLoadBalancer] max_loads_per_server: {self.max_loads_per_server}")
        print(f"[GlobalLoadBalancer] total_capacity: {self.total_capacity}")

        # Least requests load balancing
        self.weighted_serveres = [[0, server_index] for server_index in range(num_servers)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    @ray.method(concurrency_group="acquire")
    def get_server_index(self, request_id: str) -> int:
        """获取应该使用的服务器索引"""
        if self.config.actor_rollout_ref.rollout.get("load_balance", False):
            # 获取信号量许可
            self._semaphore.acquire()
            
            # 选择负载最少的服务器
            with self._lock:
                min_load_idx = min(range(self.num_servers), key=lambda i: self._current_loads[i])
                self._current_loads[min_load_idx] += 1
                server_index = min_load_idx
                
            # if random.random() < 0.002:  # 0.2% 概率打印日志，增加可见性
            #     print(f"[GlobalLoadBalancer] choose server: {server_index}, request_id: {request_id}, current_loads: {self._current_loads}")
            return server_index
        else:
            return self._choose_server_index(request_id)

    @ray.method(concurrency_group="release")
    def release_server_index(self, server_index: int):
        """释放服务器索引"""
        if self.config.actor_rollout_ref.rollout.get("load_balance", False):
            # 减少服务器负载计数
            with self._lock:
                if self._current_loads[server_index] > 0:
                    self._current_loads[server_index] -= 1
                    
            # 释放信号量许可
            self._semaphore.release()
            
            # if random.random() < 0.002:  # 0.2% 概率打印日志，增加可见性
            #     print(f"[GlobalLoadBalancer] release server: {server_index}, current_loads: {self._current_loads}")

    def _choose_server_index(self, request_id: str) -> int:
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server_index = self.weighted_serveres[0][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server_index
        return server_index

    @ray.method(concurrency_group="reset")
    def reset(self):
        """重置负载均衡器状态，包括信号量和负载计数"""
        with self._lock:
            # 重新创建信号量
            self._semaphore = threading.Semaphore(self.total_capacity)
            # 重置所有服务器的负载计数
            self._current_loads = [0] * self.num_servers

        print(f"[GlobalLoadBalancer] 已重置负载均衡器状态")


class AsyncLLMServerManager:
    """
    本地服务器管理器，负责实际的generate调用
    通过全局负载均衡器获取服务器分配
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], global_load_balancer: ray.actor.ActorHandle):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            global_load_balancer (ray.actor.ActorHandle): 全局负载均衡器的handle
        """
        self.config = config
        self.server_handles = server_handles
        self.global_load_balancer = global_load_balancer

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            List[int]: List of generated token ids.
        """
        # 从全局负载均衡器获取服务器索引（现在是同步调用）
        server_index = await self.global_load_balancer.get_server_index.remote(request_id)
        server = self.server_handles[server_index]
        output = None
        
        try:
            output = await server.generate_with_cancel.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        except asyncio.CancelledError:
            print(f"[AsyncLLMServerManager] 任务被取消: {request_id}")
            await server.cancel.remote(request_id)
        finally:
            # 确保释放服务器索引，即使出现异常（现在是同步调用）
            await self.global_load_balancer.release_server_index.remote(server_index)
        
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0

class RewardOutput(BaseModel):
    """Reward output."""

    reward: float = 0.0
    acc: float = 0.0
    pred: str = ""


class AgentLoopOutput(BaseModel):
    """Agent loop output."""
    rollout_index: int = -1
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int = 0
    metrics: AgentLoopMetrics
    reward: RewardOutput = RewardOutput()


# the config API has been changed, so we need to use the old API
# class SingleTurnAgentLoop(AgentLoopBase):
class SingleTurnAgentLoop:
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, config, server_manager, tokenizer):
        # super().__init__(config, server_manager, tokenizer)
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        # self.reward_fn = reward_fn
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()

    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        )

        with simple_timer("generate_sequences", metrics):
            response_ids = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )
        if response_ids is None:
            return None
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=2,
            metrics=metrics,
            # reward=ret,
        )
        return output

@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(self, config: DictConfig, global_load_balancer: ray.actor.ActorHandle, server_handles: list[ray.actor.ActorHandle]):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            global_load_balancer (ray.actor.ActorHandle): 全局负载均衡器的handle
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        # 创建本地的服务器管理器，使用全局负载均衡器
        self.server_manager = AsyncLLMServerManager(config, server_handles, global_load_balancer)
        self.early_stopping_coordinator = None  # 早停协调器，在generate_sequences时设置
        self.max_concurrent_prompts = config.actor_rollout_ref.rollout.get("max_concurrent_prompts", 32)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        trace_config = config.trainer.get("rollout_trace", {})

        RolloutTraceConfig.init(
            config.trainer.project_name,
            config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(self, batch: DataProto, early_stopping_coordinator: ray.actor.ActorHandle = None) -> DataProto:
        """Generate sequences from agent loop with dynamic task creation and early stopping support.

        Args:
            batch (DataProto): Input batch.
            early_stopping_coordinator: 早停协调器

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        self.early_stopping_coordinator = early_stopping_coordinator
        is_validation = batch.meta_info.get("validate", False)
        
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        agent_names = batch.non_tensor_batch["agent_name"]
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(raw_prompts))
        rollout_index = batch.non_tensor_batch["rollout_index"]
        ground_truths = [item.non_tensor_batch["reward_model"]["ground_truth"] for item in batch]

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index, rollout_index)

        # 按prompt分组任务，便于早停管理
        prompt_groups = self._group_by_prompt(agent_names, raw_prompts, ground_truths, trajectory_info, sampling_params)
        
        # 动态任务创建和管理
        completed_outputs = {}
        pending_tasks = {}  # task -> sample_index 的映射
        
        # 创建待处理的prompt队列（使用列表来保持顺序）
        pending_prompts = list(prompt_groups.items())
        
        # 设置最大并发任务数（可以配置）
        max_concurrent_tasks = self.max_concurrent_prompts
        print(f"[AgentLoopWorker] 动态任务创建模式，最大并发任务数: {max_concurrent_tasks}")
        print(f"[AgentLoopWorker] 总共需要处理 {len(pending_prompts)} 个prompt groups")
        
        # 初始创建一批任务
        created_task_count = 0
        for _ in range(min(max_concurrent_tasks, len(pending_prompts))):
            if pending_prompts:
                sample_index, group_data = pending_prompts.pop(0)
                task = asyncio.create_task(
                    self._run_prompt_group(sample_index, group_data, do_filter=not is_validation)
                )
                pending_tasks[task] = sample_index
                created_task_count += 1
                print(f"[AgentLoopWorker] 初始创建任务 {created_task_count}: prompt {sample_index}")

        # 主循环：等待任务完成并动态创建新任务
        try:
            while pending_tasks:
                # 检查早停状态
                if early_stopping_coordinator:
                    should_stop = await early_stopping_coordinator.should_stop_generation.remote()
                    if should_stop:
                        print(f"[AgentLoopWorker] 检测到早停信号，取消剩余 {len(pending_tasks)} 个运行任务和 {len(pending_prompts)} 个待创建任务")
                        # 取消所有待处理任务
                        for task in pending_tasks:
                            task.cancel()
                        break
                
                # 等待任意一个任务完成
                done, still_pending = await asyncio.wait(
                    pending_tasks.keys(), 
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.1  # 短暂超时以便定期检查早停状态
                )
                
                # 处理已完成的任务
                for task in done:
                    completed_sample_index = pending_tasks.pop(task)
                    try:
                        sample_index, outputs = await task
                        if not outputs:
                            print(f"[AgentLoopWorker] Prompt {sample_index} 完成，但是 invalid")
                            if early_stopping_coordinator:
                                await early_stopping_coordinator.report_completion.remote(sample_index, is_valid=False)
                        else:
                            completed_outputs[sample_index] = outputs
                            print(f"[AgentLoopWorker] Prompt {sample_index} 完成，输出 {len(outputs)} 个样本")
                            
                            # 向协调器报告完成状态
                            if early_stopping_coordinator:
                                await early_stopping_coordinator.report_completion.remote(sample_index, is_valid=True)
                            
                    except asyncio.CancelledError:
                        print(f"[AgentLoopWorker] 任务 {completed_sample_index} 被取消")
                    except Exception as e:
                        print(f"[AgentLoopWorker] 任务 {completed_sample_index} 执行失败: {e}")
                        raise e
                
                # 为每个完成的任务创建一个新任务（如果还有待处理的prompt）
                new_tasks_created = 0
                for _ in range(len(done)):
                    if pending_prompts and len(pending_tasks) < max_concurrent_tasks:
                        # 检查早停状态，避免在早停时还创建新任务
                        if early_stopping_coordinator:
                            should_stop = await early_stopping_coordinator.should_stop_generation.remote()
                            if should_stop:
                                print(f"[AgentLoopWorker] 检测到早停信号，停止创建新任务")
                                break
                        
                        sample_index, group_data = pending_prompts.pop(0)
                        task = asyncio.create_task(
                            self._run_prompt_group(sample_index, group_data, do_filter=not is_validation)
                        )
                        pending_tasks[task] = sample_index
                        new_tasks_created += 1
                        created_task_count += 1
                        print(f"[AgentLoopWorker] 动态创建新任务 {created_task_count}: prompt {sample_index}")
                
                if new_tasks_created > 0:
                    print(f"[AgentLoopWorker] 本轮创建了 {new_tasks_created} 个新任务，当前运行任务数: {len(pending_tasks)}")
                
                print(f"[AgentLoopWorker] 任务状态 - 运行中: {len(pending_tasks)}, 待创建: {len(pending_prompts)}, 已完成: {len(completed_outputs)}")
                
        except Exception as e:
            # 确保在异常时取消所有剩余任务
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            # 等待取消操作完成
            if pending_tasks:
                await asyncio.gather(*pending_tasks.keys(), return_exceptions=True)
            raise e
        
        # 只处理已完成的输出
        if not completed_outputs:
            # 如果没有完成的输出，返回空结果
            return None
            
        all_outputs = []
        for sample_index in sorted(completed_outputs.keys()):
            all_outputs.extend(completed_outputs[sample_index])

        output = self._postprocess(all_outputs)
        return output

    def _group_by_prompt(self, agent_names, raw_prompts, ground_truths, trajectory_info, sampling_params):
        """按prompt sample_index分组数据"""
        prompt_groups = defaultdict(list)
        
        for i, (agent_name, messages, ground_truth, trajectory) in enumerate(zip(agent_names, raw_prompts, ground_truths, trajectory_info, strict=True)):
            sample_index = trajectory["sample_index"]
            prompt_groups[sample_index].append({
                "agent_name": agent_name,
                "messages": messages.tolist(),
                "ground_truth": ground_truth,
                "trajectory": trajectory,
                "sampling_params": sampling_params
            })
        
        return prompt_groups

    async def _run_prompt_group(self, sample_index: int, group_data: list, do_filter: bool = True):
        """运行一个prompt的所有样本"""
        tasks = []
        for data in group_data:
            task = asyncio.create_task(
                self._run_agent_loop(
                    data["agent_name"], 
                    data["messages"], 
                    data["ground_truth"],
                    data["sampling_params"], 
                    data["trajectory"]
                )
            )
            tasks.append(task)
        
        try:
            # 等待所有样本完成
            outputs = await asyncio.gather(*tasks)

            if any(output is None for output in outputs):
                return sample_index, []

            # 检查是否所有样本的 reward.reward 完全一样，如果一样，则该 prompt 为 invalid
            if do_filter and all(output.reward.reward == outputs[0].reward.reward for output in outputs):
                return sample_index, []
            
            return sample_index, outputs
        except asyncio.CancelledError:
            # print(f"[_run_prompt_group] Prompt {sample_index} 被取消，正在取消 {len(tasks)} 个子任务")
            # 取消所有子任务
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # 等待所有子任务的取消操作完成
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass  # 忽略取消过程中的异常
            
            # print(f"[_run_prompt_group] Prompt {sample_index} 的所有子任务已取消")
            raise  # 重新抛出 CancelledError

    async def _run_agent_loop(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        ground_truth: str,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
    ) -> AgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"], sample_index=trajectory["sample_index"], rollout_n=trajectory["rollout_n"]
        ):
            agent_loop = SingleTurnAgentLoop(self.config, self.server_manager, self.tokenizer)
            output = await agent_loop.run(messages, sampling_params)
            if output is None:
                return None
            output.reward = self._compute_reward(ground_truth, output)
            output.rollout_index = trajectory["rollout_index"]
            return output

    def _compute_reward(self, ground_truth: str, output: AgentLoopOutput) -> RewardOutput:
        response_str = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
        
        ori_response_str = response_str

        eos_token = self.tokenizer.eos_token
        if response_str.endswith(eos_token):
            response_str = response_str[: -len(eos_token)]
        ret = compute_score(response_str, ground_truth)
        reward = ret["score"]
        acc = ret["acc"]
        pred = ret["pred"]

        # print some samples
        if random.randint(0, 1024) < 1:
            print("\n" + "="*80)
            print("🔍 [调试样例]")
            print("-"*80)
            print(f"🤖 模型回答: {ori_response_str}")
            print(f"✅ 标准答案: {ground_truth}")
            print(f"📊 评分结果: 分数={reward:.2f} | 准确率={acc:.2f} | 预测={pred}")
            print("="*80 + "\n")

        if self.config.actor_rollout_ref.rollout.overlong_buffer.enable:
            overlong_buffer_len = self.config.actor_rollout_ref.rollout.overlong_buffer.len
            expected_len = self.config.actor_rollout_ref.rollout.response_length - overlong_buffer_len
            exceed_len = len(output.response_ids) - expected_len
            overlong_penalty_factor = self.config.actor_rollout_ref.rollout.overlong_buffer.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            # print(f"[AgentLoop][DEBUG] reward: {reward}, response_len: {len(output.response_ids)}, overlong_buffer_len: {overlong_buffer_len}, exceed_len: {exceed_len}, overlong_penalty_factor: {overlong_penalty_factor}, overlong_reward: {overlong_reward}")
            reward += overlong_reward
        
        return RewardOutput(reward=reward, acc=acc, pred=pred)

    def _postprocess(self, inputs: list[AgentLoopOutput]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts
        self.tokenizer.padding_side = "left"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.prompt_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]
        valid_response_lengths = [ len(input.response_ids) for input in inputs ]
        print(f"[AgentLoop][DEBUG] max valid_response_lengths: {max(valid_response_lengths)}")
        
        # response_mask
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_mask} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = outputs["input_ids"]
        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )
        response_mask = response_mask * response_attention_mask

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        for i, input in enumerate(inputs):
            reward_tensor[i, valid_response_lengths[i]-1] = input.reward.reward
            reward_extra_info["acc"].append(input.reward.acc)
            reward_extra_info["pred"].append(input.reward.pred)
            reward_extra_info["score"].append(input.reward.reward)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
                "token_level_scores": reward_tensor,
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
        rollout_index = np.array([input.rollout_index for input in inputs], dtype=np.int32)
        metrics = [input.metrics.model_dump() for input in inputs]
        non_tensor_batch = {
            "__num_turns__": num_turns,
            "rollout_index": rollout_index,
        }
        for key, value in reward_extra_info.items():
            non_tensor_batch[key] = np.array(value)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"metrics": metrics})


async def get_trajectory_info(step, index, rollout_index):
    """Get the trajectory info (step, sample_index, rollout_n) asynchrously"""
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "rollout_index": rollout_index[i]})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_global_server_manager()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_global_server_manager(self):
        """创建全局的AsyncLLMServerManager作为Ray Actor"""
        self.global_load_balancer = GlobalLoadBalancer.options(
            name="global_async_llm_load_balancer",
        ).remote(self.config, self.rollout_dp_size)
        print("[AgentLoopManager] 创建了全局负载均衡器")

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        for i in range(self.config.actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                ).remote(self.config, self.global_load_balancer, self.async_llm_servers)
            )
        print(f"[AgentLoopManager] 创建了 {len(self.agent_loop_workers)} 个AgentLoopWorker，都使用同一个全局服务器管理器")

    def generate_sequences(self, prompts: DataProto, expected_prompt_num: int = None) -> tuple[DataProto, set]:
        """Split input batch and dispatch to agent loop workers with early stopping support.

        Args:
            prompts (DataProto): Input batch.
            expected_prompt_num (int, optional): 期望完成的prompt数量，达到后触发早停

        Returns:
            tuple[DataProto, set]: (Output batch, set of completed sample indices)
        """
        # print prompts keys for debug
        print(f"[AgentLoopManager] expected_prompt_num: {expected_prompt_num}")
        print(f"[AgentLoopManager] prompts keys: {prompts.batch.keys()} non_tensor_batch: {prompts.non_tensor_batch.keys()}")
        
        # 在每次generate调用开始时重置全局负载均衡器
        ray.get(self.global_load_balancer.reset.remote())
        print(f"[AgentLoopManager] 已重置全局负载均衡器")
        
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()

        # 创建早停协调器（如果需要）
        early_stopping_coordinator = None
        if expected_prompt_num is not None:
            early_stopping_coordinator = EarlyStoppingCoordinator.options(
                name=f"early_stopping_coordinator_{uuid4().hex[:8]}"
            ).remote(expected_prompt_num)
            print(f"[AgentLoopManager] 启用早停机制: expected_prompt_num={expected_prompt_num}")

        # 按prompt分组并分配给workers，确保同一prompt的样本在同一worker
        worker_chunks = self._split_by_prompt(prompts)
        
        # 启动所有worker任务
        worker_tasks = []
        for i, chunk in enumerate(worker_chunks):
            if chunk is not None and len(chunk) > 0:  # 只处理非空chunk
                task = self.agent_loop_workers[i].generate_sequences.remote(chunk, early_stopping_coordinator)
                worker_tasks.append(task)
            else:
                worker_tasks.append(None)

        # 等待所有worker完成
        outputs = []
        for i, task in enumerate(worker_tasks):
            if task is not None:
                try:
                    result = ray.get(task)
                    if result is None:
                        continue
                    outputs.append(result)
                except Exception as e:
                    print(f"[AgentLoopManager] Worker {i} 执行失败: {e}")
                    # 尽早抛出异常，避免后续的计算
                    raise e

        # 终止所有未完成的请求
        self.abort()

        # 合并输出
        output = DataProto.concat(outputs)
        print(f"[AgentLoopManager] 合并输出的 size: {len(output)}")

        # 获取完成的prompt集合
        completed_prompts = set()
        if early_stopping_coordinator is not None:
            try:
                completed_prompts = ray.get(early_stopping_coordinator.get_completed_prompts.remote())
                print(f"[AgentLoopManager] 早停结束，完成了 {len(completed_prompts)} 个prompts")
                ray.kill(early_stopping_coordinator)
            except Exception:
                pass

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        if len(outputs) > 0:
            metrics = [output.meta_info["metrics"] for output in outputs]  # List[List[Dict[str, str]]]
            timing = self._performance_metrics(metrics, output)
        else:
            timing = {}

        output.meta_info = {"timing": timing}
        return output

    def _split_by_prompt(self, prompts: DataProto) -> list:
        """按prompt分组并分配给workers，确保同一prompt的样本分配到同一worker"""
        # 获取sample_index信息
        if "index" in prompts.non_tensor_batch:
            indices = prompts.non_tensor_batch["index"]
        else:
            indices = np.arange(len(prompts))
        
        # 按sample_index分组
        prompt_groups = defaultdict(list)
        for i, sample_index in enumerate(indices):
            prompt_groups[sample_index].append(i)
        
        # 调试信息：显示prompt分布
        unique_prompts = list(prompt_groups.keys())
        samples_per_prompt = [len(samples) for samples in prompt_groups.values()]
        print(f"[AgentLoopManager] 总共 {len(unique_prompts)} 个unique prompts")
        print(f"[AgentLoopManager] 每个prompt的样本数范围: {min(samples_per_prompt)}-{max(samples_per_prompt)}")
        print(f"[AgentLoopManager] 总样本数: {sum(samples_per_prompt)}")

        # 分配给workers
        num_workers = len(self.agent_loop_workers)
        worker_assignments = [[] for _ in range(num_workers)]
        worker_prompt_counts = [0] * num_workers  # 记录每个worker分配到的prompt数量
        
        # 修复：使用sample_index的值而不是枚举顺序来分配
        for worker_idx, (sample_index, sample_indices) in enumerate(prompt_groups.items()):
            target_worker = worker_idx % num_workers  # 恢复原来的逻辑
            worker_assignments[target_worker].extend(sample_indices)
            worker_prompt_counts[target_worker] += 1
            
        # 调试信息：显示分配统计
        print(f"[AgentLoopManager] 每个worker分配到的prompt数量: {worker_prompt_counts}")
        print(f"[AgentLoopManager] prompt分配范围: {min(worker_prompt_counts)}-{max(worker_prompt_counts)}")
        
        # 新增：显示每个worker的样本数统计
        worker_sample_counts = [len(assignments) for assignments in worker_assignments]
        print(f"[AgentLoopManager] 每个worker的样本数: {worker_sample_counts}")
        print(f"[AgentLoopManager] 样本数范围: {min(worker_sample_counts)}-{max(worker_sample_counts)}")
        
        # 为每个worker创建数据块
        worker_chunks = []
        for worker_idx in range(num_workers):
            if worker_assignments[worker_idx]:
                indices_to_select = worker_assignments[worker_idx]
                # 创建worker的数据子集
                chunk = prompts.select_idxs(indices_to_select)
                worker_chunks.append(chunk)
                print(f"[AgentLoopManager] Worker {worker_idx} 分配到 {len(indices_to_select)} 个样本 ({worker_prompt_counts[worker_idx]} 个prompts)")
            else:
                worker_chunks.append(None)  # 该worker没有分配到任务
                print(f"[AgentLoopManager] Worker {worker_idx} 没有分配到任务")
        
        return worker_chunks

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def abort(self):
        """Abort all rollout server instances."""
        ray.get([server.abort.remote() for server in self.async_llm_servers])