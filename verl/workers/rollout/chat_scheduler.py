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
import importlib
import itertools
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import aiohttp
import numpy as np
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)


class CompletionCallback(ABC):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        self.config = config
        self.scheduler = scheduler

        # Initialize tools from config file
        self.max_turns = config.actor_rollout_ref.rollout.multi_turn.max_turns
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self._tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        print(f"Initialized tools: {self.tools}", flush=True)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    @property
    def tool_schemas(self):
        """OpenAI JSON tool schemas."""
        return self._tool_schemas

    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        return None

    @abstractmethod
    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        """Call back function to process completions.

        Args:
            messages: List of messages including raw prompt and assistant, tool response generated so far.
            completions: Chat completions from OpenAI compatible server.
            info: Any other auxiliary information pass across multi-turn.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"].
        """
        raise NotImplementedError


class ToolCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)

        # TODO: add reward manager to calculate reward score once a sample finish

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_turns and len(messages) >= self.max_turns:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!")
            return

        # STEP 1: check if the model called tools
        if finish_reason != "tool_calls":
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!")
            return

        # STEP 2: call tools
        tool_calls = completions.choices[0].message.tool_calls
        print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, done!")
            return
        messages.extend(tool_responses)

        # STEP 3: resubmit completion request with tool responses
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)

    async def _call_tool(self, tool_call) -> Dict[str, str]:
        """Call tool and return tool response."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool = self.tools[tool_name]

        instance_id = await tool.create()
        try:
            tool_response, tool_reward_score, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            await tool.release(instance_id)

        return {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": tool_call.id,
        }

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(conversation, tools=self.tool_schemas, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # response_mask: response mask with tools calling masked out
        response_mask = self._mask_out_tools_calling_tokens(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0), batch_conversations, responses["input_ids"], responses["attention_mask"])

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns})

    def _mask_out_tools_calling_tokens(
        self,
        raw_prompts: List[List[Dict[str, str]]],
        batch_conversations: List[List[Dict[str, str]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask out tools calling tokens in the responses.

        Args:
            raw_prompts: [prompt] from input dataset
            batch_conversations: [prompt + response]
            input_ids: responses tokens
            attention_mask: responses attention mask

        Returns:
            mask: (batch_size, response_length)
        """
        batch_size = input_ids.size(0)
        assert len(raw_prompts) == batch_size, f"{len(raw_prompts)} != {batch_size}"
        assert len(batch_conversations) == batch_size, f"{len(batch_conversations)} != {batch_size}"

        # Deduplicate adjacent tool calls, since they're merged into one turn.
        # [user, assistant, tool, tool, assistant] -> [user, assistant, tool, assistant]
        # TODO: it's chat_template specific, find a more generic way to do this.
        def deduplicate_adjacent_tool_calls(roles):
            result = []
            for role, group in itertools.groupby(roles):
                if role == "tool":
                    result.append(role)
                else:
                    result.extend(group)
            return result

        loss_mask = attention_mask.clone()
        for i in range(batch_size):
            responses = batch_conversations[i][len(raw_prompts[i]) :]
            assert len(responses) > 0, f"responses is empty: {responses}"

            roles = deduplicate_adjacent_tool_calls([response["role"] for response in responses])
            # Each turn should be: [BOS]...[EOS]
            eos_indices = input_ids[i].eq(self.tokenizer.eos_token_id).nonzero().squeeze(1)[: len(roles)]
            for j in range(len(roles)):
                if roles[j] == "tool":
                    bos = eos_indices[j - 1] + 1 if j > 0 else 0
                    eos = eos_indices[j]
                    loss_mask[i, bos : eos + 1] = 0

        return loss_mask


@dataclass
class TaskInfo:
    """任务信息封装"""
    task_index: int
    prompt_index: int
    task_offset: int
    server_address: str = ""
    request_id: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    reward: Optional[float] = None  # 任务的reward分数
    
    @property
    def processing_time(self) -> float:
        if self.start_time and self.completion_time:
            return self.completion_time - self.start_time
        return 0.0


class UnifiedTaskStateManager:
    """统一的任务状态管理器，集成所有任务状态相关的功能"""
    
    def __init__(self, num_prompts: int, n: int):
        """
        Args:
            num_prompts: 总的 prompt 数量
            n: 每个 prompt 的回复数量
        """
        self.num_prompts = num_prompts
        self.n = n
        
        # 原 TaskTracker 的功能
        self.active_tasks: Dict[int, TaskInfo] = {}
        self.completed_tasks: Set[int] = set()
        self.request_id_to_task: Dict[str, int] = {}
        
        # 统计每个 prompt 对应的 task 的完成情况
        self.prompt_completion_status: Dict[int, Set[int]] = {i: set() for i in range(num_prompts)}
        self.valid_prompt_indices: Set[int] = set()
        
        # 统计信息
        self.completed_prompts = 0
        self.valid_prompts = 0
        
        # 存储已完成任务的reward信息
        self.completed_task_rewards: Dict[int, float] = {}
        
        # 任务队列管理
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.task_creation_done: asyncio.Event = asyncio.Event()
        self.early_stop_signal: asyncio.Event = asyncio.Event()
    
    def add_task(self, task_info: TaskInfo):
        """添加任务"""
        self.active_tasks[task_info.task_index] = task_info
        if task_info.request_id:
            self.request_id_to_task[task_info.request_id] = task_info.task_index
    
    def update_task_reward(self, task_index: int, reward: float):
        """更新任务的reward信息"""
        if task_index in self.active_tasks:
            self.active_tasks[task_index].reward = reward
    
    def complete_task(self, task_index: int, is_validation: bool = False) -> Optional[TaskInfo]:
        """
        完成任务并更新所有相关状态
        
        Args:
            task_index: 任务索引
            is_validation: 是否为验证模式
            
        Returns:
            完成的任务信息，如果任务不存在则返回 None
        """
        if task_index not in self.active_tasks:
            return None
            
        # 获取并移除任务信息
        task_info = self.active_tasks.pop(task_index)
        task_info.completion_time = time.time()
        self.completed_tasks.add(task_index)
        
        # 存储已完成任务的reward信息
        if task_info.reward is not None:
            self.completed_task_rewards[task_index] = task_info.reward
        
        # 清理映射
        if task_info.request_id and task_info.request_id in self.request_id_to_task:
            del self.request_id_to_task[task_info.request_id]
        
        # 更新 prompt 完成状态
        prompt_index = task_info.prompt_index
        task_offset = task_info.task_offset
        self.prompt_completion_status[prompt_index].add(task_offset)
        
        # 检查这个 prompt 是否完全完成
        if len(self.prompt_completion_status[prompt_index]) == self.n:
            self.completed_prompts += 1
            
            task_validity = self._check_prompt_score_variance(prompt_index)

            # 如果提供了有效性信息，更新合法 prompt 统计
            if task_validity:
                self.valid_prompts += 1
                self.valid_prompt_indices.add(prompt_index)
            
            # 打印完成信息
            if is_validation:
                print(f"[Validation] Prompt {prompt_index} 完成 (已完成 {self.completed_prompts}/{self.num_prompts} 个)")
            else:
                if task_validity:
                    print(f"[EarlyStop] Prompt {prompt_index} 完成且合法 (已完成 {self.completed_prompts}/{self.num_prompts} 个，合法 {self.valid_prompts} 个)")
                else:
                    print(f"[EarlyStop] Prompt {prompt_index} 完成但不合法 (方差为0) (已完成 {self.completed_prompts}/{self.num_prompts} 个，合法 {self.valid_prompts} 个)")
        
        return task_info

    def is_prompt_completed(self, prompt_index: int) -> bool:
        """检查指定 prompt 是否完全完成"""
        return len(self.prompt_completion_status[prompt_index]) == self.n
    
    def get_cancelled_requests_by_server(self, task_indices: Set[int]) -> Dict[str, List[str]]:
        """获取被取消的请求按服务器分组（原 TaskTracker 方法）"""
        cancelled_by_server = defaultdict(list)
        for task_index in task_indices:
            if task_index in self.active_tasks:
                task_info = self.active_tasks[task_index]
                if task_info.request_id:
                    cancelled_by_server[task_info.server_address].append(task_info.request_id)
        return dict(cancelled_by_server)
    
    def get_continuous_completed_prefix_length(self) -> int:
        """获取从索引 0 开始的连续完成的前缀长度"""
        continuous_completed_length = 0
        for prompt_idx in range(self.num_prompts):
            if self.is_prompt_completed(prompt_idx):
                continuous_completed_length += 1
            else:
                break
        return continuous_completed_length
    
    def get_valid_prompts_in_prefix(self, prefix_length: int) -> int:
        """获取指定前缀长度中的合法 prompt 数量"""
        return len([i for i in range(min(prefix_length, self.num_prompts)) if i in self.valid_prompt_indices])
    
    def _check_prompt_score_variance(self, prompt_index: int) -> bool:
        """
        检查一个 prompt 的 n 个回复的分数方差是否不为 0
        
        Args:
            prompt_index: prompt 索引
            
        Returns:
            bool: 方差不为 0 则返回 True，否则返回 False
        """
        scores = []
        
        # 收集该 prompt 对应的 n 个回复的分数
        for task_offset in range(self.n):
            task_index = prompt_index * self.n + task_offset
            if task_index in self.completed_tasks:
                # 从已完成的任务中查找对应的TaskInfo
                # 由于任务已完成，需要从某个地方获取TaskInfo
                # 这里我们需要一个已完成任务的记录
                reward = self.completed_task_rewards.get(task_index, None)
                if reward is not None:
                    scores.append(reward)
                else:
                    logger.debug(f"Prompt {prompt_index}, task {task_offset}: 没有找到有效的reward")
                    return False  # 存在没有reward的任务，整个 prompt 无效
            else:
                # 任务尚未完成
                logger.debug(f"Prompt {prompt_index}, task {task_offset}: 任务尚未完成")
                return False
        
        return len(scores) == self.n and np.var(scores) > 1e-8

class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str],
        max_cache_size: int = 16384,
        abort_callback: callable = None,
        get_load_callback: callable = None,
    ):
        """
        Args:
            config: DictConfig.
            server_addresses: List[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
            abort_callback: callable, callback function to abort requests immediately.
        """
        self.config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

        # Least requests load balancing
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)
        self.server_addresses = server_addresses

        # LRU cache to map request_id to address
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

        # Callback for immediate request abortion
        self.abort_callback = abort_callback

        # Callback for getting load of each server
        self.get_load_callback = get_load_callback

        self.background_tasks = set()
        if self.config.multi_turn.completion_callback is None:
            self.completion_callback = ToolCompletionCallback(config, self)
            logger.warning("completion_callback is None, use ToolCompletionCallback")
        else:
            module_path, class_name = self.config.multi_turn.completion_callback.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.completion_callback = getattr(module, class_name)(config, self)

    async def _submit_chat_completions_and_callback(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        info: Dict[str, Any],
        address: Optional[str] = None,
    ):
        """Submit chat completion request, wait request finish and do callback."""
        # Check if a specific address was pre-selected
        if address is None:
            if request_id:
                request_id = request_id.removeprefix("chatcmpl-")
                if request_id in self.request_id_to_address:
                    address = self.request_id_to_address.pop(request_id)
            else:
                address = self.weighted_addresses[0][1]
                self.weighted_addresses[0][0] += 1
                heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address

        # 更新任务的 request_id 和 start_time
        task_index = info.get("__task_index__")
        if task_index is not None and hasattr(self, '_current_unified_task_manager') and self._current_unified_task_manager:
            # 找到对应的任务并更新信息
            if task_index in self._current_unified_task_manager.active_tasks:
                task_info = self._current_unified_task_manager.active_tasks[task_index]
                task_info.request_id = request_id
                task_info.start_time = time.time()
                # 更新 request_id 到任务的映射
                self._current_unified_task_manager.request_id_to_task[request_id] = task_index

        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(
                address,
                messages=messages,
                tools=self.completion_callback.tool_schemas,
                extra_body=self.completion_callback.extra_body,
                extra_headers={"x-request-id": request_id},
                **info["__sampling_params__"],
            )

            await self.completion_callback(messages, completions, info)
            
            # 在completion callback完成后，更新任务的reward信息
            task_index = info.get("__task_index__")
            if task_index is not None and hasattr(self, '_current_unified_task_manager') and self._current_unified_task_manager:
                # 从messages中提取reward信息
                if len(messages) > 0 and messages[-1].get("role") == "assistant":
                    assistant_message = messages[-1]
                    if "score" in assistant_message:
                        reward = float(assistant_message["score"])
                        self._current_unified_task_manager.update_task_reward(task_index, reward)
                        
        except Exception as e:
            logger.exception(f"completion callback failed with exception: {e}")

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        session = None
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers")
            
            # 设置合理的超时时间，避免无限等待
            timeout = aiohttp.ClientTimeout(total=600)  # 10分钟超时
            session = aiohttp.ClientSession(timeout=timeout)
            
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                # 检查响应状态
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {error_text}")
                
                # 检查响应类型
                content_type = resp.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    error_text = await resp.text()
                    # 如果是plain text响应，可能是服务器错误或请求被abort
                    raise Exception(f"Expected JSON response but got {content_type}: {error_text}")
                
                data = await resp.json()
                return ChatCompletion(**data)
        finally:
            # 只在 finally 块中关闭 session，避免重复关闭
            if session:
                try:
                    # 使用 asyncio.shield 防止 session.close() 被取消
                    await asyncio.shield(session.close())
                except Exception as e:
                    logger.warning(f"Failed to close HTTP session: {e}")
                    # 如果 shield 失败，尝试强制关闭
                    try:
                        if not session.closed:
                            session._connector.close()  # 强制关闭连接器
                    except Exception:
                        pass  # 忽略强制关闭时的异常

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        t_start = time.time()
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        is_validation = batch.meta_info.get("validate", False)
        n = 1 if is_validation else self.config.n
        batch_conversations = [None] * len(batch) * n
        
        # 创建统一的任务状态管理器
        unified_task_manager = UnifiedTaskStateManager(num_prompts=len(batch), n=n)
        
        # 将 unified_task_manager 设置为实例属性，以便在其他方法中使用
        self._current_unified_task_manager = unified_task_manager
        
        # 早停配置，验证模式下关闭早停机制
        if is_validation:
            min_valid_prompts = len(batch) * n  # 验证模式下等待所有任务完成
            print("[ChatCompletionScheduler] Validation mode: early stopping disabled")
        else:
            # 基于分数方差筛选的合法 prompt 数量
            min_valid_prompts = batch.meta_info.get("needed_valid_prompt_num", len(batch))
            print(f"[ChatCompletionScheduler] Early stop ratio: {min_valid_prompts}/{len(batch)}")
        
        # 初始化批次对话
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            batch_conversations[batch_index] = conversation.tolist()
        
        # 启动动态任务创建器
        task_creator = asyncio.create_task(
            self._dynamic_task_creator(
                batch_conversations=batch_conversations,
                sampling_params=kwargs,
                unified_task_manager=unified_task_manager,
            )
        )
        
        # 使用动态任务管理器等待任务完成
        num_prompts = len(batch)
        cancelled_request_ids_by_address = await self._wait_with_dynamic_task_creation(
            unified_task_manager=unified_task_manager,
            min_valid_prompts=min_valid_prompts,
            is_validation=is_validation,
        )
        
        # 确保任务创建器完成
        if not task_creator.done():
            task_creator.cancel()
            try:
                await task_creator
            except asyncio.CancelledError:
                pass

        completed_prompts, valid_prompts, valid_prompt_indices = unified_task_manager.completed_prompts, unified_task_manager.valid_prompts, unified_task_manager.valid_prompt_indices
        
        if is_validation:
            print(f"[ChatCompletionScheduler] Validation completed: {completed_prompts}/{num_prompts} prompts processed")
            # 验证模式或所有 prompt 都合法时，不过滤
            output_batch = self.completion_callback.postprocess(batch, batch_conversations, n=n)
            output_batch.meta_info["valid_prompt_indices"] = list(range(num_prompts))
            output_batch.meta_info["need_filtering"] = False
        else:
            print(f"[ChatCompletionScheduler] Early stop triggered: {completed_prompts}/{num_prompts} prompts completed, {valid_prompts} valid prompts")
            
            # 在训练模式下，应用 FIFO 选择原则
            valid_prompt_indices = self._apply_fifo_selection(valid_prompt_indices, min_valid_prompts)
            valid_prompts = len(valid_prompt_indices)
        
            # 过滤生成的数据
            filtered_batch, filtered_conversations = self._filter_invalid_prompts(
                batch, batch_conversations, valid_prompt_indices, n
            )
            print(f"[ChatCompletionScheduler] Filtered data: kept {len(valid_prompt_indices)}/{num_prompts} valid prompts")
            
            # 使用过滤后的数据进行 postprocess
            output_batch = self.completion_callback.postprocess(filtered_batch, filtered_conversations, n=n)
            
            # 返回合法的 prompt 索引给调用方，让调用方也过滤 new_batch
            output_batch.meta_info["valid_prompt_indices"] = sorted(list(valid_prompt_indices))
            output_batch.meta_info["need_filtering"] = True            
        
        output_batch.meta_info["timing"] = {
            "generate_sequences": time.time() - t_start,
        }
        output_batch.meta_info["completed_prompts"] = completed_prompts
        output_batch.meta_info["valid_prompts"] = valid_prompts
        output_batch.meta_info["total_prompts"] = num_prompts
        output_batch.meta_info["original_prompts"] = num_prompts
        output_batch.meta_info["cancelled_request_ids_by_address"] = cancelled_request_ids_by_address
        
        # 清理临时的统一任务管理器引用
        if hasattr(self, '_current_unified_task_manager'):
            delattr(self, '_current_unified_task_manager')
        
        print("[ChatCompletionScheduler] generate_sequences done")
        return output_batch

    async def _wait_with_dynamic_task_creation(self, unified_task_manager: UnifiedTaskStateManager, min_valid_prompts: int, is_validation: bool = False) -> tuple[int, int, set, dict]:
        """
        等待任务完成，基于合法 prompt 数量的早停功能，支持动态任务创建
        
        Args:
            unified_task_manager: 统一的任务状态管理器
            min_valid_prompts: 最少需要的合法 prompt 数量
            is_validation: 是否为验证模式
            
        Returns:
            (实际完成的 prompt 数量, 合法的 prompt 数量, 合法的 prompt 索引集合, 被取消的 request_id 集合)
        """
        # 已经发送到服务器的任务
        pending_tasks = set()
        # 收集被取消的 request_id 按地址分组
        cancelled_request_ids_by_address = {}
        
        # 动态管理任务：既要从队列获取新任务，也要等待现有任务完成
        while not unified_task_manager.task_creation_done.is_set() or pending_tasks or not unified_task_manager.task_queue.empty():
            # 检查是否达到早停条件（仅训练模式）
            # 使用 FIFO 策略：确保选择的 valid prompts 和它们之前的所有 prompts 都已完成
            if not is_validation:
                can_early_stop, early_stop_prefix_length = self._check_fifo_early_stop_condition(unified_task_manager, min_valid_prompts)
                if can_early_stop:
                    print(f"[EarlyStop] 达到 FIFO 早停条件：前 {early_stop_prefix_length} 个 prompts 都已完成，"
                            f"其中有 {unified_task_manager.get_valid_prompts_in_prefix(early_stop_prefix_length)} 个合法 prompts (需要 {min_valid_prompts} 个)")
                    unified_task_manager.early_stop_signal.set()  # 设置早停信号，通知任务创建器停止创建新任务
                    break
            
            # 从队列获取新任务（非阻塞）
            new_tasks_added = 0
            while not unified_task_manager.task_queue.empty():
                try:
                    task = unified_task_manager.task_queue.get_nowait()
                    pending_tasks.add(task)
                    new_tasks_added += 1
                except asyncio.QueueEmpty:
                    break
            
            if new_tasks_added > 0:
                print(f"[TaskManager] 添加了 {new_tasks_added} 个新任务，当前待处理任务数：{len(pending_tasks)}")
            
            # 如果没有待处理任务，等待新任务或任务创建完成
            if not pending_tasks:
                if not unified_task_manager.task_creation_done.is_set():
                    await asyncio.sleep(0.1)  # 短暂等待新任务
                    continue
                else:
                    break  # 任务创建完成且无待处理任务
            
            # 等待至少一个任务完成
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            
            # 检查完成的任务并更新 prompt 完成状态
            for task in done_tasks:
                try:
                    await task  # 获取任务结果，如果有异常会被抛出
                    
                    # 获取任务信息
                    task_index = getattr(task, '_task_index', None)
                    
                    if task_index is not None:
                        # 使用统一任务管理器标记任务完成，所有逻辑都在complete_task中处理
                        _ = unified_task_manager.complete_task(
                            task_index, 
                            is_validation=is_validation
                        )
                except Exception as e:
                    logger.exception(f"Task failed with exception: {e}")

        # 如果早停信号被设置，则取消剩余任务
        if unified_task_manager.early_stop_signal.is_set():
            print("[EarlyStop] 收到早停信号，取消剩余任务")
            # 收集被取消任务的索引
            cancelled_task_indices = set()
            
            # 取消队列中尚未开始的任务
            cancelled_from_queue = 0
            while not unified_task_manager.task_queue.empty():
                try:
                    task = unified_task_manager.task_queue.get_nowait()
                    task.cancel()
                    task_index = getattr(task, '_task_index', None)
                    if task_index is not None:
                        cancelled_task_indices.add(task_index)
                    cancelled_from_queue += 1
                except asyncio.QueueEmpty:
                    break
            
            if cancelled_from_queue > 0:
                print(f"[EarlyStop] 从队列中取消了 {cancelled_from_queue} 个尚未处理的任务")
            
            # 收集待处理任务的索引
            for task in pending_tasks:
                task_index = getattr(task, '_task_index', None)
                if task_index is not None:
                    cancelled_task_indices.add(task_index)
                task.cancel()
            
            # 清空待处理任务集合
            pending_tasks = set()
            
            # 使用统一任务管理器获取被取消的请求按服务器分组
            cancelled_request_ids_by_address = unified_task_manager.get_cancelled_requests_by_server(cancelled_task_indices)
            
            # 等待取消的任务清理完成，使用更短的超时避免无限等待
            print(f"[EarlyStop] 等待 {len(pending_tasks)} 个被取消任务的清理...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=30.0  # 30秒超时
                )
                print("[EarlyStop] 所有被取消任务已完成清理")
            except asyncio.TimeoutError:
                print("[EarlyStop] 警告：部分任务清理超时，可能存在未完成的服务器端操作")
            except Exception as e:
                logger.exception(f"[EarlyStop] 任务清理过程中发生错误: {e}")
            
            # 立即abort服务器端请求（在取消客户端任务之前）
            if cancelled_request_ids_by_address and self.abort_callback and callable(self.abort_callback):
                print(f"[EarlyStop] 立即abort {sum(len(ids) for ids in cancelled_request_ids_by_address.values())} 个服务器端请求")
                try:
                    self.abort_callback(cancelled_request_ids_by_address)
                except Exception as e:
                    logger.exception(f"Error calling abort_callback: {e}")
            
        return cancelled_request_ids_by_address

    async def _dynamic_task_creator(self, batch_conversations: List[List[Dict[str, str]]], sampling_params: Dict[str, Any], unified_task_manager: UnifiedTaskStateManager):
        """
        动态任务创建器：根据服务器负载情况分批次创建任务
        
        Args:
            batch_conversations: 批次对话列表
            sampling_params: 采样参数
            unified_task_manager: 统一的任务状态管理器
        """
        total_tasks = len(batch_conversations)
        created_tasks = 0
        n = unified_task_manager.n
        
        # 默认配置
        batch_creation_interval = getattr(self.config, 'batch_creation_interval', 10)  # 批次创建间隔（秒）
        max_concurrent_per_server = getattr(self.config, 'max_concurrent_per_server', 512)  # 每个服务器最大并发数
        
        print(f"[TaskCreator] 开始动态创建 {total_tasks} 个任务")
        print(f"[TaskCreator] 配置：batch_creation_interval={batch_creation_interval}, max_concurrent_per_server={max_concurrent_per_server}")
        
        while created_tasks < total_tasks:
            # 检查早停信号
            if unified_task_manager.early_stop_signal.is_set():
                print(f"[TaskCreator] 收到早停信号，停止创建任务。已创建 {created_tasks}/{total_tasks} 个任务")
                break
            
            # 第一步：调用 get_load_callback 获取所有服务的负载情况
            server_loads = {}
            if self.get_load_callback and callable(self.get_load_callback):
                try:
                    server_loads = await self.get_load_callback()
                    print(f"[TaskCreator] 获取服务器负载情况：{server_loads}")
                except Exception as e:
                    print(f"[TaskCreator] Error calling get_load_callback: {e}")
                    # 如果获取负载失败，使用默认策略
                    server_loads = {}
            
            # 第二步：计算每个服务器可以接收的新任务数量
            server_capacities = {}
            for address in self.server_addresses:
                current_load = server_loads.get(address, 0)
                available_capacity = max(0, max_concurrent_per_server - current_load)
                server_capacities[address] = available_capacity
            
            total_capacity = sum(server_capacities.values())
            print(f"[TaskCreator] 服务器容量情况：{server_capacities}，总容量：{total_capacity}")

            if total_capacity == 0:
                print("[TaskCreator] 所有服务器都已满载，等待...")
                await asyncio.sleep(batch_creation_interval)
                continue
            
            # 第三步：使用 round-robin 方式分配任务
            tasks_to_create = min(total_capacity, total_tasks - created_tasks)
            
            # 创建有容量的服务器地址列表
            available_servers = [address for address, capacity in server_capacities.items() if capacity > 0]
            
            # 使用 round-robin 方式分配任务
            server_index = 0
            tasks_created_this_round = 0
            
            while tasks_created_this_round < tasks_to_create and created_tasks < total_tasks:
                # 选择当前服务器
                current_server = available_servers[server_index]
                
                # 检查当前服务器是否还有容量
                if server_capacities[current_server] > 0:
                    batch_index = created_tasks
                    conversation = batch_conversations[batch_index]
                    prompt_index = batch_index // n
                    task_offset = batch_index % n
                    
                    # 创建任务信息并添加到统一任务管理器
                    task_info = TaskInfo(
                        task_index=batch_index,
                        prompt_index=prompt_index,
                        task_offset=task_offset,
                        server_address=current_server,
                        request_id=None,  # 将在 _submit_chat_completions_and_callback 中设置
                        start_time=None,  # 将在任务开始时设置
                    )
                    unified_task_manager.add_task(task_info)
                    
                    # 创建任务，指定使用当前服务器
                    task = asyncio.create_task(
                        self._submit_chat_completions_semaphore(
                            messages=conversation,
                            request_id=None,
                            sampling_params=sampling_params,
                            task_index=batch_index,
                            address=current_server,
                        )
                    )
                    
                    # 为任务添加索引信息
                    task._task_index = batch_index
                    await unified_task_manager.task_queue.put(task)
                    created_tasks += 1
                    tasks_created_this_round += 1
                    
                    # 减少该服务器的剩余容量
                    server_capacities[current_server] -= 1

                # 轮询到下一个服务器
                server_index = (server_index + 1) % len(available_servers)
                
                # 如果轮询一圈后发现所有服务器都没有容量了，退出循环
                if server_index == 0:
                    remaining_capacity = sum(server_capacities[addr] for addr in available_servers)
                    if remaining_capacity == 0:
                        break
            
            print(f"[TaskCreator] 本轮使用 round-robin 创建了 {tasks_created_this_round} 个任务，总进度：{created_tasks}/{total_tasks}")
            
            # 第四步：如果还有任务需要创建，等待一段时间
            if created_tasks < total_tasks:
                await asyncio.sleep(batch_creation_interval)
        
        print(f"[TaskCreator] 完成所有 {total_tasks} 个任务的创建")
        unified_task_manager.task_creation_done.set()

    def _filter_invalid_prompts(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], 
                               valid_prompt_indices: set, n: int) -> tuple[DataProto, List[List[Dict[str, str]]], int]:
        """
        过滤掉非法的 prompt，只保留合法的数据
        
        Args:
            batch: 原始批次数据
            batch_conversations: 所有对话列表
            valid_prompt_indices: 合法的 prompt 索引集合
            n: 每个 prompt 的回复数量
            num_prompts: 总 prompt 数量
            
        Returns:
            (过滤后的批次数据, 过滤后的对话列表, 新的 n 值)
        """        # 创建有序的合法 prompt 索引列表
        sorted_valid_prompt_indices = sorted(list(valid_prompt_indices))
        
        # 1. 构建任务级别的索引（用于过滤 batch_conversations 和任务级数据）
        task_indices = []
        for prompt_idx in sorted_valid_prompt_indices:
            for task_offset in range(n):
                task_index = prompt_idx * n + task_offset
                if task_index < len(batch_conversations):
                    task_indices.append(task_index)
        
        # 2. 过滤 batch_conversations
        filtered_conversations = [batch_conversations[i] for i in task_indices]
        
        # 3. 处理 batch 数据的过滤
        filtered_batch = batch.select_idxs(sorted_valid_prompt_indices)

        return filtered_batch, filtered_conversations
    
    def _check_fifo_early_stop_condition(self, unified_task_manager: UnifiedTaskStateManager, 
                                         min_valid_prompts: int) -> tuple[bool, int]:
        """
        检查基于 FIFO 策略的早停条件
        
        找到从 prompt index 0 开始的连续完成前缀，如果在这个前缀中有足够的 valid prompts，
        则可以进行早停，以保证数据分布的完整性。
        
        Args:
            unified_task_manager: 统一的任务状态管理器
            min_valid_prompts: 最少需要的合法 prompt 数量
            
        Returns:
            (是否可以早停, 连续完成的前缀长度)
        """
        # 使用 UnifiedTaskStateManager 的方法获取连续完成的前缀长度
        continuous_completed_length = unified_task_manager.get_continuous_completed_prefix_length()
        
        if continuous_completed_length == 0:
            return False, 0
        
        # 使用 UnifiedTaskStateManager 的方法计算前缀中的 valid prompts 数量
        valid_prompts_in_prefix = unified_task_manager.get_valid_prompts_in_prefix(continuous_completed_length)
        
        # 如果前缀中的 valid prompts 数量满足要求，则可以早停
        return valid_prompts_in_prefix >= min_valid_prompts, continuous_completed_length

    def _apply_fifo_selection(self, valid_prompt_indices: set, min_valid_prompts: int) -> set:
        """
        按照 FIFO 原则选择前 min_valid_prompts 个 valid prompts
        
        Args:
            valid_prompt_indices: 所有合法的 prompt 索引集合
            min_valid_prompts: 需要的合法 prompt 数量
            
        Returns:
            按顺序选择的 valid prompt 索引集合
        """
        if len(valid_prompt_indices) <= min_valid_prompts:
            return valid_prompt_indices
        
        # 按照 prompt index 排序，选择前 min_valid_prompts 个
        sorted_valid_indices = sorted(list(valid_prompt_indices))
        selected_indices = sorted_valid_indices[:min_valid_prompts]
        
        print(f"[FIFO] 从 {len(valid_prompt_indices)} 个合法 prompts 中按顺序选择前 {len(selected_indices)} 个：{selected_indices}")
        
        return set(selected_indices)

    async def _submit_chat_completions_semaphore(self, messages: List[Dict[str, str]], request_id: str, sampling_params: Dict[str, Any], task_index: int, address: str=None):
        """Submit chat completion request to a specific server and wait for completion"""
        info = {
            "__depth__": 0,  # indicate how many ongoing completion requests
            "__sampling_params__": sampling_params,
            "__task_index__": task_index,
        }

        # Use the unified submit_chat_completions method with specific address
        await self._submit_chat_completions_and_callback(messages=messages, request_id=request_id, info=info, address=address)
