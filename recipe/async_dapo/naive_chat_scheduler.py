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
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import requests
import torch
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.async_server import ChatCompletionScheduler


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


class NaiveChatCompletionScheduler(ChatCompletionScheduler):
    """
    A very naive implementation of ChatCompletionScheduler for demo purpose,
    only do single-turn chat completion.
    """
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig, rollout config.
            model_path: str, model path.
            server_addresses: List[str], server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        super().__init__(config, model_path, server_addresses, max_cache_size)
        # self.compute_score = get_custom_reward_fn(config)
        # if self.compute_score is None:
            # raise ValueError("No custom score function provided")
        self.reward_fn_key = "data_source"
        print(f'[DEBUG] NaiveChatCompletionScheduler config max_response_length: {config.response_length}')

    async def compute_score(self, solution_str, ground_truth):
        """调用 math_verify_service.py 中的评分函数计算分数"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post("http://localhost:8000/score", json={
                    "solution_str": solution_str,
                    "ground_truth": ground_truth,
                }) as response:
                    result = await response.json()
                    
                    return {
                        "score": result["score"],
                        "acc": result["acc"],
                        "pred": result["pred"]
                    }
            
        except Exception as e:
            print(f"Error in compute_score: {e}")
            return {
                "score": -1.0,
                "acc": 0.0,
                "pred": ""
            }

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        if self.config.get("stop_token_ids", None) is not None:
            kwargs["stop_token_ids"] = list(self.config.stop_token_ids)

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        # if not do_sample or is_validate:
        #     kwargs["n"] = 1
        #     kwargs["temperature"] = 0
        if not do_sample:
            kwargs.update({
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            })
        elif is_validate:
            # TODO: try **
            kwargs.update({
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            })

        kwargs.update(sampling_params)
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            assert exception is None, f"exception: {exception}"
            conversation, batch_conversations, batch_scores, batch_index, ground_truth, data_source, extra_info = (
                info["conversation"],
                info["batch_conversations"],
                info["batch_scores"],
                info["batch_index"],
                info["ground_truth"],
                info["data_source"],
                info["extra_info"],
            )

            conversations = []
            scores = []
            for choice in completions.choices:
                chat = conversation.copy()
                chat.append({"role": choice.message.role, "content": choice.message.content})
                conversations.append(chat)

                # if the content is very very short, we should print it and its corresponding information and highlight it
                if len(choice.message.content) < 20:
                    print(f"\n{'='*60}")
                    print(f"\n🚨 检测到超短响应 (批次 {batch_index}):")
                    print(f"  📏 长度: {len(choice.message.content)} 字符")
                    print(f"  📝 内容: \033[93m'{choice.message.content}'\033[0m")
                    print(f"  🎯 参考答案: {str(ground_truth)[:100]}{'...' if len(str(ground_truth)) > 100 else ''}")
                    print(f"  📊 数据源: {data_source}")
                    print(f"{'='*60}")

                result = await self.compute_score(
                    solution_str=choice.message.content,
                    ground_truth=ground_truth,
                )
                scores.append(result)

            batch_scores[batch_index] = scores
            batch_conversations[batch_index] = conversations

            should_log = random.randint(0, 256) == 1
            if should_log:
                print(f"\n{'='*60}")
                print(f"📊 批次调试信息 - 索引: {batch_index}")
                print(f"{'='*60}")

                # print completions.usage
                print(f"🔍 使用情况:")
                print(f"  请求数: {completions.usage.prompt_tokens}")
                print(f"  响应数: {completions.usage.completion_tokens}")
                print(f"  总令牌数: {completions.usage.total_tokens}")
                
                # 分数信息
                print(f"🎯 评分结果:")
                for i, score in enumerate(scores):
                    if isinstance(score, dict):
                        print(f"  响应 {i+1}: 分数={score.get('score', 'N/A'):.3f}, "
                              f"准确率={score.get('acc', 'N/A')}, "
                              f"预测='{score.get('pred', 'N/A')[:50]}{'...' if len(str(score.get('pred', ''))) > 50 else ''}'")
                    else:
                        print(f"  响应 {i+1}: 分数={score}")
                
                # 对话内容
                print(f"\n💬 对话内容:")
                for conv_idx, conv in enumerate(conversations):
                    print(f"  ┌─ 对话 {conv_idx + 1} ─────────────────────")
                    for msg_idx, msg in enumerate(conv):
                        role_emoji = "👤" if msg['role'] == 'user' else "🤖" if msg['role'] == 'assistant' else "🔧"
                        content = msg['content']
                        # 如果内容太长，显示开头和结尾，隐藏中间部分
                        if len(content) > 200:
                            head_length = 80
                            tail_length = 80
                            content = (content[:head_length] + 
                                     f"\n  │     ... (隐藏了 {len(content) - head_length - tail_length} 个字符) ...\n  │     " + 
                                     content[-tail_length:])
                        
                        # 处理多行内容的显示
                        content_lines = content.split('\n')
                        if len(content_lines) > 1:
                            print(f"  │ {role_emoji} {msg['role']}: {content_lines[0]}")
                            for line in content_lines[1:]:
                                print(f"  │     {line}")
                        else:
                            print(f"  │ {role_emoji} {msg['role']}: {content}")
                    print(f"  └{'─' * 35}")
                
                print(f"{'='*60}\n")

            # NOTE: we can call tools and resubmit chat completions here.
            # call_tools(completions, info)
            # await self.submit_chat_completions(callback2, ...)

        # TODO: we may need to control max concurrent requests here, or it will harm prefix cache hit rate.
        tasks, batch_conversations, batch_scores = [], [None] * len(batch), [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            data_item = batch[batch_index]  # DataProtoItem

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            conversation = data_item.non_tensor_batch["raw_prompt"]

            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_scores": batch_scores,
                            "batch_index": batch_index,
                            "conversation": list(conversation),
                            "ground_truth": ground_truth,
                            "data_source": data_source,
                            "extra_info": extra_info,
                        },
                        model=self.model_name,
                        messages=conversation.tolist(),
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print("[NaiveChatCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations, batch_scores, kwargs["n"])

    def _postprocess(self, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], batch_scores: List[List[float]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]

        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)
        batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        # Qwen 2.5 tokenizer will add a newline at the end of the response, we need to strip it
        sequences = [self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False).rstrip() for conversation in batch_conversations]

        # responses: [response]
        # TODO: mask out tools calling tokens?
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        # prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")

        prompt_input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        prompt_attention_mask = batch.batch["attention_mask"]
        prompt_position_ids = batch.batch["position_ids"]

        responses = self.tokenizer(responses, return_tensors="pt", padding_side="right", max_length=self.config.response_length, padding="max_length", truncation=True)
        if n > 1:
            prompt_input_ids =prompt_input_ids.repeat_interleave(n, dim=0)
            prompt_position_ids = prompt_position_ids.repeat_interleave(n, dim=0)
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(n, dim=0)

        valid_response_length_list = responses["attention_mask"].sum(dim=-1).view(-1).tolist()

        # if responses["input_ids"].shape[1] > 1024*4:
        print(f"[DEBUG] {responses['input_ids'].shape}")

        response_input_ids = pad_sequence_to_length(responses["input_ids"], self.config.response_length, self.tokenizer.pad_token_id)
        response_attention_mask = pad_sequence_to_length(responses["attention_mask"], self.config.response_length, 0)

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        reward_tensor = torch.zeros_like(response_input_ids, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        # flatten batch_scores
        batch_scores = [score for scores in batch_scores for score in scores]
        for i in range(len(batch_scores)):
            result, valid_resp_length = batch_scores[i], valid_response_length_list[i]
            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score
            reward_tensor[i, valid_resp_length - 1] = reward

        if n > 1:
            repeated_non_tensor_batch = {}
            for key, val in batch.non_tensor_batch.items():
                repeated_non_tensor_batch[key] = np.repeat(val, n, axis=0)
        else:
            repeated_non_tensor_batch = batch.non_tensor_batch

        for k, v in reward_extra_info.items():
            repeated_non_tensor_batch[k] = np.array(v)

        batch = TensorDict(
            {
                "prompts": prompt_input_ids,
                "responses": response_input_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_level_scores": reward_tensor,
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch, non_tensor_batch=repeated_non_tensor_batch)
