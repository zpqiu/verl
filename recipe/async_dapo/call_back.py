import asyncio
import itertools
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler, CompletionCallback

logger = logging.getLogger(__file__)


class RewardCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)
        self.max_response_length = config.data.max_response_length
        
        print(f"[Callback] 数据集加载逻辑已移至 math_verify_service.py")

        # TODO: add reward manager to calculate reward score once a sample finish

    async def compute_score(self, solution_str, prompt):
        """调用 math_verify_service.py 中的评分函数计算分数"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post("http://localhost:8000/score", json={
                    "solution_str": "<think>" + solution_str,
                    "prompt": prompt,
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

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
            message["score"] = -1.0
            message["acc"] = 0.0
            message["pred"] = ""
            return
        
        question = messages[1]["content"]
        ret = await self.compute_score(message["content"], question)

        message["score"] = ret["score"]
        message["acc"] = ret["acc"]
        message["pred"] = ret["pred"]
        
        messages.append(message)

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        batch_responses, batch_scores = [], []
        for conversation in batch_conversations:
            assistant_response = conversation[-1]
            score = {
                "score": assistant_response["score"],
                "acc": assistant_response["acc"],
                "pred": assistant_response["pred"]
            }
            batch_responses.append(assistant_response["content"] + self.tokenizer.eos_token)
            batch_scores.append(score)

        prompt_input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        prompt_attention_mask = batch.batch["attention_mask"]
        prompt_position_ids = batch.batch["position_ids"]

        responses = self.tokenizer(batch_responses, return_tensors="pt", padding=True, 
                                   max_length=self.max_response_length, 
                                   truncation=True, add_special_tokens=False)

        if n > 1:
            prompt_input_ids =prompt_input_ids.repeat_interleave(n, dim=0)
            prompt_position_ids = prompt_position_ids.repeat_interleave(n, dim=0)
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(n, dim=0)

        valid_response_length_list = responses["attention_mask"].sum(dim=-1).view(-1).tolist()

        # if responses["input_ids"].shape[1] > 1024*4:
        print(f"[DEBUG] {responses['input_ids'].shape}")

        response_input_ids = pad_sequence_to_length(responses["input_ids"], self.max_response_length, self.tokenizer.pad_token_id)
        response_attention_mask = pad_sequence_to_length(responses["attention_mask"], self.max_response_length, 0)

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        reward_tensor = torch.zeros_like(response_input_ids, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        # flatten batch_scores
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
