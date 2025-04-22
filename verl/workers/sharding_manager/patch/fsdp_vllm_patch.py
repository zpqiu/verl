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

import re
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.utils import is_pp_missing_parameter


def get_layer_index(layer_name: str) -> int:
    pattern = r"layers\.(\d+)"
    match = re.search(pattern, layer_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unable to parse layer index from '{layer_name}'")


def get_layer_index(layer_name: str) -> int:
    pattern = r"layers\.(\d+)"
    match = re.search(pattern, layer_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unable to parse layer index from '{layer_name}'")


def patched_ds_v3_load_weights(model: nn.Module, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
    def get_spec_layer_idx_from_weight_name(config: PretrainedConfig, weight_name: str) -> Optional[int]:
        if hasattr(config, "num_nextn_predict_layers") and (config.num_nextn_predict_layers > 0):
            layer_idx = config.num_hidden_layers
            for i in range(config.num_nextn_predict_layers):
                if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                    return layer_idx + i
        return None

    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=model.config.n_routed_experts,
    )
    params_dict = dict(model.named_parameters())
    loaded_params: Set[str] = set()

    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue

        spec_layer = get_spec_layer_idx_from_weight_name(model.config, name)
        if spec_layer is not None:
            continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, model):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, model):
                    continue
                param = params_dict[name]

                # custom weight_loader·
                layer_idx = get_layer_index(name)
                weight_loader = model.model.layers[layer_idx].mlp.experts.weight_loader
                # replace
                # weight_loader = param.weight_loader

                weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue

                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, model):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
    return loaded_params


def patched_qwen_moe_load_weights(model: nn.Module, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=model.config.num_experts,
    )

    params_dict = dict(model.named_parameters())
    loaded_params: Set[str] = set()
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if "mlp.experts" in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                continue
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, model):
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, model):
                    continue
                # Skip loading extra bias for GPTQ models.
                if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                    continue
                # custom weight_loader
                param = params_dict[name]
                layer_idx = get_layer_index(name)
                weight_loader = model.model.layers[layer_idx].mlp.experts.weight_loader
                # replace
                # weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, model):
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        continue
                    else:
                        name = remapped_kv_scale_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params
