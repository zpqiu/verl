# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Online export utilities for FSDP1 + modelopt quantized models.

This module provides utilities to export quantized state_dict from FSDP1 wrapped
models without modifying the original model weights. This is useful for RL training
scenarios where you need to export weights at each step for rollout.

Example usage:
    ```python
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export.online_export_fsdp import OnlineQuantExporter

    # Step 1: 量化模型
    actor_module = mtq.quantize(actor_module, mtq.NVFP4_DEFAULT_CFG, forward_loop)

    # Step 2: 创建 exporter 并执行预处理（会修改模型，处理 QKV fusion 等）
    # 必须在 FSDP wrap 之前调用！
    exporter = OnlineQuantExporter(actor_module)

    # Step 3: FSDP wrap
    actor_module_fsdp = FSDP(actor_module, ...)
    FSDP.set_state_dict_type(actor_module_fsdp, ...)

    # Step 4: 在每个 RL step 导出量化后的 state_dict
    def rollout_mode(self):
        state_dict = self.actor_module_fsdp.state_dict()
        quantized_state_dict = exporter.export(state_dict)
        # Send quantized_state_dict to rollout workers
    ```
"""

import re
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization import set_quantizer_by_cfg_context

from modelopt.torch.export.layer_utils import is_layernorm, is_quantlinear
from modelopt.torch.export.model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
)
from modelopt.torch.export.quant_utils import (
    get_quantization_format,
    get_weight_block_size,
    preprocess_linear_fusion,
    fuse_prequant_layernorm,
)


def _is_enabled_quantizer(quantizer):
    """检查 quantizer 是否启用"""
    if hasattr(quantizer, "is_enabled") and quantizer.is_enabled:
        return True
    if isinstance(quantizer, SequentialQuantizer):
        return any(q.is_enabled for q in quantizer)
    return False


class OnlineQuantExporter:
    """用于在线导出量化模型 state_dict 的工具类。

    此类专为 FSDP1 + modelopt 场景设计，支持：
    - 处理 QKV fusion 等 fused layers
    - 不修改原始模型权重（训练可继续）
    - 每个 step 根据当前权重动态计算 scaling factors
    - 导出 input_quantizer 的 scale 信息

    注意：
    - 必须在 FSDP wrap 之前创建 exporter
    - 初始化时会执行 requantize_resmooth（会修改模型，但只做一次）
    - 之后每个 step 的导出不会修改模型
    """

    SUPPORTED_FORMATS = {QUANTIZATION_NVFP4, QUANTIZATION_NVFP4_AWQ}

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = None,
        run_resmooth: bool = True,
    ):
        """初始化 exporter。

        必须在 FSDP wrap 之前调用。

        Args:
            model: 量化后的模型（FSDP wrap 之前）
            dtype: 权重的目标数据类型，默认为 bfloat16
            run_resmooth: 是否执行 requantize_resmooth 处理 fused layers
        """
        self.dtype = dtype or torch.bfloat16
        self.model = model

        # 执行 resmooth 和 fusion（修改模型，但只做一次）
        if run_resmooth:
            self._requantize_resmooth_fused_layers()

        # 提取量化层的元信息
        self.quant_layer_configs = self._extract_quant_layer_configs()

        if not self.quant_layer_configs:
            raise ValueError(
                "未找到任何量化层。请确保：\n"
                "1. 模型已经通过 mtq.quantize() 量化\n"
                "2. 在 FSDP wrap 之前创建 exporter"
            )
        
        print(f"======================DEBUG: fused_layer_to_group ==========================")
        for k, v in self.fused_layer_to_group.items():
            print(f"DEBUG: {k}: {', '.join(v)}")

        print(f"======================DEBUG: quant_layer_configs ==========================")
        for k, v in self.quant_layer_configs.items():
            print(f"DEBUG: {k}: {v}")

    def _requantize_resmooth_fused_layers(self):
        """执行 requantize 和 resmooth，处理 fused layers（如 QKV）。

        此方法会修改模型：
        - 统一 fused layers 的 pre_quant_scale 和 amax
        - 将 pre_quant_scale 融合到 layernorm 权重中

        只在初始化时调用一次。
        """
        model = self.model
        input_to_linear = defaultdict(list)
        output_to_layernorm = {}
        quantization_format = get_quantization_format(model)

        def is_quantized_module(module):
            return is_quantlinear(module) and (
                _is_enabled_quantizer(getattr(module, "input_quantizer", None))
                or _is_enabled_quantizer(getattr(module, "weight_quantizer", None))
            )

        qkv_group = ["q_proj", "k_proj", "v_proj"]
        mlp_group = ["gate_proj", "up_proj"]
        for name, module in model.named_modules():
            if "quantizer" in name:
                continue
            # print(f"DEBUG: {name}, {type(module)}")
            if ("self_attn" not in name and "mlp" not in name) or not is_quantized_module(module):
                continue
            module.name = name
            for param_name in [*qkv_group, *mlp_group]:
                if param_name in name:
                    input_to_linear[name.split(param_name)[0]].append(module)
                    break

        fused_linear_groups = []  # 存储 fused layer 组，用于导出时统一 amax

        # 处理 fused layers
        for tensor, modules in input_to_linear.items():
            quant_format = get_quantization_format(modules[0])
            if len(modules) > 1 and quant_format not in [
                QUANTIZATION_FP8,
                QUANTIZATION_NONE,
                QUANTIZATION_FP8_PB_REAL,
            ]:
                # preprocess_linear_fusion(modules)
                # 记录 fused layer 组（用于导出时统一 amax）
                fused_linear_groups.append([m.name for m in modules])

        # 记录 fused layers 信息：每个 layer 属于哪个 fusion group
        # fused_layer_to_group: layer_name -> list of all layer names in the same group
        self.fused_layer_to_group = {}
        for group in fused_linear_groups:
            for layer_name in group:
                self.fused_layer_to_group[layer_name] = group

    def _extract_quant_layer_configs(self) -> dict[str, dict[str, Any]]:
        """从模型中提取每个量化层的元信息。

        Returns:
            dict: key 为模块名称，value 为量化配置
        """
        configs = {}

        for name, module in self.model.named_modules():
            quant_format = get_quantization_format(module)
            if quant_format in [QUANTIZATION_NONE, None]:
                continue
            if quant_format not in self.SUPPORTED_FORMATS:
                continue

            weight_quantizer = getattr(module, "weight_quantizer", None)
            if weight_quantizer is None:
                continue

            # 处理 SequentialQuantizer
            if isinstance(weight_quantizer, SequentialQuantizer):
                actual_quantizer = weight_quantizer[0]
            else:
                actual_quantizer = weight_quantizer

            config = {
                "quantization_format": quant_format,
                "block_size": get_weight_block_size(module),
                # 记录是否有 input_quantizer
                "has_input_quantizer": hasattr(module, "input_quantizer")
                and _is_enabled_quantizer(module.input_quantizer),
                # 记录是否已经 fused with layernorm
                "fused_with_layernorm": getattr(module, "fused_with_layernorm", False),
            }
            configs[name] = config

        return configs

    def export(
        self,
        state_dict: dict[str, torch.Tensor],
        fsdp_key_prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        """将 state_dict 中的权重转换为量化格式。

        完全对应 preprocess_linear_fusion 的逻辑：
        1. 对 fused layers 统一 pre_quant_scale（计算平均值）
        2. Resmooth 权重：weight = weight * old_scale / avg_scale
        3. 重新计算 weight amax（因为权重被缩放了）
        4. 统一 input amax（取 max）
        5. 统一 weight amax（仅对标量 amax，取 max）

        不修改原始 state_dict。

        Args:
            state_dict: 从 FSDP 模型获取的 state_dict
            fsdp_key_prefix: FSDP wrapper 可能添加的 key 前缀

        Returns:
            量化后的 state_dict，包含：
            - {name}.weight: packed uint8 量化权重
            - {name}.weight_scale: per-block 缩放因子
            - {name}.weight_scale_2: per-tensor 缩放因子
            - {name}.input_scale: 输入缩放因子（如果有）
            - {name}.pre_quant_scale: 预量化缩放因子（如果有且未 fused）
        """
        new_state_dict = {}

        # 第一遍：收集所有权重、quantizer 的 amax 值和 pre_quant_scale
        weight_cache = {}  # module_name -> weight tensor (克隆，用于 resmooth)
        input_amax_cache = {}  # module_name -> input_quantizer amax
        pre_quant_scale_cache = {}  # module_name -> pre_quant_scale

        for key, value in state_dict.items():
            clean_key = self._clean_fsdp_key(key, fsdp_key_prefix)

            if clean_key.endswith(".weight") and "_quantizer" not in clean_key:
                module_name = clean_key[:-7]  # 去掉 ".weight"
                if module_name in self.quant_layer_configs:
                    # 克隆权重，因为后面可能需要 resmooth
                    weight_cache[module_name] = value.clone().to(self.dtype)
            elif "input_quantizer._amax" in clean_key:
                module_name = clean_key.replace(".input_quantizer._amax", "")
                input_amax_cache[module_name] = value.clone()
            elif "input_quantizer._pre_quant_scale" in clean_key:
                module_name = clean_key.replace(".input_quantizer._pre_quant_scale", "")
                pre_quant_scale_cache[module_name] = value.clone()

        # 第二遍：对 fused layers 处理 pre_quant_scale 和 resmooth 权重
        # 完全对应 preprocess_linear_fusion 的逻辑：
        # 1. 统一 pre_quant_scale 为平均值
        # 2. Resmooth 权重：weight = weight * old_scale / avg_scale
        # 3. 重新计算 weight amax（因为权重被缩放了）
        # 4. 统一 input amax（取 max）
        # 5. 统一 weight amax（仅对标量 amax，取 max）
        #
        unified_weight_amax = {}  # module_name -> unified weight amax
        unified_input_amax = {}  # module_name -> unified input amax
        unified_pre_quant_scale = {}  # module_name -> unified pre_quant_scale

        # 处理已记录的 fused layer groups
        processed_groups = set()
        for module_name in self.quant_layer_configs:
            if module_name in self.fused_layer_to_group:
                group = self.fused_layer_to_group[module_name]
                group_key = tuple(sorted(group))

                if group_key in processed_groups:
                    continue
                processed_groups.add(group_key)

                # 收集 group 中所有 layer 的 pre_quant_scale
                group_pre_quant_scales = []
                for layer_name in group:
                    if layer_name in pre_quant_scale_cache:
                        group_pre_quant_scales.append(
                            (layer_name, pre_quant_scale_cache[layer_name])
                        )

                # 如果有 pre_quant_scale，计算平均值并 resmooth 权重
                if group_pre_quant_scales:
                    avg_pre_quant_scale = torch.mean(
                        torch.stack([pqs for _, pqs in group_pre_quant_scales]),
                        dim=0,
                    )

                    # 对每个 layer 进行 resmooth
                    for layer_name, old_scale in group_pre_quant_scales:
                        if layer_name in weight_cache:
                            weight = weight_cache[layer_name]
                            device = weight.device

                            # 检查是否需要 resmooth
                            if not torch.equal(old_scale.to(device), avg_pre_quant_scale.to(device)):
                                # Resmooth 权重：new_weight = old_weight * old_scale / avg_scale
                                weight_cache[layer_name] = (
                                    weight
                                    * old_scale.to(dtype=weight.dtype, device=device)
                                    / avg_pre_quant_scale.to(dtype=weight.dtype, device=device)
                                )

                        # 记录统一后的 pre_quant_scale
                        unified_pre_quant_scale[layer_name] = avg_pre_quant_scale

                # 根据 resmooth 后的权重重新计算 weight amax
                group_weight_amaxs = []
                for layer_name in group:
                    if layer_name in weight_cache:
                        weight = weight_cache[layer_name]
                        amax = weight.abs().max()
                        # 只对标量 amax 进行统一
                        if amax.numel() == 1:
                            group_weight_amaxs.append(amax)

                # 统一 weight amax（取 max）
                if group_weight_amaxs:
                    unified_amax = torch.max(torch.stack(group_weight_amaxs))
                    for layer_name in group:
                        unified_weight_amax[layer_name] = unified_amax

                # 统一 input amax（取 max）
                group_input_amaxs = []
                for layer_name in group:
                    if layer_name in input_amax_cache:
                        amax = input_amax_cache[layer_name]
                        if amax.numel() == 1:
                            group_input_amaxs.append(amax)

                if group_input_amaxs:
                    unified_input = torch.max(torch.stack(group_input_amaxs))
                    for layer_name in group:
                        unified_input_amax[layer_name] = unified_input

        # 处理非 fused layers
        for module_name in self.quant_layer_configs:
            if module_name not in self.fused_layer_to_group:
                # 非 fused layer，根据当前权重计算 amax
                if module_name in weight_cache:
                    unified_weight_amax[module_name] = weight_cache[module_name].abs().max()
                if module_name in input_amax_cache:
                    unified_input_amax[module_name] = input_amax_cache[module_name]
                if module_name in pre_quant_scale_cache:
                    unified_pre_quant_scale[module_name] = pre_quant_scale_cache[module_name]

        # print(f"======================DEBUG: unified_weight_amax ==========================")
        # for k, v in unified_weight_amax.items():
        #     print(f"DEBUG: {k}: {v}")
        # print(f"======================DEBUG: unified_input_amax ==========================")
        # for k, v in unified_input_amax.items():
        #     print(f"DEBUG: {k}: {v}")
        # print(f"======================DEBUG: unified_pre_quant_scale ==========================")
        # for k, v in unified_pre_quant_scale.items():
        #     print(f"DEBUG: {k}: {v}")

        # 第三遍：处理权重和生成量化输出
        for key, value in state_dict.items():
            clean_key = self._clean_fsdp_key(key, fsdp_key_prefix)

            # 跳过 quantizer 内部的 buffer（已经在第一遍处理了）
            if "_quantizer." in clean_key:
                continue

            # 检查是否是权重
            if not clean_key.endswith(".weight"):
                new_state_dict[key] = value
                continue

            module_name = clean_key[:-7]  # 去掉 ".weight"

            # 检查是否需要量化
            if module_name not in self.quant_layer_configs:
                new_state_dict[key] = value
                continue

            config = self.quant_layer_configs[module_name]
            quant_format = config["quantization_format"]
            block_size = config["block_size"]

            # 使用 resmooth 后的权重（已经在第二遍处理过了）
            weight = weight_cache.get(module_name)
            if weight is None:
                weight = value.clone().to(self.dtype)
            device = weight.device

            # 获取统一后的 weight amax
            weight_amax = unified_weight_amax.get(module_name)
            if weight_amax is None:
                # 如果没有缓存的 amax，使用权重的 max 值
                weight_amax = weight.abs().max()

            weight_amax = weight_amax.to(device).float()

            # 计算 weight_scaling_factor_2: amax / (6 * 448)
            weight_scaling_factor_2 = weight_amax / (6.0 * 448.0)

            # 根据 resmooth 后的权重计算 per-block scaling factor
            weight_scaling_factor, _ = NVFP4QTensor.get_weights_scaling_factor(
                weight, block_size, weight_scaling_factor_2
            )

            # 量化权重
            quantized_result = NVFP4QTensor.quantize(
                weight,
                block_size,
                weight_scaling_factor,
                weight_scaling_factor_2.view(-1, 1, 1)
                if weight_scaling_factor_2.dim() != 0
                else weight_scaling_factor_2,
            )
            quantized_weight = quantized_result[0]._quantized_data

            # 添加量化后的权重
            new_state_dict[key] = quantized_weight

            # 添加 weight scales
            key_prefix = key[:-6]  # 去掉 "weight"
            new_state_dict[key_prefix + "weight_scale"] = weight_scaling_factor
            new_state_dict[key_prefix + "weight_scale_2"] = weight_scaling_factor_2.squeeze()

            # 处理 input_quantizer
            if config["has_input_quantizer"]:
                # 使用统一后的 input amax
                input_amax = unified_input_amax.get(module_name)
                if input_amax is not None:
                    # input_scale = amax / (maxbound * 448)
                    # 对于 NVFP4，maxbound = 6
                    input_scale = input_amax.float().to(device) / (6.0 * 448.0)
                    new_state_dict[key_prefix + "input_scale"] = input_scale.squeeze()

                # 处理 pre_quant_scale（如果没有 fused with layernorm）
                # 使用统一后的 pre_quant_scale
                if not config["fused_with_layernorm"]:
                    pre_quant_scale = unified_pre_quant_scale.get(module_name)
                    if pre_quant_scale is not None:
                        new_state_dict[key_prefix + "pre_quant_scale"] = pre_quant_scale

        return new_state_dict

    def _clean_fsdp_key(self, key: str, prefix: str) -> str:
        """移除 FSDP 添加的 key 前缀"""
        if prefix and key.startswith(prefix):
            return key[len(prefix) :]
        return key

    def get_quant_config_json(self) -> dict[str, Any]:
        """生成量化配置的 JSON 格式"""
        if not self.quant_layer_configs:
            return {"quant_algo": None}

        first_config = next(iter(self.quant_layer_configs.values()))

        return {
            "quant_algo": (
                "NVFP4"
                if first_config["quantization_format"] == QUANTIZATION_NVFP4
                else "NVFP4_AWQ"
            ),
            "group_size": first_config["block_size"],
            "quantized_layers": list(self.quant_layer_configs.keys()),
        }
