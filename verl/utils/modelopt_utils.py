import logging
from typing import Optional
from unittest.mock import patch

import torch

logger = logging.getLogger(__name__)

# copied from modelopt
_default_disabled_quantizer_cfg = {
    "nn.BatchNorm1d": {"*": {"enable": False}},
    "nn.BatchNorm2d": {"*": {"enable": False}},
    "nn.BatchNorm3d": {"*": {"enable": False}},
    "nn.LeakyReLU": {"*": {"enable": False}},
    "*lm_head*": {"enable": False},
    "*proj_out.*": {"enable": False},  # In Whisper model, lm_head has key name proj_out
    "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
    "*router*": {"enable": False},  # Skip the MOE router
    "*mlp.gate.*": {"enable": False},  # Skip the MOE router
    "*mlp.shared_expert_gate.*": {"enable": False},  # Skip the MOE router
    "*linear_attn.conv1d*": {"enable": False},
    "*mixer.conv1d*": {"enable": False},
    "*output_layer*": {"enable": False},
    "output.*": {"enable": False},
    "default": {"enable": False},
}

# customized weight only cfg for modelopt
NVFP4_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": False,
        },
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

# used by vllm initilization
HF_NVFP4_WEIGHT_ONLY_CFG = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16
            },
            "weights": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16
            },
            "targets": [
                "Linear"
            ]
        }
    },
    "ignore": [
        "lm_head",
        "lm_head"
    ],
    "quant_algo": "NVFP4",
    "producer": {
        "name": "modelopt",
        "version": "0.39.0"
    },
    "quant_method": "modelopt"
}

NVFP4_MLP_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*mlp*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3),
            }, 
            "enable": True,
            "pass_through_bwd": True,
        },
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

# used by vllm initilization
HF_NVFP4_MLP_WEIGHT_ONLY_CFG = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16
            },
            "weights": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16
            },
            "targets": [
                "Linear"
            ]
        }
    },
    "ignore": [
        "lm_head",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.10.self_attn.q_proj",
        "model.layers.10.self_attn.k_proj",
        "model.layers.10.self_attn.v_proj",
        "model.layers.10.self_attn.o_proj",
        "model.layers.11.self_attn.q_proj",
        "model.layers.11.self_attn.k_proj",
        "model.layers.11.self_attn.v_proj",
        "model.layers.11.self_attn.o_proj",
        "model.layers.12.self_attn.q_proj",
        "model.layers.12.self_attn.k_proj",
        "model.layers.12.self_attn.v_proj",
        "model.layers.12.self_attn.o_proj",
        "model.layers.13.self_attn.q_proj",
        "model.layers.13.self_attn.k_proj",
        "model.layers.13.self_attn.v_proj",
        "model.layers.13.self_attn.o_proj",
        "model.layers.14.self_attn.q_proj",
        "model.layers.14.self_attn.k_proj",
        "model.layers.14.self_attn.v_proj",
        "model.layers.14.self_attn.o_proj",
        "model.layers.15.self_attn.q_proj",
        "model.layers.15.self_attn.k_proj",
        "model.layers.15.self_attn.v_proj",
        "model.layers.15.self_attn.o_proj",
        "model.layers.16.self_attn.q_proj",
        "model.layers.16.self_attn.k_proj",
        "model.layers.16.self_attn.v_proj",
        "model.layers.16.self_attn.o_proj",
        "model.layers.17.self_attn.q_proj",
        "model.layers.17.self_attn.k_proj",
        "model.layers.17.self_attn.v_proj",
        "model.layers.17.self_attn.o_proj",
        "model.layers.18.self_attn.q_proj",
        "model.layers.18.self_attn.k_proj",
        "model.layers.18.self_attn.v_proj",
        "model.layers.18.self_attn.o_proj",
        "model.layers.19.self_attn.q_proj",
        "model.layers.19.self_attn.k_proj",
        "model.layers.19.self_attn.v_proj",
        "model.layers.19.self_attn.o_proj",
        "model.layers.2.self_attn.q_proj",
        "model.layers.2.self_attn.k_proj",
        "model.layers.2.self_attn.v_proj",
        "model.layers.2.self_attn.o_proj",
        "model.layers.20.self_attn.q_proj",
        "model.layers.20.self_attn.k_proj",
        "model.layers.20.self_attn.v_proj",
        "model.layers.20.self_attn.o_proj",
        "model.layers.21.self_attn.q_proj",
        "model.layers.21.self_attn.k_proj",
        "model.layers.21.self_attn.v_proj",
        "model.layers.21.self_attn.o_proj",
        "model.layers.22.self_attn.q_proj",
        "model.layers.22.self_attn.k_proj",
        "model.layers.22.self_attn.v_proj",
        "model.layers.22.self_attn.o_proj",
        "model.layers.23.self_attn.q_proj",
        "model.layers.23.self_attn.k_proj",
        "model.layers.23.self_attn.v_proj",
        "model.layers.23.self_attn.o_proj",
        "model.layers.24.self_attn.q_proj",
        "model.layers.24.self_attn.k_proj",
        "model.layers.24.self_attn.v_proj",
        "model.layers.24.self_attn.o_proj",
        "model.layers.25.self_attn.q_proj",
        "model.layers.25.self_attn.k_proj",
        "model.layers.25.self_attn.v_proj",
        "model.layers.25.self_attn.o_proj",
        "model.layers.26.self_attn.q_proj",
        "model.layers.26.self_attn.k_proj",
        "model.layers.26.self_attn.v_proj",
        "model.layers.26.self_attn.o_proj",
        "model.layers.27.self_attn.q_proj",
        "model.layers.27.self_attn.k_proj",
        "model.layers.27.self_attn.v_proj",
        "model.layers.27.self_attn.o_proj",
        "model.layers.28.self_attn.q_proj",
        "model.layers.28.self_attn.k_proj",
        "model.layers.28.self_attn.v_proj",
        "model.layers.28.self_attn.o_proj",
        "model.layers.29.self_attn.q_proj",
        "model.layers.29.self_attn.k_proj",
        "model.layers.29.self_attn.v_proj",
        "model.layers.29.self_attn.o_proj",
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.self_attn.k_proj",
        "model.layers.3.self_attn.v_proj",
        "model.layers.3.self_attn.o_proj",
        "model.layers.30.self_attn.q_proj",
        "model.layers.30.self_attn.k_proj",
        "model.layers.30.self_attn.v_proj",
        "model.layers.30.self_attn.o_proj",
        "model.layers.31.self_attn.q_proj",
        "model.layers.31.self_attn.k_proj",
        "model.layers.31.self_attn.v_proj",
        "model.layers.31.self_attn.o_proj",
        "model.layers.32.self_attn.q_proj",
        "model.layers.32.self_attn.k_proj",
        "model.layers.32.self_attn.v_proj",
        "model.layers.32.self_attn.o_proj",
        "model.layers.33.self_attn.q_proj",
        "model.layers.33.self_attn.k_proj",
        "model.layers.33.self_attn.v_proj",
        "model.layers.33.self_attn.o_proj",
        "model.layers.34.self_attn.q_proj",
        "model.layers.34.self_attn.k_proj",
        "model.layers.34.self_attn.v_proj",
        "model.layers.34.self_attn.o_proj",
        "model.layers.35.self_attn.q_proj",
        "model.layers.35.self_attn.k_proj",
        "model.layers.35.self_attn.v_proj",
        "model.layers.35.self_attn.o_proj",
        "model.layers.4.self_attn.q_proj",
        "model.layers.4.self_attn.k_proj",
        "model.layers.4.self_attn.v_proj",
        "model.layers.4.self_attn.o_proj",
        "model.layers.5.self_attn.q_proj",
        "model.layers.5.self_attn.k_proj",
        "model.layers.5.self_attn.v_proj",
        "model.layers.5.self_attn.o_proj",
        "model.layers.6.self_attn.q_proj",
        "model.layers.6.self_attn.k_proj",
        "model.layers.6.self_attn.v_proj",
        "model.layers.6.self_attn.o_proj",
        "model.layers.7.self_attn.q_proj",
        "model.layers.7.self_attn.k_proj",
        "model.layers.7.self_attn.v_proj",
        "model.layers.7.self_attn.o_proj",
        "model.layers.8.self_attn.q_proj",
        "model.layers.8.self_attn.k_proj",
        "model.layers.8.self_attn.v_proj",
        "model.layers.8.self_attn.o_proj",
        "model.layers.9.self_attn.q_proj",
        "model.layers.9.self_attn.k_proj",
        "model.layers.9.self_attn.v_proj",
        "model.layers.9.self_attn.o_proj",
        "lm_head"
    ],
    "quant_algo": "NVFP4",
    "producer": {
        "name": "modelopt",
        "version": "0.39.0"
    },
    "quant_method": "modelopt"
}


def process_weights_after_loading_modelopt(self, layer: torch.nn.Module) -> None:
    if getattr(layer, "prefix", None) == "model.layers.27.mlp.gate_up_proj" or getattr(layer, "prefix", "").startswith("model.layers.27.self_attn"):
        print(f"##VLLM##: {getattr(layer, 'prefix', None)}: {layer.params_dtype} bias: {getattr(layer, 'bias', None)} {layer.weight.data[0, :4]}, scale: {layer.weight_scale.data[0, :4]}, scale_2: {layer.weight_scale_2.data[0]}")
    from vllm.model_executor.layers.quantization.utils.quant_utils import swizzle_blockscale
    from torch.nn import Parameter
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_bias,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_scales,
        nvfp4_marlin_process_global_scale,
        mxfp4_marlin_process_scales,
    )

    def _create_param_from_subclass_attributes(custom_data, custom_weight):
        param = Parameter(custom_data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_weight_dir = dir(custom_weight)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_weight_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_weight, attr))

        return param

    def prepare_fp4_layer_for_marlin(layer: torch.nn.Module, weight_scale_2_max: torch.Tensor) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression will "
            "be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )

        is_nvfp4 = hasattr(layer, "weight_scale_2")
        group_size = 16 if is_nvfp4 else 32

        part_size_n = layer.output_size_per_partition
        part_size_k = layer.input_size_per_partition
        param_dtype = layer.params_dtype

        assert layer.weight.shape == (part_size_n, part_size_k // 2)

        device = layer.weight.device

        # WORKSPACE
        if getattr(layer, "workspace", None) is None:
            layer.workspace = marlin_make_workspace_new(device)

        # WEIGHT
        # Repack weights to marlin format
        perm = torch.empty(0, dtype=torch.int, device=device)
        qweight = layer.weight.view(torch.int32).T.contiguous()

        marlin_qweight = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=part_size_k,
            size_n=part_size_n,
            num_bits=4,
        )
        layer.marlin_weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

        # WEIGHT SCALES
        # Permute scales
        weight_scale = layer.weight_scale.T.contiguous()

        if not is_nvfp4:
            weight_scale = weight_scale.view(torch.float8_e8m0fnu)

        weight_scale = weight_scale.to(param_dtype)
        weight_scale = marlin_permute_scales(
            s=weight_scale, size_k=part_size_k, size_n=part_size_n, group_size=group_size
        )

        if is_nvfp4:
            weight_scale = nvfp4_marlin_process_scales(weight_scale)
            layer.marlin_weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

            weight_scale_2 = weight_scale_2_max.to(param_dtype)
            weight_scale_2 = nvfp4_marlin_process_global_scale(weight_scale_2)
            layer.marlin_weight_scale_2 = torch.nn.Parameter(weight_scale_2, requires_grad=False)
        else:
            weight_scale = mxfp4_marlin_process_scales(weight_scale)
            layer.marlin_weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        if hasattr(layer, "bias") and layer.bias is not None:
            assert layer.bias.shape == (part_size_n,)
            bias = marlin_permute_bias(layer.bias)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        return

    # global scales:
    input_scale_2 = layer.input_scale.data
    layer.input_scale = _create_param_from_subclass_attributes(input_scale_2, layer.input_scale)
    input_scale_2_max = input_scale_2.max().to(torch.float32)

    weight_scale_2 = layer.weight_scale_2.data
    layer.weight_scale_2 = _create_param_from_subclass_attributes(weight_scale_2, layer.weight_scale_2)
    weight_scale_2_max = weight_scale_2.max().to(torch.float32)

    layer.alpha = Parameter(input_scale_2_max * weight_scale_2_max,
                            requires_grad=False)

    # Calculate `1 / input_scale` so that we don't need to do so at runtime
    layer.input_scale_inv = Parameter(
        (1 / layer.input_scale).to(torch.float32), requires_grad=False)

    # Swizzle the weight blockscale.
    # contracting dimension is input dimension
    # block_size = 16;
    assert (layer.weight_scale.dtype == torch.float8_e4m3fn), (
        "Weight Block scale must be represented as FP8-E4M3")

    if self.backend == "marlin":
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        layer.weight = _create_param_from_subclass_attributes(weight, layer.weight)
        layer.weight_scale = _create_param_from_subclass_attributes(weight_scale, layer.weight_scale)
        prepare_fp4_layer_for_marlin(layer, weight_scale_2_max)

        if getattr(layer, "prefix", None) == "model.layers.27.mlp.gate_up_proj" or getattr(layer, "prefix", "").startswith("model.layers.27.self_attn"):
            print(f"##VLLM-MARLIN##: {getattr(layer, 'prefix', None)}: {layer.marlin_weight.data[0, :4]}, scale: {layer.marlin_weight_scale.data[0, :4]}, scale_2: {layer.marlin_weight_scale_2.data}")

        del layer.alpha
        # del layer.input_scale
    elif self.backend == "flashinfer-trtllm":
        # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
        # FlashInfer provides nvfp4_quantize to quantize + shuffle the
        # layout but we use our own quantization so we have to call
        # shuffles ourselves.
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        epilogue_tile_m = 128
        weight = shuffle_matrix_a(weight.view(torch.uint8),
                                    epilogue_tile_m)
        weight_scale = (shuffle_matrix_sf_a(weight_scale.view(
            torch.uint8), epilogue_tile_m).reshape(
                weight_scale.shape).view(torch.float8_e4m3fn))

        layer.weight_scale = _create_param_from_subclass_attributes(weight_scale, layer.weight_scale)
        layer.weight = _create_param_from_subclass_attributes(weight, layer.weight)
    else:
        swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
        layer.weight_scale = _create_param_from_subclass_attributes(swizzled_weight_scale, layer.weight_scale)
        layer.weight = _create_param_from_subclass_attributes(layer.weight.data, layer.weight)


def process_weights_after_loading_kv(self, layer) -> None:
    """Modified version of BaseKVCacheMethod.process_weights_after_loading.

    Doesn't delete k_scale, v_scale, q_scale, and prob_scale parameters to allow
    for dynamic updates during refit.
    """
    # If the kv-cache dtype is auto, we enforce the k/v_scale to be 1.0
    # regardless whether the kv-scale is available in the checkpoint.
    # No need to process kv scales after loading if we are going to
    # calculate them on the fly.
    from vllm.platforms import current_platform

    if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            # We prefer to use separate k_scale and v_scale if present
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            # If no scales were loaded (both scales are invalid negative
            # values), use the default value of 1.0
            k_scale = 1.0
            v_scale = 1.0
        else:
            # If we find a single kv_scale in the checkpoint, we remap
            # kv_scale to k_scale during weight loading, and duplicate
            # k_scale to v_scale here
            assert layer.k_scale > 0.0
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

        if layer.q_scale < 0.0:
            layer._q_scale.copy_(k_scale)
            layer._q_scale_float = k_scale

        # These are used in the final Attention.forward()
        layer._k_scale.copy_(k_scale)
        layer._v_scale.copy_(v_scale)
        layer._k_scale_float = k_scale
        layer._v_scale_float = v_scale

    if layer.q_scale > 0.0:
        q_scale = layer.q_scale
        if current_platform.is_fp8_fnuz():
            q_scale *= 2
        layer.calculate_kv_scales = False
    else:
        q_scale = 1.0
    if layer.prob_scale > 0.0:
        prob_scale = layer.prob_scale
        if current_platform.is_fp8_fnuz():
            prob_scale *= 2
    else:
        prob_scale = 1.0

    is_singleton_float = (
        lambda x: isinstance(x, float)
        or isinstance(x, torch.Tensor)
        and x.numel() == 1
        and x.is_floating_point()
    )
    if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
        raise ValueError(
            "Only support per-tensor scaling factorfor fp8-quantized Q/prob"
        )

    # These are used in the final Attention.forward()
    layer._q_scale.copy_(q_scale)
    layer._q_scale_float = (
        q_scale.item() if isinstance(q_scale, torch.Tensor) else q_scale
    )

    layer._prob_scale.copy_(prob_scale)

    # IMPORTANT: We DON'T delete the parameters here to allow for dynamic updates
    # Original code deleted: layer.k_scale, layer.v_scale, layer.q_scale, layer.prob_scale


def apply_modelopt(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
    from vllm.utils.flashinfer import (flashinfer_scaled_fp4_mm)
    if self.backend == "marlin":
        # if getattr(layer, "prefix", None) == "model.layers.27.mlp.gate_up_proj" or getattr(layer, "prefix", "").startswith("model.layers.27.self_attn"):
            # print(f"##VLLM-MARLIN##: {getattr(layer, 'prefix', None)}: {layer.marlin_weight.data[0, :4]}, scale: {layer.marlin_weight_scale.data[0, :4]}, scale_2: {layer.marlin_weight_scale_2.data}")
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.marlin_weight,
            weight_scale=layer.marlin_weight_scale,
            weight_scale_2=layer.marlin_weight_scale_2,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias)

    output_dtype = x.dtype
    output_shape = [x.shape[0], layer.weight.shape[0]]

    # quantize BF16 or FP16 to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = scaled_fp4_quant(x, layer.input_scale_inv)

    # validate dtypes of quantized input, input block scale,
    # weight and weight_blockscale
    assert (x_fp4.dtype == torch.uint8)
    assert (layer.weight.dtype == torch.uint8)
    assert (x_blockscale.dtype == torch.float8_e4m3fn)
    assert (layer.weight_scale.dtype == torch.float8_e4m3fn)
    assert (layer.alpha.dtype == torch.float32)

    mm_args = (
        x_fp4,
        layer.weight,
        x_blockscale,
        layer.weight_scale,
        layer.alpha,
        output_dtype,
    )
    if self.backend == "flashinfer-trtllm":
        out = flashinfer_scaled_fp4_mm(*mm_args, backend="trtllm")
    elif self.backend == "flashinfer-cutlass":
        out = flashinfer_scaled_fp4_mm(*mm_args, backend="cutlass")
    else:
        out = cutlass_scaled_fp4_mm(*mm_args)

    if bias is not None:
        out = out + bias
    return out.view(*output_shape)

def apply_vllm_modelopt_patches():
    func1_path = "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4LinearMethod.process_weights_after_loading"
    patcher1 = patch(func1_path, process_weights_after_loading_modelopt)
    patcher1.start()
    func2_path = "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4LinearMethod.apply"
    patcher2 = patch(func2_path, apply_modelopt)
    patcher2.start()
    # Static scales mode: patch process_weights_after_loading to preserve k_scale/v_scale for manual updates
    func5_path = "vllm.model_executor.layers.quantization.kv_cache.BaseKVCacheMethod.process_weights_after_loading"
    patcher5 = patch(func5_path, process_weights_after_loading_kv)
    patcher5.start()
