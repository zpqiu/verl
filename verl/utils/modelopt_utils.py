import logging
from typing import Optional
from unittest.mock import patch

import torch

logger = logging.getLogger(__name__)

def process_weights_after_loading_modelopt(self, layer: torch.nn.Module) -> None:
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

    def prepare_fp4_layer_for_marlin(layer: torch.nn.Module) -> None:
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

            weight_scale_2 = layer.weight_scale_2.to(param_dtype)
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
    input_scale_2 = layer.input_scale.max().to(torch.float32)
    layer.input_scale = _create_param_from_subclass_attributes(input_scale_2, layer.input_scale)

    weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)
    layer.weight_scale_2 = _create_param_from_subclass_attributes(weight_scale_2, layer.weight_scale_2)

    layer.alpha = Parameter(layer.input_scale * layer.weight_scale_2,
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
        prepare_fp4_layer_for_marlin(layer)
        del layer.alpha
        del layer.input_scale
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
