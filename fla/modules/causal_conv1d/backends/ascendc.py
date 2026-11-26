# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from torch.autograd.function import once_differentiable

from fla.backends import BaseBackend, register
from fla.modules.causal_conv1d.ops import causal_conv1d
from fla.utils import input_guard


class _AscendCCausalConv1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, activation):
        from fla_npu.ops.ascendc import causal_conv1d_fn

        ctx.save_for_backward(x, weight, bias)
        ctx.activation = activation
        return causal_conv1d_fn(x, weight, bias, activation=activation)

    @staticmethod
    @once_differentiable
    @input_guard
    def backward(ctx, dy):
        from fla_npu.ops.ascendc import causal_conv1d_bwd, causal_conv1d_fn

        x, weight, bias = ctx.saved_tensors
        # The fused forward returns the activated output. Backward needs the preactivation.
        preactivation = causal_conv1d_fn(x, weight, bias, activation=None) if ctx.activation else None
        dx, dw, db, _ = causal_conv1d_bwd(x, preactivation, weight, dy, activation=int(ctx.activation is not None))
        return dx, dw, db if bias is not None else None, None


@register(causal_conv1d, env_var='FLA_CONV1D_ASCENDC')
class AscendCCausalConv1dBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    def causal_conv1d_verifier(
        self, x, weight=None, bias=None, residual=None, initial_state=None, output_final_state=False,
        activation=None, backend='triton', cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None,
        cp_context=None, **kwargs,
    ):
        if (
            x.device.type != 'npu' or x.ndim != 3 or min(x.shape) <= 0 or x.shape[-1] % 16
            or x.dtype not in (torch.bfloat16, torch.float16) or backend != 'triton' or kwargs
            or output_final_state or activation not in (None, 'silu', 'swish')
            or any(t is not None for t in (residual, initial_state, cu_seqlens, cu_seqlens_cpu, chunk_indices, cp_context))
            or weight is None or weight.ndim != 2 or weight.shape[0] != x.shape[-1] or weight.shape[1] not in (2, 3, 4)
            or weight.device != x.device or weight.dtype != x.dtype
            or (bias is not None and (bias.shape != (x.shape[-1],) or bias.device != x.device or bias.dtype != x.dtype))
            or torch.npu.is_current_stream_capturing()
        ):
            return False, 'AscendC causal convolution requires dense FP16/BF16 input without state or residual'
        from fla_npu.ops import ascendc

        if not all(hasattr(ascendc, op) for op in ('causal_conv1d_fn', 'causal_conv1d_bwd')):
            return False, 'installed fla_npu wheel lacks the causal convolution forward or backward entry'
        return True, None

    def causal_conv1d(
        self, x, weight=None, bias=None, residual=None, initial_state=None, output_final_state=False,
        activation=None, backend='triton', cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None,
        cp_context=None, **kwargs,
    ):
        y = _AscendCCausalConv1dFunction.apply(x.contiguous(), weight.T.contiguous(), bias, activation)
        return y, None
