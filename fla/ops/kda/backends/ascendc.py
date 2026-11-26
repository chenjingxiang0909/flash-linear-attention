# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Paired KDA forward and backward using the optional fla_npu wheel."""

from math import isfinite

import torch

from fla.backends import BaseBackend, register
from fla.modules.norm.l2norm import l2norm
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.kda.chunk import chunk_kda
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class _AscendCChunkKDAFunction(torch.autograd.Function):

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, beta, a_log, dt_bias, scale, lower_bound, disable_recompute, offsets):
        from fla_npu.ops.ascendc import chunk_kda_fwd

        out, _, gk, Aqk, Akk, w, _, qg, kg, v_new, h, _ = chunk_kda_fwd(
            q, k, v, g, beta, scale, layout='BSND', cu_seqlens=offsets,
            safe_gate=True, lower_bound=lower_bound, use_gate_in_kernel=True,
            A_log=a_log, dt_bias=dt_bias, disable_recompute=disable_recompute,
        )
        ctx.save_for_backward(q, k, v, g, beta, a_log, dt_bias, gk, Aqk, Akk, w, qg, kg, v_new, h)
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.disable_recompute = disable_recompute
        ctx.offsets = offsets
        return out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        from fla_npu.ops.ascendc import chunk_kda_bwd

        q, k, v, g, beta, a_log, dt_bias, gk, Aqk, Akk, w, qg, kg, v_new, h = ctx.saved_tensors
        dq, dk, dv, db, dg, _, da_log, dbias = chunk_kda_bwd(
            q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, do, ctx.scale,
            layout='BSND', raw_g=g, A_log=a_log, dt_bias=dt_bias, lower_bound=ctx.lower_bound,
            cu_seqlens=ctx.offsets, disable_recompute=ctx.disable_recompute,
        )
        return dq, dk, dv, dg, db, da_log, dbias, None, None, None, None


@register(chunk_kda, env_var='FLA_KDA_ASCENDC')
@register(kda_gate_chunk_cumsum, env_var='FLA_KDA_ASCENDC')
class AscendCKDABackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    def kda_gate_chunk_cumsum_verifier(
        self, g, A_log, chunk_size, scale=None, dt_bias=None, cu_seqlens=None,
        output_dtype=torch.float, chunk_indices=None, lower_bound=None, use_graph=False, **kwargs,
    ):
        if (
            use_graph or kwargs or g.device.type != 'npu' or g.ndim != 4 or min(g.shape) <= 0
            or torch.npu.is_current_stream_capturing()
            or g.shape[-1] > 256 or chunk_size not in (32, 64, 128)
            or g.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or (output_dtype or g.dtype) not in (torch.float32, torch.bfloat16, torch.float16)
            or (lower_bound is not None and not -5 <= lower_bound < 0)
            or (scale is not None and not isfinite(scale))
        ):
            return False, 'AscendC KDA gate cumsum requires eager execution, K<=256, chunk=32/64/128 and supported gate options'
        heads, width = g.shape[2:]
        for tensor, size in ((A_log, heads), (dt_bias, heads * width)):
            if tensor is not None and (
                tensor.numel() != size or tensor.dtype != torch.float32 or tensor.device != g.device
            ):
                return False, 'AscendC KDA gate cumsum requires FP32 A_log and dt_bias on the input NPU'
        if cu_seqlens is None:
            if chunk_indices is not None:
                return False, 'AscendC KDA gate cumsum requires packed offsets for chunk indices'
        else:
            if (
                g.shape[0] != 1 or cu_seqlens.ndim != 1 or cu_seqlens.device != g.device
                or cu_seqlens.dtype not in (torch.int32, torch.int64)
            ):
                return False, 'AscendC KDA gate cumsum requires packed B=1 and one-dimensional integer offsets'
            offsets = cu_seqlens.tolist()
            if (
                len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != g.shape[1]
                or any(b <= a for a, b in zip(offsets, offsets[1:]))
            ):
                return False, 'AscendC KDA gate cumsum requires nonempty sequences covering the input'
            if chunk_indices is not None and not torch.equal(
                chunk_indices, prepare_chunk_indices(cu_seqlens, chunk_size),
            ):
                return False, 'AscendC KDA gate cumsum requires canonical chunk indices'
        from fla_npu.ops import ascendc

        if not hasattr(ascendc, 'kda_gate_cumsum'):
            return False, 'installed fla_npu wheel has no KDA gate cumsum entry point'
        return True, None

    @input_guard
    def kda_gate_chunk_cumsum(
        self, g, A_log, chunk_size, scale=None, dt_bias=None, cu_seqlens=None,
        output_dtype=torch.float, chunk_indices=None, lower_bound=None, use_graph=False, **kwargs,
    ):
        from fla_npu.ops.ascendc import kda_gate_cumsum

        a_log = g.new_zeros(g.shape[2], dtype=torch.float32) if A_log is None else A_log.flatten()
        result = kda_gate_cumsum(
            g.transpose(1, 2).contiguous(), chunk_size, A_log=a_log,
            dt_bias=dt_bias.flatten() if dt_bias is not None else None,
            cu_seqlens=cu_seqlens.tolist() if cu_seqlens is not None else None,
            use_gate_in_kernel=True, safe_gate=lower_bound is not None, lower_bound=lower_bound,
        )
        # The wheel converts to log2 before accumulating. FLA applies an optional scale.
        if scale != RCP_LN2:
            result = result * ((1.0 if scale is None else scale) / RCP_LN2)
        return result.transpose(1, 2).to(output_dtype or g.dtype).contiguous()

    def chunk_kda_verifier(
        self, q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False, safe_gate=False, lower_bound=None, disable_recompute=False,
        return_intermediate_states=False, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
        cp_context=None, use_graph=False, max_num_seqs=None, **kwargs,
    ) -> tuple[bool, str | None]:
        if cp_context is not None or use_graph or max_num_seqs is not None or return_intermediate_states:
            return False, 'AscendC KDA requires eager execution without CP or intermediate state outputs'
        if kwargs.keys() - {'chunk_size', 'A_log', 'dt_bias'} or kwargs.get('chunk_size', 64) != 64:
            return False, 'AscendC KDA requires chunk_size=64 and supported options'
        if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
            return False, 'allow_neg_eigval requires beta sigmoid'
        if q.device.type != 'npu':
            return False, 'AscendC KDA requires an NPU'
        if q.ndim != 4 or min(q.shape) <= 0 or q.shape[-1] != 128 or k.shape != q.shape or v.shape != q.shape:
            return False, 'AscendC KDA requires nonempty matching [B, T, H, 128] Q/K/V'
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            return False, 'AscendC KDA requires BF16 Q/K/V'
        if g.shape != q.shape or beta.shape != q.shape[:-1]:
            return False, 'AscendC KDA requires matching gate and beta shapes'
        if any(t.dtype not in (torch.bfloat16, torch.float32) for t in (g, beta)):
            return False, 'AscendC KDA requires BF16 or FP32 gate and beta'
        if any(t.device != q.device for t in (k, v, g, beta)):
            return False, 'AscendC KDA requires all inputs on the same NPU'
        if scale is not None and (not isinstance(scale, (int, float)) or not isfinite(scale)):
            return False, 'AscendC KDA requires a finite scalar scale'

        batch, _, heads, width = q.shape
        sequences = batch
        if cu_seqlens is not None:
            if batch != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
                return False, 'AscendC KDA requires packed inputs with B=1 and cu_seqlens [N+1]'
            if cu_seqlens.dtype not in (torch.int32, torch.int64) or cu_seqlens.device not in (q.device, torch.device('cpu')):
                return False, 'AscendC KDA requires integer cu_seqlens on CPU or the input device'
            sequences = cu_seqlens.numel() - 1
        if cu_seqlens_cpu is not None and (
            cu_seqlens is None or cu_seqlens_cpu.shape != cu_seqlens.shape
            or cu_seqlens_cpu.device.type != 'cpu' or cu_seqlens_cpu.dtype not in (torch.int32, torch.int64)
        ):
            return False, 'AscendC KDA requires a CPU integer mirror of cu_seqlens'
        if initial_state is not None and (
            initial_state.shape != (sequences, heads, width, width)
            or initial_state.dtype != torch.float32 or initial_state.device != q.device
        ):
            return False, 'AscendC KDA requires FP32 initial_state [N, H, 128, 128]'

        a_log = kwargs.get('A_log') if use_gate_in_kernel else None
        dt_bias = kwargs.get('dt_bias') if use_gate_in_kernel else None
        if use_gate_in_kernel and (not safe_gate or lower_bound is None or not -5 <= lower_bound < 0):
            return False, 'AscendC KDA raw gates require safe_gate and -5 <= lower_bound < 0'
        if not use_gate_in_kernel and lower_bound is not None:
            return False, 'AscendC KDA requires a raw gate when lower_bound is supplied'
        for name, tensor, shape in (('A_log', a_log, (heads,)), ('dt_bias', dt_bias, (heads * width,))):
            if tensor is not None and (tensor.shape != shape or tensor.dtype != torch.float32 or tensor.device != q.device):
                return False, f'AscendC KDA requires FP32 {name} of shape {shape} on the input device'

        needs_grad = torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in (q, k, v, g, beta, a_log, dt_bias, initial_state)
        )
        if needs_grad:
            if not use_gate_in_kernel or initial_state is not None or output_final_state:
                return False, 'AscendC KDA backward requires raw safe gates and no initial or final state'
            if cu_seqlens is not None and a_log is not None and a_log.requires_grad:
                return False, 'AscendC packed KDA backward requires frozen A_log'
            if not disable_recompute and (heads % 8 != 0 or heads > 256):
                return False, 'AscendC KDA recompute requires H divisible by 8 and at most 256'
        return True, None

    @input_guard
    def chunk_kda(
        self, q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False, safe_gate=False, lower_bound=None, disable_recompute=False,
        return_intermediate_states=False, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
        cp_context=None, use_graph=False, max_num_seqs=None, **kwargs,
    ):
        from fla_npu.ops.ascendc import chunk_kda_fwd

        if use_qk_l2norm_in_kernel:
            q, k = l2norm(q), l2norm(k)
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta, scale=2.0 if allow_neg_eigval else 1.0)

        a_log = kwargs.get('A_log') if use_gate_in_kernel else None
        dt_bias = kwargs.get('dt_bias') if use_gate_in_kernel else None
        if use_gate_in_kernel and a_log is None:
            a_log = q.new_zeros(q.shape[2], dtype=torch.float32)
        offsets = None
        if cu_seqlens is not None:
            host_offsets = cu_seqlens.cpu() if cu_seqlens_cpu is None else cu_seqlens_cpu
            offsets = host_offsets.tolist()
        if scale is None:
            scale = k.shape[-1] ** -0.5

        if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, g, beta, a_log, dt_bias)):
            out = _AscendCChunkKDAFunction.apply(q, k, v, g, beta, a_log, dt_bias, scale,
                                                 lower_bound, disable_recompute, offsets)
            return out, None

        # The wheel may update the supplied state, which FLA treats as an input.
        state = initial_state.clone() if initial_state is not None else None
        out, final_state, *_ = chunk_kda_fwd(
            q, k, v, g, beta, scale, layout='BSND', initial_state=state,
            output_final_state=output_final_state, cu_seqlens=offsets, safe_gate=safe_gate,
            lower_bound=lower_bound, use_gate_in_kernel=use_gate_in_kernel, A_log=a_log,
            dt_bias=dt_bias, disable_recompute=disable_recompute, state_v_first=state_v_first,
        )
        return out, final_state
