# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from math import isfinite

import torch

from fla.backends import BaseBackend, register
from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
from fla.utils import input_guard

_INTS = (torch.int32, torch.int64)
_SCALAR_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _sequence_lengths(q, cu_seqlens):
    batch, tokens = q.shape[:2]
    if cu_seqlens is None:
        return [tokens] * batch
    if batch != 1 or cu_seqlens.ndim != 1 or cu_seqlens.dtype not in _INTS or cu_seqlens.device != q.device:
        return None
    offsets = cu_seqlens.tolist()
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != tokens:
        return None
    lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    return lengths if min(lengths) >= 0 else None


def _valid_indexed_state(q, state, lengths, indices, accepted, unique_tokens):
    if indices.device != q.device or indices.dtype not in _INTS:
        return False
    if indices.ndim == 1 and max(lengths) == 1 and min(lengths) == 1 and indices.numel() >= len(lengths):
        slots = [[i] for i in indices[:len(lengths)].tolist()]
    elif indices.ndim == 2 and indices.shape[0] == len(lengths) and indices.shape[1] >= max(lengths):
        slots = [row[:length] for row, length in zip(indices.tolist(), lengths)]
    else:
        return False
    if accepted is not None and (
        accepted.dtype not in _INTS or accepted.device != q.device or accepted.shape != (len(lengths),)
        or any(length and not 1 <= count <= length for count, length in zip(accepted.tolist(), lengths))
    ):
        return False
    used = set()
    for row in slots:
        if any(not 0 <= i < state.shape[0] or i in used for i in row):
            return False
        if unique_tokens and len(set(row)) != len(row):
            return False
        used.update(row)
    return True


@register(fused_recurrent_kda_fwd, env_var='FLA_KDA_ASCENDC')
class AscendCRecurrentKDABackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    def fused_recurrent_kda_fwd_verifier(
        self, q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, scale=None,
        output_final_state=False, inplace_final_state=True, state_v_first=False,
        cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, lower_bound=None,
        out=None, **kwargs,
    ):
        if kwargs or q.device.type != 'npu':
            return False, 'AscendC recurrent KDA requires an NPU and supported options'
        from fla_npu.ops import ascendc

        if not hasattr(ascendc, 'recurrent_kda'):
            return False, 'installed fla_npu wheel has no recurrent KDA entry point'
        if (
            q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or min(q.shape) <= 0
            or q.shape[:2] != v.shape[:2] or q.shape[-1] != 128 or v.shape[-1] not in (128, 256)
            or v.shape[2] == 0 or v.shape[2] % q.shape[2]
            or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
        ):
            return False, 'AscendC recurrent KDA requires BF16 Q/K/V with K=128 and V=128/256'
        tensors = (k, v, g, beta, A_log, dt_bias, initial_state, out)
        if any(x is not None and x.device != q.device for x in tensors):
            return False, 'AscendC recurrent KDA requires tensors on the same NPU'
        if torch.is_grad_enabled() and any(x is not None and x.requires_grad for x in (q, *tensors)):
            return False, 'AscendC recurrent KDA is inference-only'
        if g.shape != (*v.shape[:-1], q.shape[-1]) or beta.shape != v.shape[:-1]:
            return False, 'AscendC recurrent KDA requires key-wise gate and scalar beta'
        if g.dtype not in _SCALAR_DTYPES or beta.dtype not in _SCALAR_DTYPES:
            return False, 'AscendC recurrent KDA requires floating-point gate and beta'
        if scale is not None and (not isinstance(scale, (float, int)) or not isfinite(scale)):
            return False, 'AscendC recurrent KDA requires a finite scalar scale'
        if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
            return False, 'allow_neg_eigval requires beta sigmoid'
        heads = v.shape[2]
        if use_gate_in_kernel:
            if A_log is not None and (A_log.shape != (heads,) or A_log.dtype != torch.float32):
                return False, 'AscendC recurrent KDA requires FP32 A_log[HV]'
            if dt_bias is not None and (
                dt_bias.shape not in ((heads * 128,), (heads, 128)) or dt_bias.dtype != torch.float32
            ):
                return False, 'AscendC recurrent KDA requires FP32 key-wise bias'
            if lower_bound is not None and not -5 <= lower_bound < 0:
                return False, 'AscendC recurrent KDA requires -5 <= lower_bound < 0'
        if out is not None and (out.shape != v.shape or out.dtype != v.dtype or not out.is_contiguous()):
            return False, 'AscendC recurrent KDA requires a contiguous output buffer matching V'
        lengths = _sequence_lengths(q, cu_seqlens)
        if lengths is None:
            return False, 'AscendC recurrent KDA requires valid packed offsets covering all tokens'
        tail = (v.shape[-1], 128) if state_v_first else (128, v.shape[-1])
        if initial_state is not None and (
            initial_state.ndim != 4 or initial_state.shape[1:] != (heads, *tail)
            or initial_state.shape[0] == 0 or initial_state.dtype not in (torch.bfloat16, torch.float32)
            or not initial_state.is_contiguous() or (inplace_final_state and initial_state.requires_grad)
        ):
            return False, 'AscendC recurrent KDA requires a contiguous BF16 or FP32 state pool'
        if (inplace_final_state or ssm_state_indices is not None) and initial_state is None:
            return False, 'AscendC recurrent KDA requires a state pool for indexed or in-place calls'
        if ssm_state_indices is None:
            if num_accepted_tokens is not None or (initial_state is not None and initial_state.shape[0] != len(lengths)):
                return False, 'AscendC recurrent KDA requires one initial state per sequence without slot indices'
        elif not _valid_indexed_state(
            q, initial_state, lengths, ssm_state_indices, num_accepted_tokens,
            output_final_state and not inplace_final_state,
        ):
            return False, 'AscendC recurrent KDA requires valid non-overlapping indexed state slots'
        return True, None

    @input_guard(no_guard_contiguous=('initial_state', 'out'))
    def fused_recurrent_kda_fwd(
        self, q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, scale=None,
        output_final_state=False, inplace_final_state=True, state_v_first=False,
        cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, lower_bound=None,
        out=None, **kwargs,
    ):
        from fla_npu.ops.ascendc import recurrent_kda

        state = initial_state
        if state is None:
            count = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
            tail = (v.shape[-1], 128) if state_v_first else (128, v.shape[-1])
            state = q.new_zeros((count, v.shape[2], *tail), dtype=torch.float32)
        elif output_final_state and not inplace_final_state:
            state = state.float()
        if use_gate_in_kernel and A_log is None:
            A_log = q.new_zeros(v.shape[2], dtype=torch.float32)

        result, final = recurrent_kda(
            q, k, v, g, beta, state,
            scale=scale, cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            A_log=A_log if use_gate_in_kernel else None, dt_bias=dt_bias if use_gate_in_kernel else None,
            output_final_state=output_final_state, inplace_final_state=inplace_final_state,
            state_v_first=state_v_first, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel, use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval, safe_gate=use_gate_in_kernel and lower_bound is not None,
            lower_bound=lower_bound if use_gate_in_kernel else None,
        )
        if out is not None:
            result = out.copy_(result)
        if inplace_final_state:
            final = initial_state
        elif output_final_state and ssm_state_indices is not None:
            # The wheel writes cache slots. FLA returns one checkpoint per token.
            if ssm_state_indices.ndim == 1:
                slots = ssm_state_indices[:q.shape[0] * q.shape[1]]
            else:
                lengths = _sequence_lengths(q, cu_seqlens)
                slots = torch.cat([row[:length] for row, length in zip(ssm_state_indices, lengths)])
            final = final.index_select(0, slots.long())
        elif output_final_state and cu_seqlens is not None:
            empty = cu_seqlens[1:] == cu_seqlens[:-1]
            final = torch.where(empty[:, None, None, None], state, final)
        return result, final
