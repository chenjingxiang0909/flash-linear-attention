# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from math import isfinite, log

import torch

from fla.backends import BaseBackend, register
from fla.ops.common.backends.intracard import IntraCardCPBackend
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.utils import input_guard

_LN2 = log(2.0)
_LOW_PRECISION = (torch.bfloat16, torch.float16)


def _chunk_metadata(key, cu_seqlens, chunk_indices, chunk_size, cu_seqlens_cpu=None):
    if cu_seqlens is None:
        if chunk_indices is not None:
            raise ValueError('chunk indices require sequence offsets')
        return None, None, key.shape[0], (key.shape[1] + chunk_size - 1) // chunk_size
    if (
        key.shape[0] != 1 or cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or cu_seqlens.device != key.device
    ):
        raise ValueError('packed AscendC GDN requires B=1 and one-dimensional integer offsets')
    offsets = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != key.shape[1]:
        raise ValueError('sequence offsets must cover all tokens')
    lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    if min(lengths) <= 0:
        raise ValueError('AscendC GDN standalone ops require nonempty sequences')
    pairs = [(i, j) for i, length in enumerate(lengths) for j in range((length + chunk_size - 1) // chunk_size)]
    if chunk_indices is not None and (
        chunk_indices.device != key.device or chunk_indices.dtype not in (torch.int32, torch.int64)
        or chunk_indices.tolist() != [list(x) for x in pairs]
    ):
        raise ValueError('AscendC GDN standalone ops require canonical chunk indices')
    return offsets, [index for pair in pairs for index in pair], len(lengths), len(pairs)


def _tensor(tensor, shape, device, dtypes):
    return tensor is not None and tensor.shape == shape and tensor.device == device and tensor.dtype in dtypes


def _contract(
    name, key, value, g, chunk_size, cu_seqlens, chunk_indices, *,
    v_sizes=(128,), chunk_sizes=(64,), cu_seqlens_cpu=None,
):
    if (
        key.device.type != 'npu'
        or key.ndim != 4 or value.ndim != 4 or min(key.shape) <= 0 or value.shape[2] == 0
        or key.shape[:2] != value.shape[:2] or key.shape[-1] != 128 or value.shape[-1] not in v_sizes
        or value.shape[2] % key.shape[2] or value.shape[2] // key.shape[2] > 16
        or key.dtype not in _LOW_PRECISION or value.dtype != key.dtype or value.device != key.device
        or chunk_size not in chunk_sizes
        or not _tensor(g, value.shape[:-1], key.device, (torch.float32, key.dtype))
        or torch.npu.is_current_stream_capturing()
    ):
        return None
    from fla_npu.ops import ascendc

    if not hasattr(ascendc, name):
        return None
    return _chunk_metadata(key, cu_seqlens, chunk_indices, chunk_size, cu_seqlens_cpu)


def _head_first(tensor):
    return tensor.transpose(1, 2).contiguous()


def _natural_gate(g):
    # FLA stores the cumulative gate in log2 space; these wheel entries use exp.
    return _head_first(g.float() * _LN2)


def _nonincreasing(g, offsets, chunk_size):
    # Gate resets at sequence and chunk boundaries are outside the native scan.
    offsets = offsets or [0, g.shape[1]]
    increases = g[:, 1:] > g[:, :-1]
    for start, end in zip(offsets, offsets[1:]):
        if start:
            increases[:, start - 1] = False
        increases[:, start + chunk_size - 1:end - 1:chunk_size] = False
    return not bool(increases.any())


def _supports_wy(name, k, v, beta, A, g, cu_seqlens, chunk_indices):
    return (
        _contract(name, k, v, g, 64, cu_seqlens, chunk_indices)
        and _tensor(A, (*v.shape[:-1], 64), k.device, (k.dtype,))
        and g.dtype == torch.float32
        and _tensor(beta, v.shape[:-1], k.device, (torch.float32,))
    )


@register(chunk_fwd_o, env_var='FLA_GDN_ASCENDC')
@register(chunk_bwd_dv_local, env_var='FLA_GDN_ASCENDC')
@register(chunk_bwd_dqkwg, env_var='FLA_GDN_ASCENDC')
@register(chunk_local_cumsum, env_var='FLA_GDN_ASCENDC')
@register(chunk_gated_delta_rule_fwd_h, env_var='FLA_GDN_ASCENDC')
@register(chunk_gated_delta_rule_bwd_dhu, env_var='FLA_GDN_ASCENDC')
@register(recompute_w_u_fwd, env_var='FLA_GDN_ASCENDC')
@register(prepare_wy_repr_bwd, env_var='FLA_GDN_ASCENDC')
class AscendCGDNSubopsBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    def chunk_local_cumsum_verifier(
        self, g, chunk_size, reverse=False, scale=None, cu_seqlens=None, output_dtype=torch.float,
        chunk_indices=None, use_graph=False, **kwargs,
    ):
        if (
            g.device.type != 'npu' or g.ndim != 3 or min(g.shape) <= 0
            or g.dtype not in (torch.float32, *_LOW_PRECISION) or chunk_size not in (16, 32, 64)
            or (output_dtype or g.dtype) not in (torch.float32, *_LOW_PRECISION)
            or use_graph or kwargs or (scale is not None and not isfinite(scale))
            or torch.npu.is_current_stream_capturing()
        ):
            return False, 'AscendC GDN cumsum requires an eager scalar gate and chunk=16/32/64'
        from fla_npu.ops import ascendc

        if not hasattr(ascendc, 'chunk_local_cumsum'):
            return False, 'installed fla_npu wheel has no scalar cumsum entry point'
        _chunk_metadata(g, cu_seqlens, chunk_indices, chunk_size)
        return True, None

    @input_guard
    def chunk_local_cumsum(
        self, g, chunk_size, reverse=False, scale=None, cu_seqlens=None, output_dtype=torch.float,
        chunk_indices=None, use_graph=False, **kwargs,
    ):
        from fla_npu.ops.ascendc import chunk_local_cumsum

        offsets, indices, _, _ = _chunk_metadata(g, cu_seqlens, chunk_indices, chunk_size)
        dtype = output_dtype or g.dtype
        result = chunk_local_cumsum(
            _head_first(g), chunk_size, cu_seqlens=offsets, chunk_indices_out=indices,
            reverse=reverse, scale=1.0 if scale is None else scale, head_first=True,
            output_dtype='bfloat16' if dtype == torch.bfloat16 else 'float32',
        )
        return _head_first(result).to(dtype)

    def chunk_bwd_dv_local_verifier(
        self, q, k, do, g=None, g_gamma=None, A=None, scale=None, cu_seqlens=None,
        chunk_size=64, chunk_indices=None,
    ):
        meta = _contract(
            'chunk_bwd_dv_local', k, do, g, chunk_size, cu_seqlens, chunk_indices,
            v_sizes=(128, 256), chunk_sizes=(64, 128),
        )
        if (
            not meta or g_gamma is not None or A is not None
            or not _tensor(q, k.shape, k.device, (k.dtype,))
            or not isinstance(scale, (int, float)) or not isfinite(scale)
        ):
            return False, 'AscendC DvLocal requires K=128, V=128/256, chunk=64/128, a scalar gate and scale, without A or g_gamma'
        # The wheel clamps positive gate differences before exponentiation.
        if not _nonincreasing(g, meta[0], chunk_size):
            return False, 'AscendC DvLocal requires a nonincreasing gate within each chunk'
        return True, None

    @input_guard
    def chunk_bwd_dv_local(
        self, q, k, do, g=None, g_gamma=None, A=None, scale=None, cu_seqlens=None,
        chunk_size=64, chunk_indices=None,
    ):
        from fla_npu.ops.ascendc import chunk_bwd_dv_local

        offsets, indices, _, _ = _chunk_metadata(k, cu_seqlens, chunk_indices, chunk_size)
        dv = chunk_bwd_dv_local(
            _head_first(q), _head_first(k), _head_first(do), _natural_gate(g), scale, chunk_size,
            cu_seqlens=offsets, chunk_indices=indices,
        )
        return _head_first(dv)

    def chunk_bwd_dqkwg_verifier(
        self, q, k, v, do, h, dh, w=None, g=None, g_gamma=None, dv=None, scale=None, state_v_first=False,
        cu_seqlens=None, chunk_size=64, chunk_indices=None,
    ):
        meta = _contract('chunk_bwd_dqkwg', k, v, g, chunk_size, cu_seqlens, chunk_indices)
        if (
            not meta or w is not None or g_gamma is not None or state_v_first or k.shape[2] != v.shape[2]
            or g.dtype != torch.float32 or not _tensor(q, k.shape, k.device, (k.dtype,))
            or not all(_tensor(t, v.shape, k.device, (k.dtype,)) for t in (do, dv))
            or not all(_tensor(t, (k.shape[0], meta[3], v.shape[2], 128, 128), k.device, (k.dtype,)) for t in (h, dh))
            or not isinstance(scale, (int, float)) or not isfinite(scale)
        ):
            return False, 'AscendC Dqkwg requires equal heads, K=V=128, K-first states, dv and scalar gate, without w'
        if bool((g > 0).any()) or not _nonincreasing(g, meta[0], chunk_size):
            return False, 'AscendC Dqkwg requires a nonpositive and nonincreasing gate within each chunk'
        return True, None

    @input_guard
    def chunk_bwd_dqkwg(
        self, q, k, v, do, h, dh, w=None, g=None, g_gamma=None, dv=None, scale=None, state_v_first=False,
        cu_seqlens=None, chunk_size=64, chunk_indices=None,
    ):
        from fla_npu.ops.ascendc import chunk_bwd_dqkwg

        offsets, indices, _, _ = _chunk_metadata(k, cu_seqlens, chunk_indices, chunk_size)
        dq, dk, _, dg = chunk_bwd_dqkwg(
            _head_first(q), _head_first(k), _head_first(v), _natural_gate(g), h, _head_first(do), dh,
            _head_first(dv), chunk_size, cu_seqlens=offsets, chunk_indices=indices, scale=scale,
        )
        return _head_first(dq), _head_first(dk), None, _head_first(dg)

    def chunk_fwd_o_verifier(
        self, q, k, v, h, g=None, g_gamma=None, scale=None, state_v_first=False,
        cu_seqlens=None, chunk_size=64, chunk_indices=None,
    ):
        meta = _contract('chunk_fwd_o', k, v, g, chunk_size, cu_seqlens, chunk_indices)
        if (
            not meta or g_gamma is not None or k.dtype != torch.bfloat16
            or not _tensor(q, k.shape, k.device, (k.dtype,))
            or not _tensor(h, (k.shape[0], meta[3], v.shape[2], 128, 128), k.device, (k.dtype,))
            or (scale is not None and not isfinite(scale))
        ):
            return False, 'AscendC FwdO requires BF16 K=V=128, chunk=64 and scalar gate'
        return True, None

    @input_guard
    def chunk_fwd_o(
        self, q, k, v, h, g=None, g_gamma=None, scale=None, state_v_first=False,
        cu_seqlens=None, chunk_size=64, chunk_indices=None,
    ):
        from fla_npu.ops.ascendc import chunk_fwd_o

        offsets, indices, _, _ = _chunk_metadata(k, cu_seqlens, chunk_indices, chunk_size)
        return chunk_fwd_o(
            _head_first(q), _head_first(k), _head_first(v), h,
            128 ** -0.5 if scale is None else scale, g=_head_first(g),
            cu_seqlens=offsets, chunk_indices=indices, chunk_size=chunk_size,
            transpose_state_layout=state_v_first, use_exp2=True, output_layout='BSND',
        )

    def chunk_gated_delta_rule_fwd_h_verifier(
        self, k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=64,
        save_new_value=True, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
        chunk_indices=None, chunk_offsets=None,
    ):
        if cu_seqlens is not None and torch.is_inference_mode_enabled() and IntraCardCPBackend().is_enabled():
            return False, 'IntraCardCP owns the packed inference state scan when enabled'
        if cu_seqlens_cpu is not None and (
            cu_seqlens is None or cu_seqlens_cpu.device.type != 'cpu'
            or cu_seqlens_cpu.shape != cu_seqlens.shape or cu_seqlens_cpu.dtype not in (torch.int32, torch.int64)
        ):
            return False, 'AscendC FwdH requires a matching CPU offsets mirror'
        meta = _contract(
            'chunk_fwd_h', k, u, g, chunk_size, cu_seqlens, chunk_indices, cu_seqlens_cpu=cu_seqlens_cpu,
        )
        if (
            not meta or gk is not None or not save_new_value or chunk_offsets is not None
            or k.dtype != torch.bfloat16 or not _tensor(w, u.shape, k.device, (k.dtype,))
            or (initial_state is not None and not _tensor(
                initial_state, (meta[2], u.shape[2], 128, 128), k.device, (torch.float32,),
            ))
        ):
            return False, 'AscendC FwdH requires BF16 K=V=128, scalar gate and optional FP32 initial state'
        return True, None

    @input_guard
    def chunk_gated_delta_rule_fwd_h(
        self, k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=64,
        save_new_value=True, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
        chunk_indices=None, chunk_offsets=None,
    ):
        from fla_npu.ops.ascendc import chunk_fwd_h

        offsets, indices, _, _ = _chunk_metadata(k, cu_seqlens, chunk_indices, chunk_size, cu_seqlens_cpu)
        h, v_new, ht = chunk_fwd_h(
            _head_first(k), _head_first(w), _head_first(u), g=_head_first(g), initial_state=initial_state,
            output_final_state=output_final_state, chunk_size=chunk_size, state_v_first=state_v_first,
            save_new_value=save_new_value, cu_seqlens=offsets, chunk_indices=indices, use_exp2=True,
        )
        return h, _head_first(v_new), ht

    def chunk_gated_delta_rule_bwd_dhu_verifier(
        self, q, k, w, do, dv, g=None, gk=None, h0=None, dht=None, scale=None, state_v_first=False,
        cu_seqlens=None, chunk_size=64, chunk_indices=None, chunk_offsets=None, use_graph=False,
    ):
        meta = _contract('chunk_gated_delta_rule_bwd_dhu', k, do, g, chunk_size, cu_seqlens, chunk_indices)
        if (
            not meta or gk is not None or h0 is not None or chunk_offsets is not None or use_graph
            or not _tensor(q, k.shape, k.device, (k.dtype,))
            or not _tensor(w, do.shape, k.device, (k.dtype,))
            or not _tensor(dv, do.shape, k.device, (k.dtype,))
            or (dht is not None and not _tensor(dht, (meta[2], do.shape[2], 128, 128), k.device, (k.dtype,)))
            or not isinstance(scale, (int, float)) or not isfinite(scale)
        ):
            return False, 'AscendC Dhu requires scalar gate and K=V=128 without an initial-state gradient'
        return True, None

    @input_guard
    def chunk_gated_delta_rule_bwd_dhu(
        self, q, k, w, do, dv, g=None, gk=None, h0=None, dht=None, scale=None, state_v_first=False,
        cu_seqlens=None, chunk_size=64, chunk_indices=None, chunk_offsets=None, use_graph=False,
    ):
        from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu

        offsets, indices, _, _ = _chunk_metadata(k, cu_seqlens, chunk_indices, chunk_size)
        dh, _, dv2 = chunk_gated_delta_rule_bwd_dhu(
            *map(_head_first, (q, k, w, do, dv)), scale, chunk_size,
            g=_head_first(g), dht=dht, cu_seqlens=offsets, chunk_indices=indices,
            use_exp2=True, transpose_state_layout=state_v_first,
        )
        if state_v_first:
            dh = dh.transpose(-1, -2).contiguous()
        return dh, None, _head_first(dv2)

    def recompute_w_u_fwd_verifier(self, k, v, beta, A, g=None, cu_seqlens=None, chunk_indices=None):
        if not _supports_wy('recompute_w_u_fwd', k, v, beta, A, g, cu_seqlens, chunk_indices):
            return False, 'AscendC WY recompute requires a supported wheel entry and scalar gate'
        return True, None

    @input_guard
    def recompute_w_u_fwd(self, k, v, beta, A, g=None, cu_seqlens=None, chunk_indices=None):
        from fla_npu.ops.ascendc import recompute_w_u_fwd

        offsets, indices, _, _ = _chunk_metadata(k, cu_seqlens, chunk_indices, 64)
        w, u = recompute_w_u_fwd(
            *map(_head_first, (k, v, beta, A)), 64, g=_natural_gate(g),
            cu_seqlens=offsets, chunk_indices=indices,
        )
        return _head_first(w), _head_first(u)

    def prepare_wy_repr_bwd_verifier(self, k, v, beta, A, dw, du, g=None, cu_seqlens=None, chunk_indices=None):
        if (
            not _supports_wy('prepare_wy_repr_bwd', k, v, beta, A, g, cu_seqlens, chunk_indices)
            or k.shape[2] != v.shape[2] or not _tensor(dw, k.shape, k.device, (k.dtype,))
            or not _tensor(du, v.shape, k.device, (k.dtype,))
        ):
            return False, 'AscendC WY backward requires equal head counts and a supported scalar gate'
        return True, None

    @input_guard
    def prepare_wy_repr_bwd(self, k, v, beta, A, dw, du, g=None, cu_seqlens=None, chunk_indices=None):
        from fla_npu.ops.ascendc import prepare_wy_repr_bwd

        offsets, indices, _, _ = _chunk_metadata(k, cu_seqlens, chunk_indices, 64)
        dk, dv, db, dg = prepare_wy_repr_bwd(
            *map(_head_first, (k, v, beta, A, dw, du)), _natural_gate(g), 64,
            cu_seqlens=offsets, chunk_indices=indices,
        )
        # Both backward interfaces return the gradient with respect to the natural-log gate.
        return tuple(map(_head_first, (dk, dv, db, dg)))
