# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from math import isfinite

import torch

from fla.backends import BaseBackend


class AscendCCPBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    env_var = 'FLA_CP_ASCENDC'
    default_enable = False


def is_enabled():
    backend = AscendCCPBackend()
    return backend.is_enabled() and backend.is_available() and not torch.npu.is_current_stream_capturing()


def _head_first(tensor):
    return tensor.transpose(1, 2).contiguous()


def _get_op(name):
    try:
        from fla_npu.ops import ascendc
    except ImportError:
        return None
    return getattr(ascendc, name, None)


def _window(k, cu_seqlens):
    if (
        cu_seqlens is None or cu_seqlens.shape != (2,) or cu_seqlens.device != k.device
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
    ):
        return None
    start, end = cu_seqlens.tolist()
    if 0 <= start < end <= k.shape[1]:
        return [start, end]
    return None


def _supports_preprocess(k, w, u, g, gk, bg, chunk_size, dtypes, gate_dtypes):
    if (
        k.device.type != 'npu' or chunk_size != 64 or bg is not None or (g is None) == (gk is None)
        or k.shape[0] != 1 or k.shape[-1] != 128 or k.dtype not in dtypes or min(k.shape) <= 0
        or u.shape[:2] != k.shape[:2] or u.shape[-1] != 128 or u.shape[2] % k.shape[2]
        or u.shape[2] // k.shape[2] not in (1, 2, 3, 4)
        or w.shape != u.shape or w.dtype != k.dtype or u.dtype != k.dtype
        or w.device != k.device or u.device != k.device
    ):
        return False
    if gk is not None:
        return gk.shape == w.shape and gk.dtype in gate_dtypes and gk.device == k.device
    # The Triton CP caller already packs the scalar gate head-first when HV > 1.
    gate_shape = (1, u.shape[2], k.shape[1]) if u.shape[2] > 1 else (1, k.shape[1], 1)
    return g.shape == gate_shape and g.dtype in (torch.float32, k.dtype) and g.device == k.device


def preprocess_fwd(k, w, u, g, gk, bg, v, hm, cu_seqlens, chunk_size):
    if v is not None or not _supports_preprocess(
        k, w, u, g, gk, bg, chunk_size, (torch.bfloat16,), (torch.float32, torch.bfloat16),
    ):
        return False
    op = _get_op('pre_process_fwd_kernel_merged')
    if op is None:
        return False
    offsets = _window(k, cu_seqlens)
    if offsets is None:
        return False
    result = op(
        _head_first(k), _head_first(w), _head_first(u),
        g=None if g is None else g.reshape(1, u.shape[2], k.shape[1]),
        gk=None if gk is None else _head_first(gk), cu_seqlens=offsets, chunk_size=chunk_size,
    )
    hm.copy_(result[0])
    return True


def preprocess_bwd(q, k, w, do, dv, g, gk, bg, dhm, cu_seqlens, scale, chunk_size):
    if (
        not _supports_preprocess(
            k, w, do, g, gk, bg, chunk_size, (torch.bfloat16, torch.float16), (k.dtype,),
        )
        or q.shape != k.shape or dv.shape != do.shape or q.dtype != k.dtype or dv.dtype != k.dtype
        or q.device != k.device or dv.device != k.device
        or not isinstance(scale, (int, float)) or not isfinite(scale)
    ):
        return False
    op = _get_op('pre_process_bwd_kernel_merged')
    if op is None:
        return False
    offsets = _window(k, cu_seqlens)
    if offsets is None:
        return False
    result = op(
        _head_first(q), _head_first(k), _head_first(w), _head_first(do), _head_first(dv), scale, chunk_size,
        g=None if g is None else g.reshape(1, do.shape[2], k.shape[1]),
        gk=None if gk is None else _head_first(gk), cu_seqlens=offsets,
    )
    dhm.copy_(result)
    return True


def merge(h, ag_hm, num_ranks, rank, state_v_first, forward):
    if (
        h.ndim != 3 or h.shape[-2:] != (128, 128) or not 1 <= h.shape[0] <= 256
        or h.dtype != torch.float32 or not h.is_contiguous() or not 1 <= ag_hm.shape[0] <= 1024
        or ag_hm.shape[1:] != (h.shape[0], 128, 256) or ag_hm.dtype != h.dtype or ag_hm.device != h.device
        or not isinstance(num_ranks, int) or not 0 <= rank < ag_hm.shape[0]
        or not 1 <= num_ranks <= (rank if forward else ag_hm.shape[0] - rank - 1)
    ):
        return False
    op = _get_op('merge_fwd_bwd_kernel')
    if op is None:
        return False
    op(h, ag_hm.contiguous(), num_ranks, rank, forward=forward, state_v_first=state_v_first)
    return True
