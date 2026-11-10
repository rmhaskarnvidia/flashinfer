# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""BF16 x MXFP8 (W8A16) dense GEMM public API.

``out = a @ dequant(b).T`` with BF16 activations and MXFP8 weights: E4M3
values with one UE8M0 (power-of-two) scale per 32 consecutive K elements.
"""

import functools
from typing import Literal, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..jit.gemm import gen_bf16_mxfp8_hopper_low_latency_module
from ..utils import (
    backend_requirement,
    get_device_sm_count,
    register_custom_op,
    supported_compute_capability,
)

_Bf16Mxfp8Backend = Literal["hopper-low-latency"]

_MXFP8_BLOCK_SIZE = 32
# Largest M the hopper-low-latency kernel serves (two n8 token tiles).
_HOPPER_LOW_LATENCY_MAX_M = 16

# (kw, nt, wk, wn, ksplit, gx) for M <= 8 and for 8 < M <= 16, measured on
# H100; other shapes use the heuristic in _hopper_low_latency_config.
_HOPPER_LOW_LATENCY_TUNED = {
    (1536, 6144): ((256, 2, 4, 1, 6, 0), (96, 2, 16, 1, 4, 33)),
    (6144, 1024): ((64, 4, 16, 1, 1, 0), (64, 2, 16, 1, 1, 132)),
    (768, 6144): ((384, 1, 2, 2, 8, 0), (96, 2, 16, 1, 4, 0)),
    (6144, 384): ((192, 1, 2, 1, 1, 0), (32, 4, 12, 1, 1, 0)),
    (3072, 6144): ((192, 2, 8, 1, 4, 66), (96, 4, 8, 1, 8, 33)),
    (6144, 1536): ((192, 1, 8, 1, 1, 264), (128, 1, 12, 1, 1, 132)),
}


@functools.cache
def _hopper_low_latency_config(
    n: int, k: int, m: int, sm_count: int
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Tile configuration ``(kw, nt, wk, wn, ksplit, gx)`` for an ``(N, K)`` weight.

    Each warp covers ``kw`` of K for ``nt`` 16-row tiles; a CTA has ``wk``
    warps along K and ``wn`` along N; ``ksplit`` CTAs (one cluster) split K.
    ``gx > 0`` launches ``gx`` persistent CTA columns that loop over the N
    blocks; ``gx == 0`` launches one CTA column per N block.
    The heuristic targets ~24 KB of weights per CTA and ~3 CTAs per SM.
    Returns None when no configuration tiles the shape exactly.
    """
    if (n, k) in _HOPPER_LOW_LATENCY_TUNED:
        return _HOPPER_LOW_LATENCY_TUNED[(n, k)][m > 8]
    target_ctas = 3 * sm_count
    best = None
    for kw in (128, 64, 192, 256, 96, 32, 384):
        for wk in (8, 4, 6, 3, 2, 1, 12, 16):
            if k % (kw * wk):
                continue
            ksplit = k // (kw * wk)
            if ksplit > 8:
                continue
            for nt in (2, 1, 4):
                for wn in (1, 2, 4):
                    if wk * wn > 16 or n % (wn * nt * 16):
                        continue
                    ctas = (n // (wn * nt * 16)) * ksplit
                    cta_bytes = n * k // ctas
                    score = abs(ctas / target_ctas - 1) + 0.5 * abs(
                        cta_bytes / 24576 - 1
                    )
                    if best is None or score < best[0]:
                        best = (score, (kw, nt, wk, wn, ksplit, 0))
    return None if best is None else best[1]


@functools.cache
def _get_hopper_low_latency_module():
    module = gen_bf16_mxfp8_hopper_low_latency_module().build_and_load()

    @register_custom_op(
        "flashinfer::bf16_mxfp8_hopper_low_latency_gemm",
        mutates_args=["out"],
    )
    def gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        b_descale: torch.Tensor,
        out: torch.Tensor,
        kw: int,
        nt: int,
        wk: int,
        wn: int,
        ksplit: int,
        gx: int,
        enable_pdl: bool,
    ) -> None:
        module.bf16_mxfp8_hopper_low_latency_gemm(
            a, b, b_descale, out, kw, nt, wk, wn, ksplit, gx, enable_pdl
        )

    return gemm


def _scales_2d(b_descale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The ``(N, K / 32)`` uint8 scales from a 2-D linear or 1-D 128x4 layout."""
    b_descale = b_descale.view(torch.uint8)
    if b_descale.dim() == 2:
        if tuple(b_descale.shape) != (n, k // _MXFP8_BLOCK_SIZE):
            raise ValueError(
                f"b_descale must be (N, K / 32) = {(n, k // _MXFP8_BLOCK_SIZE)}; "
                f"got {tuple(b_descale.shape)}"
            )
        return b_descale
    if b_descale.dim() == 1:
        from ..batch_invariant.quantization.layout import unswizzle_sf_128x4

        return unswizzle_sf_128x4(b_descale, n, k, _MXFP8_BLOCK_SIZE)
    raise ValueError(
        "b_descale must be 2-D (N, K / 32) linear or 1-D 128x4-swizzled scales; "
        f"got shape {tuple(b_descale.shape)}"
    )


def _pack_hopper_low_latency(
    b: torch.Tensor, b_descale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Weights: 512 B blocks of 16 rows x 32 K in mma.m16n8k16 A-fragment order,
    # so each lane loads its fragments of two k16 steps with one 16 B load.
    # Scales: per (16-row tile, 32-K group), lane row r reads rows r and r + 8.
    n, k = b.shape
    w = b.view(torch.uint8).reshape(n // 16, 2, 8, k // 32, 2, 2, 4, 2)
    # (nt, h, r, kg, s, j, c, p) -> (nt, kg, r, c, s, j, p, h)
    w = w.permute(0, 3, 2, 6, 4, 5, 7, 1).contiguous().view(n, k)
    sc = _scales_2d(b_descale, n, k).reshape(n // 16, 2, 8, k // 32)
    sc = sc.permute(0, 3, 2, 1).contiguous().view(n, k // 32)
    return w, sc


@flashinfer_api
def prepare_bf16_mxfp8_weights(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    *,
    backend: _Bf16Mxfp8Backend = "hopper-low-latency",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare MXFP8 weights for :func:`mm_bf16_mxfp8` with ``backend``.

    Args:
        b: ``(N, K)`` weight in ``torch.float8_e4m3fn`` (or its ``uint8``
            bytes), row-major as in ``nn.Linear``.
        b_descale: UE8M0 scales of 32-element K blocks, as ``(N, K / 32)``
            linear scales or the 1-D 128x4-swizzled buffer that
            ``mxfp8_quantize(..., is_sf_swizzled_layout=True)`` returns, in
            ``uint8`` or ``torch.float8_e8m0fnu``.
        backend: The backend the prepared weights are for.

    Returns:
        ``(b, b_descale)`` in the backend's layout. Pass them to
        :func:`mm_bf16_mxfp8` with the same ``backend``.
    """
    if backend != "hopper-low-latency":
        raise ValueError(f"Unsupported backend: {backend}")
    if b.dim() != 2 or b.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise ValueError(
            f"b must be a 2-D float8_e4m3fn or uint8 (N, K) tensor; got "
            f"{b.dtype} {tuple(b.shape)}"
        )
    n, k = b.shape
    if n % 16 or k % _MXFP8_BLOCK_SIZE:
        raise ValueError(
            f"hopper-low-latency requires N % 16 == 0 and K % 32 == 0; got N={n}, K={k}"
        )
    return _pack_hopper_low_latency(b.contiguous(), b_descale)


def _check_mm_bf16_mxfp8_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    *,
    backend: _Bf16Mxfp8Backend = "hopper-low-latency",
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    enable_pdl: bool = True,
):
    if a.dim() != 2 or a.dtype != torch.bfloat16:
        raise ValueError(
            f"a must be a 2-D bfloat16 (M, K) tensor; got {a.dtype} {tuple(a.shape)}"
        )
    if b.dim() != 2 or b.shape[1] != a.shape[1]:
        raise ValueError(
            f"b must be the prepared (N, K) weight with K = {a.shape[1]}; got "
            f"{tuple(b.shape)}"
        )
    if out_dtype not in (None, torch.bfloat16):
        raise ValueError(f"out_dtype must be bfloat16; got {out_dtype}")
    if out is not None and (
        out.dtype != torch.bfloat16
        or tuple(out.shape) != (a.shape[0], b.shape[0])
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"out must be a contiguous bfloat16 {(a.shape[0], b.shape[0])} tensor"
        )
    return True


@supported_compute_capability([90])
def _hopper_low_latency_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    *,
    backend: _Bf16Mxfp8Backend = "hopper-low-latency",
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    enable_pdl: bool = True,
):
    m, k = a.shape
    n = b.shape[0]
    if m > _HOPPER_LOW_LATENCY_MAX_M:
        raise ValueError(
            f"hopper-low-latency serves M <= {_HOPPER_LOW_LATENCY_MAX_M}; got M={m}"
        )
    if b.dtype != torch.uint8 or tuple(b_descale.shape) != (n, k // 32):
        raise ValueError(
            "hopper-low-latency expects the uint8 weight and (N, K / 32) scales "
            "returned by prepare_bf16_mxfp8_weights(..., backend='hopper-low-latency')"
        )
    if a.stride(1) != 1 or a.stride(0) % 2 or a.data_ptr() % 4:
        raise ValueError(
            "hopper-low-latency needs a with unit K stride, an even row stride "
            "and 4-byte alignment"
        )
    if _hopper_low_latency_config(n, k, m, get_device_sm_count(a.device)) is None:
        raise ValueError(f"hopper-low-latency has no tiling for N={n}, K={k}")
    return True


@backend_requirement(
    {"hopper-low-latency": _hopper_low_latency_requirement},
    common_check=_check_mm_bf16_mxfp8_problem_size,
)
@flashinfer_api
def mm_bf16_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    *,
    backend: _Bf16Mxfp8Backend = "hopper-low-latency",
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """BF16 x MXFP8 GEMM: ``out = a @ dequant(b).T``.

    Supports **W8A16** workloads: BF16 activations with MXFP8 weights (E4M3
    values, one UE8M0 scale per 32 K elements). The activations are not
    quantized.

    Example:
        .. code-block:: python

            # 1) Prepare weights once, at model load.
            b_p, sf_p = flashinfer.prepare_bf16_mxfp8_weights(
                b, b_descale, backend="hopper-low-latency"
            )
            # 2) Run the GEMM with the same backend.
            out = flashinfer.mm_bf16_mxfp8(
                a, b_p, sf_p, backend="hopper-low-latency"
            )

    Backends:
        ``hopper-low-latency`` (SM90): a weight-streaming decode kernel for
        ``M <= 16``, tuned for ``M <= 8``; at larger M a tiled GEMM is usually
        faster. Weights are dequantized exactly to BF16 in registers and
        multiplied with the BF16 activations by ``mma.sync`` in FP32; the
        power-of-two block scales are applied exactly to the FP32 partials.
        Split-K reduces across a thread-block cluster, so
        the call needs no workspace and is CUDA graph safe. Requires
        ``N % 16 == 0`` and ``K % 32 == 0``.

    Args:
        a: ``(M, K)`` activations in ``torch.bfloat16`` with unit K stride.
        b: Prepared weight from :func:`prepare_bf16_mxfp8_weights`.
        b_descale: Prepared scales from :func:`prepare_bf16_mxfp8_weights`.
        backend: Same identifier passed to :func:`prepare_bf16_mxfp8_weights`.
        out: Optional preallocated contiguous ``(M, N)`` bfloat16 output.
        out_dtype: Output dtype; only ``torch.bfloat16`` is supported.
        enable_pdl: Launch with Programmatic Dependent Launch; the weight
            loads then overlap the previous kernel's tail.

    Returns:
        ``(M, N)`` bfloat16 tensor.
    """
    m, k = a.shape
    n = b.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    if m == 0:
        return out
    config = _hopper_low_latency_config(n, k, m, get_device_sm_count(a.device))
    _get_hopper_low_latency_module()(a, b, b_descale, out, *config, enable_pdl)
    return out
