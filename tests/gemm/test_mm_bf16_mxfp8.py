# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BF16 x MXFP8 GEMM API ``mm_bf16_mxfp8``."""

import pytest
import torch

from flashinfer import mm_bf16_mxfp8, prepare_bf16_mxfp8_weights
from flashinfer.batch_invariant.quantization.layout import swizzle_sf_128x4
from flashinfer.utils import get_compute_capability

BACKEND = "hopper-low-latency"


def _skip_unless_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if get_compute_capability(torch.device("cuda"))[0] != 9:
        pytest.skip("hopper-low-latency requires SM90")


def _quantized_weight(n, k, swizzled):
    """Random MXFP8 weight: E4M3 values, UE8M0 scales spread over several
    exponents, as (b, b_descale, dequantized fp32 weight)."""
    w = torch.randn(n, k // 32, 32, device="cuda")
    w *= torch.exp2(torch.randint(-6, 7, (n, k // 32, 1), device="cuda").float())
    # OCP MX scale: 2^(floor(log2(amax)) - 8), so |w / scale| < 512 rounds into E4M3.
    exp = torch.floor(torch.log2(w.abs().amax(dim=-1, keepdim=True))) - 8
    w_q = (w * torch.exp2(-exp)).clamp(-448, 448).to(torch.float8_e4m3fn)
    sf = (exp.squeeze(-1) + 127).to(torch.uint8)
    w_deq = (w_q.float() * torch.exp2(exp)).reshape(n, k)
    w_q = w_q.reshape(n, k)
    if swizzled:
        sf = swizzle_sf_128x4(sf, n, k, 32)
    return w_q, sf, w_deq


def _reference(a, w_deq):
    return (a.float() @ w_deq.T).to(torch.bfloat16)


def _assert_close(out, ref):
    torch.testing.assert_close(
        out.float(), ref.float(), rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )


# (N, K): tuned shapes (some persistent) plus shapes that take the heuristic
# configuration.
SHAPES = [
    (1536, 6144),
    (6144, 1024),
    (768, 6144),
    (6144, 384),
    (3072, 6144),
    (6144, 1536),
    (256, 4096),
    (4096, 512),
    (2048, 7168),
    (512, 2880),
]


@pytest.mark.parametrize("m", [1, 2, 4, 7, 8, 9, 16])
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("swizzled", [False, True])
def test_mm_bf16_mxfp8_matches_reference(m, n, k, swizzled):
    _skip_unless_sm90()
    torch.manual_seed(m * 7 + n + k)
    w_q, sf, w_deq = _quantized_weight(n, k, swizzled)
    b, b_descale = prepare_bf16_mxfp8_weights(w_q, sf, backend=BACKEND)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    out = mm_bf16_mxfp8(a, b, b_descale, backend=BACKEND)
    _assert_close(out, _reference(a, w_deq))


def test_mm_bf16_mxfp8_all_e4m3_codes():
    """Every finite E4M3 code (zeros and subnormals included) dequantizes exactly."""
    _skip_unless_sm90()
    n, k = 256, 4096
    codes = torch.tensor(
        [c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8, device="cuda"
    )
    w_q = codes[torch.randint(len(codes), (n, k), device="cuda")]
    w_q = w_q.view(torch.float8_e4m3fn)
    sf = torch.randint(120, 131, (n, k // 32), dtype=torch.uint8, device="cuda")
    w_deq = w_q.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)
    b, b_descale = prepare_bf16_mxfp8_weights(w_q, sf, backend=BACKEND)
    # One-hot activations select single weights, so the output is exact.
    a = torch.zeros(8, k, dtype=torch.bfloat16, device="cuda")
    a[torch.arange(8), torch.randint(k, (8,))] = 1
    out = mm_bf16_mxfp8(a, b, b_descale, backend=BACKEND)
    torch.testing.assert_close(out, _reference(a, w_deq), rtol=0, atol=0)


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_mm_bf16_mxfp8_strided_input_and_out(enable_pdl):
    _skip_unless_sm90()
    m, n, k = 5, 1536, 6144
    w_q, sf, w_deq = _quantized_weight(n, k, swizzled=False)
    b, b_descale = prepare_bf16_mxfp8_weights(
        w_q, sf.view(torch.float8_e8m0fnu), backend=BACKEND
    )
    a = torch.randn(m, k + 64, dtype=torch.bfloat16, device="cuda")[:, :k]
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    ret = mm_bf16_mxfp8(
        a, b, b_descale, backend=BACKEND, out=out, enable_pdl=enable_pdl
    )
    assert ret.data_ptr() == out.data_ptr()
    _assert_close(out, _reference(a, w_deq))


def test_mm_bf16_mxfp8_cuda_graph():
    _skip_unless_sm90()
    m, n, k = 4, 768, 6144
    w_q, sf, w_deq = _quantized_weight(n, k, swizzled=True)
    b, b_descale = prepare_bf16_mxfp8_weights(w_q, sf, backend=BACKEND)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    mm_bf16_mxfp8(a, b, b_descale, backend=BACKEND, out=out)  # build and warm up
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mm_bf16_mxfp8(a, b, b_descale, backend=BACKEND, out=out)
    for _ in range(3):
        a.copy_(torch.randn_like(a))
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(out, _reference(a, w_deq))


def test_mm_bf16_mxfp8_empty_m():
    _skip_unless_sm90()
    n, k = 256, 4096
    w_q, sf, _ = _quantized_weight(n, k, swizzled=False)
    b, b_descale = prepare_bf16_mxfp8_weights(w_q, sf, backend=BACKEND)
    a = torch.empty(0, k, dtype=torch.bfloat16, device="cuda")
    assert mm_bf16_mxfp8(a, b, b_descale, backend=BACKEND).shape == (0, n)


def test_mm_bf16_mxfp8_rejects_unsupported():
    _skip_unless_sm90()
    n, k = 256, 4096
    w_q, sf, _ = _quantized_weight(n, k, swizzled=False)
    b, b_descale = prepare_bf16_mxfp8_weights(w_q, sf, backend=BACKEND)
    with pytest.raises(ValueError, match="M <= 16"):
        mm_bf16_mxfp8(
            torch.randn(17, k, dtype=torch.bfloat16, device="cuda"),
            b,
            b_descale,
            backend=BACKEND,
        )
    with pytest.raises(ValueError, match="bfloat16"):
        mm_bf16_mxfp8(
            torch.randn(4, k, dtype=torch.float16, device="cuda"),
            b,
            b_descale,
            backend=BACKEND,
        )
    with pytest.raises(ValueError, match="prepare_bf16_mxfp8_weights"):
        mm_bf16_mxfp8(
            torch.randn(4, k, dtype=torch.bfloat16, device="cuda"),
            w_q,
            sf,
            backend=BACKEND,
        )
    with pytest.raises(ValueError, match="N % 16"):
        prepare_bf16_mxfp8_weights(w_q[:250], sf[:250], backend=BACKEND)
