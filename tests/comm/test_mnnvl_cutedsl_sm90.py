# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LL protocol on Hopper: MoE finalize of Marlin-layout (route-major, FP32-weight)
expert outputs, and the H200 profile.

Run with ``torchrun --nproc_per_node=<2|4|8>``.
"""

import os

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion
from flashinfer.comm.mnnvl_cutedsl import (
    DEFAULT_CONFIG,
    KernelTarget,
    MNNVLCuteDSLConfig,
    MRangeDispatch,
    ProtocolKind,
    StaticProfile,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_ll import (
    LL_ALL_REDUCE_H200_TP8_H6144,
    LL_FINALIZE_H200_TP8_H6144_K4,
)
from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace

TOP_K = 4
RMS_EPS = 1e-6
CAPACITY = 16
pytestmark = [pytest.mark.gpu_2, pytest.mark.arch_hopper]


@pytest.fixture(scope="module")
def distributed_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8):
        pytest.skip("Run this test with 2, 4 or 8 distributed ranks")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("SM90 is required")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=device)
    try:
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.destroy_process_group()


def _config(tp: int, hidden: int) -> MNNVLCuteDSLConfig:
    """The H200 presets for any TP size and hidden size."""
    return MNNVLCuteDSLConfig(
        profiles=(
            StaticProfile(
                tp_size=tp,
                hidden_size=hidden,
                top_k=TOP_K,
                dtype=torch.bfloat16,
                finalize_routes=MRangeDispatch(
                    upper_bounds=(None,),
                    targets=(
                        KernelTarget(ProtocolKind.LL, LL_FINALIZE_H200_TP8_H6144_K4),
                    ),
                ),
                all_reduce_routes=MRangeDispatch(
                    upper_bounds=(None,),
                    targets=(
                        KernelTarget(ProtocolKind.LL, LL_ALL_REDUCE_H200_TP8_H6144),
                    ),
                ),
            ),
        )
    )


def _inputs(m: int, hidden: int, rank: int):
    common = torch.Generator(device="cuda").manual_seed(6144 + m)
    local = torch.Generator(device="cuda").manual_seed(6144 + m + 100 * (rank + 1))
    residual = torch.randn(m, hidden, generator=common, device="cuda").bfloat16()
    gamma = (torch.randn(hidden, generator=common, device="cuda") * 0.1).bfloat16()
    routed = (
        torch.randn(m * TOP_K, hidden, generator=local, device="cuda") * 0.5
    ).bfloat16()
    weights = torch.rand(m, TOP_K, generator=local, device="cuda")
    shared = (torch.randn(m, hidden, generator=local, device="cuda") * 0.5).bfloat16()
    return residual, gamma, routed, weights, shared


def _gemma_norm(prenorm: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    values = prenorm.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + RMS_EPS)
    return (normalized * (1.0 + gamma.float())).bfloat16()


@pytest.mark.parametrize("route_major", (True, False))
@torch.inference_mode()
def test_finalize_marlin_layout(distributed_group, route_major):
    group = distributed_group
    rank, tp = dist.get_rank(group), dist.get_world_size(group)
    hidden = 6144
    weight_dtype = torch.float32 if route_major else torch.bfloat16
    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
        tp,
        rank,
        CAPACITY,
        hidden,
        torch.bfloat16,
        group=group,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        weight_bias=1.0,
        config=_config(tp, hidden),
        expert_weight_dtype=weight_dtype,
        route_major=route_major,
    )
    for m in (1, 4, 16):
        residual, gamma, routed, weights, shared = _inputs(m, hidden, rank)
        weights = weights.to(weight_dtype)
        # Permuted rows: reverse each token's routes to exercise the index.
        if route_major:
            index, rows = None, routed
        else:
            index = torch.arange(m * TOP_K, dtype=torch.int32, device="cuda")
            index = index.view(m, TOP_K).flip(1).contiguous()
            rows = routed.view(m, TOP_K, hidden).flip(1).reshape(m * TOP_K, hidden)
        residual_out = torch.empty_like(residual)
        norm_out = torch.empty_like(residual)
        allreduce_fusion(
            input=rows,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
            launch_with_pdl=True,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=gamma,
            rms_eps=RMS_EPS,
            expanded_idx_to_permuted_idx=index,
            expert_scale_factor=weights,
            shared_expert_output=shared,
            weight_bias=1.0,
        )
        partial = (
            routed.view(m, TOP_K, hidden).float() * weights.float().unsqueeze(-1)
        ).sum(1) + shared.float()
        partial = partial.bfloat16().float()
        dist.all_reduce(partial, group=group)
        prenorm = (partial + residual.float()).bfloat16()
        torch.testing.assert_close(residual_out, prenorm, rtol=0.02, atol=0.02)
        torch.testing.assert_close(
            norm_out, _gemma_norm(residual_out, gamma), rtol=0.02, atol=0.02
        )
    workspace.destroy()


@torch.inference_mode()
def test_h200_default_profile(distributed_group):
    """DEFAULT_CONFIG serves MiniMax-M3's TP8 shape on H200, under CUDA graphs."""
    group = distributed_group
    rank, tp = dist.get_rank(group), dist.get_world_size(group)
    if tp != 8:
        pytest.skip("The H200 profile is for TP8")
    hidden = 6144
    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
        tp,
        rank,
        CAPACITY,
        hidden,
        torch.bfloat16,
        group=group,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        weight_bias=1.0,
        config=DEFAULT_CONFIG,
        expert_weight_dtype=torch.float32,
        route_major=True,
    )
    m = 8
    residual, gamma, routed, weights, shared = _inputs(m, hidden, rank)
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual)

    def invoke():
        allreduce_fusion(
            input=routed,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
            launch_with_pdl=True,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=gamma,
            rms_eps=RMS_EPS,
            expert_scale_factor=weights,
            shared_expert_output=shared,
            weight_bias=1.0,
        )

    invoke()
    expected = (residual_out.clone(), norm_out.clone())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for _ in range(3):
        residual_out.zero_()
        norm_out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(residual_out, expected[0])
        assert torch.equal(norm_out, expected[1])
    workspace.destroy()
