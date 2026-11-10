/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace bf16_mxfp8_hopper_low_latency {

/*
 * BF16 activation x MXFP8 weight GEMM for small M (<= 16) on SM90.
 *
 * out[M, N] = a[M, K] @ dequant(w[N, K])^T, where w holds E4M3 values with one
 * UE8M0 (power-of-two) scale per 32 consecutive K elements of a row.
 *
 * Hopper has no block-scaled MMA, so the weights are dequantized in registers:
 * E4M3 -> BF16 is exact, mma.sync.m16n8k16 multiplies them with the BF16
 * activations (used as stored) and accumulates in FP32, with the weights as the
 * 16-row A operand and up to 8 tokens per n8 tile, and each 32-K group's
 * power-of-two scale is applied to the FP32 partial (exact).
 *
 * The kernel is shaped for a weight-streaming decode GEMM: each warp keeps its
 * K slice of the activation in registers as MMA B fragments and issues all of
 * its 16 B weight loads up front (before griddepcontrol.wait under PDL, since
 * they do not depend on the previous kernel). When there are more N blocks than
 * resident CTAs, a persistent variant loops over N blocks and issues the next
 * block's weight loads before reducing the current one. Warps along K reduce through
 * shared memory; the split-K CTAs of a thread-block cluster push their partials
 * into rank 0's shared memory with st.async, completing on an mbarrier that only
 * rank 0 waits on, so there is no workspace and the launch is CUDA graph safe.
 *
 * Weights use a prepared layout (see prepare_bf16_mxfp8_weights): 512 B blocks
 * of 16 rows x 32 K ordered as mma A fragments, and 16 B scale blocks holding
 * the 16 rows' scales of one 32-K group.
 */

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// E4M3 in bits 8-15 of each 16-bit half -> bf16x2 (exact, subnormals included):
// move exponent and mantissa into place, then rebias the exponent by 2^(127 - 7).
__device__ __forceinline__ uint32_t e4m3x2_hi_to_bf16x2(uint32_t t) {
  uint32_t v = (t & 0x80008000u) | ((t >> 4) & 0x07F007F0u), r;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(0x7B807B80u));
  return r;
}

// 4 E4M3 bytes ordered (row r, k), (row r + 8, k), (row r, k + 1),
// (row r + 8, k + 1) -> the bf16x2 fragments of rows r and r + 8.
__device__ __forceinline__ void e4m3x4_to_bf16x4(uint32_t w, uint32_t& lo, uint32_t& hi) {
  lo = e4m3x2_hi_to_bf16x2(w << 8);
  hi = e4m3x2_hi_to_bf16x2(w);
}

__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float e8m0_to_scale(uint32_t s) {
  return __uint_as_float(s << 23);  // 2^(s - 127)
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// MT: n8 token tiles (M <= 8 * MT). KW: K per warp (multiple of 32).
// NT: 16-row weight tiles per warp. Block: WK x WN warps, WK along K.
// Grid: (GX, ksplit); the ksplit CTAs of a column form a cluster. Without
// PERSISTENT, GX = N / n_cta and each CTA computes one N block; with it, each
// CTA loops over N blocks blockIdx.x, blockIdx.x + GX, ... and streams the next
// block's weights while the current one is reduced.
template <int MT, int KW, int NT, bool PERSISTENT>
__global__ void Bf16Mxfp8LowLatencyKernel(const uint4* __restrict__ W,
                                          const uint16_t* __restrict__ S,
                                          const __nv_bfloat16* __restrict__ X,
                                          __nv_bfloat16* __restrict__ Y, int M, int N, int K,
                                          int WK, int WN, int x_stride, int ksplit, bool pdl) {
  constexpr int G = KW / 32;  // k-groups per warp
  extern __shared__ __align__(16) float red[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wk = warp % WK, wn = warp / WK;
  const int r = lane >> 2, c = lane & 3;
  const int KG = K >> 5;
  const int n_cta = WN * NT * 16;
  const int num_blocks = N / n_cta;
  const int kg0 = (blockIdx.y * WK + wk) * G;
  const bool clustered = ksplit > 1;
  const int outs = n_cta * M;
  // Partials are staged [wk][tok][n] with a padded row (n_cta + 4): conflict-free
  // for both the fragment stores and the n-contiguous reduction reads. The
  // persistent kernel alternates two such buffers between N blocks.
  const int ld = n_cta + 4;
  const int total = MT * 8 * ld;
  // Split-K: rank 0 of the cluster gathers the other ranks' partials with
  // st.async into gather[], completing on an mbarrier it alone waits on.
  float* gather = red + (PERSISTENT ? 2 : 1) * WK * total;
  uint64_t* gather_bar = reinterpret_cast<uint64_t*>(gather + (ksplit - 1) * outs + 2) - 1;
  uint32_t rank = 0;
  if (clustered) {
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    if (rank == 0 && threadIdx.x == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_addr(gather_bar)));
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  }

  // Weight loads do not depend on the previous kernel, so they go out first.
  // Offsets are 32-bit (N * K / 16 < 2^31) so each load is base + immediate.
  uint4 w[NT][G];
  uint32_t s[NT][G];
  auto load_weights = [&](int xi) {
    const int nt0 = xi * WN * NT + wn * NT;
    const uint4* wp = W + ((uint32_t)nt0 * KG + kg0) * 32 + lane;
    const uint16_t* sp = S + ((uint32_t)nt0 * KG + kg0) * 8 + r;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int g = 0; g < G; ++g) {
        w[t][g] = ldg_stream(wp + (t * KG + g) * 32);
        s[t][g] = __ldg(sp + (t * KG + g) * 8);
      }
    }
  };
  int xi = blockIdx.x;
  load_weights(xi);
  // The wait sits in its own basic block: otherwise ptxas sinks the weight loads
  // below it and nothing is prefetched while the previous kernel finishes.
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  uint32_t b[2 * G][MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int tok = mt * 8 + r;
    const uint32_t* xp =
        reinterpret_cast<const uint32_t*>(X + (size_t)tok * x_stride + kg0 * 32 + 2 * c);
    const bool valid = tok < M;
#pragma unroll
    for (int kk = 0; kk < 2 * G; ++kk) {
      b[kk][mt][0] = valid ? xp[kk * 8] : 0u;
      b[kk][mt][1] = valid ? xp[kk * 8 + 4] : 0u;
    }
  }
  asm volatile("griddepcontrol.launch_dependents;");

  // Thread -> (tok, n) without per-element division: the first
  // tok_step * n_cta threads each own one n and every tok_step-th token (or,
  // when blockDim < n_cta, every blockDim-th n of every token).
  const int nthr = blockDim.x;
  const int tn = threadIdx.x % n_cta;
  const int tok_step = nthr >= n_cta ? nthr / n_cta : 1;
  const int n_step = nthr >= n_cta ? n_cta : nthr;
  const int tt = threadIdx.x < tok_step * n_cta ? threadIdx.x / n_cta : M;

  for (int it = 0;; ++it) {
    float* buf = red + (PERSISTENT ? (it & 1) * WK * total : 0);
    float* my_red = buf + wk * total;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      float acc[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[mt][i] = 0.f;
#pragma unroll
      for (int g = 0; g < G; ++g) {
        uint32_t a0[4], a1[4];
        e4m3x4_to_bf16x4(w[t][g].x, a0[0], a0[1]);
        e4m3x4_to_bf16x4(w[t][g].y, a0[2], a0[3]);
        e4m3x4_to_bf16x4(w[t][g].z, a1[0], a1[1]);
        e4m3x4_to_bf16x4(w[t][g].w, a1[2], a1[3]);
        float f0 = e8m0_to_scale(s[t][g] & 0xFFu);
        float f1 = e8m0_to_scale(s[t][g] >> 8);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          float d[4] = {0.f, 0.f, 0.f, 0.f};
          mma_bf16(d, a0, b[2 * g][mt][0], b[2 * g][mt][1]);
          mma_bf16(d, a1, b[2 * g + 1][mt][0], b[2 * g + 1][mt][1]);
          acc[mt][0] = fmaf(d[0], f0, acc[mt][0]);
          acc[mt][1] = fmaf(d[1], f0, acc[mt][1]);
          acc[mt][2] = fmaf(d[2], f1, acc[mt][2]);
          acc[mt][3] = fmaf(d[3], f1, acc[mt][3]);
        }
      }
      const int n_local = (wn * NT + t) * 16 + r;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int tok = mt * 8 + 2 * c;
        my_red[tok * ld + n_local] = acc[mt][0];
        my_red[(tok + 1) * ld + n_local] = acc[mt][1];
        my_red[tok * ld + n_local + 8] = acc[mt][2];
        my_red[(tok + 1) * ld + n_local + 8] = acc[mt][3];
      }
    }
    const bool more = PERSISTENT && xi + (int)gridDim.x < num_blocks;
    if (more) load_weights(xi + gridDim.x);
    __syncthreads();

    const int n_base = xi * n_cta;
    auto partial = [&](int tok, int n) {
      float v = 0.f;
      for (int q = 0; q < WK; ++q) v += buf[q * total + tok * ld + n];
      return v;
    };
    if (!clustered) {
      for (int tok = tt; tok < M; tok += tok_step)
        for (int n = tn; n < n_cta; n += n_step)
          Y[(size_t)tok * N + n_base + n] = __float2bfloat16(partial(tok, n));
    } else {
      // Completes once every rank is done with the previous N block (rank 0 has
      // read the gather buffer) and, the first time, rank 0's mbarrier is ready.
      asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
      if (rank != 0) {
        uint32_t dst, bar;
        asm("mapa.shared::cluster.u32 %0, %1, 0;"
            : "=r"(dst)
            : "r"(smem_addr(gather + (rank - 1) * outs)));
        asm("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(bar) : "r"(smem_addr(gather_bar)));
        for (int tok = tt; tok < M; tok += tok_step)
          for (int n = tn; n < n_cta; n += n_step)
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(
                    dst + (tok * n_cta + n) * 4),
                "f"(partial(tok, n)), "r"(bar)
                : "memory");
      } else {
        if (threadIdx.x == 0)
          asm volatile(
              "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(gather_bar)),
              "r"((ksplit - 1) * outs * 4)
              : "memory");
        for (int tok = tt; tok < M; tok += tok_step)
          for (int n = tn; n < n_cta; n += n_step) {
            float v = partial(tok, n);
            asm volatile(
                "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], "
                "%1;\n @!p bra WAIT;\n}" ::"r"(smem_addr(gather_bar)),
                "r"(it & 1)
                : "memory");
            for (int q = 0; q < ksplit - 1; ++q) v += gather[q * outs + tok * n_cta + n];
            Y[(size_t)tok * N + n_base + n] = __float2bfloat16(v);
          }
      }
      // After the last block, rank 0 waits on the gather and the others may exit.
      if (more) asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    }
    if (!more) return;
    xi += gridDim.x;
  }
}

struct LaunchParams {
  const void* w;
  const void* s;
  const void* x;
  void* y;
  int m, n, k, x_stride;
  int kw, nt, wk, wn, ksplit;
  int gx;  // persistent CTA columns; 0 (or >= N / n_cta) for one N block per CTA
  bool pdl;
};

inline size_t SmemBytes(int mt, int n_cta, int m, int wk, int ksplit, bool persistent) {
  // Reduction buffer(s), then (split-K) the gathered partials of ranks 1.. and an mbarrier.
  return ((size_t)(persistent ? 2 : 1) * wk * mt * 8 * (n_cta + 4) +
          (ksplit > 1 ? (size_t)(ksplit - 1) * n_cta * m + 4 : 0)) *
         sizeof(float);
}

template <int MT, int KW, int NT>
cudaError_t LaunchConfigured(const LaunchParams& p, cudaStream_t stream) {
  const int n_cta = p.wn * NT * 16;
  const int num_blocks = p.n / n_cta;
  const bool persistent = p.gx > 0 && p.gx < num_blocks;
  auto kernel = persistent ? Bf16Mxfp8LowLatencyKernel<MT, KW, NT, true>
                           : Bf16Mxfp8LowLatencyKernel<MT, KW, NT, false>;
  size_t smem = SmemBytes(MT, n_cta, p.m, p.wk, p.ksplit, persistent);
  if (smem > 48 * 1024) {
    cudaError_t status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (status != cudaSuccess) return status;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(persistent ? p.gx : num_blocks, p.ksplit, 1);
  cfg.blockDim = dim3(32 * p.wk * p.wn, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  int na = 0;
  attrs[na].id = cudaLaunchAttributeClusterDimension;
  attrs[na].val.clusterDim.x = 1;
  attrs[na].val.clusterDim.y = p.ksplit;
  attrs[na].val.clusterDim.z = 1;
  ++na;
  if (p.pdl) {
    attrs[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attrs;
  cfg.numAttrs = na;
  return cudaLaunchKernelEx(
      &cfg, kernel, reinterpret_cast<const uint4*>(p.w), reinterpret_cast<const uint16_t*>(p.s),
      reinterpret_cast<const __nv_bfloat16*>(p.x), reinterpret_cast<__nv_bfloat16*>(p.y), p.m, p.n,
      p.k, p.wk, p.wn, p.x_stride, p.ksplit, p.pdl);
}

template <int MT, int KW>
cudaError_t DispatchNT(const LaunchParams& p, cudaStream_t stream) {
  switch (p.nt) {
    case 1:
      return LaunchConfigured<MT, KW, 1>(p, stream);
    case 2:
      return LaunchConfigured<MT, KW, 2>(p, stream);
    case 4:
      return LaunchConfigured<MT, KW, 4>(p, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

template <int MT>
cudaError_t DispatchKW(const LaunchParams& p, cudaStream_t stream) {
  switch (p.kw) {
    case 32:
      return DispatchNT<MT, 32>(p, stream);
    case 64:
      return DispatchNT<MT, 64>(p, stream);
    case 96:
      return DispatchNT<MT, 96>(p, stream);
    case 128:
      return DispatchNT<MT, 128>(p, stream);
    case 192:
      return DispatchNT<MT, 192>(p, stream);
    case 256:
      return DispatchNT<MT, 256>(p, stream);
    case 384:
      return DispatchNT<MT, 384>(p, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

// Validates the tile configuration against the problem and launches.
inline cudaError_t Launch(const LaunchParams& p, cudaStream_t stream) {
  if (p.m < 1 || p.m > 16 || p.wk < 1 || p.wn < 1 || p.ksplit < 1 || p.ksplit > 8 ||
      p.wk * p.wn > 32 || p.gx < 0 || p.n % (p.wn * p.nt * 16) != 0 ||
      p.k != p.ksplit * p.wk * p.kw) {
    return cudaErrorInvalidValue;
  }
  return p.m <= 8 ? DispatchKW<1>(p, stream) : DispatchKW<2>(p, stream);
}

}  // namespace bf16_mxfp8_hopper_low_latency
}  // namespace flashinfer
