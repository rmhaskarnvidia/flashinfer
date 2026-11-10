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
#include "flashinfer/gemm/bf16_mxfp8_hopper_low_latency.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::bf16_mxfp8_hopper_low_latency;

// a: [M, K] bf16 (row stride may exceed K); b: prepared [N, K] uint8;
// b_descale: prepared [N, K / 32] uint8; out: [M, N] bf16, contiguous.
void bf16_mxfp8_hopper_low_latency_gemm(TensorView a, TensorView b, TensorView b_descale,
                                        TensorView out, int64_t kw, int64_t nt, int64_t wk,
                                        int64_t wn, int64_t ksplit, int64_t gx, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(b_descale);
  CHECK_INPUT(out);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(2, b_descale);
  CHECK_DIM(2, out);
  CHECK_INPUT_TYPE(a, dl_bfloat16);
  CHECK_INPUT_TYPE(out, dl_bfloat16);
  CHECK_INPUT_TYPE(b, dl_uint8);
  CHECK_INPUT_TYPE(b_descale, dl_uint8);

  const int64_t m = a.size(0), k = a.size(1), n = b.size(0);
  TVM_FFI_ICHECK_EQ(b.size(1), k) << "b must be [N, K]";
  TVM_FFI_ICHECK_EQ(b_descale.size(0), n) << "b_descale must be [N, K / 32]";
  TVM_FFI_ICHECK_EQ(b_descale.size(1), k / 32) << "b_descale must be [N, K / 32]";
  TVM_FFI_ICHECK(out.size(0) == m && out.size(1) == n) << "out must be [M, N]";
  // Activations are read as 32-bit pairs.
  TVM_FFI_ICHECK(a.stride(0) % 2 == 0 && reinterpret_cast<uintptr_t>(a.data_ptr()) % 4 == 0)
      << "a must be 4-byte aligned with an even row stride";
  if (m == 0) return;

  LaunchParams p;
  p.w = b.data_ptr();
  p.s = b_descale.data_ptr();
  p.x = a.data_ptr();
  p.y = out.data_ptr();
  p.m = static_cast<int>(m);
  p.n = static_cast<int>(n);
  p.k = static_cast<int>(k);
  p.x_stride = static_cast<int>(a.stride(0));
  p.kw = static_cast<int>(kw);
  p.nt = static_cast<int>(nt);
  p.wk = static_cast<int>(wk);
  p.wn = static_cast<int>(wn);
  p.ksplit = static_cast<int>(ksplit);
  p.gx = static_cast<int>(gx);
  p.pdl = enable_pdl;

  cudaError_t status = Launch(p, get_stream(a.device()));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "bf16 x mxfp8 hopper-low-latency GEMM failed (M=" << m << ", N=" << n << ", K=" << k
      << ", kw=" << kw << ", nt=" << nt << ", wk=" << wk << ", wn=" << wn << ", ksplit=" << ksplit
      << ", gx=" << gx << "): " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_mxfp8_hopper_low_latency_gemm,
                              bf16_mxfp8_hopper_low_latency_gemm);
