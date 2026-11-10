from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm90a_nvcc_flags


def gen_bf16_mxfp8_hopper_low_latency_module() -> JitSpec:
    """BF16 x MXFP8 (W8A16) small-M GEMM for SM90."""
    return gen_jit_spec(
        "bf16_mxfp8_hopper_low_latency",
        [jit_env.FLASHINFER_CSRC_DIR / "bf16_mxfp8_hopper_low_latency.cu"],
        extra_cuda_cflags=sm90a_nvcc_flags,
    )
