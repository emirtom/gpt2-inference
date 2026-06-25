import os

import torch
from torch.utils.cpp_extension import load_inline

# =========================================================================
# CUDA kernel source (the actual kernel goes here)
# =========================================================================
fused_ffn_cuda_source = r"""
#include <cuda_runtime.h>

// Fused FFN kernel: LayerNorm -> c_fc -> GELU -> c_proj -> residual
//
// Shapes:
//   x:    [M, 768]        M = batch * seq_len
//   ln_w, ln_b: [768]
//   w1:   [768, 3072]     (c_fc weight, stored as [in, out] by HuggingFace Conv1D)
//   b1:   [3072]
//   w2:   [3072, 768]     (c_proj weight)
//   b2:   [768]
//   out:  [M, 768]
//
// Tiling parameters (suggested starting point):
//   TILE_M = 64         rows per block
//   TILE_K = 64         input-dim tile (across 768)
//   TILE_N = 64         intermediate-dim tile (across 3072)
//   BLOCK_SIZE = 256    threads per block
//
// ====== PHASE 1: LayerNorm ======
// Distribute 768 elements across 256 threads (3 per thread).
// Load input into registers, block-reduce to get per-row mean and variance,
// compute inv_std. Store mean/inv_std for 64 rows.
//
// ====== PHASE 2: Fused double-GEMM + GELU ======
// For each intermediate-dim tile (n from 0 to 3072 step TILE_N):
//   a) First matmul: compute inter[:, n:n+TILE_N] = x_ln @ w1[:, n:n+TILE_N]
//      - Iterate over input-dim tiles (k from 0 to 768 step TILE_K)
//      - Apply LayerNorm params to loaded input tile on-the-fly
//      - Accumulate into register tiles
//   b) Add bias b1 slice and apply GELU in registers
//   c) Second matmul: out += inter @ w2[n:n+TILE_N, :]
//      - Store inter to shared memory so all threads can read it
//      - Each thread multiplies inter values by w2 tiles for its output columns
//
// ====== PHASE 3: Epilogue ======
// Add b2 bias, add residual (original x), write to global memory.

__global__ void fused_ffn_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ ln_w,
    const float* __restrict__ ln_b,
    const float* __restrict__ w1,
    const float* __restrict__ b1,
    const float* __restrict__ w2,
    const float* __restrict__ b2,
    int M
) {
    // --- Placeholder ---
    // Replace this with the 3-phase implementation described above.
    // The grid and block dimensions are set in fused_ffn_cuda below.
    //
    // Hints:
    //   - blockIdx.x gives the row tile index (each tile has TILE_M rows)
    //   - Declare __shared__ float smem[...]; for input tiles and intermediate values
    //   - Use warp shuffle (__shfl_xor_sync) for block-level reductions
    //   - GELU tanh approx: 0.5 * x * (1 + tanh(0.79788456 * x * (1 + 0.044715 * x*x)))
}

// --- Host launcher (infrastructure — sets up grid/block) ---
void fused_ffn_cuda(
    torch::Tensor out,
    const torch::Tensor x,
    const torch::Tensor ln_w,
    const torch::Tensor ln_b,
    const torch::Tensor w1,
    const torch::Tensor b1,
    const torch::Tensor w2,
    const torch::Tensor b2,
    int M)
{
    const int TILE_M = 64;
    const int BLOCK_SIZE = 256;

    dim3 block(BLOCK_SIZE);
    dim3 grid((M + TILE_M - 1) / TILE_M);

    fused_ffn_kernel<<<grid, block>>>(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        ln_w.data_ptr<float>(),
        ln_b.data_ptr<float>(),
        w1.data_ptr<float>(),
        b1.data_ptr<float>(),
        w2.data_ptr<float>(),
        b2.data_ptr<float>(),
        M
    );
}
"""

# =========================================================================
# C++ host source (pybind11 glue — infrastructure)
# =========================================================================
fused_ffn_cpp_source = r"""
#include <torch/extension.h>

void fused_ffn_cuda(
    torch::Tensor out,
    const torch::Tensor x,
    const torch::Tensor ln_w,
    const torch::Tensor ln_b,
    const torch::Tensor w1,
    const torch::Tensor b1,
    const torch::Tensor w2,
    const torch::Tensor b2,
    int M);

torch::Tensor fused_ffn_forward(
    torch::Tensor x,
    torch::Tensor ln_w,
    torch::Tensor ln_b,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2)
{
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, S, D)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    int B = x.size(0);
    int S = x.size(1);
    int M = B * S;

    auto x_flat = x.reshape({M, 768}).contiguous();
    auto out_flat = torch::empty({M, 768}, x.options());

    fused_ffn_cuda(out_flat, x_flat, ln_w, ln_b, w1, b1, w2, b2, M);

    return out_flat.reshape({B, S, 768});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_ffn_forward, "Fused FFN (LN + FC + GELU + FC + residual)");
}
"""

# =========================================================================
# Python wrapper
# =========================================================================
_module = None


def _get_module():
    global _module
    if _module is None:
        # nvcc 12.0 only supports up to compute_90, but the GPU is Blackwell (12.0).
        # Override auto-detection to compile for a compatible arch (Hopper sm_90).
        if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
            os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

        _module = load_inline(
            name='fused_ffn',
            cpp_sources=[fused_ffn_cpp_source],
            cuda_sources=[fused_ffn_cuda_source],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=True,
        )
    return _module


def fused_mlp(h, i, weights_dict):
    """Fused FFN: LayerNorm -> c_fc -> GELU -> c_proj -> +residual.

    Same interface as simple.mlp.mlp().
    """
    device = h.device

    ln2_w = weights_dict[f'transformer.h.{i}.ln_2.weight'].to(device=device, dtype=torch.float32)
    ln2_b = weights_dict[f'transformer.h.{i}.ln_2.bias'].to(device=device, dtype=torch.float32)
    c_fc_w = weights_dict[f'transformer.h.{i}.mlp.c_fc.weight'].to(device=device, dtype=torch.float32)
    c_fc_b = weights_dict[f'transformer.h.{i}.mlp.c_fc.bias'].to(device=device, dtype=torch.float32)
    c_proj_w = weights_dict[f'transformer.h.{i}.mlp.c_proj.weight'].to(device=device, dtype=torch.float32)
    c_proj_b = weights_dict[f'transformer.h.{i}.mlp.c_proj.bias'].to(device=device, dtype=torch.float32)
    h = h.to(dtype=torch.float32).contiguous()

    mod = _get_module()
    return mod.forward(h, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_w, c_proj_b)
