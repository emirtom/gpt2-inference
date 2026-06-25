#include <cuda_runtime.h>
#include <cstdio>
#include <torch/extension.h>

// ============================================================================
// Phase 1: LayerNorm
// ============================================================================
// Computes per-row mean and inv_std for 64 rows, stores at smem[2048..2175].
//
// smem layout during this phase:
//   [w*128 + r]      = warp w partial sum_x  for row r
//   [w*128 + 64 + r] = warp w partial sum_x2 for row r
//   [r]              = final mean for row r        (after cross-warp reduction)
//   [64 + r]         = final inv_std for row r
//   [2048 + r]       = saved mean (after copy)
//   [2112 + r]       = saved inv_std
__device__ void phase1_layernorm(
    float* smem,
    const float* __restrict__ x,
    int block_row_start,
    int col,
    int M)
{
    float sum_x[64] = {0};
    float sum_x2[64] = {0};

    // 1a. Load input, compute per-row partial sums
    for (int r = 0; r < 64; r++) {
        int global_row = block_row_start + r;
        if (global_row < M) {
            for (int i = 0; i < 3; i++) {
                float val = x[global_row * 768 + col + i];
                sum_x[r]  += val;
                sum_x2[r] += val * val;
            }
        }
    }

    // 1b. Within-warp reduction (shuffle)
    for (int offset = 16; offset > 0; offset /= 2) {
        for (int r = 0; r < 64; r++) {
            sum_x[r]  += __shfl_xor_sync(0xffffffff, sum_x[r], offset);
            sum_x2[r] += __shfl_xor_sync(0xffffffff, sum_x2[r], offset);
        }
    }

    // 1c. Lane 0 of each warp writes its partials to shared memory
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane_id == 0) {
        int base = warp_id * 128;
        for (int r = 0; r < 64; r++) {
            smem[base + r]      = sum_x[r];
            smem[base + 64 + r] = sum_x2[r];
        }
    }
    __syncthreads();

    // 1d. Cross-warp reduction: warp 0 sums 8 warp partials → mean/inv_std
    if (warp_id == 0) {
        for (int r = 0; r < 64; r++) {
            if (block_row_start + r < M) {
                float total_x = 0.0f, total_x2 = 0.0f;
                for (int w = 0; w < 8; w++) {
                    int base = w * 128;
                    total_x  += smem[base + r];
                    total_x2 += smem[base + 64 + r];
                }
                float mean = total_x / 768.0f;
                float var  = total_x2 / 768.0f - mean * mean;
                float inv_std = rsqrtf(var + 1e-5f);

                smem[r]      = mean;
                smem[64 + r] = inv_std;
            }
        }
    }
    __syncthreads();

    // Save stats to safe region (they'd be overwritten by Phase 2 tiles)
    if (threadIdx.x < 128) {
        smem[2048 + threadIdx.x] = smem[threadIdx.x];
    }
    __syncthreads();
}

// ============================================================================
// Phase 2a+b: First matmul + GELU for one intermediate tile.
// ============================================================================
// Computes inter_tile[16][64] = GELU(x_ln @ w1[:, n:n+64] + b1[n:n+64]).
// x_tile goes to smem[0..1023], inter_tile to smem[1024..2047].
// LN stats read from smem[2048..2175].
//
//   n: start of the intermediate-dim tile (0, 64, ..., 3024)
__device__ void phase2_compute_inter(
    float* smem,
    const float* __restrict__ x,
    const float* __restrict__ ln_w,
    const float* __restrict__ ln_b,
    const float* __restrict__ w1,
    const float* __restrict__ b1,
    int block_row_start,
    int r_start,
    int M,
    int n)
{
    const int SUB_M = 16;

    // Zero inter_tile
    for (int i = threadIdx.x; i < SUB_M * 64; i += 256) {
        smem[1024 + i] = 0.0f;
    }
    __syncthreads();

    // Tile over input dimension (768)
    for (int k = 0; k < 768; k += 64) {

        // Load x_tile[SUB_M][64] into smem[0..1023], apply LN on the fly
        for (int i = threadIdx.x; i < SUB_M * 64; i += 256) {
            int local_r = i / 64;
            int kk  = i % 64;
            int global_r = block_row_start + r_start + local_r;
            if (global_r < M && (k + kk) < 768) {
                int c = k + kk;
                float val = x[global_r * 768 + c];
                float mean    = smem[2048 + r_start + local_r];
                float inv_std = smem[2112 + r_start + local_r];
                smem[i] = (val - mean) * inv_std * ln_w[c] + ln_b[c];
            } else {
                smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // x_tile @ w1[k:k+64, n:n+64] → accumulate into inter_tile
        for (int i = threadIdx.x; i < SUB_M * 64; i += 256) {
            int local_r = i / 64;
            int nn = i % 64;
            float sum = 0.0f;
            for (int kk = 0; kk < 64; kk++) {
                sum += smem[local_r * 64 + kk] * w1[(k + kk) * 3072 + (n + nn)];
            }
            smem[1024 + i] += sum;
        }
        __syncthreads();
    }

    // Add bias + GELU
    for (int i = threadIdx.x; i < SUB_M * 64; i += 256) {
        int nn = i % 64;
        float val = smem[1024 + i] + b1[n + nn];
        float tmp = 0.79788456f * val * (1.0f + 0.044715f * val * val);
        smem[1024 + i] = 0.5f * val * (1.0f + tanhf(tmp));
    }
    __syncthreads();
}

// ============================================================================
// Phase 2c: Second matmul accumulation.
// ============================================================================
// out_reg += inter_tile @ w2[n:n+64, :]
// inter_tile is read from smem[1024..2047], w2 from global.
__device__ void phase2_second_matmul(
    float (*out_reg)[3],
    const float* smem,
    const float* __restrict__ w2,
    int col,
    int n)
{
    for (int local_r = 0; local_r < 16; local_r++) {
        for (int ci = 0; ci < 3; ci++) {
            int c = col + ci;
            float sum = 0.0f;
            for (int nn = 0; nn < 64; nn++) {
                sum += smem[1024 + local_r * 64 + nn] * w2[(n + nn) * 768 + c];
            }
            out_reg[local_r][ci] += sum;
        }
    }
}

// ============================================================================
// Phase 3: Write output with b2 bias and residual.
// ============================================================================
__device__ void phase3_write_output(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ b2,
    float (*out_reg)[3],
    int block_row_start,
    int r_start,
    int col,
    int M)
{
    for (int local_r = 0; local_r < 16; local_r++) {
        int global_r = block_row_start + r_start + local_r;
        if (global_r < M) {
            for (int ci = 0; ci < 3; ci++) {
                int c = col + ci;
                out[global_r * 768 + c] = out_reg[local_r][ci] + b2[c] + x[global_r * 768 + c];
            }
        }
    }
}

// ============================================================================
// Main kernel
// ============================================================================
__global__ void fused_ffn_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ ln_w,
    const float* __restrict__ ln_b,
    const float* __restrict__ w1,
    const float* __restrict__ b1,
    const float* __restrict__ w2,
    const float* __restrict__ b2,
    int M)
{
    __shared__ float smem[4096];
    int block_row_start = blockIdx.x * 64;
    int col = threadIdx.x * 3;

    phase1_layernorm(smem, x, block_row_start, col, M);

    for (int r_start = 0; r_start < 64; r_start += 16) {
        float out_reg[16][3] = {{0}};

        for (int n = 0; n < 3072; n += 64) {
            phase2_compute_inter(smem, x, ln_w, ln_b, w1, b1,
                                 block_row_start, r_start, M, n);
            phase2_second_matmul(out_reg, smem, w2, col, n);
            __syncthreads();
        }

        phase3_write_output(out, x, b2, out_reg, block_row_start, r_start, col, M);
    }
}

// ============================================================================
// Host launcher
// ============================================================================
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel sync error: %s\n", cudaGetErrorString(err));
    }
}
