/******************************************************************************
 * Per-Dimension Batched Newton-Schulz Backward Kernel
 * For MuonMamba: Exact backward pass through per-dimension NS
 * Using float32 precision throughout for maximum numerical stability
 * Copyright (c) 2024
 ******************************************************************************/

 #pragma once

 #include <c10/util/BFloat16.h>
 #include <cuda_runtime.h>
 #include <c10/cuda/CUDAException.h>
 #include <algorithm>
 #include "selective_scan.h"
 
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  // Block reduction helper
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  template<int BLOCK_SIZE>
//  __device__ __forceinline__ float blockReduceSum_perdim(float val) {
//      __shared__ float shared[BLOCK_SIZE];
//      shared[threadIdx.x] = val;
//      __syncthreads();
     
//      for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
//          if (threadIdx.x < stride) {
//              shared[threadIdx.x] += shared[threadIdx.x + stride];
//          }
//          __syncthreads();
//      }
     
//      return shared[0];
//  }
 
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  // Kernel 1: Recompute forward intermediates (A, B) for all dimensions
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  template<int kBlockSize = 256>
//  __global__ void recompute_ns_intermediates_batched(
//      const float* __restrict__ V_raw,       // [B, D, L, N]
//      float* __restrict__ norms_out,         // [D]
//      float* __restrict__ A_out,             // [D, work_M, work_M] where work_M = min(M, N)
//      float* __restrict__ B_out,             // [D, work_M, work_M]
//      int B, int D, int L, int N
//  ) {
//      const int tid = threadIdx.x;
//      const int dim_id = blockIdx.x;  // Each block handles one dimension
     
//      if (dim_id >= D) return;
     
//      const int M = B * L;  // Total rows for this dimension
//      const int work_M = min(M, N);  // Smaller dimension
//      const int work_N = max(M, N);  // Larger dimension
//      const bool transposed = (M > N);
     
//      // Ensure work_M <= 64 for shared memory efficiency
//      if (work_M > 64) return;
     
//      // Coefficients for Newton-Schulz 1-step
//      constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
     
//      // Shared memory for intermediate computations
//      extern __shared__ float smem[];
//      float* smem_A = smem;
//      float* smem_A2 = smem_A + work_M * work_M;
     
//      // === STEP 1: Compute Frobenius norm ===
//      float local_sum = 0.0f;
//      for (int idx = tid; idx < M * N; idx += kBlockSize) {
//          int row = idx / N;
//          int col = idx % N;
         
//          if (row >= M || col >= N) continue;
         
//          int batch_idx = row / L;
//          int seq_idx = row % L;
//          int global_idx = batch_idx * D * L * N + dim_id * L * N + seq_idx * N + col;
         
//          if (global_idx >= B * D * L * N) continue;
         
//         float val = V_raw[global_idx];
//         // CRITICAL: Check for NaN/Inf before computing norm
//         if (!isfinite(val)) {
//             val = 0.0f;  // Replace NaN/Inf with 0
//         }
//         local_sum += val * val;
//     }
    
//     float norm_sq = blockReduceSum_perdim<kBlockSize>(local_sum);
    
//     // CRITICAL: Check for NaN/Inf in norm_sq before sqrt
//     if (!isfinite(norm_sq)) {
//         norm_sq = 0.0f;
//     }
    
//     // Broadcast norm via shared memory first, then write to global
//     __shared__ float shared_norm;
//     if (tid == 0) {
//         float norm_val;
//         // CRITICAL FIX: Use larger epsilon and clamp norm to prevent very small values
//         // BF16 rounding can make b_t values very small or zero, causing norm to be tiny
//         if (norm_sq < 1e-10f) {
//             // All b_t values rounded to zero or extremely small - use safe default
//             norm_val = 1e-4f;  // Safe minimum norm
//         } else {
//             norm_val = sqrtf(norm_sq + 1e-5f);  // Add epsilon INSIDE sqrt (more stable)
//             norm_val = fmaxf(norm_val, 1e-4f);  // Clamp to minimum to prevent overflow
//         }
//         // CRITICAL: Ensure norm is finite
//         if (!isfinite(norm_val)) {
//             norm_val = 1e-4f;  // Fallback to safe minimum
//         }
//         shared_norm = norm_val;
//         norms_out[dim_id] = norm_val;
//     }
//      __syncthreads();  // Broadcast norm to all threads
     
//      // Now ALL threads have correct norm from shared memory
//      float norm = shared_norm;
     
//      // === STEP 2: Compute covariance matrix A ===
//      // Initialize A to zero
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          smem_A[idx] = 0.0f;
//      }
//      __syncthreads();
     
//      // Compute A using efficient thread mapping
//      const int total_elements = work_M * work_M;
//      for (int idx = tid; idx < total_elements; idx += kBlockSize) {
//          const int i = idx / work_M;
//          const int j = idx % work_M;
         
//          if (i >= work_M || j >= work_M) continue;
         
//          float sum = 0.0f;
         
//          for (int k = 0; k < work_N; ++k) {
//              float g_ik, g_jk;
//              int idx_i, idx_j;
             
//              if (transposed) {
//                  // Tall: A[i,j] = sum_k G[k,i] * G[k,j] (i,j cols 0..N-1, k rows 0..M-1)
//                  int batch_k = k / L;
//                  int seq_k = k % L;
//                  idx_i = batch_k * D * L * N + dim_id * L * N + seq_k * N + i;
//                  idx_j = batch_k * D * L * N + dim_id * L * N + seq_k * N + j;
//              } else {
//                  // Fat: A[i,j] = sum_k G[i,k] * G[j,k] (i,j rows 0..M-1, k cols 0..N-1)
//                  int batch_i = i / L;
//                  int seq_i = i % L;
//                  int batch_j = j / L;
//                  int seq_j = j % L;
//                  idx_i = batch_i * D * L * N + dim_id * L * N + seq_i * N + k;
//                  idx_j = batch_j * D * L * N + dim_id * L * N + seq_j * N + k;
//              }
             
//              // Skip if out of bounds
//              if (idx_i >= B * D * L * N || idx_j >= B * D * L * N) continue;
             
//              g_ik = V_raw[idx_i] / norm;
//              g_jk = V_raw[idx_j] / norm;
//              sum += g_ik * g_jk;
//          }
         
//          smem_A[i * work_M + j] = sum;
//      }
//      __syncthreads();
     
//      // === STEP 3: Compute A^2 ===
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          int i = idx / work_M;
//          int j = idx % work_M;
//          float sum = 0.0f;
         
//          for (int k = 0; k < work_M; ++k) {
//              sum += smem_A[i * work_M + k] * smem_A[k * work_M + j];
//          }
//          smem_A2[idx] = sum;
//      }
//      __syncthreads();
     
//      // === STEP 4: Compute B = b*A + c*A^2 ===
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          int global_idx = dim_id * work_M * work_M + idx;
//          B_out[global_idx] = b * smem_A[idx] + c * smem_A2[idx];
//          A_out[global_idx] = smem_A[idx];  // Store A for backward pass
//      }
//  }
 
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  // Kernel 2: Backward through polynomial and matmul to get dG_norm (COMPLETE with Term2)
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  template<int kBlockSize = 256>
//  __global__ void ns_backward_to_normalized_batched(
//      const float* __restrict__ V_raw,       // [B, D, L, N]
//      const float* __restrict__ norms,       // [D]
//      const float* __restrict__ A_matrices,  // [D, work_M, work_M]
//      const float* __restrict__ B_matrices,  // [D, work_M, work_M]
//      const float* __restrict__ dV_ortho,    // [B, D, L, N]
//      float* __restrict__ dG_norm,           // [B, D, L, N] - intermediate gradient
//      int B, int D, int L, int N
//  ) {
//      const int tid = threadIdx.x;
//      const int dim_id = blockIdx.x;  // Each block handles one dimension
     
//      if (dim_id >= D) return;
     
//      const int M = B * L;  // Total rows for this dimension
//      const float norm = norms[dim_id];
//      const int work_M = min(M, N);
//      const int work_N = max(M, N);
//      const bool transposed = (M > N);
     
//      // Ensure work_M <= 64 for shared memory efficiency
//      if (work_M > 64) return;
     
//      constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
     
//      // Shared memory layout: [A, B, M, H, S] all work_M × work_M
//      extern __shared__ float smem[];
//      float* smem_A = smem;
//      float* smem_B = smem_A + work_M * work_M;
//      float* smem_M = smem_B + work_M * work_M;
//      float* smem_H = smem_M + work_M * work_M;
//      float* smem_S = smem_H + work_M * work_M;
     
//      // Load A and B matrices for this dimension
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          smem_A[idx] = A_matrices[dim_id * work_M * work_M + idx];
//          smem_B[idx] = B_matrices[dim_id * work_M * work_M + idx];
//      }
//      __syncthreads();
     
//      // === STEP 1: Compute M = dG' @ G_norm^T [work_M, work_M] ===
//      // Only needed for Term2
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          int i = idx / work_M;
//          int j = idx % work_M;
//          float sum = 0.0f;
         
//          for (int k = 0; k < work_N; ++k) {
//              float dG_ik, G_jk;
//              int dG_idx, G_idx;
             
//              if (transposed) {
//                  // Tall: M[i,j] = sum_k dG'[k,i] * G[k,j]
//                  int batch_k = k / L;
//                  int seq_k = k % L;
//                  dG_idx = batch_k * D * L * N + dim_id * L * N + seq_k * N + i;
//                  G_idx = batch_k * D * L * N + dim_id * L * N + seq_k * N + j;
//              } else {
//                  // Fat: M[i,j] = sum_k dG'[i,k] * G[j,k]
//                  int batch_i = i / L;
//                  int seq_i = i % L;
//                  int batch_j = j / L;
//                  int seq_j = j % L;
//                  dG_idx = batch_i * D * L * N + dim_id * L * N + seq_i * N + k;
//                  G_idx = batch_j * D * L * N + dim_id * L * N + seq_j * N + k;
//              }
             
//              if (dG_idx < B * D * L * N && G_idx < B * D * L * N) {
//                  dG_ik = dV_ortho[dG_idx];
//                  G_jk = V_raw[G_idx] / norm;  // Normalized
//                  sum += dG_ik * G_jk;
//              }
//          }
         
//          // Safety check for M matrix
//          smem_M[idx] = isfinite(sum) ? sum : 0.0f;
//      }
//      __syncthreads();
     
//      // === STEP 2: Compute H = b*M + c*(M@A + A@M) [work_M, work_M] ===
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          int i = idx / work_M;
//          int j = idx % work_M;
         
//          // Compute (M@A)[i,j]
//          float MA_ij = 0.0f;
//          for (int k = 0; k < work_M; ++k) {
//              MA_ij += smem_M[i * work_M + k] * smem_A[k * work_M + j];
//          }
         
//          // Compute (A@M)[i,j]
//          float AM_ij = 0.0f;
//          for (int k = 0; k < work_M; ++k) {
//              AM_ij += smem_A[i * work_M + k] * smem_M[k * work_M + j];
//          }
         
//          float H_val = b * smem_M[idx] + c * (MA_ij + AM_ij);
//          smem_H[idx] = isfinite(H_val) ? H_val : 0.0f;
//      }
//      __syncthreads();
     
//      // === STEP 3: Compute S = H + H^T [work_M, work_M] ===
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          int i = idx / work_M;
//          int j = idx % work_M;
//          smem_S[i * work_M + j] = smem_H[i * work_M + j] + smem_H[j * work_M + i];
//      }
//      __syncthreads();
     
//      // === STEP 4: Compute final dG_norm = Term1 + Term2 ===
//      // Term1 = a*dG' + B @ dG' (or dG' @ B for tall)
//      // Term2 = S @ G_norm (or G_norm @ S for tall)
//      for (int idx = tid; idx < M * N; idx += kBlockSize) {
//          int row = idx / N;
//          int col = idx % N;
         
//          if (row >= M || col >= N) continue;
         
//          int batch_idx = row / L;
//          int seq_idx = row % L;
//          int input_idx = batch_idx * D * L * N + dim_id * L * N + seq_idx * N + col;
         
//          if (input_idx >= B * D * L * N) continue;
         
//          float dG_prime = dV_ortho[input_idx];
         
//          // Term1: B @ dG' (fat) or dG' @ B (tall)
//          float term1 = 0.0f;
//          for (int k = 0; k < work_M; ++k) {
//              float b_val, dG_val;
//              int dG_idx;
             
//              if (transposed) {
//                  int batch_row = row / L;
//                  int seq_row = row % L;
//                  dG_idx = batch_row * D * L * N + dim_id * L * N + seq_row * N + k;
//                  b_val = smem_B[k * work_M + col];
//              } else {
//                  int batch_k = k / L;
//                  int seq_k = k % L;
//                  dG_idx = batch_k * D * L * N + dim_id * L * N + seq_k * N + col;
//                  b_val = smem_B[row * work_M + k];
//              }
             
//              if (dG_idx < B * D * L * N) {
//                  dG_val = dV_ortho[dG_idx];
//                  term1 += b_val * dG_val;
//              }
//          }
         
//          // Term2: S @ G_norm (fat) or G_norm @ S (tall)
//          float term2 = 0.0f;
//          for (int k = 0; k < work_M; ++k) {
//              float s_val, G_val;
//              int G_idx;
             
//              if (transposed) {
//                  int batch_row = row / L;
//                  int seq_row = row % L;
//                  G_idx = batch_row * D * L * N + dim_id * L * N + seq_row * N + k;
//                  s_val = smem_S[k * work_M + col];
//              } else {
//                  int batch_k = k / L;
//                  int seq_k = k % L;
//                  G_idx = batch_k * D * L * N + dim_id * L * N + seq_k * N + col;
//                  s_val = smem_S[row * work_M + k];
//              }
             
//              if (G_idx < B * D * L * N) {
//                  G_val = V_raw[G_idx] / norm;  // Normalized
//                  term2 += s_val * G_val;
//              }
//          }
         
//          // Complete backward: dG_norm = a*dG' + Term1 + Term2
//          float result = a * dG_prime + term1 + term2;
//          dG_norm[input_idx] = isfinite(result) ? result : 0.0f;
//      }
//  }
 
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  // Kernel 3: Backward through normalization
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  template<int kBlockSize = 256>
//  __global__ void ns_backward_through_norm_batched(
//      const float* __restrict__ V_raw,       // [B, D, L, N]
//      const float* __restrict__ norms,       // [D]
//      const float* __restrict__ dG_norm,    // [B, D, L, N]
//      float* __restrict__ dV_raw,           // [B, D, L, N]
//      int B, int D, int L, int N
//  ) {
//      const int tid = threadIdx.x;
//      const int dim_id = blockIdx.x;  // Each block handles one dimension
     
//      if (dim_id >= D) return;
     
//      const int M = B * L;  // Total rows for this dimension
//      const float norm = norms[dim_id];
     
//      // Compute dot product <G, dG_norm> for normalization backward
//      // CRITICAL FIX: Use RAW (unnormalized) values for correct chain rule!
//      // Formula: dG = dG_norm / ||G|| - G * <G, dG_norm> / ||G||^3
//      float local_dot = 0.0f;
//      for (int idx = tid; idx < M * N; idx += kBlockSize) {
//          int row = idx / N;
//          int col = idx % N;
         
//          if (row >= M || col >= N) continue;
         
//          int batch_idx = row / L;
//          int seq_idx = row % L;
//          int global_idx = batch_idx * D * L * N + dim_id * L * N + seq_idx * N + col;
         
//          if (global_idx >= B * D * L * N) continue;
         
//          float g_raw_val = V_raw[global_idx];  // Use RAW value (not normalized)!
//          float dG_norm_val = dG_norm[global_idx];
//          local_dot += g_raw_val * dG_norm_val;  // <G, dG_norm>
//      }
     
//      float dot_product = blockReduceSum_perdim<kBlockSize>(local_dot);
     
//      // Broadcast dot_product to all threads via shared memory
//      __shared__ float shared_dot;
//      if (tid == 0) {
//          shared_dot = dot_product;
//      }
//      __syncthreads();
//      dot_product = shared_dot;
     
//      // Apply normalization backward: dG = dG_norm / norm - G * (G : dG_norm) / norm^3
//      for (int idx = tid; idx < M * N; idx += kBlockSize) {
//          int row = idx / N;
//          int col = idx % N;
         
//          if (row >= M || col >= N) continue;
         
//          int batch_idx = row / L;
//          int seq_idx = row % L;
//          int global_idx = batch_idx * D * L * N + dim_id * L * N + seq_idx * N + col;
         
//          if (global_idx >= B * D * L * N) continue;
         
//          float g_norm_val = V_raw[global_idx] / norm;  // Normalized value
//          float dG_norm_val = dG_norm[global_idx];
         
//          // Backward through normalization with safety clamp
//          float norm_safe = fmaxf(norm, 1e-4f);  // Prevent division by very small numbers
//          float dG_val = dG_norm_val / norm_safe - g_norm_val * dot_product / (norm_safe * norm_safe);
//          dV_raw[global_idx] = dG_val;
//      }
//  }
 
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  // Launch wrapper
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  inline void launch_newton_schulz_per_dim_backward(
//      const float* V_raw,   // [batch, dim, seqlen, dstate]
//      const float* dV_ortho, // [batch, dim, seqlen, dstate]
//      float* dV_raw,        // [batch, dim, seqlen, dstate]
//      int batch, int dim, int seqlen, int dstate,
//      cudaStream_t stream
//  ) {
//      constexpr int kBlockSize = 256;
     
//      // Determine work matrix size (min(M, N) where M = batch * seqlen)
//      const int M = batch * seqlen;
//      const int work_M = (M > dstate) ? dstate : M;
     
//      // For very large matrices, fall back to identity transform
//      if (work_M > 64) {
//          cudaMemcpyAsync(dV_raw, dV_ortho, batch * dim * seqlen * dstate * sizeof(float), 
//                         cudaMemcpyDeviceToDevice, stream);
//          return;
//      }
     
//      // Allocate temporary buffers with correct sizes
//      float *norms, *A_matrices, *B_matrices, *dG_norm;
//      C10_CUDA_CHECK(cudaMalloc(&norms, dim * sizeof(float)));
//      C10_CUDA_CHECK(cudaMalloc(&A_matrices, dim * work_M * work_M * sizeof(float)));
//      C10_CUDA_CHECK(cudaMalloc(&B_matrices, dim * work_M * work_M * sizeof(float)));
//      C10_CUDA_CHECK(cudaMalloc(&dG_norm, batch * dim * seqlen * dstate * sizeof(float)));
     
//      // Kernel 1: Recompute forward intermediates
//      int smem_size_intermediates = 2 * work_M * work_M * sizeof(float);
//      recompute_ns_intermediates_batched<kBlockSize><<<dim, kBlockSize, smem_size_intermediates, stream>>>(
//          V_raw, norms, A_matrices, B_matrices, batch, dim, seqlen, dstate
//      );
//      C10_CUDA_KERNEL_LAUNCH_CHECK();
     
//      // Kernel 2: Backward through polynomial and matmul (needs 5 matrices: A, B, M, H, S)
//      int smem_size_backward = 5 * work_M * work_M * sizeof(float);
//      ns_backward_to_normalized_batched<kBlockSize><<<dim, kBlockSize, smem_size_backward, stream>>>(
//          V_raw, norms, A_matrices, B_matrices, dV_ortho, dG_norm, batch, dim, seqlen, dstate
//      );
//      C10_CUDA_KERNEL_LAUNCH_CHECK();
     
//      // Kernel 3: Backward through normalization
//      ns_backward_through_norm_batched<kBlockSize><<<dim, kBlockSize, 0, stream>>>(
//          V_raw, norms, dG_norm, dV_raw, batch, dim, seqlen, dstate
//      );
//      C10_CUDA_KERNEL_LAUNCH_CHECK();
     
//     // Clean up temporary memory
//     C10_CUDA_CHECK(cudaFree(norms));
//     C10_CUDA_CHECK(cudaFree(A_matrices));
//     C10_CUDA_CHECK(cudaFree(B_matrices));
//     C10_CUDA_CHECK(cudaFree(dG_norm));
// }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Velocity 5-Step Newton-Schulz Backward Pass
// Computes gradients through last NS iteration only (first 4 steps detached)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper: Convert float to bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16_ns(float x) {
    return __float2bfloat16(x);
}

// Helper: Convert bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float_ns(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Helper: Convert weight_t to float (handles complex)
template <typename T>
__device__ __forceinline__ float to_float_ns(T x) {
    return float(x);
}

template <typename T>
__device__ __forceinline__ float to_float_ns(c10::complex<T> x) {
    return float(x.real());  // For complex, use real part
}

// Helper: Reinterpret float as bfloat16 without rounding
__device__ __forceinline__ __nv_bfloat16 float_to_bf16_reinterpret_ns(float f) {
    unsigned int f_bits = __float_as_uint(f);
    unsigned short bf16_raw = static_cast<unsigned short>(f_bits >> 16);
    unsigned int reconstructed = static_cast<unsigned int>(bf16_raw) << 16;
    float bf16_as_fp32 = __uint_as_float(reconstructed);
    return __float2bfloat16(bf16_as_fp32);
}

template<typename input_t, typename weight_t, int kBlockSize = 256, int kTileSize = 64>
__global__ void newton_schulz_velocity_5step_backward_kernel(
    const float* __restrict__ grad_output,  // [B, D, L, N] - gradient from scan
    const input_t* __restrict__ u,          // [B, D, L] - input (for recomputation)
    const input_t* __restrict__ delta,      // [B, D, L] - input
    const weight_t* __restrict__ B,         // [D, N] or [B, G, L, N] if variable
    float* __restrict__ grad_u,             // [B, D, L] - output gradient
    float* __restrict__ grad_delta,         // [B, D, L] - output gradient
    float* __restrict__ grad_B,             // [D, N] or [B, G, L, N] - output gradient
    float* __restrict__ X_temp,             // [B, D, L, N] - temporary buffer for X_4
    float* __restrict__ dX_4_temp,          // [B, D, L, N] - temporary buffer for dX_4
    float alpha,
    int B_dim, int D, int L, int dstate, int t_start,
    int u_batch_stride, int u_d_stride,
    int delta_batch_stride, int delta_d_stride,
    int B_batch_stride, int B_group_stride,
    int B_d_stride, int B_dstate_stride,
    bool is_variable_B, int n_groups
) {
    const int batch_idx = blockIdx.x;
    const int time_local = blockIdx.y;
    const int time_idx = t_start + time_local;
    
    if (batch_idx >= B_dim || time_idx >= L) return;
    
    const int tid = threadIdx.x;
    
    // Newton-Schulz coefficients
    constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    
    // Determine transpose
    const bool transposed = (D > dstate);
    const int gram_size = transposed ? dstate : D;
    
    // Shared memory layout
    extern __shared__ float smem[];
    __nv_bfloat16* tile_buffer_bf16 = (__nv_bfloat16*)smem;
    const int tile_buffer_size = kTileSize * (transposed ? D : dstate);
    float* gram_A_fp32 = (float*)(tile_buffer_bf16 + tile_buffer_size);
    float* partial_sums = gram_A_fp32 + gram_size * gram_size;
    
    // Additional space for gradient accumulators (reuse after forward recomputation)
    // Allocated as max(gram_size_sq, 2*kTileSize) floats - used for grad_u_partial and grad_delta_partial
    float* dX_accumulator = partial_sums + kBlockSize;
    
    // ========== PHASE 1: Recompute X_0 → X_4 (Detached, 4 iterations) ==========
    
    // Step 1: Compute b_t, convert to BF16, compute norm
    float norm_sq_local = 0.0f;
    
    for (int d_start = 0; d_start < D; d_start += kTileSize) {
        const int d_end = min(d_start + kTileSize, D);
        const int tile_rows = d_end - d_start;
        
        for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
            const int local_row = idx / dstate;
            const int col = idx % dstate;
            const int global_row = d_start + local_row;
            
            int u_idx = batch_idx * u_batch_stride + global_row * u_d_stride + time_idx;
            float u_val = to_float_ns(u[u_idx]);
            
            int delta_idx = batch_idx * delta_batch_stride + global_row * delta_d_stride + time_idx;
            float delta_val = to_float_ns(delta[delta_idx]);
            
            float B_val;
            if (!is_variable_B) {
                B_val = to_float_ns(B[global_row * B_d_stride + col * B_dstate_stride]);
            } else {
                // Variable B: [B, G, N, L] - FIXED: Use correct indexing
                int group_size = (D + n_groups - 1) / n_groups;
                int group_id = min(global_row / group_size, n_groups - 1);
                // For [B, G, N, L]: B[b, g, n, t] = base + n * B_dstate_stride + t
                B_val = to_float_ns(B[batch_idx * B_batch_stride + 
                                   group_id * B_group_stride +
                                   col * B_dstate_stride + time_idx]);
            }
            
            float b_t_val = alpha * delta_val * B_val * u_val;
            __nv_bfloat16 b_t_bf16 = __float2bfloat16(b_t_val);
            float b_t_rounded = __bfloat162float(b_t_bf16);
            
            norm_sq_local += b_t_rounded * b_t_rounded;
            
            // Store in tile buffer temporarily
            tile_buffer_bf16[local_row * dstate + col] = b_t_bf16;
        }
    }
    
    // Block reduction for norm
    partial_sums[tid] = norm_sq_local;
    __syncthreads();
    
    for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // CRITICAL FIX: Use larger epsilon and clamp norm to prevent very small values
    // BF16 rounding can make b_t values very small or zero, causing norm to be tiny
    // Very small norms cause overflow when dividing in normalization backward
    // __shared__ float shared_norm;
    // if (tid == 0) {
    //     float norm_sq = partial_sums[0];
    //     float norm_val;
    //     if (norm_sq < 1e-10f) {
    //         // All b_t values rounded to zero or extremely small - use safe default
    //         norm_val = 1e-4f;  // Safe minimum norm
    //     } else {
    //         norm_val = sqrtf(norm_sq + 1e-5f);  // Larger epsilon (1e-5 instead of 1e-8)
    //         norm_val = fmaxf(norm_val, 1e-4f);  // Clamp to minimum to prevent overflow
    //     }
    //     shared_norm = norm_val;
    // }
    __shared__ float shared_norm;
    if (tid == 0) {
        // This logic MUST match the forward pass exactly
        shared_norm = sqrtf(partial_sums[0] + 1e-8f); 
    }
    __syncthreads();

    // 'norm' is the original, recomputed value (can be small)
    float norm = shared_norm; 

    // 'norm_safe' is clamped for safe division *later* in the gradient calculation
    float norm_safe = fmaxf(norm, 1e-4f);

    
    // Step 2: Normalize to get X_0, store in global memory
    for (int d_start = 0; d_start < D; d_start += kTileSize) {
        const int d_end = min(d_start + kTileSize, D);
        const int tile_rows = d_end - d_start;
        
        for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
            const int local_row = idx / dstate;
            const int col = idx % dstate;
            const int global_row = d_start + local_row;
            
            __nv_bfloat16 b_t_bf16 = tile_buffer_bf16[local_row * dstate + col];
            float normalized = __bfloat162float(b_t_bf16) / norm;
            __nv_bfloat16 normalized_bf16 = __float2bfloat16(normalized);
            float normalized_as_float = __bfloat162float(normalized_bf16);
            
            int buffer_idx = batch_idx * D * L * dstate + 
                            global_row * L * dstate + 
                            time_idx * dstate + col;
            X_temp[buffer_idx] = normalized_as_float;
        }
    }
    __syncthreads();
    
    // Step 3: Run 4 NS iterations (same as forward, but only 4 iterations)
    for (int step = 0; step < 4; ++step) {
        // Compute A = X @ X.T
        for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
            gram_A_fp32[idx] = 0.0f;
        }
        __syncthreads();
        
        if (!transposed) {
            for (int d_start = 0; d_start < D; d_start += kTileSize) {
                const int d_end = min(d_start + kTileSize, D);
                const int tile_rows = d_end - d_start;
                
                for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                    const int local_row = idx / dstate;
                    const int col = idx % dstate;
                    const int global_row = d_start + local_row;
                    
                    int buffer_idx = batch_idx * D * L * dstate + 
                                    global_row * L * dstate + 
                                    time_idx * dstate + col;
                    float stored_val = X_temp[buffer_idx];
                    tile_buffer_bf16[local_row * dstate + col] = float_to_bf16_reinterpret_ns(stored_val);
                }
                __syncthreads();
                
                for (int ij = tid; ij < tile_rows * gram_size; ij += kBlockSize) {
                    const int local_i = ij / gram_size;
                    const int j = ij % gram_size;
                    const int global_i = d_start + local_i;
                    
                    if (global_i < gram_size && j < gram_size) {
                        float sum = 0.0f;
                        if (j >= d_start && j < d_end) {
                            for (int k = 0; k < dstate; ++k) {
                                float a_val = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
                                float b_val = __bfloat162float(tile_buffer_bf16[(j - d_start) * dstate + k]);
                                sum += a_val * b_val;
                            }
                        } else {
                            for (int k = 0; k < dstate; ++k) {
                                int j_idx = batch_idx * D * L * dstate + j * L * dstate + time_idx * dstate + k;
                                float a_val = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
                                float b_val = X_temp[j_idx];
                                sum += a_val * b_val;
                            }
                        }
                        atomicAdd(&gram_A_fp32[global_i * gram_size + j], sum);
                    }
                }
                __syncthreads();
            }
        } else {
            for (int d_start = 0; d_start < D; d_start += kTileSize) {
                const int d_end = min(d_start + kTileSize, D);
                const int tile_cols = d_end - d_start;
                
                for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
                    const int row = idx / tile_cols;
                    const int local_col = idx % tile_cols;
                    const int global_col = d_start + local_col;
                    
                    int buffer_idx = batch_idx * D * L * dstate + 
                                    global_col * L * dstate + 
                                    time_idx * dstate + row;
                    float stored_val = X_temp[buffer_idx];
                    tile_buffer_bf16[row * tile_cols + local_col] = float_to_bf16_reinterpret_ns(stored_val);
                }
                __syncthreads();
                
                for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
                    const int i = ij / gram_size;
                    const int j = ij % gram_size;
                    
                    float sum = 0.0f;
                    for (int k = 0; k < tile_cols; ++k) {
                        float a_val = __bfloat162float(tile_buffer_bf16[i * tile_cols + k]);
                        float b_val = __bfloat162float(tile_buffer_bf16[j * tile_cols + k]);
                        sum += a_val * b_val;
                    }
                    atomicAdd(&gram_A_fp32[ij], sum);
                }
                __syncthreads();
            }
        }
        
        // Convert A to BF16, compute A², then B = b*A + c*A²
        const int gram_storage_needed = 2 * gram_size * gram_size;
        __nv_bfloat16* gram_A_bf16 = tile_buffer_bf16;
        
        for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
            gram_A_bf16[ij] = __float2bfloat16(gram_A_fp32[ij]);
        }
        __syncthreads();
        
        __nv_bfloat16* temp_A2_bf16 = gram_A_bf16 + gram_size * gram_size;
        
        for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
            const int i = ij / gram_size;
            const int j = ij % gram_size;
            
            float A2_ij_fp32 = 0.0f;
            for (int k = 0; k < gram_size; ++k) {
                float a_val = __bfloat162float(gram_A_bf16[i * gram_size + k]);
                float b_val = __bfloat162float(gram_A_bf16[k * gram_size + j]);
                A2_ij_fp32 += a_val * b_val;
            }
            temp_A2_bf16[ij] = __float2bfloat16(A2_ij_fp32);
        }
        __syncthreads();
        
        for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
            float A_ij = __bfloat162float(gram_A_bf16[ij]);
            float A2_ij = __bfloat162float(temp_A2_bf16[ij]);
            float B_fp32 = b * A_ij + c * A2_ij;
            gram_A_bf16[ij] = __float2bfloat16(B_fp32);
        }
        __syncthreads();
        
        // Apply X = a*X + B@X
        __nv_bfloat16* x_tile_buffer = tile_buffer_bf16 + gram_storage_needed;
        
        if (!transposed) {
            for (int d_start = 0; d_start < D; d_start += kTileSize) {
                const int d_end = min(d_start + kTileSize, D);
                const int tile_rows = d_end - d_start;
                
                for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                    const int local_row = idx / dstate;
                    const int col = idx % dstate;
                    const int global_row = d_start + local_row;
                    
                    int buffer_idx = batch_idx * D * L * dstate + 
                                    global_row * L * dstate + 
                                    time_idx * dstate + col;
                    float stored_val = X_temp[buffer_idx];
                    x_tile_buffer[local_row * dstate + col] = float_to_bf16_reinterpret_ns(stored_val);
                }
                __syncthreads();
                
                for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                    const int local_row = idx / dstate;
                    const int col = idx % dstate;
                    const int global_row = d_start + local_row;
                    
                    float x_val = __bfloat162float(x_tile_buffer[local_row * dstate + col]);
                    
                    float sum = 0.0f;
                    for (int k = 0; k < gram_size; ++k) {
                        float x_kj;
                        if (k >= d_start && k < d_end) {
                            x_kj = __bfloat162float(x_tile_buffer[(k - d_start) * dstate + col]);
                        } else {
                            int idx_kj = batch_idx * D * L * dstate + k * L * dstate + time_idx * dstate + col;
                            x_kj = X_temp[idx_kj];
                        }
                        float b_ik = __bfloat162float(gram_A_bf16[global_row * gram_size + k]);
                        sum += b_ik * x_kj;
                    }
                    
                    float x_new_fp32 = a * x_val + sum;
                    __nv_bfloat16 x_new_bf16 = __float2bfloat16(x_new_fp32);
                    float x_new_rounded = __bfloat162float(x_new_bf16);
                    
                    int buffer_idx = batch_idx * D * L * dstate + 
                                    global_row * L * dstate + 
                                    time_idx * dstate + col;
                    X_temp[buffer_idx] = x_new_rounded;
                }
                __syncthreads();
            }
        } else {
            for (int d_start = 0; d_start < D; d_start += kTileSize) {
                const int d_end = min(d_start + kTileSize, D);
                const int tile_cols = d_end - d_start;
                
                for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
                    const int row = idx / tile_cols;
                    const int local_col = idx % tile_cols;
                    const int global_col = d_start + local_col;
                    
                    int buffer_idx = batch_idx * D * L * dstate + 
                                    global_col * L * dstate + 
                                    time_idx * dstate + row;
                    float stored_val = X_temp[buffer_idx];
                    x_tile_buffer[row * tile_cols + local_col] = float_to_bf16_reinterpret_ns(stored_val);
                }
                __syncthreads();
                
                for (int idx = tid; idx < gram_size * tile_cols; idx += kBlockSize) {
                    const int n = idx / tile_cols;
                    const int local_d = idx % tile_cols;
                    const int d = d_start + local_d;
                    
                    float x_val = __bfloat162float(x_tile_buffer[n * tile_cols + local_d]);
                    
                    float sum = 0.0f;
                    for (int k = 0; k < gram_size; ++k) {
                        float x_dk;
                        if (d >= d_start && d < d_end) {
                            x_dk = __bfloat162float(x_tile_buffer[k * tile_cols + local_d]);
                        } else {
                            int idx_dk = batch_idx * D * L * dstate + d * L * dstate + time_idx * dstate + k;
                            x_dk = X_temp[idx_dk];
                        }
                        float b_nk = __bfloat162float(gram_A_bf16[n * gram_size + k]);
                        sum += b_nk * x_dk;
                    }
                    
                    float x_new_fp32 = a * x_val + sum;
                    __nv_bfloat16 x_new_bf16 = __float2bfloat16(x_new_fp32);
                    float x_new_rounded = __bfloat162float(x_new_bf16);
                    
                    int buffer_idx = batch_idx * D * L * dstate + 
                                    d * L * dstate + 
                                    time_idx * dstate + n;
                    X_temp[buffer_idx] = x_new_rounded;
                }
                __syncthreads();
            }
        }
    }
    
    // Now X_temp contains X_4, ready for backward pass through 5th iteration
    
    // ========== PHASE 2: Backward Through 5th Iteration ==========
    
    // Compute A_4 = X_4 @ X_4.T (needed for gradient computation)
    for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
        gram_A_fp32[idx] = 0.0f;
    }
    __syncthreads();
    
    if (!transposed) {
        for (int d_start = 0; d_start < D; d_start += kTileSize) {
            const int d_end = min(d_start + kTileSize, D);
            const int tile_rows = d_end - d_start;
            
            for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                const int local_row = idx / dstate;
                const int col = idx % dstate;
                const int global_row = d_start + local_row;
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_row * L * dstate + 
                                time_idx * dstate + col;
                float stored_val = X_temp[buffer_idx];
                tile_buffer_bf16[local_row * dstate + col] = float_to_bf16_reinterpret_ns(stored_val);
            }
            __syncthreads();
            
            for (int ij = tid; ij < tile_rows * gram_size; ij += kBlockSize) {
                const int local_i = ij / gram_size;
                const int j = ij % gram_size;
                const int global_i = d_start + local_i;
                
                if (global_i < gram_size && j < gram_size) {
                    float sum = 0.0f;
                    if (j >= d_start && j < d_end) {
                        for (int k = 0; k < dstate; ++k) {
                            float a_val = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
                            float b_val = __bfloat162float(tile_buffer_bf16[(j - d_start) * dstate + k]);
                            sum += a_val * b_val;
                        }
                    } else {
                        for (int k = 0; k < dstate; ++k) {
                            int j_idx = batch_idx * D * L * dstate + j * L * dstate + time_idx * dstate + k;
                            float a_val = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
                            float b_val = X_temp[j_idx];
                            sum += a_val * b_val;
                        }
                    }
                    atomicAdd(&gram_A_fp32[global_i * gram_size + j], sum);
                }
            }
            __syncthreads();
        }
    } else {
        for (int d_start = 0; d_start < D; d_start += kTileSize) {
            const int d_end = min(d_start + kTileSize, D);
            const int tile_cols = d_end - d_start;
            
            for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
                const int row = idx / tile_cols;
                const int local_col = idx % tile_cols;
                const int global_col = d_start + local_col;
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_col * L * dstate + 
                                time_idx * dstate + row;
                float stored_val = X_temp[buffer_idx];
                tile_buffer_bf16[row * tile_cols + local_col] = float_to_bf16_reinterpret_ns(stored_val);
            }
            __syncthreads();
            
            for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
                const int i = ij / gram_size;
                const int j = ij % gram_size;
                
                float sum = 0.0f;
                for (int k = 0; k < tile_cols; ++k) {
                    float a_val = __bfloat162float(tile_buffer_bf16[i * tile_cols + k]);
                    float b_val = __bfloat162float(tile_buffer_bf16[j * tile_cols + k]);
                    sum += a_val * b_val;
                }
                atomicAdd(&gram_A_fp32[ij], sum);
            }
            __syncthreads();
        }
    }
    
    // Convert A_4 to BF16, compute A_4², then B_4 = b*A_4 + c*A_4²
    const int gram_storage_needed = 2 * gram_size * gram_size;
    __nv_bfloat16* gram_A_bf16 = tile_buffer_bf16;
    
    for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
        gram_A_bf16[ij] = __float2bfloat16(gram_A_fp32[ij]);
    }
    __syncthreads();
    
    __nv_bfloat16* temp_A2_bf16 = gram_A_bf16 + gram_size * gram_size;
    
    for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
        const int i = ij / gram_size;
        const int j = ij % gram_size;
        
        float A2_ij_fp32 = 0.0f;
        for (int k = 0; k < gram_size; ++k) {
            float a_val = __bfloat162float(gram_A_bf16[i * gram_size + k]);
            float b_val = __bfloat162float(gram_A_bf16[k * gram_size + j]);
            A2_ij_fp32 += a_val * b_val;
        }
        temp_A2_bf16[ij] = __float2bfloat16(A2_ij_fp32);
    }
    __syncthreads();
    
    // Compute B_4 with BF16 rounding to match forward pass, store in gram_A_fp32
    for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
        float A_ij = __bfloat162float(gram_A_bf16[ij]);
        float A2_ij = __bfloat162float(temp_A2_bf16[ij]);
        float B_4_fp32 = b * A_ij + c * A2_ij;
        // Apply BF16 rounding to match forward pass
        __nv_bfloat16 B_4_bf16 = __float2bfloat16(B_4_fp32);
        gram_A_fp32[ij] = __bfloat162float(B_4_bf16);
    }
    __syncthreads();
    
    // Now we have: X_4 in X_temp, A_4 in gram_A_bf16, B_4 in gram_A_fp32
    
    // Step 1: Load grad_output (dL/db_t_ortho) directly as dL/dX_5
    // Per official PyTorch NS5: forward returns normalized result WITHOUT scaling by norm
    // So backward: dL/dX_5 = dL/db_t_ortho (no norm multiplication needed)
    if (!transposed) {
        // Non-transposed: X is [D, N], storage is [D, N]
        for (int d_start = 0; d_start < D; d_start += kTileSize) {
            const int d_end = min(d_start + kTileSize, D);
            const int tile_rows = d_end - d_start;
            
            for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                const int local_row = idx / dstate;
                const int col = idx % dstate;
                const int global_row = d_start + local_row;
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_row * L * dstate + 
                                time_idx * dstate + col;
                
                // Load grad_output directly (no scaling by norm)
                float dX_5 = grad_output[buffer_idx];
                
                // Initialize dX_4 = a * dX_5
                dX_4_temp[buffer_idx] = a * dX_5;
                
                // DEBUG: Print grad_output reading (only for first few timesteps to avoid spam)
                if (batch_idx == 0 && time_idx < 4 && global_row == 0 && col == 0) {
                    printf("NS_BWD: batch=%d time=%d dim=%d state=%d dX_5=%.6f dX_4=%.6f buffer_idx=%d\n",
                           batch_idx, time_idx, global_row, col, dX_5, dX_4_temp[buffer_idx], buffer_idx);
                }
            }
        }
    } else {
        // Transposed: X_storage is [D, N] (logical [N, D])
        for (int d_start = 0; d_start < D; d_start += kTileSize) {
            const int d_end = min(d_start + kTileSize, D);
            const int tile_cols = d_end - d_start;
            
            for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
                const int row = idx / tile_cols;  // n index
                const int local_col = idx % tile_cols;
                const int global_col = d_start + local_col;  // d index
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_col * L * dstate + 
                                time_idx * dstate + row;
                
                // Load grad_output directly (no scaling by norm)
                float dX_5 = grad_output[buffer_idx];
                
                // Initialize dX_4 = a * dX_5
                dX_4_temp[buffer_idx] = a * dX_5;
                
                // DEBUG: Print grad_output reading (transposed case)
                if (batch_idx == 0 && time_idx < 4 && global_col == 0 && row == 0) {
                    printf("NS_BWD_TRANS: batch=%d time=%d dim=%d state=%d dX_5=%.6f buffer_idx=%d\n",
                           batch_idx, time_idx, global_col, row, dX_5, buffer_idx);
                }
            }
        }
    }
    __syncthreads();
    
    // Step 2: Compute dX_4 += gradient through B_4@X_4
    // Note: B_4 is treated as a constant (detached), so we only backprop through its application
    if (!transposed) {
        // Not transposed: X is [D, N], B_4 is [D, D]
        // Forward: X_new = a*X + B_4 @ X (B_4 is detached/constant)
        // Backward: dX_4 += B_4.T @ dX_5 (treating B_4 as constant)
        
        for (int d_start = 0; d_start < D; d_start += kTileSize) {
            const int d_end = min(d_start + kTileSize, D);
            const int tile_rows = d_end - d_start;
            
            // Load X_4 and dX_5 tiles
            for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                const int local_row = idx / dstate;
                const int col = idx % dstate;
                const int global_row = d_start + local_row;
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_row * L * dstate + 
                                time_idx * dstate + col;
                
                // Store X_4 in tile_buffer
                float x_4_val = X_temp[buffer_idx];
                tile_buffer_bf16[local_row * dstate + col] = float_to_bf16_reinterpret_ns(x_4_val);
            }
            __syncthreads();
            
            // Forward: X_5[i, j] = a*X_4[i, j] + sum_k B_4[i, k] * X_4[k, j]
            // Backward: dX_4[i, j] = a*dX_5[i, j] + sum_k B_4[i, k] * dX_5[k, j]
            // Note: We need dX_5[k, j] for all k that contributed to X_5[i, j]
            // Also: X_5[k, j] = a*X_4[k, j] + sum_l B_4[k, l] * X_4[l, j]
            // So dX_4[i, j] gets contribution from dX_5[k, j] with coefficient B_4[k, i]
            // dX_4[i, j] = a*dX_5[i, j] + sum_k B_4[k, i] * dX_5[k, j]
            // This is: dX_4 = a*dX_5 + B_4.T @ dX_5
            for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                const int local_row = idx / dstate;
                const int col = idx % dstate;
                const int global_row = d_start + local_row;
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_row * L * dstate + 
                                time_idx * dstate + col;
                
                // Initialize with a*dX_5 (already done in Step 1, so here we add B_4.T @ dX_5)
                // (B_4.T @ dX_5)[i,j] = sum_k B_4[k,i] * dX_5[k,j]
                float sum = 0.0f;
                for (int k = 0; k < gram_size; ++k) {
                    int k_idx = batch_idx * D * L * dstate + k * L * dstate + time_idx * dstate + col;
                    float dX_5_kj = grad_output[k_idx];  // dX_5[k, j]
                    float B_4_ki = gram_A_fp32[k * gram_size + global_row];  // B_4[k, i]
                    sum += B_4_ki * dX_5_kj;
                }
                dX_4_temp[buffer_idx] += sum;
            }
            __syncthreads();
        }
    } else {
        // Transposed: X_storage is [D, N] (logical [N, D]), B_4 is [N, N]
        // Forward: X_storage @ B_4.T (right multiply, B_4 is detached/constant)
        // Backward: dX_4_storage += dX_5_storage @ B_4 (treating B_4 as constant)
        
        for (int d_start = 0; d_start < D; d_start += kTileSize) {
            const int d_end = min(d_start + kTileSize, D);
            const int tile_cols = d_end - d_start;
            
            // Load X_4_storage tile [dstate, tile_cols]
            for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
                const int row = idx / tile_cols;
                const int local_col = idx % tile_cols;
                const int global_col = d_start + local_col;
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_col * L * dstate + 
                                time_idx * dstate + row;
                float x_4_val = X_temp[buffer_idx];
                tile_buffer_bf16[row * tile_cols + local_col] = float_to_bf16_reinterpret_ns(x_4_val);
            }
            __syncthreads();
            
            // Compute dX_4_storage += dX_5_storage @ B_4
            for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
                const int row = idx / tile_cols;  // n index
                const int local_col = idx % tile_cols;
                const int global_col = d_start + local_col;  // d index
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_col * L * dstate + 
                                time_idx * dstate + row;
                
                // Forward: X_5_storage[d, n] = a*X_4_storage[d, n] + sum_k B_4[n, k] * X_4_storage[d, k]
                //         (This is X_4 @ B_4.T in matrix form)
                // Backward: dX_4_storage[d, n] += sum_k dX_5_storage[d, k] * B_4[k, n]
                //          (This is dX_5 @ B_4 in matrix form)
                float sum = 0.0f;
                for (int k = 0; k < gram_size; ++k) {
                    int k_idx = batch_idx * D * L * dstate + global_col * L * dstate + time_idx * dstate + k;
                    float dX_5_dk = grad_output[k_idx];  // dX_5_storage[d, k]
                    float B_4_kn = gram_A_fp32[k * gram_size + row];  // B_4[k, n] where n=row (FIXED!)
                    sum += dX_5_dk * B_4_kn;
                }
                dX_4_temp[buffer_idx] += sum;
            }
            __syncthreads();
        }
    }
    
    // Step 3: Compute gradient through normalization
    // d(b_t)[i,j] = (dX_4[i,j] - (sum_kl dX_4[k,l] * X_4_norm[k,l]) * X_4_norm[i,j]) / norm
    // First compute dot product: dnorm_from_loss = sum dX_4 * X_4_norm
    
    float dnorm_local = 0.0f;
    if (!transposed) {
        // Non-transposed: X is [D, N]
        for (int d_start = 0; d_start < D; d_start += kTileSize) {
            const int d_end = min(d_start + kTileSize, D);
            const int tile_rows = d_end - d_start;
            
            for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                const int local_row = idx / dstate;
                const int col = idx % dstate;
                const int global_row = d_start + local_row;
                
                // Recompute X_0 (normalized input) for dot product
                int u_idx = batch_idx * u_batch_stride + global_row * u_d_stride + time_idx;
                float u_val = to_float_ns(u[u_idx]);
                
                int delta_idx = batch_idx * delta_batch_stride + global_row * delta_d_stride + time_idx;
                float delta_val = to_float_ns(delta[delta_idx]);
                
                float B_val;
                if (!is_variable_B) {
                    B_val = to_float_ns(B[global_row * B_d_stride + col * B_dstate_stride]);
                } else {
                    // Variable B: [B, G, N, L] - FIXED: Use correct indexing
                    int group_size = (D + n_groups - 1) / n_groups;
                    int group_id = min(global_row / group_size, n_groups - 1);
                    B_val = to_float_ns(B[batch_idx * B_batch_stride + 
                                           group_id * B_group_stride +
                                           col * B_dstate_stride + time_idx]);
                }
                
                float b_t_val = alpha * delta_val * B_val * u_val;
                __nv_bfloat16 b_t_bf16 = __float2bfloat16(b_t_val);
                float b_t_rounded = __bfloat162float(b_t_bf16);
                float X_0_fp32 = b_t_rounded / norm;
                __nv_bfloat16 X_0_bf16 = __float2bfloat16(X_0_fp32);
                float X_0_val = __bfloat162float(X_0_bf16);
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_row * L * dstate + 
                                time_idx * dstate + col;
                
                float dX_4_val = dX_4_temp[buffer_idx];
                dnorm_local += dX_4_val * X_0_val;
            }
        }
    } else {
        // Transposed: X_storage is [D, N] (logical [N, D])
        for (int d_start = 0; d_start < D; d_start += kTileSize) {
            const int d_end = min(d_start + kTileSize, D);
            const int tile_cols = d_end - d_start;
            
            for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
                const int row = idx / tile_cols;  // n index
                const int local_col = idx % tile_cols;
                const int global_col = d_start + local_col;  // d index
                
                // Recompute X_0 (normalized input) for dot product
                int u_idx = batch_idx * u_batch_stride + global_col * u_d_stride + time_idx;
                float u_val = to_float_ns(u[u_idx]);
                
                int delta_idx = batch_idx * delta_batch_stride + global_col * delta_d_stride + time_idx;
                float delta_val = to_float_ns(delta[delta_idx]);
                
                float B_val;
                if (!is_variable_B) {
                    B_val = to_float_ns(B[global_col * B_d_stride + row * B_dstate_stride]);
                } else {
                    // Variable B: [B, G, N, L] - FIXED: Use correct indexing
                    int group_size = (D + n_groups - 1) / n_groups;
                    int group_id = min(global_col / group_size, n_groups - 1);
                    B_val = to_float_ns(B[batch_idx * B_batch_stride + 
                                           group_id * B_group_stride +
                                           row * B_dstate_stride + time_idx]);
                }
                
                float b_t_val = alpha * delta_val * B_val * u_val;
                __nv_bfloat16 b_t_bf16 = __float2bfloat16(b_t_val);
                float b_t_rounded = __bfloat162float(b_t_bf16);
                float X_0_fp32 = b_t_rounded / norm;
                __nv_bfloat16 X_0_bf16 = __float2bfloat16(X_0_fp32);
                float X_0_val = __bfloat162float(X_0_bf16);
                
                int buffer_idx = batch_idx * D * L * dstate + 
                                global_col * L * dstate + 
                                time_idx * dstate + row;
                
                float dX_4_val = dX_4_temp[buffer_idx];
                dnorm_local += dX_4_val * X_0_val;
            }
        }
    }
    
    // Block reduction for dnorm
    partial_sums[tid] = dnorm_local;
    __syncthreads();
    
    for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    float dnorm_from_loss = partial_sums[0];
    
    // CRITICAL: Check for NaN/Inf in dnorm_from_loss before using it
    if (!isfinite(dnorm_from_loss)) {
        dnorm_from_loss = 0.0f;  // Replace NaN/Inf with 0
    }
    
    __syncthreads();
    
    // ========== OPTIMIZED: Compute gradients for all inputs in parallel tiles ==========
    // Process dimensions in tiles (like forward pass) for better parallelism
    // Shared memory layout: [kTileSize] for grad_u/grad_delta partial sums per dimension
    // Reuse dX_accumulator space (allocated as max(gram_size_sq, 2*kTileSize) floats)
    float* grad_u_partial = dX_accumulator;  // Uses kTileSize floats
    float* grad_delta_partial = grad_u_partial + kTileSize;  // Uses kTileSize floats (total: 2*kTileSize)
    
    for (int d_start = 0; d_start < D; d_start += kTileSize) {
        const int d_end = min(d_start + kTileSize, D);
        const int tile_size = d_end - d_start;
        
        // Initialize partial sums for this tile
        for (int idx = tid; idx < tile_size; idx += kBlockSize) {
            grad_u_partial[idx] = 0.0f;
            grad_delta_partial[idx] = 0.0f;
        }
        __syncthreads();
        
        // Each thread processes multiple (d, n) pairs in this tile
        for (int idx = tid; idx < tile_size * dstate; idx += kBlockSize) {
            const int local_d = idx / dstate;
            const int n = idx % dstate;
            const int global_d = d_start + local_d;
            
            // Load inputs
            int u_idx = batch_idx * u_batch_stride + global_d * u_d_stride + time_idx;
            float u_val = to_float_ns(u[u_idx]);
            
            int delta_idx = batch_idx * delta_batch_stride + global_d * delta_d_stride + time_idx;
            float delta_val = to_float_ns(delta[delta_idx]);
            
            float B_val;
            int B_idx;
            if (!is_variable_B) {
                B_idx = global_d * B_d_stride + n * B_dstate_stride;
                B_val = to_float_ns(B[B_idx]);
            } else {
                int group_size = (D + n_groups - 1) / n_groups;
                int group_id = min(global_d / group_size, n_groups - 1);
                B_idx = batch_idx * B_batch_stride + 
                        group_id * B_group_stride +
                        n * B_dstate_stride + time_idx;
                B_val = to_float_ns(B[B_idx]);
            }
            
            // Recompute X_0 for gradient through normalization
            float b_t_val = alpha * delta_val * B_val * u_val;
            __nv_bfloat16 b_t_bf16 = __float2bfloat16(b_t_val);
            float b_t_rounded = __bfloat162float(b_t_bf16);
            float X_0_fp32 = b_t_rounded / norm;
            __nv_bfloat16 X_0_bf16 = __float2bfloat16(X_0_fp32);
            float X_0_val = __bfloat162float(X_0_bf16);
            
            // CRITICAL: Check for NaN/Inf in X_0_val before using it
            if (!isfinite(X_0_val)) {
                X_0_val = 0.0f;  // Replace NaN/Inf with 0
            }
            
            int buffer_idx = batch_idx * D * L * dstate + global_d * L * dstate + time_idx * dstate + n;
            float dX_4_val = dX_4_temp[buffer_idx];
            
            // CRITICAL: Check for NaN/Inf in dX_4_val before using it
            if (!isfinite(dX_4_val)) {
                dX_4_val = 0.0f;  // Replace NaN/Inf with 0
            }
            
            // Gradient through normalization
            // CRITICAL: Clamp norm to prevent division by zero or very small values
            // Very small norms can cause overflow when dividing
            float norm_safe = fmaxf(norm, 1e-4f);  // Prevent division by very small numbers
            
            // CRITICAL: Check for NaN/Inf in inputs before computation
            float dX_4_safe = isfinite(dX_4_val) ? dX_4_val : 0.0f;
            float X_0_safe = isfinite(X_0_val) ? X_0_val : 0.0f;
            float dnorm_safe = isfinite(dnorm_from_loss) ? dnorm_from_loss : 0.0f;
            
            // Compute gradient: d(b_t) = (dX_4 - dnorm * X_0) / norm
            // Note: This formula assumes X_4 ≈ X_0 (detached iterations approximation)
            float d_b_t = (dX_4_safe - dnorm_safe * X_0_safe) / norm_safe;
            
            // CRITICAL: Final check for NaN/Inf in result
            if (!isfinite(d_b_t)) {
                d_b_t = 0.0f;  // Replace NaN/Inf with 0
            }
            
            // Accumulate gradients
            float grad_u_contrib = alpha * delta_val * B_val * d_b_t;
            float grad_delta_contrib = alpha * B_val * u_val * d_b_t;
            float grad_B_contrib = alpha * delta_val * u_val * d_b_t;
            
            // CRITICAL: Check for NaN/Inf before atomic accumulation
            if (!isfinite(grad_u_contrib)) grad_u_contrib = 0.0f;
            if (!isfinite(grad_delta_contrib)) grad_delta_contrib = 0.0f;
            if (!isfinite(grad_B_contrib)) grad_B_contrib = 0.0f;
            
            // Atomic accumulation into shared memory for u and delta (one value per dimension)
            atomicAdd(&grad_u_partial[local_d], grad_u_contrib);
            atomicAdd(&grad_delta_partial[local_d], grad_delta_contrib);
            
            // Direct atomic to global for grad_B (many values per dimension)
            atomicAdd(&grad_B[B_idx], grad_B_contrib);
        }
        __syncthreads();
        
        // Write accumulated gradients for u and delta to global memory
        for (int local_d = tid; local_d < tile_size; local_d += kBlockSize) {
            const int global_d = d_start + local_d;
            
            int u_idx = batch_idx * u_batch_stride + global_d * u_d_stride + time_idx;
            int delta_idx = batch_idx * delta_batch_stride + global_d * delta_d_stride + time_idx;
            
            // CRITICAL: Check for NaN/Inf in accumulated gradients before writing to global
            float grad_u_val = grad_u_partial[local_d];
            float grad_delta_val = grad_delta_partial[local_d];
            
            if (!isfinite(grad_u_val)) grad_u_val = 0.0f;
            if (!isfinite(grad_delta_val)) grad_delta_val = 0.0f;
            
            // DEBUG: Print for first dimension
            if (batch_idx == 0 && time_idx < 4 && global_d == 0) {
                printf("NS_BWD_OPT: batch=%d time=%d dim=%d grad_u=%.6f grad_delta=%.6f\n",
                       batch_idx, time_idx, global_d, grad_u_val, grad_delta_val);
            }
            
            atomicAdd(&grad_u[u_idx], grad_u_val);
            atomicAdd(&grad_delta[delta_idx], grad_delta_val);
        }
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launch wrapper for velocity 5-step backward
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename input_t, typename weight_t>
inline void launch_newton_schulz_velocity_5step_backward(
    const float* grad_output,
    const input_t* u, const input_t* delta, const weight_t* B,
    float* grad_u, float* grad_delta, float* grad_B,
    float* X_temp, float* dX_4_temp,  // Pre-allocated temporary buffers (CRITICAL: avoid cudaMalloc/cudaFree)
    float alpha, int batch, int dim, int seqlen, int dstate,
    int t_start, int t_end,
    int u_batch_stride, int u_d_stride,
    int delta_batch_stride, int delta_d_stride,
    int B_batch_stride, int B_group_stride,
    int B_d_stride, int B_dstate_stride,
    bool is_variable_B, int n_groups,
    cudaStream_t stream
) {
    constexpr int kBlockSize = 256;
    constexpr int kTileSize = 64;
    
    const int num_timesteps = t_end - t_start;
    if (num_timesteps <= 0) return;
    
    dim3 grid(batch, num_timesteps);
    dim3 block(kBlockSize);
    
    // Shared memory calculation (similar to forward, but need extra space for gradients)
    const bool transposed = (dim > dstate);
    const int gram_size = transposed ? dstate : dim;
    
    const int tile_buffer_elements = kTileSize * (transposed ? dim : dstate);
    const int gram_size_sq = gram_size * gram_size;
    
    const int required_tile_buffer_for_poly = 2 * gram_size_sq;
    const int actual_tile_buffer_size = max(tile_buffer_elements, required_tile_buffer_for_poly);
    
    // Additional space needed:
    // - partial_sums: kBlockSize floats
    // - dX_accumulator: reused for grad_u_partial and grad_delta_partial (2 * kTileSize floats)
    //   We allocate max(gram_size_sq, 2 * kTileSize) to ensure sufficient space
    const int grad_partial_size = max(gram_size_sq, 2 * kTileSize);
    const int smem_size = actual_tile_buffer_size * sizeof(__nv_bfloat16) + 
                          gram_size_sq * sizeof(float) +  // gram_A_fp32
                          kBlockSize * sizeof(float) +     // partial_sums
                          grad_partial_size * sizeof(float);  // dX_accumulator (reused for grad_u_partial + grad_delta_partial)
    
    if (smem_size > 48 * 1024) {
        #ifndef USE_ROCM
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            newton_schulz_velocity_5step_backward_kernel<input_t, weight_t, kBlockSize, kTileSize>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        ));
        #endif
    }
    
    // CRITICAL: Zero out dX_4_temp since we use += operations
    // Use pre-allocated buffers passed from caller (allocated in selective_scan.cpp)
    // This avoids host-synchronizing cudaMalloc/cudaFree calls during training
    const size_t temp_buffer_size = batch * dim * seqlen * dstate * sizeof(float);
    C10_CUDA_CHECK(cudaMemsetAsync(dX_4_temp, 0, temp_buffer_size, stream));
    
    newton_schulz_velocity_5step_backward_kernel<input_t, weight_t, kBlockSize, kTileSize><<<grid, block, smem_size, stream>>>(
        grad_output, u, delta, B,
        grad_u, grad_delta, grad_B,
        X_temp, dX_4_temp,
        alpha, batch, dim, seqlen, dstate, t_start,
        u_batch_stride, u_d_stride,
        delta_batch_stride, delta_d_stride,
        B_batch_stride, B_group_stride,
        B_d_stride, B_dstate_stride,
        is_variable_B, n_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}