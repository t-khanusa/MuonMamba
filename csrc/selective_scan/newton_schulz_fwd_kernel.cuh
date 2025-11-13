/******************************************************************************
 * Per-Dimension Batched Newton-Schulz Forward Kernel
 * For MuonMamba: Apply NS independently for each dimension on [B*L, N] matrices
 * Using float32 precision throughout for maximum numerical stability
 * Copyright (c) 2024
 ******************************************************************************/

 #pragma once

 #include <c10/util/BFloat16.h>
 #include <cuda_runtime.h>
 #include <c10/cuda/CUDAException.h>
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
//  // Kernel 1: Compute Frobenius norms for all D dimensions in parallel
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  template<int kBlockSize = 256>
//  __global__ void compute_norms_batched(
//      const float* __restrict__ V_in,        // [B, D, L, N]
//      float* __restrict__ norms_out,         // [D] - one norm per dimension
//      int B, int D, int L, int N
//  ) {
//      const int tid = threadIdx.x;
//      const int dim_id = blockIdx.x;  // Each block handles one dimension
     
//      if (dim_id >= D) return;
     
//      // Compute Frobenius norm for this dimension
//      float local_sum = 0.0f;
//      const int M = B * L;  // Total rows for this dimension
     
//      for (int idx = tid; idx < M * N; idx += kBlockSize) {
//          int row = idx / N;
//          int col = idx % N;
         
//          // Access V_in[batch, dim, seqlen, dstate] -> V_in[b*D*L*N + d*L*N + l*N + n]
//          int batch_idx = row / L;
//          int seq_idx = row % L;
//          int global_idx = batch_idx * D * L * N + dim_id * L * N + seq_idx * N + col;
         
//          float val = V_in[global_idx];
//          local_sum += val * val;
//      }
     
//      // Block reduction
//      float norm_sq = blockReduceSum_perdim<kBlockSize>(local_sum);
     
//      if (tid == 0) {
//          float norm = sqrtf(norm_sq + 1e-8f);  // Add epsilon inside sqrt for better stability
//          norms_out[dim_id] = norm;
//      }
//  }
 
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  // Kernel 2: Compute covariance matrices A_d = (G_d^T @ G_d) / M for all D dimensions
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  template<int kBlockSize = 256>
//  __global__ void compute_covariance_batched(
//      const float* __restrict__ V_in,        // [B, D, L, N]
//      const float* __restrict__ norms,       // [D]
//      float* __restrict__ A_out,             // [D, work_M, work_M] where work_M = min(M, N)
//      int B, int D, int L, int N
//  ) {
//      const int tid = threadIdx.x;
//      const int dim_id = blockIdx.x;  // Each block handles one dimension
     
//      if (dim_id >= D) return;
     
//      const int M = B * L;  // Total rows for this dimension
//      const float norm = norms[dim_id];
     
//      // Handle transpose for tall matrices (make matrix "fat": rows <= cols)
//      const int work_M = min(M, N);  // Always the smaller dimension
//      const int work_N = max(M, N);  // Always the larger dimension
//      const bool transposed = (M > N);
     
//      // Ensure work_M <= 64 for shared memory efficiency
//      if (work_M > 64) {
//          // For very large matrices, we'd need a different approach
//          // For now, clamp to 64 and handle gracefully
//          return;
//      }
     
//      // Shared memory for covariance matrix A [work_M, work_M]
//      extern __shared__ float smem[];
//      float* smem_A = smem;
     
//      // Initialize A to zero
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          smem_A[idx] = 0.0f;
//      }
//      __syncthreads();
     
//      // Use 2D thread mapping for efficiency: each thread handles one (i,j) pair
//      // Map threadIdx.x to 2D coordinates (i, j) in work_M x work_M
//      const int total_elements = work_M * work_M;
//      for (int idx = tid; idx < total_elements; idx += kBlockSize) {
//          const int i = idx / work_M;
//          const int j = idx % work_M;
         
//          if (i >= work_M || j >= work_M) continue;
         
//          float sum = 0.0f;
         
//          // Sum over all work_N columns
//          // After potential transposition, we always work with fat matrices
//          // A = X @ X.T where X is work_M × work_N
//          for (int k = 0; k < work_N; ++k) {
//              float g_ik, g_jk;
             
//              if (transposed) {
//                  // Tall: A[i,j] = sum_k G[k,i] * G[k,j] (i,j cols 0..N-1, k rows 0..M-1)
//                  int batch_k = k / L;
//                  int seq_k = k % L;
//                  int idx_i = batch_k * D * L * N + dim_id * L * N + seq_k * N + i;
//                  int idx_j = batch_k * D * L * N + dim_id * L * N + seq_k * N + j;
                 
//                  // Skip if out of bounds (rare, but safe)
//                  if (idx_i >= B * D * L * N || idx_j >= B * D * L * N) continue;
                 
//                  g_ik = V_in[idx_i] / norm;
//                  g_jk = V_in[idx_j] / norm;
//              } else {
//                  // Fat: A[i,j] = sum_k G[i,k] * G[j,k] (i,j rows 0..M-1, k cols 0..N-1)
//                  int batch_i = i / L;
//                  int seq_i = i % L;
//                  int batch_j = j / L;
//                  int seq_j = j % L;
//                  int idx_i = batch_i * D * L * N + dim_id * L * N + seq_i * N + k;
//                  int idx_j = batch_j * D * L * N + dim_id * L * N + seq_j * N + k;
                 
//                  // Skip if out of bounds (rare, but safe)
//                  if (idx_i >= B * D * L * N || idx_j >= B * D * L * N) continue;
                 
//                  g_ik = V_in[idx_i] / norm;
//                  g_jk = V_in[idx_j] / norm;
//              }
             
//              sum += g_ik * g_jk;
//          }
         
//          // Direct assignment instead of atomicAdd (threads have unique (i,j) pairs)
//          smem_A[i * work_M + j] = sum;
//      }
//      __syncthreads();
     
//      // Store to global memory (no additional normalization by M to avoid double normalization)
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          int global_idx = dim_id * work_M * work_M + idx;
//          A_out[global_idx] = smem_A[idx];
//      }
//  }
 
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  // Kernel 3: Compute A^2 and B = b*A + c*A^2 for all D dimensions in parallel
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  template<int kBlockSize = 256>
//  __global__ void compute_polynomial_batched(
//      const float* __restrict__ A_in,        // [D, work_M, work_M] where work_M = min(M, N)
//      float* __restrict__ B_out,             // [D, work_M, work_M]
//      int D, int work_M
//  ) {
//      const int tid = threadIdx.x;
//      const int dim_id = blockIdx.x;  // Each block handles one dimension
     
//      if (dim_id >= D) return;
     
//      // Coefficients for Newton-Schulz 1-step (original values)
//      constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
     
//      // Shared memory for A and A^2
//      extern __shared__ float smem[];
//      float* smem_A = smem;
//      float* smem_A2 = smem_A + work_M * work_M;
     
//      // Load A matrix for this dimension
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          smem_A[idx] = A_in[dim_id * work_M * work_M + idx];
//      }
//      __syncthreads();
     
//      // Compute A^2
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
     
//      // Compute B = b*A + c*A^2
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          int global_idx = dim_id * work_M * work_M + idx;
//          B_out[global_idx] = b * smem_A[idx] + c * smem_A2[idx];
//      }
//  }
 
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  // Kernel 4: Apply orthogonalization G'_d = a*G_d + B_d @ G_d for all D dimensions
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  template<int kBlockSize = 256>
//  __global__ void apply_orthogonalization_batched(
//      const float* __restrict__ V_in,        // [B, D, L, N]
//      const float* __restrict__ norms,       // [D]
//      const float* __restrict__ B_matrices,  // [D, work_M, work_M] where work_M = min(M, N)
//      float* __restrict__ V_out,             // [B, D, L, N]
//      int B, int D, int L, int N
//  ) {
//      const int tid = threadIdx.x;
//      const int dim_id = blockIdx.x;  // Each block handles one dimension
     
//      if (dim_id >= D) return;
     
//      const int M = B * L;  // Total rows for this dimension
//      const float norm = norms[dim_id];
     
//      // Handle transpose for tall matrices (make matrix "fat": rows <= cols)
//      const int work_M = min(M, N);  // Always the smaller dimension
//      const int work_N = max(M, N);  // Always the larger dimension
//      const bool transposed = (M > N);
     
//      // Ensure work_M <= 64 for shared memory efficiency
//      if (work_M > 64) {
//          return;
//      }
     
//      // Coefficients for Newton-Schulz 1-step
//      constexpr float a = 3.4445f;
     
//      // Shared memory for B matrix
//      extern __shared__ float smem[];
//      float* smem_B = smem;
     
//      // Load B matrix for this dimension
//      for (int idx = tid; idx < work_M * work_M; idx += kBlockSize) {
//          smem_B[idx] = B_matrices[dim_id * work_M * work_M + idx];
//      }
//      __syncthreads();
     
//      // Apply orthogonalization: G' = a*G_norm + B @ G_norm
//      // Use efficient thread mapping: each thread handles one output element
//      for (int idx = tid; idx < M * N; idx += kBlockSize) {
//          int row = idx / N;
//          int col = idx % N;
         
//          // Boundary checks
//          if (row >= M || col >= N) continue;
         
//          // Access input
//          int batch_idx = row / L;
//          int seq_idx = row % L;
//          int input_idx = batch_idx * D * L * N + dim_id * L * N + seq_idx * N + col;
         
//          // Additional boundary check
//          if (input_idx >= B * D * L * N) continue;
         
//          float g_norm = V_in[input_idx] / (norm + 1e-8f);  // Add epsilon for numerical stability
         
//          // Compute B @ G_norm for this element
//          // Formula: X = a * X + B @ X where B is work_M × work_M
//          float sum = 0.0f;
//          for (int k = 0; k < work_M; ++k) {
//              float b_row_k, g_k_col;
             
//              if (transposed) {
//                  // Tall: sum_k G[row, k] * B[k, col] (right multiply)
//                  // row 0..M-1, col 0..N-1, k 0..N-1
//                  int batch_row = row / L;  // Fixed row
//                  int seq_row = row % L;
//                  int g_idx = batch_row * D * L * N + dim_id * L * N + seq_row * N + k;  // Vary k as col
//                  float b_val = smem_B[k * work_M + col];  // B[k, col]
                 
//                  if (g_idx >= B * D * L * N) continue;
                 
//                  b_row_k = b_val;
//                  g_k_col = V_in[g_idx] / norm;
//              } else {
//                  // Fat: sum_k B[row, k] * G[k, col] (left multiply)
//                  // row 0..M-1, col 0..N-1, k 0..M-1
//                  int batch_k = k / L;
//                  int seq_k = k % L;
//                  int g_idx = batch_k * D * L * N + dim_id * L * N + seq_k * N + col;
//                  float b_val = smem_B[row * work_M + k];  // B[row, k]
                 
//                  if (g_idx >= B * D * L * N) continue;
                 
//                  b_row_k = b_val;
//                  g_k_col = V_in[g_idx] / norm;
//              }
             
//              sum += b_row_k * g_k_col;
//          }
         
//          // Apply Newton-Schulz update
//          float result = a * g_norm + sum;
         
//          // Store output
//          V_out[input_idx] = result;
//      }
//  }
 
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  // Launch wrapper
//  ////////////////////////////////////////////////////////////////////////////////////////////////////
 
//  inline void launch_newton_schulz_per_dim(
//      const float* V_in,   // [batch, dim, seqlen, dstate]
//      float* V_out,        // [batch, dim, seqlen, dstate]
//      int batch, int dim, int seqlen, int dstate,
//      cudaStream_t stream
//  ) {
//      constexpr int kBlockSize = 256;
     
//      // Determine work matrix size (min(M, N) where M = batch * seqlen)
//      const int M = batch * seqlen;
//      const int work_M = (M > dstate) ? dstate : M;
     
//      // Ensure work_M <= 64 for shared memory efficiency
//      if (work_M > 64) {
//          // For very large matrices, fall back to identity transform
//          // This is a limitation of the current implementation
//          cudaMemcpyAsync(V_out, V_in, batch * dim * seqlen * dstate * sizeof(float), 
//                         cudaMemcpyDeviceToDevice, stream);
//          return;
//      }
     
//      // Allocate temporary buffers with correct sizes
//      float *norms, *A_matrices, *B_matrices;
//      C10_CUDA_CHECK(cudaMalloc(&norms, dim * sizeof(float)));
//      C10_CUDA_CHECK(cudaMalloc(&A_matrices, dim * work_M * work_M * sizeof(float)));
//      C10_CUDA_CHECK(cudaMalloc(&B_matrices, dim * work_M * work_M * sizeof(float)));
     
//      // Kernel 1: Compute norms for all dimensions
//      compute_norms_batched<kBlockSize><<<dim, kBlockSize, 0, stream>>>(
//          V_in, norms, batch, dim, seqlen, dstate
//      );
//      C10_CUDA_KERNEL_LAUNCH_CHECK();
     
//      // Kernel 2: Compute covariance matrices
//      int smem_size_cov = work_M * work_M * sizeof(float);
//      compute_covariance_batched<kBlockSize><<<dim, kBlockSize, smem_size_cov, stream>>>(
//          V_in, norms, A_matrices, batch, dim, seqlen, dstate
//      );
//      C10_CUDA_KERNEL_LAUNCH_CHECK();
     
//      // Kernel 3: Compute polynomial B = b*A + c*A^2
//      int smem_size_poly = 2 * work_M * work_M * sizeof(float);
//      compute_polynomial_batched<kBlockSize><<<dim, kBlockSize, smem_size_poly, stream>>>(
//          A_matrices, B_matrices, dim, work_M
//      );
//      C10_CUDA_KERNEL_LAUNCH_CHECK();
     
//      // Kernel 4: Apply orthogonalization
//      int smem_size_ortho = work_M * work_M * sizeof(float);
//      apply_orthogonalization_batched<kBlockSize><<<dim, kBlockSize, smem_size_ortho, stream>>>(
//          V_in, norms, B_matrices, V_out, batch, dim, seqlen, dstate
//      );
//      C10_CUDA_KERNEL_LAUNCH_CHECK();
     
//     // Clean up temporary memory
//     C10_CUDA_CHECK(cudaFree(norms));
//     C10_CUDA_CHECK(cudaFree(A_matrices));
//     C10_CUDA_CHECK(cudaFree(B_matrices));
// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // Tiled Newton-Schulz kernel for velocity orthogonalization
// // Process [D, N] matrices independently for each (batch, timestep) pair
// ////////////////////////////////////////////////////////////////////////////////////////////////////

// template<int kBlockSize = 256, int kTileSize = 64>
// __global__ void newton_schulz_velocity_tiled_kernel(
//     float* __restrict__ velocity_buffer,  // [B, D, L, dstate]
//     int B, int D, int L, int dstate,
//     int t_start  // Starting timestep offset
// ) {
//     // Block indices
//     const int batch_idx = blockIdx.x;
//     const int time_local = blockIdx.y;
//     const int time_idx = t_start + time_local;
    
//     if (batch_idx >= B || time_idx >= L) return;
    
//     const int tid = threadIdx.x;
    
//     // Newton-Schulz coefficients
//     constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    
//     // Shared memory layout
//     extern __shared__ float smem[];
//     float* tile_buffer = smem;                           // [kTileSize, dstate]
//     float* gram_A = tile_buffer + kTileSize * dstate;    // [dstate, dstate]
//     float* matrix_B = gram_A + dstate * dstate;          // [dstate, dstate]
//     float* partial_sums = matrix_B + dstate * dstate;    // [kBlockSize]
    
//     // Velocity buffer layout: [B, D, L, dstate]
//     // For fixed (batch_idx, time_idx), we process matrix [D, dstate]
//     // Access: velocity_buffer[b][d][t][s] = velocity_buffer[b*D*L*dstate + d*L*dstate + t*dstate + s]
    
//     // ========== STEP 1: Compute Frobenius Norm ==========
//     float local_sum = 0.0f;
    
//     for (int d_start = 0; d_start < D; d_start += kTileSize) {
//         const int d_end = min(d_start + kTileSize, D);
//         const int tile_rows = d_end - d_start;
        
//         // Each thread accumulates partial norm for this tile
//         for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
//             const int local_row = idx / dstate;
//             const int col = idx % dstate;
//             const int global_row = d_start + local_row;
            
//             // Access velocity_buffer[batch_idx][global_row][time_idx][col]
//             float val = velocity_buffer[batch_idx * D * L * dstate + global_row * L * dstate + time_idx * dstate + col];
//             local_sum += val * val;
//         }
//     }
    
//     // Block reduction for norm
//     partial_sums[tid] = local_sum;
//     __syncthreads();
    
//     for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
//         if (tid < stride) {
//             partial_sums[tid] += partial_sums[tid + stride];
//         }
//         __syncthreads();
//     }
    
//     float norm = sqrtf(partial_sums[0] + 1e-8f);
//     __syncthreads();
    
//     // ========== STEP 2: Compute Gram Matrix A = V^T @ V ==========
//     // A is [dstate, dstate], accumulated from all tiles
    
//     // Initialize A to zero
//     for (int idx = tid; idx < dstate * dstate; idx += kBlockSize) {
//         gram_A[idx] = 0.0f;
//     }
//     __syncthreads();
    
//     // Accumulate contributions from each tile
//     for (int d_start = 0; d_start < D; d_start += kTileSize) {
//         const int d_end = min(d_start + kTileSize, D);
//         const int tile_rows = d_end - d_start;
        
//         // Load tile into shared memory: [tile_rows, dstate]
//         for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
//             const int local_row = idx / dstate;
//             const int col = idx % dstate;
//             const int global_row = d_start + local_row;
            
//             // Load and normalize
//             float val = velocity_buffer[batch_idx * D * L * dstate + global_row * L * dstate + time_idx * dstate + col];
//             tile_buffer[local_row * dstate + col] = val / norm;
//         }
//         __syncthreads();
        
//         // Compute partial Gram matrix: A[i,j] += sum_k tile[k,i] * tile[k,j]
//         for (int ij = tid; ij < dstate * dstate; ij += kBlockSize) {
//             const int i = ij / dstate;
//             const int j = ij % dstate;
            
//             float sum = 0.0f;
//             for (int k = 0; k < tile_rows; ++k) {
//                 sum += tile_buffer[k * dstate + i] * tile_buffer[k * dstate + j];
//             }
            
//             atomicAdd(&gram_A[i * dstate + j], sum);
//         }
//         __syncthreads();
//     }
    
//     // ========== STEP 3: Compute A^2 and B = b*A + c*A^2 ==========
//     for (int ij = tid; ij < dstate * dstate; ij += kBlockSize) {
//         const int i = ij / dstate;
//         const int j = ij % dstate;
        
//         float sum = 0.0f;
//         for (int k = 0; k < dstate; ++k) {
//             sum += gram_A[i * dstate + k] * gram_A[k * dstate + j];
//         }
//         matrix_B[ij] = sum;  // Temporarily store A^2
//     }
//     __syncthreads();
    
//     // Compute B = b*A + c*A^2
//     for (int ij = tid; ij < dstate * dstate; ij += kBlockSize) {
//         matrix_B[ij] = b * gram_A[ij] + c * matrix_B[ij];
//     }
//     __syncthreads();
    
//     // ========== STEP 4: Apply Orthogonalization V' = a*V + V*B ==========
//     for (int d_start = 0; d_start < D; d_start += kTileSize) {
//         const int d_end = min(d_start + kTileSize, D);
//         const int tile_rows = d_end - d_start;
        
//         // Load tile (normalized)
//         for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
//             const int local_row = idx / dstate;
//             const int col = idx % dstate;
//             const int global_row = d_start + local_row;
            
//             float val = velocity_buffer[batch_idx * D * L * dstate + global_row * L * dstate + time_idx * dstate + col];
//             tile_buffer[local_row * dstate + col] = val / norm;
//         }
//         __syncthreads();
        
//         // Compute V_tile' = a * V_tile + V_tile @ B
//         for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
//             const int local_row = idx / dstate;
//             const int col = idx % dstate;
//             const int global_row = d_start + local_row;
            
//             float v_norm = tile_buffer[local_row * dstate + col];
            
//             // Compute (V_tile @ B)[row, col] = sum_k V_tile[row, k] * B[k, col]
//             float sum = 0.0f;
//             for (int k = 0; k < dstate; ++k) {
//                 sum += tile_buffer[local_row * dstate + k] * matrix_B[k * dstate + col];
//             }
            
//             float result = a * v_norm + sum;
            
//             // Scale back by norm: V' = G' * norm
//             velocity_buffer[batch_idx * D * L * dstate + global_row * L * dstate + time_idx * dstate + col] = result * norm;
//         }
//         __syncthreads();
//     }
// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // Launch wrapper for tiled NS on velocity buffer
// ////////////////////////////////////////////////////////////////////////////////////////////////////

// inline void launch_newton_schulz_velocity_tiled(
//     float* velocity_buffer,       // [batch, dim, seqlen, dstate]
//     int batch, int dim, int seqlen, int dstate,
//     int t_start, int t_end,       // timestep range to process
//     cudaStream_t stream
// ) {
//     constexpr int kBlockSize = 256;
//     constexpr int kTileSize = 64;  // Process 64 dimension rows at a time
    
//     // Launch grid: one block per (batch, timestep) pair in range
//     const int num_timesteps = t_end - t_start;
//     if (num_timesteps <= 0) return;
    
//     dim3 grid(batch, num_timesteps);
//     dim3 block(kBlockSize);
    
//     // Shared memory requirements:
//     // - Tile buffer: kTileSize * dstate floats
//     // - Gram matrix A: dstate * dstate floats
//     // - Matrix B: dstate * dstate floats
//     // - Partial sums for norm: kBlockSize floats
//     const int smem_size = (kTileSize * dstate + 2 * dstate * dstate + kBlockSize) * sizeof(float);
    
//     // For dim=128, dstate=64: 64*64 + 2*64*64 + 256 = 12544 floats = 50KB
//     // If this exceeds 48KB, increase shared memory limit
//     if (smem_size > 48 * 1024) {
//         // May need cudaFuncSetAttribute to increase shared memory limit
//         // For now, continue and let it potentially overflow
//     }
    
//     newton_schulz_velocity_tiled_kernel<kBlockSize, kTileSize><<<grid, block, smem_size, stream>>>(
//         velocity_buffer, batch, dim, seqlen, dstate, t_start
//     );
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
// }

// // ////////////////////////////////////////////////////////////////////////////////////////////////////
// // // 5-Step Newton-Schulz with On-the-Fly b_t Computation (Speed Optimized)
// // // Uses bfloat16 for NS iterations (as per official Muon paper for numerical stability)
// // // Outputs float32 for scan operations (to avoid accumulation errors)
// // ////////////////////////////////////////////////////////////////////////////////////////////////////

// // // Helper: Convert float to bfloat16
// // __device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float x) {
// //     return __float2bfloat16(x);
// // }

// // // Helper: Convert bfloat16 to float
// //     const int time_idx = t_start + time_local;
    
// //     if (batch_idx >= B_dim || time_idx >= L) return;
    
// //     const int tid = threadIdx.x;
    
// //     // Newton-Schulz coefficients
// //     constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    
// //     // Determine transpose
// //     const bool transposed = (D > dstate);
// //     const int gram_size = transposed ? dstate : D;
    
// //     // Shared memory layout
// //     extern __shared__ float smem[];
// //     __nv_bfloat16* tile_buffer_bf16 = (__nv_bfloat16*)smem;
// //     const int tile_buffer_size = kTileSize * (transposed ? D : dstate);
// //     float* gram_A_fp32 = (float*)(tile_buffer_bf16 + tile_buffer_size);
// //     float* partial_sums = gram_A_fp32 + gram_size * gram_size;
    
// //     // Additional space for gradient accumulators (reuse after forward recomputation)
// //     float* dX_accumulator = partial_sums + kBlockSize;  // [kTileSize, dstate] or [dstate, kTileSize]
    
// //     // ========== PHASE 1: Recompute X_0 → X_4 (Detached, 4 iterations) ==========
    
// //     // Step 1: Compute b_t, convert to BF16, compute norm
// //     float norm_sq_local = 0.0f;
    
// //     for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //         const int d_end = min(d_start + kTileSize, D);
// //         const int tile_rows = d_end - d_start;
        
// //         for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //             const int local_row = idx / dstate;
// //             const int col = idx % dstate;
// //             const int global_row = d_start + local_row;
            
// //             int u_idx = batch_idx * u_batch_stride + global_row * u_d_stride + time_idx;
// //             float u_val = to_float(u[u_idx]);
            
// //             int delta_idx = batch_idx * delta_batch_stride + global_row * delta_d_stride + time_idx;
// //             float delta_val = to_float(delta[delta_idx]);
            
// //             float B_val;
// //             if (!is_variable_B) {
// //                 B_val = to_float(B[global_row * B_d_stride + col * B_dstate_stride]);
// //             } else {
// //                 int group_size = (D + n_groups - 1) / n_groups;
// //                 int group_id = min(global_row / group_size, n_groups - 1);
// //                 B_val = to_float(B[batch_idx * B_batch_stride + 
// //                                    group_id * B_group_stride +
// //                                    time_idx * dstate + col]);
// //             }
            
// //             float b_t_val = alpha * delta_val * B_val * u_val;
// //             __nv_bfloat16 b_t_bf16 = __float2bfloat16(b_t_val);
// //             float b_t_rounded = __bfloat162float(b_t_bf16);
            
// //             norm_sq_local += b_t_rounded * b_t_rounded;
            
// //             // Store in tile buffer temporarily
// //             tile_buffer_bf16[local_row * dstate + col] = b_t_bf16;
// //         }
// //     }
    
// //     // Block reduction for norm
// //     partial_sums[tid] = norm_sq_local;
// //     __syncthreads();
    
// //     for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
// //         if (tid < stride) {
// //             partial_sums[tid] += partial_sums[tid + stride];
// //         }
// //         __syncthreads();
// //     }
    
// //     float norm = sqrtf(partial_sums[0] + 1e-8f);
// //     __syncthreads();
    
// //     // Step 2: Normalize to get X_0, store in global memory (reuse grad_u as temp buffer)
// //     float* X_temp = grad_u;  // Temporary storage for X during recomputation
    
// //     for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //         const int d_end = min(d_start + kTileSize, D);
// //         const int tile_rows = d_end - d_start;
        
// //         for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //             const int local_row = idx / dstate;
// //             const int col = idx % dstate;
// //             const int global_row = d_start + local_row;
            
// //             __nv_bfloat16 b_t_bf16 = tile_buffer_bf16[local_row * dstate + col];
// //             float normalized = __bfloat162float(b_t_bf16) / norm;
// //             __nv_bfloat16 normalized_bf16 = __float2bfloat16(normalized);
// //             float normalized_as_float = __bfloat162float(normalized_bf16);
            
// //             int buffer_idx = batch_idx * D * L * dstate + 
// //                             global_row * L * dstate + 
// //                             time_idx * dstate + col;
// //             X_temp[buffer_idx] = normalized_as_float;
// //         }
// //     }
// //     __syncthreads();
    
// //     // Step 3: Run 4 NS iterations (same as forward, but only 4 iterations)
// //     for (int step = 0; step < 4; ++step) {
// //         // Compute A = X @ X.T
// //         for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
// //             gram_A_fp32[idx] = 0.0f;
// //         }
// //         __syncthreads();
        
// //         if (!transposed) {
// //             for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //                 const int d_end = min(d_start + kTileSize, D);
// //                 const int tile_rows = d_end - d_start;
                
// //                 for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //                     const int local_row = idx / dstate;
// //                     const int col = idx % dstate;
// //                     const int global_row = d_start + local_row;
                    
// //                     int buffer_idx = batch_idx * D * L * dstate + 
// //                                     global_row * L * dstate + 
// //                                     time_idx * dstate + col;
// //                     float stored_val = X_temp[buffer_idx];
// //                     tile_buffer_bf16[local_row * dstate + col] = float_to_bf16_reinterpret(stored_val);
// //                 }
// //                 __syncthreads();
                
// //                 for (int ij = tid; ij < tile_rows * gram_size; ij += kBlockSize) {
// //                     const int local_i = ij / gram_size;
// //                     const int j = ij % gram_size;
// //                     const int global_i = d_start + local_i;
                    
// //                     if (global_i < gram_size && j < gram_size) {
// //                         float sum = 0.0f;
// //                         if (j >= d_start && j < d_end) {
// //                             for (int k = 0; k < dstate; ++k) {
// //                                 float a_val = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
// //                                 float b_val = __bfloat162float(tile_buffer_bf16[(j - d_start) * dstate + k]);
// //                                 sum += a_val * b_val;
// //                             }
// //                         } else {
// //                             for (int k = 0; k < dstate; ++k) {
// //                                 int j_idx = batch_idx * D * L * dstate + j * L * dstate + time_idx * dstate + k;
// //                                 float a_val = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
// //                                 float b_val = X_temp[j_idx];
// //                                 sum += a_val * b_val;
// //                             }
// //                         }
// //                         atomicAdd(&gram_A_fp32[global_i * gram_size + j], sum);
// //                     }
// //                 }
// //                 __syncthreads();
// //             }
// //         } else {
// //             for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //                 const int d_end = min(d_start + kTileSize, D);
// //                 const int tile_cols = d_end - d_start;
                
// //                 for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
// //                     const int row = idx / tile_cols;
// //                     const int local_col = idx % tile_cols;
// //                     const int global_col = d_start + local_col;
                    
// //                     int buffer_idx = batch_idx * D * L * dstate + 
// //                                     global_col * L * dstate + 
// //                                     time_idx * dstate + row;
// //                     float stored_val = X_temp[buffer_idx];
// //                     tile_buffer_bf16[row * tile_cols + local_col] = float_to_bf16_reinterpret(stored_val);
// //                 }
// //                 __syncthreads();
                
// //                 for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
// //                     const int i = ij / gram_size;
// //                     const int j = ij % gram_size;
                    
// //                     float sum = 0.0f;
// //                     for (int k = 0; k < tile_cols; ++k) {
// //                         float a_val = __bfloat162float(tile_buffer_bf16[i * tile_cols + k]);
// //                         float b_val = __bfloat162float(tile_buffer_bf16[j * tile_cols + k]);
// //                         sum += a_val * b_val;
// //                     }
// //                     atomicAdd(&gram_A_fp32[ij], sum);
// //                 }
// //                 __syncthreads();
// //             }
// //         }
        
// //         // Convert A to BF16, compute A², then B = b*A + c*A²
// //         const int gram_storage_needed = 2 * gram_size * gram_size;
// //         __nv_bfloat16* gram_A_bf16 = tile_buffer_bf16;
        
// //         for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
// //             gram_A_bf16[ij] = __float2bfloat16(gram_A_fp32[ij]);
// //         }
// //         __syncthreads();
        
// //         __nv_bfloat16* temp_A2_bf16 = gram_A_bf16 + gram_size * gram_size;
        
// //         for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
// //             const int i = ij / gram_size;
// //             const int j = ij % gram_size;
            
// //             float A2_ij_fp32 = 0.0f;
// //             for (int k = 0; k < gram_size; ++k) {
// //                 float a_val = __bfloat162float(gram_A_bf16[i * gram_size + k]);
// //                 float b_val = __bfloat162float(gram_A_bf16[k * gram_size + j]);
// //                 A2_ij_fp32 += a_val * b_val;
// //             }
// //             temp_A2_bf16[ij] = __float2bfloat16(A2_ij_fp32);
// //         }
// //         __syncthreads();
        
// //         for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
// //             float A_ij = __bfloat162float(gram_A_bf16[ij]);
// //             float A2_ij = __bfloat162float(temp_A2_bf16[ij]);
// //             float B_fp32 = b * A_ij + c * A2_ij;
// //             gram_A_bf16[ij] = __float2bfloat16(B_fp32);
// //         }
// //         __syncthreads();
        
// //         // Apply X = a*X + B@X
// //         __nv_bfloat16* x_tile_buffer = tile_buffer_bf16 + gram_storage_needed;
        
// //         if (!transposed) {
// //             for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //                 const int d_end = min(d_start + kTileSize, D);
// //                 const int tile_rows = d_end - d_start;
                
// //                 for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //                     const int local_row = idx / dstate;
// //                     const int col = idx % dstate;
// //                     const int global_row = d_start + local_row;
                    
// //                     int buffer_idx = batch_idx * D * L * dstate + 
// //                                     global_row * L * dstate + 
// //                                     time_idx * dstate + col;
// //                     float stored_val = X_temp[buffer_idx];
// //                     x_tile_buffer[local_row * dstate + col] = float_to_bf16_reinterpret(stored_val);
// //                 }
// //                 __syncthreads();
                
// //                 for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //                     const int local_row = idx / dstate;
// //                     const int col = idx % dstate;
// //                     const int global_row = d_start + local_row;
                    
// //                     float x_val = __bfloat162float(x_tile_buffer[local_row * dstate + col]);
                    
// //                     float sum = 0.0f;
// //                     for (int k = 0; k < gram_size; ++k) {
// //                         float x_kj;
// //                         if (k >= d_start && k < d_end) {
// //                             x_kj = __bfloat162float(x_tile_buffer[(k - d_start) * dstate + col]);
// //                         } else {
// //                             int idx_kj = batch_idx * D * L * dstate + k * L * dstate + time_idx * dstate + col;
// //                             x_kj = X_temp[idx_kj];
// //                         }
// //                         float b_ik = __bfloat162float(gram_A_bf16[global_row * gram_size + k]);
// //                         sum += b_ik * x_kj;
// //                     }
                    
// //                     float x_new_fp32 = a * x_val + sum;
// //                     __nv_bfloat16 x_new_bf16 = __float2bfloat16(x_new_fp32);
// //                     float x_new_rounded = __bfloat162float(x_new_bf16);
                    
// //                     int buffer_idx = batch_idx * D * L * dstate + 
// //                                     global_row * L * dstate + 
// //                                     time_idx * dstate + col;
// //                     X_temp[buffer_idx] = x_new_rounded;
// //                 }
// //                 __syncthreads();
// //             }
// //         } else {
// //             for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //                 const int d_end = min(d_start + kTileSize, D);
// //                 const int tile_cols = d_end - d_start;
                
// //                 for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
// //                     const int row = idx / tile_cols;
// //                     const int local_col = idx % tile_cols;
// //                     const int global_col = d_start + local_col;
                    
// //                     int buffer_idx = batch_idx * D * L * dstate + 
// //                                     global_col * L * dstate + 
// //                                     time_idx * dstate + row;
// //                     float stored_val = X_temp[buffer_idx];
// //                     x_tile_buffer[row * tile_cols + local_col] = float_to_bf16_reinterpret(stored_val);
// //                 }
// //                 __syncthreads();
                
// //                 for (int idx = tid; idx < gram_size * tile_cols; idx += kBlockSize) {
// //                     const int n = idx / tile_cols;
// //                     const int local_d = idx % tile_cols;
// //                     const int d = d_start + local_d;
                    
// //                     float x_val = __bfloat162float(x_tile_buffer[n * tile_cols + local_d]);
                    
// //                     float sum = 0.0f;
// //                     for (int k = 0; k < gram_size; ++k) {
// //                         float x_dk;
// //                         if (d >= d_start && d < d_end) {
// //                             x_dk = __bfloat162float(x_tile_buffer[k * tile_cols + local_d]);
// //                         } else {
// //                             int idx_dk = batch_idx * D * L * dstate + d * L * dstate + time_idx * dstate + k;
// //                             x_dk = X_temp[idx_dk];
// //                         }
// //                         float b_nk = __bfloat162float(gram_A_bf16[n * gram_size + k]);
// //                         sum += b_nk * x_dk;
// //                     }
                    
// //                     float x_new_fp32 = a * x_val + sum;
// //                     __nv_bfloat16 x_new_bf16 = __float2bfloat16(x_new_fp32);
// //                     float x_new_rounded = __bfloat162float(x_new_bf16);
                    
// //                     int buffer_idx = batch_idx * D * L * dstate + 
// //                                     d * L * dstate + 
// //                                     time_idx * dstate + n;
// //                     X_temp[buffer_idx] = x_new_rounded;
// //                 }
// //                 __syncthreads();
// //             }
// //         }
// //     }
    
// //     // Now X_temp contains X_4, ready for backward pass through 5th iteration
    
// //     // ========== PHASE 2: Backward Through 5th Iteration ==========
    
// //     // Compute A_4 = X_4 @ X_4.T (needed for gradient computation)
// //     for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
// //         gram_A_fp32[idx] = 0.0f;
// //     }
// //     __syncthreads();
    
// //     if (!transposed) {
// //         for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //             const int d_end = min(d_start + kTileSize, D);
// //             const int tile_rows = d_end - d_start;
            
// //             for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //                 const int local_row = idx / dstate;
// //                 const int col = idx % dstate;
// //                 const int global_row = d_start + local_row;
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_row * L * dstate + 
// //                                 time_idx * dstate + col;
// //                 float stored_val = X_temp[buffer_idx];
// //                 tile_buffer_bf16[local_row * dstate + col] = float_to_bf16_reinterpret(stored_val);
// //             }
// //             __syncthreads();
            
// //             for (int ij = tid; ij < tile_rows * gram_size; ij += kBlockSize) {
// //                 const int local_i = ij / gram_size;
// //                 const int j = ij % gram_size;
// //                 const int global_i = d_start + local_i;
                
// //                 if (global_i < gram_size && j < gram_size) {
// //                     float sum = 0.0f;
// //                     if (j >= d_start && j < d_end) {
// //                         for (int k = 0; k < dstate; ++k) {
// //                             float a_val = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
// //                             float b_val = __bfloat162float(tile_buffer_bf16[(j - d_start) * dstate + k]);
// //                             sum += a_val * b_val;
// //                         }
// //                     } else {
// //                         for (int k = 0; k < dstate; ++k) {
// //                             int j_idx = batch_idx * D * L * dstate + j * L * dstate + time_idx * dstate + k;
// //                             float a_val = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
// //                             float b_val = X_temp[j_idx];
// //                             sum += a_val * b_val;
// //                         }
// //                     }
// //                     atomicAdd(&gram_A_fp32[global_i * gram_size + j], sum);
// //                 }
// //             }
// //             __syncthreads();
// //         }
// //     } else {
// //         for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //             const int d_end = min(d_start + kTileSize, D);
// //             const int tile_cols = d_end - d_start;
            
// //             for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
// //                 const int row = idx / tile_cols;
// //                 const int local_col = idx % tile_cols;
// //                 const int global_col = d_start + local_col;
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_col * L * dstate + 
// //                                 time_idx * dstate + row;
// //                 float stored_val = X_temp[buffer_idx];
// //                 tile_buffer_bf16[row * tile_cols + local_col] = float_to_bf16_reinterpret(stored_val);
// //             }
// //             __syncthreads();
            
// //             for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
// //                 const int i = ij / gram_size;
// //                 const int j = ij % gram_size;
                
// //                 float sum = 0.0f;
// //                 for (int k = 0; k < tile_cols; ++k) {
// //                     float a_val = __bfloat162float(tile_buffer_bf16[i * tile_cols + k]);
// //                     float b_val = __bfloat162float(tile_buffer_bf16[j * tile_cols + k]);
// //                     sum += a_val * b_val;
// //                 }
// //                 atomicAdd(&gram_A_fp32[ij], sum);
// //             }
// //             __syncthreads();
// //         }
// //     }
    
// //     // Convert A_4 to BF16, compute A_4², then B_4 = b*A_4 + c*A_4²
// //     const int gram_storage_needed = 2 * gram_size * gram_size;
// //     __nv_bfloat16* gram_A_bf16 = tile_buffer_bf16;
    
// //     for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
// //         gram_A_bf16[ij] = __float2bfloat16(gram_A_fp32[ij]);
// //     }
// //     __syncthreads();
    
// //     __nv_bfloat16* temp_A2_bf16 = gram_A_bf16 + gram_size * gram_size;
    
// //     for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
// //         const int i = ij / gram_size;
// //         const int j = ij % gram_size;
        
// //         float A2_ij_fp32 = 0.0f;
// //         for (int k = 0; k < gram_size; ++k) {
// //             float a_val = __bfloat162float(gram_A_bf16[i * gram_size + k]);
// //             float b_val = __bfloat162float(gram_A_bf16[k * gram_size + j]);
// //             A2_ij_fp32 += a_val * b_val;
// //         }
// //         temp_A2_bf16[ij] = __float2bfloat16(A2_ij_fp32);
// //     }
// //     __syncthreads();
    
// //     // Compute B_4, store in gram_A_fp32 (reuse as FP32 storage)
// //     for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
// //         float A_ij = __bfloat162float(gram_A_bf16[ij]);
// //         float A2_ij = __bfloat162float(temp_A2_bf16[ij]);
// //         gram_A_fp32[ij] = b * A_ij + c * A2_ij;  // B_4 in FP32
// //     }
// //     __syncthreads();
    
// //     // Now we have: X_4 in X_temp, A_4 in gram_A_bf16, B_4 in gram_A_fp32
    
// //     // Step 1: Load grad_output (dX_5) and initialize dX_4
// //     // Use grad_delta as temporary buffer for dX_4
// //     float* dX_4_temp = grad_delta;
    
// //     for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //         const int d_end = min(d_start + kTileSize, D);
// //         const int tile_rows = d_end - d_start;
        
// //         for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //             const int local_row = idx / dstate;
// //             const int col = idx % dstate;
// //             const int global_row = d_start + local_row;
            
// //             int buffer_idx = batch_idx * D * L * dstate + 
// //                             global_row * L * dstate + 
// //                             time_idx * dstate + col;
            
// //             // Load grad_output and initialize dX_4 = a * dX_5
// //             float dX_5 = grad_output[buffer_idx];
// //             dX_4_temp[buffer_idx] = a * dX_5;
// //         }
// //     }
// //     __syncthreads();
    
// //     // Step 2: Compute dX_4 += gradient through B_4@X_4
// //     // Also accumulate dB_4
// //     // Reuse partial_sums area for dB_4 accumulator (need gram_size²)
// //     float* dB_4_accum = partial_sums;
// //     for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
// //         dB_4_accum[idx] = 0.0f;
// //     }
// //     __syncthreads();
    
// //     if (!transposed) {
// //         // Not transposed: X is [D, N], B_4 is [D, D]
// //         // Forward: X_new = a*X + B_4 @ X
// //         // Backward: dX_4 += B_4.T @ dX_5, dB_4 = dX_5 @ X_4.T
        
// //         for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //             const int d_end = min(d_start + kTileSize, D);
// //             const int tile_rows = d_end - d_start;
            
// //             // Load X_4 and dX_5 tiles
// //             for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //                 const int local_row = idx / dstate;
// //                 const int col = idx % dstate;
// //                 const int global_row = d_start + local_row;
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_row * L * dstate + 
// //                                 time_idx * dstate + col;
                
// //                 // Store X_4 in tile_buffer
// //                 float x_4_val = X_temp[buffer_idx];
// //                 tile_buffer_bf16[local_row * dstate + col] = float_to_bf16_reinterpret(x_4_val);
// //             }
// //             __syncthreads();
            
// //             // Compute dX_4 += B_4.T @ dX_5 for this tile
// //             for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //                 const int local_row = idx / dstate;
// //                 const int col = idx % dstate;
// //                 const int global_row = d_start + local_row;
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_row * L * dstate + 
// //                                 time_idx * dstate + col;
                
// //                 // (B_4.T @ dX_5)[i,j] = sum_k B_4[k,i] * dX_5[k,j]
// //                 float sum = 0.0f;
// //                 for (int k = 0; k < gram_size; ++k) {
// //                     int k_idx = batch_idx * D * L * dstate + k * L * dstate + time_idx * dstate + col;
// //                     float dX_5_kj = grad_output[k_idx];
// //                     float B_4_ki = gram_A_fp32[k * gram_size + global_row];
// //                     sum += B_4_ki * dX_5_kj;
// //                 }
// //                 dX_4_temp[buffer_idx] += sum;
// //             }
            
// //             // Compute dB_4 contribution: dB_4[i,j] += sum_k dX_5[i,k] * X_4[j,k]
// //             for (int ij = tid; ij < tile_rows * gram_size; ij += kBlockSize) {
// //                 const int local_i = ij / gram_size;
// //                 const int j = ij % gram_size;
// //                 const int global_i = d_start + local_i;
                
// //                 if (global_i < gram_size) {
// //                     float sum = 0.0f;
// //                     for (int k = 0; k < dstate; ++k) {
// //                         int i_idx = batch_idx * D * L * dstate + global_i * L * dstate + time_idx * dstate + k;
// //                         float dX_5_ik = grad_output[i_idx];
                        
// //                         float X_4_jk;
// //                         if (j >= d_start && j < d_end) {
// //                             X_4_jk = __bfloat162float(tile_buffer_bf16[(j - d_start) * dstate + k]);
// //                         } else {
// //                             int j_idx = batch_idx * D * L * dstate + j * L * dstate + time_idx * dstate + k;
// //                             X_4_jk = X_temp[j_idx];
// //                         }
// //                         sum += dX_5_ik * X_4_jk;
// //                     }
// //                     atomicAdd(&dB_4_accum[global_i * gram_size + j], sum);
// //                 }
// //             }
// //             __syncthreads();
// //         }
// //     } else {
// //         // Transposed: X_storage is [D, N] (logical [N, D]), B_4 is [N, N]
// //         // Forward: X_storage @ B_4.T (right multiply)
// //         // Backward: dX_4_storage += dX_5_storage @ B_4, dB_4 = dX_5_storage.T @ X_4_storage
        
// //         for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //             const int d_end = min(d_start + kTileSize, D);
// //             const int tile_cols = d_end - d_start;
            
// //             // Load X_4_storage tile [dstate, tile_cols]
// //             for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
// //                 const int row = idx / tile_cols;
// //                 const int local_col = idx % tile_cols;
// //                 const int global_col = d_start + local_col;
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_col * L * dstate + 
// //                                 time_idx * dstate + row;
// //                 float x_4_val = X_temp[buffer_idx];
// //                 tile_buffer_bf16[row * tile_cols + local_col] = float_to_bf16_reinterpret(x_4_val);
// //             }
// //             __syncthreads();
            
// //             // Compute dX_4_storage += dX_5_storage @ B_4
// //             for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
// //                 const int row = idx / tile_cols;  // n index
// //                 const int local_col = idx % tile_cols;
// //                 const int global_col = d_start + local_col;  // d index
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_col * L * dstate + 
// //                                 time_idx * dstate + row;
                
// //                 // (dX_5_storage @ B_4)[d,n] = sum_k dX_5_storage[d,k] * B_4[k,n]
// //                 float sum = 0.0f;
// //                 for (int k = 0; k < gram_size; ++k) {
// //                     int k_idx = batch_idx * D * L * dstate + global_col * L * dstate + time_idx * dstate + k;
// //                     float dX_5_dk = grad_output[k_idx];
// //                     float B_4_kn = gram_A_fp32[k * gram_size + row];
// //                     sum += dX_5_dk * B_4_kn;
// //                 }
// //                 dX_4_temp[buffer_idx] += sum;
// //             }
            
// //             // Compute dB_4: dB_4[n1,n2] += sum_d dX_5_storage[d,n1] * X_4_storage[d,n2]
// //             for (int n1n2 = tid; n1n2 < gram_size * gram_size; n1n2 += kBlockSize) {
// //                 const int n1 = n1n2 / gram_size;
// //                 const int n2 = n1n2 % gram_size;
                
// //                 float sum = 0.0f;
// //                 for (int local_d = 0; local_d < tile_cols; ++local_d) {
// //                     const int d = d_start + local_d;
// //                     int d_n1_idx = batch_idx * D * L * dstate + d * L * dstate + time_idx * dstate + n1;
// //                     float dX_5_d_n1 = grad_output[d_n1_idx];
// //                     float X_4_d_n2 = __bfloat162float(tile_buffer_bf16[n2 * tile_cols + local_d]);
// //                     sum += dX_5_d_n1 * X_4_d_n2;
// //                 }
// //                 atomicAdd(&dB_4_accum[n1 * gram_size + n2], sum);
// //             }
// //             __syncthreads();
// //         }
// //     }
    
// //     // Step 3: Compute dA_4 from dB_4
// //     // dA_4 = b*dB_4 + c*(dB_4 @ A_4.T + dB_4.T @ A_4)
// //     // Reuse dX_accumulator for dA_4
// //     float* dA_4_accum = dX_accumulator;
    
// //     for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
// //         const int i = ij / gram_size;
// //         const int j = ij % gram_size;
        
// //         float dB_4_ij = dB_4_accum[ij];
// //         float dA_4_from_linear = b * dB_4_ij;
        
// //         // Compute (dB_4 @ A_4.T)[i,j] = sum_k dB_4[i,k] * A_4[j,k]
// //         float sum1 = 0.0f;
// //         for (int k = 0; k < gram_size; ++k) {
// //             float dB_4_ik = dB_4_accum[i * gram_size + k];
// //             float A_4_jk = __bfloat162float(gram_A_bf16[j * gram_size + k]);
// //             sum1 += dB_4_ik * A_4_jk;
// //         }
        
// //         // Compute (dB_4.T @ A_4)[i,j] = sum_k dB_4[k,i] * A_4[k,j]
// //         float sum2 = 0.0f;
// //         for (int k = 0; k < gram_size; ++k) {
// //             float dB_4_ki = dB_4_accum[k * gram_size + i];
// //             float A_4_kj = __bfloat162float(gram_A_bf16[k * gram_size + j]);
// //             sum2 += dB_4_ki * A_4_kj;
// //         }
        
// //         dA_4_accum[ij] = dA_4_from_linear + c * (sum1 + sum2);
// //     }
// //     __syncthreads();
    
// //     // Step 4: Compute dX_4 from dA_4
// //     // dX_4 += (dA_4 + dA_4.T) @ X_4 (or right multiply for transposed)
    
// //     if (!transposed) {
// //         // dX_4 += (dA_4 + dA_4.T) @ X_4
// //         for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //             const int d_end = min(d_start + kTileSize, D);
// //             const int tile_rows = d_end - d_start;
            
// //             // Load X_4 tile
// //             for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //                 const int local_row = idx / dstate;
// //                 const int col = idx % dstate;
// //                 const int global_row = d_start + local_row;
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_row * L * dstate + 
// //                                 time_idx * dstate + col;
// //                 float x_4_val = X_temp[buffer_idx];
// //                 tile_buffer_bf16[local_row * dstate + col] = float_to_bf16_reinterpret(x_4_val);
// //             }
// //             __syncthreads();
            
// //             // Compute contribution
// //             for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //                 const int local_row = idx / dstate;
// //                 const int col = idx % dstate;
// //                 const int global_row = d_start + local_row;
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_row * L * dstate + 
// //                                 time_idx * dstate + col;
                
// //                 // ((dA_4 + dA_4.T) @ X_4)[i,j] = sum_k (dA_4[i,k] + dA_4[k,i]) * X_4[k,j]
// //                 float sum = 0.0f;
// //                 for (int k = 0; k < gram_size; ++k) {
// //                     float dA_sym = dA_4_accum[global_row * gram_size + k] + dA_4_accum[k * gram_size + global_row];
                    
// //                     float X_4_kj;
// //                     if (k >= d_start && k < d_end) {
// //                         X_4_kj = __bfloat162float(tile_buffer_bf16[(k - d_start) * dstate + col]);
// //                     } else {
// //                         int k_idx = batch_idx * D * L * dstate + k * L * dstate + time_idx * dstate + col;
// //                         X_4_kj = X_temp[k_idx];
// //                     }
// //                     sum += dA_sym * X_4_kj;
// //                 }
// //                 dX_4_temp[buffer_idx] += sum;
// //             }
// //             __syncthreads();
// //         }
// //     } else {
// //         // dX_4_storage += X_4_storage @ (dA_4 + dA_4.T)
// //         for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //             const int d_end = min(d_start + kTileSize, D);
// //             const int tile_cols = d_end - d_start;
            
// //             // Load X_4_storage tile
// //             for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
// //                 const int row = idx / tile_cols;
// //                 const int local_col = idx % tile_cols;
// //                 const int global_col = d_start + local_col;
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_col * L * dstate + 
// //                                 time_idx * dstate + row;
// //                 float x_4_val = X_temp[buffer_idx];
// //                 tile_buffer_bf16[row * tile_cols + local_col] = float_to_bf16_reinterpret(x_4_val);
// //             }
// //             __syncthreads();
            
// //             // Compute contribution
// //             for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
// //                 const int row = idx / tile_cols;  // n index
// //                 const int local_col = idx % tile_cols;
// //                 const int global_col = d_start + local_col;  // d index
                
// //                 int buffer_idx = batch_idx * D * L * dstate + 
// //                                 global_col * L * dstate + 
// //                                 time_idx * dstate + row;
                
// //                 // (X_4_storage @ (dA_4 + dA_4.T))[d,n] = sum_k X_4_storage[d,k] * (dA_4[k,n] + dA_4[n,k])
// //                 float sum = 0.0f;
// //                 for (int k = 0; k < gram_size; ++k) {
// //                     float dA_sym = dA_4_accum[k * gram_size + row] + dA_4_accum[row * gram_size + k];
// //                     float X_4_dk = __bfloat162float(tile_buffer_bf16[k * tile_cols + local_col]);
// //                     sum += X_4_dk * dA_sym;
// //                 }
// //                 dX_4_temp[buffer_idx] += sum;
// //             }
// //             __syncthreads();
// //         }
// //     }
    
// //     // Step 5: Compute gradient through normalization
// //     // d(b_t)[i,j] = (dX_4[i,j] - (sum_kl dX_4[k,l] * X_4_norm[k,l]) * X_4_norm[i,j]) / norm
// //     // First compute dot product: dnorm_from_loss = sum dX_4 * X_4_norm
    
// //     float dnorm_local = 0.0f;
// //     for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //         const int d_end = min(d_start + kTileSize, D);
// //         const int tile_rows = d_end - d_start;
        
// //         for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //             const int local_row = idx / dstate;
// //             const int col = idx % dstate;
// //             const int global_row = d_start + local_row;
            
// //             int buffer_idx = batch_idx * D * L * dstate + 
// //                             global_row * L * dstate + 
// //                             time_idx * dstate + col;
            
// //             float dX_4_val = dX_4_temp[buffer_idx];
// //             float X_4_norm_val = X_temp[buffer_idx];  // X_4 is normalized
// //             dnorm_local += dX_4_val * X_4_norm_val;
// //         }
// //     }
    
// //     // Block reduction for dnorm
// //     partial_sums[tid] = dnorm_local;
// //     __syncthreads();
    
// //     for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
// //         if (tid < stride) {
// //             partial_sums[tid] += partial_sums[tid + stride];
// //         }
// //         __syncthreads();
// //     }
    
// //     float dnorm_from_loss = partial_sums[0];
// //     __syncthreads();
    
// //     // Compute d(b_t_bf16) and accumulate gradients for u, delta, B
// //     float grad_u_local = 0.0f;
// //     float grad_delta_local = 0.0f;
    
// //     for (int d_start = 0; d_start < D; d_start += kTileSize) {
// //         const int d_end = min(d_start + kTileSize, D);
// //         const int tile_rows = d_end - d_start;
        
// //         for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
// //             const int local_row = idx / dstate;
// //             const int col = idx % dstate;
// //             const int global_row = d_start + local_row;
            
// //             int buffer_idx = batch_idx * D * L * dstate + 
// //                             global_row * L * dstate + 
// //                             time_idx * dstate + col;
            
// //             float dX_4_val = dX_4_temp[buffer_idx];
// //             float X_4_norm_val = X_temp[buffer_idx];
            
// //             // Gradient through normalization
// //             float d_b_t_bf16 = (dX_4_val - dnorm_from_loss * X_4_norm_val) / norm;
            
// //             // Straight-through for BF16: gradient passes unchanged
// //             float d_b_t = d_b_t_bf16;
            
// //             // Load u, delta, B values
// //             int u_idx = batch_idx * u_batch_stride + global_row * u_d_stride + time_idx;
// //             float u_val = to_float(u[u_idx]);
            
// //             int delta_idx = batch_idx * delta_batch_stride + global_row * delta_d_stride + time_idx;
// //             float delta_val = to_float(delta[delta_idx]);
            
// //             float B_val;
// //             int B_idx;
// //             if (!is_variable_B) {
// //                 B_idx = global_row * B_d_stride + col * B_dstate_stride;
// //                 B_val = to_float(B[B_idx]);
// //             } else {
// //                 int group_size = (D + n_groups - 1) / n_groups;
// //                 int group_id = min(global_row / group_size, n_groups - 1);
// //                 B_idx = batch_idx * B_batch_stride + 
// //                         group_id * B_group_stride +
// //                         time_idx * dstate + col;
// //                 B_val = to_float(B[B_idx]);
// //             }
            
// //             // Gradients: b_t = alpha * delta * B * u
// //             float grad_u_contrib = alpha * delta_val * B_val * d_b_t;
// //             float grad_delta_contrib = alpha * B_val * u_val * d_b_t;
// //             float grad_B_contrib = alpha * delta_val * u_val * d_b_t;
            
// //             // Accumulate for u and delta (sum over dstate dimension)
// //             grad_u_local += grad_u_contrib;
// //             grad_delta_local += grad_delta_contrib;
            
// //             // For grad_B, use atomicAdd (handle both constant and variable B)
// //             atomicAdd(&grad_B[B_idx], grad_B_contrib);
// //         }
// //     }
    
// //     // Final: write accumulated gradients for u and delta
// //     // Each thread accumulates over dstate, need to distribute back
// //     // Actually, we need to sum over dstate for each (batch, dim, time)
// //     // This requires more careful handling - use shared memory reduction per dim
    
// //     // For simplicity in this first implementation, write directly (will be inefficient but correct)
// //     for (int d = 0; d < D; ++d) {
// //         float grad_u_sum = 0.0f;
// //         float grad_delta_sum = 0.0f;
        
// //         for (int n = tid; n < dstate; n += kBlockSize) {
// //             int buffer_idx = batch_idx * D * L * dstate + d * L * dstate + time_idx * dstate + n;
            
// //             float dX_4_val = dX_4_temp[buffer_idx];
// //             float X_4_norm_val = X_temp[buffer_idx];
// //             float d_b_t_bf16 = (dX_4_val - dnorm_from_loss * X_4_norm_val) / norm;
// //             float d_b_t = d_b_t_bf16;
            
// //             int u_idx = batch_idx * u_batch_stride + d * u_d_stride + time_idx;
// //             float u_val = to_float(u[u_idx]);
            
// //             int delta_idx = batch_idx * delta_batch_stride + d * delta_d_stride + time_idx;
// //             float delta_val = to_float(delta[delta_idx]);
            
// //             float B_val;
// //             if (!is_variable_B) {
// //                 B_val = to_float(B[d * B_d_stride + n * B_dstate_stride]);
// //             } else {
// //                 int group_size = (D + n_groups - 1) / n_groups;
// //                 int group_id = min(d / group_size, n_groups - 1);
// //                 B_val = to_float(B[batch_idx * B_batch_stride + 
// //                                    group_id * B_group_stride +
// //                                    time_idx * dstate + n]);
// //             }
            
// //             grad_u_sum += alpha * delta_val * B_val * d_b_t;
// //             grad_delta_sum += alpha * B_val * u_val * d_b_t;
// //         }
        
// //         // Block reduction
// //         partial_sums[tid] = grad_u_sum;
// //         __syncthreads();
// //         for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
// //             if (tid < stride) {
// //                 partial_sums[tid] += partial_sums[tid + stride];
// //             }
// //             __syncthreads();
// //         }
// //         if (tid == 0) {
// //             int u_idx = batch_idx * u_batch_stride + d * u_d_stride + time_idx;
// //             atomicAdd(&grad_u[u_idx], partial_sums[0]);
// //         }
// //         __syncthreads();
        
// //         partial_sums[tid] = grad_delta_sum;
// //         __syncthreads();
// //         for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
// //             if (tid < stride) {
// //                 partial_sums[tid] += partial_sums[tid + stride];
// //             }
// //             __syncthreads();
// //         }
// //         if (tid == 0) {
// //             int delta_idx = batch_idx * delta_batch_stride + d * delta_d_stride + time_idx;
// //             atomicAdd(&grad_delta[delta_idx], partial_sums[0]);
// //         }
// //         __syncthreads();
// //     }
// // }

////////////////////////////////////////////////////////////////////////////////////////////////////
// 5-Step Newton-Schulz with On-the-Fly b_t Computation (Speed Optimized)
// Uses bfloat16 for NS iterations (as per official Muon paper for numerical stability)
// Outputs float32 for scan operations (to avoid accumulation errors)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper: Convert float to bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float x) {
    return __float2bfloat16(x);
}

// Helper: Convert bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Helper: Convert weight_t to float (handles complex)
template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return float(x);
}

template <typename T>
__device__ __forceinline__ float to_float(c10::complex<T> x) {
    return float(x.real());  // For complex, use real part
}

// Helper: Check if weight_t is complex
template<typename T>
struct is_complex_type {
    static constexpr bool value = false;
};

template<typename T>
struct is_complex_type<c10::complex<T>> {
    static constexpr bool value = true;
};

// Helper: Get complex value from weight_t (returns complex_t for complex, float for real)
template<typename weight_t>
__device__ __forceinline__ complex_t get_complex_value(weight_t x) {
    if constexpr (is_complex_type<weight_t>::value) {
        return complex_t(x.real(), x.imag());
    } else {
        return complex_t(float(x), 0.0f);
    }
}

// Helper: Reinterpret float as bfloat16 without rounding
// Used when reading values that are already in BF16 precision stored as float
// When we store via __bfloat162float, the BF16 bits are in upper 16 bits of float
__device__ __forceinline__ __nv_bfloat16 float_to_bf16_reinterpret(float f) {
    // BF16 representation is in upper 16 bits of FP32
    // We need to reconstruct the __nv_bfloat16 from these bits
    // Use __uint_as_float and bit manipulation
    unsigned int f_bits = __float_as_uint(f);
    unsigned short bf16_raw = static_cast<unsigned short>(f_bits >> 16);
    
    // Convert back to float with only upper 16 bits (BF16 format)
    unsigned int reconstructed = static_cast<unsigned int>(bf16_raw) << 16;
    float bf16_as_fp32 = __uint_as_float(reconstructed);
    
    // Now convert to __nv_bfloat16 (this won't add rounding since bits are already BF16)
    return __float2bfloat16(bf16_as_fp32);
}

// Helper: True atomic add for bfloat16 using atomicCAS on unsigned int
__device__ __forceinline__ void atomicAddBF16(__nv_bfloat16* address, float val) {
    // Use unsigned int atomic operations (atomicCAS requires int/uint)
    // BF16 is 16 bits, we'll manipulate as part of 32-bit word
    unsigned int* base_address = (unsigned int*)((size_t)address & ~3);
    unsigned int offset = ((size_t)address & 2) ? 16 : 0;  // 0 or 16 bits offset
    
    unsigned int old_val = *base_address;
    unsigned int assumed;
    
    do {
        assumed = old_val;
        
        // Extract the BF16 value from the 32-bit word
        unsigned short old_bf16_bits = (unsigned short)((assumed >> offset) & 0xFFFF);
        __nv_bfloat16 old_bf16 = *reinterpret_cast<__nv_bfloat16*>(&old_bf16_bits);
        
        // Add in FP32
        float old_fp32 = __bfloat162float(old_bf16);
        float new_fp32 = old_fp32 + val;
        __nv_bfloat16 new_bf16 = __float2bfloat16(new_fp32);
        unsigned short new_bf16_bits = *reinterpret_cast<unsigned short*>(&new_bf16);
        
        // Insert new BF16 value into 32-bit word
        unsigned int new_val = (assumed & ~(0xFFFF << offset)) | (((unsigned int)new_bf16_bits) << offset);
        
        // Atomic compare-and-swap on 32-bit word
        old_val = atomicCAS(base_address, assumed, new_val);
        
    } while (assumed != old_val);
}

template<typename input_t, typename weight_t, int kBlockSize = 256, int kTileSize = 128>
__global__ void newton_schulz_velocity_5step_kernel(
    const input_t* __restrict__ u,          // [B, D, L] - input type (float16/bfloat16/float32)
    const input_t* __restrict__ delta,      // [B, D, L] - input type
    const weight_t* __restrict__ B,         // [D, N] or [B, G, L, N] if variable - weight type
    float* __restrict__ velocity_ortho,     // [B, D, L, N] - output in FLOAT32 for scan
    float* __restrict__ X_4_buffer,         // [B, D, L, N] - store X_4 for backward
    float alpha,
    int B_dim, int D, int L, int dstate, int t_start,
    int u_batch_stride, int u_d_stride,
    int delta_batch_stride, int delta_d_stride,
    int B_batch_stride, int B_group_stride,
    int B_d_stride, int B_dstate_stride,
    bool is_variable_B, int n_groups
) {
    // Block indices
    const int batch_idx = blockIdx.x;
    const int time_local = blockIdx.y;
    const int time_idx = t_start + time_local;
    
    if (batch_idx >= B_dim || time_idx >= L) return;
    
    const int tid = threadIdx.x;
    
    // Newton-Schulz coefficients
    constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    
    // Determine transpose: PyTorch transposes tall matrices (rows > cols)
    // For b_t with shape [D, N]: transpose if D > N
    const bool transposed = (D > dstate);
    const int gram_size = transposed ? dstate : D;  // min(D, dstate)
    
    // Shared memory layout - HYBRID APPROACH (matches PyTorch accumulation behavior)
    // tile_buffer_bf16: stores X values as native bfloat16 during NS iterations
    // gram_A_fp32: stores Gram matrix A in FP32 during accumulation (like PyTorch internal)
    // gram_B_bf16: stores matrix B in BF16 after computation
    // partial_sums: for reductions (only used for norm computation)
    extern __shared__ float smem[];
    __nv_bfloat16* tile_buffer_bf16 = (__nv_bfloat16*)smem;         // [kTileSize, max(D,N)] in BF16
    
    // Calculate offset for gram matrix - use FP32 for accumulation to match PyTorch
    const int tile_buffer_size = kTileSize * (transposed ? D : dstate);
    float* gram_A_fp32 = (float*)(tile_buffer_bf16 + tile_buffer_size);  // [gram_size, gram_size] in FP32
    float* partial_sums = gram_A_fp32 + gram_size * gram_size;           // [kBlockSize] in FP32
    
    // ========== STEP 0: Compute b_t, convert to BF16, then normalize ==========
    // CRITICAL: Match PyTorch order exactly - convert to BF16 FIRST, then normalize
    // For complex weights, b_t is complex and we store both real and imag parts
    constexpr bool kIsComplex = is_complex_type<weight_t>::value;
    float norm_sq_local = 0.0f;
    float norm_sq_fp32_local = 0.0f;  // For comparison
    
    // Phase A: Compute b_t, convert to BF16 immediately, store to global, accumulate norm
    for (int d_start = 0; d_start < D; d_start += kTileSize) {
        const int d_end = min(d_start + kTileSize, D);
        const int tile_rows = d_end - d_start;
        
        // Each thread computes b_t elements for this tile
        for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
            const int local_row = idx / dstate;
            const int col = idx % dstate;
            const int global_row = d_start + local_row;
            
            // Load u[batch, dim, time] and convert to float
            int u_idx = batch_idx * u_batch_stride + global_row * u_d_stride + time_idx;
            float u_val = to_float(u[u_idx]);
            
            // Load delta[batch, dim, time] and convert to float
            int delta_idx = batch_idx * delta_batch_stride + global_row * delta_d_stride + time_idx;
            float delta_val = to_float(delta[delta_idx]);
            
            // Load B (handle constant vs variable) - keep as weight_t for complex support
            weight_t B_val;
            if (!is_variable_B) {
                // Constant B: [D, N] - assume contiguous row-major layout
                B_val = B[global_row * B_d_stride + col * B_dstate_stride];
            } else {
                // Variable B: [B, G, N, L] - FIXED: Use correct indexing
                int group_size = (D + n_groups - 1) / n_groups;
                int group_id = min(global_row / group_size, n_groups - 1);
                // For [B, G, N, L]: B[b, g, n, t] = base + n * B_dstate_stride + t
                B_val = B[batch_idx * B_batch_stride + 
                          group_id * B_group_stride +
                          col * B_dstate_stride + time_idx];
            }
            
            // Compute b_t = alpha * delta * B * u
            // For complex: b_t is complex, for real: b_t is real
            complex_t b_t_complex;
            if constexpr (kIsComplex) {
                complex_t B_complex = get_complex_value(B_val);
                b_t_complex = complex_t(alpha * delta_val * u_val, 0.0f) * B_complex;
            } else {
                float B_val_float = to_float(B_val);
                b_t_complex = complex_t(alpha * delta_val * B_val_float * u_val, 0.0f);
            }
            
            // For debugging: accumulate FP32 norm (magnitude squared)
            float b_t_mag_sq = b_t_complex.real_ * b_t_complex.real_ + b_t_complex.imag_ * b_t_complex.imag_;
            norm_sq_fp32_local += b_t_mag_sq;
            
            // Convert to BF16 BEFORE normalization (matches PyTorch: X = G.bfloat16())
            // Round-trip through BF16 to match PyTorch semantics exactly
            __nv_bfloat16 b_t_real_bf16 = __float2bfloat16(b_t_complex.real_);
            __nv_bfloat16 b_t_imag_bf16 = __float2bfloat16(b_t_complex.imag_);
            float b_t_real_rounded = __bfloat162float(b_t_real_bf16);  // FP32 with BF16 precision
            float b_t_imag_rounded = __bfloat162float(b_t_imag_bf16);  // FP32 with BF16 precision
            
            // Accumulate norm from BF16-rounded values
            float b_t_rounded_mag_sq = b_t_real_rounded * b_t_real_rounded + b_t_imag_rounded * b_t_imag_rounded;
            norm_sq_local += b_t_rounded_mag_sq;
            
            // Store BF16-rounded value (real part only for complex, full value for real)
            // Buffer layout: [batch, dim, seqlen, dstate] - always real values
            // For complex: store only real part (imag part discarded)
            int buffer_idx = batch_idx * D * L * dstate +
                            global_row * L * dstate +
                            time_idx * dstate +
                            col;
            velocity_ortho[buffer_idx] = b_t_real_rounded;
        }
    }
    
    // Phase B: Block reduction for norm (computed from BF16 values)
    partial_sums[tid] = norm_sq_local;
    __syncthreads();
    
    for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    float norm_bf16 = sqrtf(partial_sums[0] + 1e-8f);
    __syncthreads();
    
    // Also reduce FP32 norm for comparison
    partial_sums[tid] = norm_sq_fp32_local;
    __syncthreads();
    
    for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    float norm_fp32 = sqrtf(partial_sums[0] + 1e-8f);
    float norm = norm_bf16;  // Use BF16 norm for normalization
    __syncthreads();
    
    // Debug: Print both norms for first batch/time (disabled for testing)
    if (false && tid == 0 && batch_idx == 0 && time_idx == 0) {
        printf("[NS DEBUG] Norm (FP32 before BF16): %.6f\n", norm_fp32);
        printf("[NS DEBUG] Norm (after BF16 conversion): %.6f\n", norm_bf16);
        printf("[NS DEBUG] Using norm: %.6f for normalization\n", norm);
        printf("[NS DEBUG] Matrix shape: D=%d, N=%d, transposed=%s, gram_size=%d\n", D, dstate, transposed ? "true" : "false", gram_size);
        printf("[NS DEBUG] Starting 5 NS iterations...\n");
    }
    
    // Phase C: Normalize BF16 values and store back to global memory
    // Matches PyTorch: X = X / X.norm()
    // For complex: normalize both real and imag parts by the same norm
    for (int d_start = 0; d_start < D; d_start += kTileSize) {
        const int d_end = min(d_start + kTileSize, D);
        const int tile_rows = d_end - d_start;
        
        for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
            const int local_row = idx / dstate;
            const int col = idx % dstate;
            const int global_row = d_start + local_row;
            
            int buffer_idx = batch_idx * D * L * dstate +
                            global_row * L * dstate +
                            time_idx * dstate +
                            col;
            
            // Load BF16-rounded value from global memory (real part only for now)
            float val_bf16_as_float = velocity_ortho[buffer_idx];
            
            // Normalize and round to BF16
            float normalized = val_bf16_as_float / norm;
            __nv_bfloat16 normalized_bf16 = __float2bfloat16(normalized);
            float normalized_as_float = __bfloat162float(normalized_bf16);
            
            // Store normalized BF16 value (as float) for NS iterations
            velocity_ortho[buffer_idx] = normalized_as_float;
            
            // Debug: Print first few values for batch 0, time 0 (disabled for testing)
            if (false && batch_idx == 0 && time_idx == 0 && global_row < 3 && col < 3) {
                printf("[NS DEBUG] After norm: X[%d,%d] = %.6f (before_norm=%.6f, norm=%.6f)\n", 
                       global_row, col, normalized_as_float, val_bf16_as_float, norm);
            }
        }
    }
    __syncthreads();
    
    // ========== STEPS 1-5: Newton-Schulz Iterations ==========
    for (int step = 0; step < 5; ++step) {
        // ===== Compute A = X @ X.T (accumulated from tiles) =====
        // A size depends on transpose: [gram_size, gram_size]
        // Initialize A to zero in FP32 (accumulate like PyTorch internal matmul)
        for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
            gram_A_fp32[idx] = 0.0f;
        }
        __syncthreads();
        
        if (!transposed) {
            // Case 1: D ≤ N (fat/square matrix), A is [D, D]
            // A[i,j] = sum_k X[i,k] * X[j,k] where k ranges over dstate
            // Tile over D dimension (rows)
            for (int d_start = 0; d_start < D; d_start += kTileSize) {
                const int d_end = min(d_start + kTileSize, D);
                const int tile_rows = d_end - d_start;
                
                // Load tile [tile_rows, dstate] into shared memory as BF16
                // Global buffer stores BF16-precision values as floats
                // For complex: load both real and imag, but for NS we'll use real part for now
                // TODO: Implement proper complex NS using Hermitian transpose
                for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                    const int local_row = idx / dstate;
                    const int col = idx % dstate;
                    const int global_row = d_start + local_row;
                    
                    int buffer_idx = batch_idx * D * L * dstate +
                                    global_row * L * dstate +
                                    time_idx * dstate +
                                    col;
                    
                    // Load BF16-precision float (real part only for now)
                    float stored_val = velocity_ortho[buffer_idx];
                    tile_buffer_bf16[local_row * dstate + col] = float_to_bf16_reinterpret(stored_val);
                }
                __syncthreads();
                
                // Compute partial Gram: for rows in this tile
                // A[i,j] += sum_k X_bf16[i,k] * X_bf16[j,k]
                // Accumulate in FP32 (matches PyTorch internal matmul behavior)
                for (int ij = tid; ij < tile_rows * gram_size; ij += kBlockSize) {
                    const int local_i = ij / gram_size;
                    const int j = ij % gram_size;
                    const int global_i = d_start + local_i;
                    
                    if (global_i < gram_size && j < gram_size) {
                        float sum = 0.0f;
                        // Compute A[i,j] for all j
                        if (j >= d_start && j < d_end) {
                            // Both in current tile - use BF16 from shared memory
                            for (int k = 0; k < dstate; ++k) {
                                float a = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
                                float b = __bfloat162float(tile_buffer_bf16[(j - d_start) * dstate + k]);
                                sum += a * b;
                            }
                            atomicAdd(&gram_A_fp32[global_i * gram_size + j], sum);
                        } else {
                            // j from other tiles - load from global as BF16
                            for (int k = 0; k < dstate; ++k) {
                                int j_idx = batch_idx * D * L * dstate + j * L * dstate + time_idx * dstate + k;
                                float a = __bfloat162float(tile_buffer_bf16[local_i * dstate + k]);
                                // velocity_ortho stores BF16-precision values as float
                                float b = velocity_ortho[j_idx];
                                sum += a * b;
                            }
                            atomicAdd(&gram_A_fp32[global_i * gram_size + j], sum);
                        }
                    }
                }
                __syncthreads();
            }
        } else {
            // Case 2: D > N (tall matrix, logically transposed to [N, D])
            // A is [N, N] where N=dstate
            // A[i,j] = sum_k X[i,k] * X[j,k] where k ranges over D
            // Tile over D dimension (now columns in transposed view)
            for (int d_start = 0; d_start < D; d_start += kTileSize) {
                const int d_end = min(d_start + kTileSize, D);
                const int tile_cols = d_end - d_start;
                
                // Load tile [dstate, tile_cols] as BF16 - transposed view
                for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
                    const int row = idx / tile_cols;
                    const int local_col = idx % tile_cols;
                    const int global_col = d_start + local_col;
                    
                    int buffer_idx = batch_idx * D * L * dstate +
                                    global_col * L * dstate +
                                    time_idx * dstate +
                                    row;
                    
                    // Load BF16-precision float (real part only for now)
                    float stored_val = velocity_ortho[buffer_idx];
                    tile_buffer_bf16[row * tile_cols + local_col] = float_to_bf16_reinterpret(stored_val);
                }
                __syncthreads();
                
                // Compute partial Gram: A[i,j] += sum_k X_bf16[i,k] * X_bf16[j,k]
                // Accumulate in FP32 (matches PyTorch internal matmul behavior)
                for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
                    const int i = ij / gram_size;
                    const int j = ij % gram_size;
                    
                    float sum = 0.0f;
                    for (int k = 0; k < tile_cols; ++k) {
                        float a = __bfloat162float(tile_buffer_bf16[i * tile_cols + k]);
                        float b = __bfloat162float(tile_buffer_bf16[j * tile_cols + k]);
                        sum += a * b;
                    }
                    
                    atomicAdd(&gram_A_fp32[ij], sum);
                }
                __syncthreads();
            }
        }
        
        // ===== Convert A from FP32 to BF16, then compute B = b*A + c*A^2 =====
        // Step 1: Convert accumulated FP32 Gram matrix to BF16 (matches PyTorch matmul output)
        // CRITICAL: Store A and A² at START of tile_buffer (offset 0)
        // X tiles will be loaded at an OFFSET to avoid overwriting these gram matrices
        // We need 2 * gram_size * gram_size BF16 elements for A and A²
        const int gram_storage_needed = 2 * gram_size * gram_size;
        __nv_bfloat16* gram_A_bf16 = tile_buffer_bf16;  // [gram_size, gram_size] at offset 0
        
        // Convert A from FP32 to BF16 (single conversion, matches PyTorch)
        for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
            gram_A_bf16[ij] = __float2bfloat16(gram_A_fp32[ij]);
        }
        __syncthreads();
        
        // Debug: Print Gram matrix values - ALWAYS for batch 0, time 0, all steps (disabled for testing)
        if (false && tid == 0 && batch_idx == 0 && time_idx == 0) {
            printf("\n[NS DEBUG] Iteration %d: Gram matrix A (first 3x3):\n", step + 1);
            for (int i = 0; i < min(3, gram_size); ++i) {
                for (int j = 0; j < min(3, gram_size); ++j) {
                    float val = __bfloat162float(gram_A_bf16[i * gram_size + j]);
                    printf("  A[%d,%d] = %.6f", i, j, val);
                }
                printf("\n");
            }
            // Print trace
            float trace = 0.0f;
            for (int i = 0; i < gram_size; ++i) {
                trace += __bfloat162float(gram_A_bf16[i * gram_size + i]);
            }
            printf("  A.trace() = %.6f (should approach %d)\n", trace, gram_size);
        }
        
        // Step 2: Compute A² in BF16 (A @ A with BF16 inputs, FP32 accumulation, BF16 output)
        // Store A² temporarily after A in tile buffer
        __nv_bfloat16* temp_A2_bf16 = gram_A_bf16 + gram_size * gram_size;
        
        for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
            const int i = ij / gram_size;
            const int j = ij % gram_size;
            
            // BF16 matmul: accumulate in FP32, output BF16
            float A2_ij_fp32 = 0.0f;
            for (int k = 0; k < gram_size; ++k) {
                float a = __bfloat162float(gram_A_bf16[i * gram_size + k]);
                float b_val = __bfloat162float(gram_A_bf16[k * gram_size + j]);
                A2_ij_fp32 += a * b_val;
            }
            temp_A2_bf16[ij] = __float2bfloat16(A2_ij_fp32);
        }
        __syncthreads();
        
        // Step 3: Compute B = b*A + c*A² in BF16, store back in gram_A_fp32 space as BF16
        // Reuse gram_A_bf16 to store B
        for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
            float A_ij = __bfloat162float(gram_A_bf16[ij]);
            float A2_ij = __bfloat162float(temp_A2_bf16[ij]);
            
            // Compute B in FP32, convert to BF16
            float B_fp32 = b * A_ij + c * A2_ij;
            gram_A_bf16[ij] = __float2bfloat16(B_fp32);
        }
        __syncthreads();
        // Now gram_A_bf16 contains B matrix in BF16
        
        // Debug: Print B matrix values (disabled for testing)
        if (false && tid == 0 && batch_idx == 0 && time_idx == 0) {
            printf("[NS DEBUG] Iteration %d: B matrix (first 3x3):\n", step + 1);
            for (int i = 0; i < min(3, gram_size); ++i) {
                for (int j = 0; j < min(3, gram_size); ++j) {
                    float val = __bfloat162float(gram_A_bf16[i * gram_size + j]);
                    printf("  B[%d,%d] = %.6f", i, j, val);
                }
                printf("\n");
            }
        }
        
        // Note: X_4 saving removed for forward pass since we use the same buffer
        // For backward pass, we'll recompute X_4 from X_5 (stored in velocity_ortho)
        
        // ===== Apply X = a*X + B@X (in tiles) =====
        // Formula depends on transpose
        // CRITICAL: Load X tiles at OFFSET to avoid overwriting gram matrices (A and A²)
        __nv_bfloat16* x_tile_buffer = tile_buffer_bf16 + gram_storage_needed;  // Start after gram matrices
        
        if (!transposed) {
            // Case 1: D ≤ N (fat matrix), X is [D, N], B is [D, D]
            // X_new[i, j] = a*X[i, j] + sum_k B[i, k] * X[k, j]
            // Tile over rows (D dimension)
            for (int d_start = 0; d_start < D; d_start += kTileSize) {
                const int d_end = min(d_start + kTileSize, D);
                const int tile_rows = d_end - d_start;
                
                // Load tile [tile_rows, dstate] as BF16 into x_tile_buffer (at offset)
                for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                    const int local_row = idx / dstate;
                    const int col = idx % dstate;
                    const int global_row = d_start + local_row;
                    
                    int buffer_idx = batch_idx * D * L * dstate +
                                    global_row * L * dstate +
                                    time_idx * dstate +
                                    col;
                    
                    // Load BF16-precision float (real part only for now)
                    float stored_val = velocity_ortho[buffer_idx];
                    x_tile_buffer[local_row * dstate + col] = float_to_bf16_reinterpret(stored_val);
                }
                __syncthreads();
                
                // Compute X_new = a*X + B@X for this tile, store as BF16
                for (int idx = tid; idx < tile_rows * dstate; idx += kBlockSize) {
                    const int local_row = idx / dstate;
                    const int col = idx % dstate;
                    const int global_row = d_start + local_row;
                    
                    float x_val = __bfloat162float(x_tile_buffer[local_row * dstate + col]);
                    
                    // (B@X)[i, j] = sum_k B[i, k] * X[k, j]
                    float sum = 0.0f;
                    for (int k = 0; k < gram_size; ++k) {
                        // Load X[k, j] as BF16
                        float x_kj;
                        if (k >= d_start && k < d_end) {
                            x_kj = __bfloat162float(x_tile_buffer[(k - d_start) * dstate + col]);
                        } else {
                            int idx_kj = batch_idx * D * L * dstate + k * L * dstate + time_idx * dstate + col;
                            // velocity_ortho stores BF16 values as float, read directly without double conversion
                            x_kj = velocity_ortho[idx_kj];
                        }
                        // B is BF16, multiply and accumulate in FP32 (mimics PyTorch)
                        float b_ik = __bfloat162float(gram_A_bf16[global_row * gram_size + k]);
                        sum += b_ik * x_kj;
                    }
                    
                    // X_new = a*X + sum in FP32, then convert to BF16
                    // CRITICAL: Round to BF16 after each iteration for stability
                    float x_new_fp32 = a * x_val + sum;
                    __nv_bfloat16 x_new_bf16 = __float2bfloat16(x_new_fp32);
                    float x_new_rounded = __bfloat162float(x_new_bf16);
                    
                    // Store rounded BF16 value (as float) - critical for iteration stability
                    int buffer_idx = batch_idx * D * L * dstate + 
                                    global_row * L * dstate + 
                                    time_idx * dstate + col;
                    velocity_ortho[buffer_idx] = x_new_rounded;
                }
                __syncthreads();
            }
        } else {
            // Case 2: D > N (tall matrix, logically transposed to [N, D])
            // Logical: X is [N, D], B is [N, N]
            // Storage: X is [D, N], element X_storage[d, n] = X_logical[n, d]
            // Logical: X_new[n, d] = a*X[n, d] + sum_k B[n, k] * X[k, d]
            // Storage: X_new_storage[d, n] = a*X_storage[d, n] + sum_k B[n, k] * X_storage[d, k]
            // Tile over D dimension (columns in logical view)
            for (int d_start = 0; d_start < D; d_start += kTileSize) {
                const int d_end = min(d_start + kTileSize, D);
                const int tile_cols = d_end - d_start;
                
                // Load tile [dstate, tile_cols] as BF16 into x_tile_buffer (at offset)
                for (int idx = tid; idx < dstate * tile_cols; idx += kBlockSize) {
                    const int row = idx / tile_cols;
                    const int local_col = idx % tile_cols;
                    const int global_col = d_start + local_col;
                    
                    int buffer_idx = batch_idx * D * L * dstate + 
                                    global_col * L * dstate + 
                                    time_idx * dstate + row;
                    
                    // Load BF16-precision float, reinterpret bits to __nv_bfloat16
                    float stored_val = velocity_ortho[buffer_idx];
                    x_tile_buffer[row * tile_cols + local_col] = float_to_bf16_reinterpret(stored_val);
                }
                __syncthreads();
                
                // Compute X_new for this tile, store as BF16
                for (int idx = tid; idx < gram_size * tile_cols; idx += kBlockSize) {
                    const int n = idx / tile_cols;  // Row in logical view
                    const int local_d = idx % tile_cols;  // Column in logical view (local)
                    const int d = d_start + local_d;
                    
                    float x_val = __bfloat162float(x_tile_buffer[n * tile_cols + local_d]);
                    
                    // sum_k B[n, k] * X_storage[d, k]
                    float sum = 0.0f;
                    for (int k = 0; k < gram_size; ++k) {
                        // X_storage[d, k] as BF16
                        // Tile buffer stores X_storage[d_start:d_end, 0:gram_size]
                        // We can read X_storage[d, k] from tile_buffer[k, local_d] if d is in tile
                        float x_dk;
                        if (d >= d_start && d < d_end) {
                            // d is in current tile, read from x_tile_buffer
                            x_dk = __bfloat162float(x_tile_buffer[k * tile_cols + local_d]);
                        } else {
                            // d is outside tile (shouldn't happen in single-tile case)
                            int idx_dk = batch_idx * D * L * dstate + d * L * dstate + time_idx * dstate + k;
                            x_dk = velocity_ortho[idx_dk];
                        }
                        // B is BF16, multiply and accumulate in FP32 (mimics PyTorch)
                        float b_nk = __bfloat162float(gram_A_bf16[n * gram_size + k]);
                        sum += b_nk * x_dk;
                    }
                    
                    // X_new = a*X + sum in FP32, then convert to BF16
                    // CRITICAL: Round to BF16 after each iteration for stability
                    float x_new_fp32 = a * x_val + sum;
                    __nv_bfloat16 x_new_bf16 = __float2bfloat16(x_new_fp32);
                    float x_new_rounded = __bfloat162float(x_new_bf16);
                    
                    // Store rounded BF16 value (as float) - critical for iteration stability
                    int buffer_idx = batch_idx * D * L * dstate + 
                                    d * L * dstate + 
                                    time_idx * dstate + n;
                    velocity_ortho[buffer_idx] = x_new_rounded;
                }
                __syncthreads();
            }
        }
    }
    
    // ========== STEP 6: Values already stored as BF16-precision floats ==========
    // velocity_ortho already contains BF16-precision values stored as float32
    // No additional conversion needed - scan will use these directly
    
    // Debug: Print final values for first batch/time (disabled for testing)
    if (false && tid == 0 && batch_idx == 0 && time_idx == 0) {
        printf("[NS DEBUG] After 5 NS iterations, first few values:\n");
        for (int d = 0; d < min(3, D); ++d) {
            for (int n = 0; n < min(3, dstate); ++n) {
                int idx = batch_idx * D * L * dstate + d * L * dstate + time_idx * dstate + n;
                printf("[NS DEBUG] X_final[%d,%d] = %.6f\n", d, n, velocity_ortho[idx]);
            }
        }
    }
    __syncthreads();
    
    // velocity_ortho now contains FP32 values ready for scan
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launch wrapper for 5-step NS
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename input_t, typename weight_t>
inline void launch_newton_schulz_velocity_5step(
    const input_t* u, const input_t* delta, const weight_t* B,
    float* velocity_ortho, float* X_4_buffer,
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
    
    // Shared memory depends on transpose
    // Transpose if dim > dstate (tall matrix)
    const bool transposed = (dim > dstate);
    const int gram_size = transposed ? dstate : dim;
    
    // Shared memory layout:
    // - tile_buffer_bf16: kTileSize * (transposed ? dim : dstate) in BF16 (2 bytes each)
    //   During polynomial step, reused for gram_A_bf16 and temp_A2_bf16
    // - gram_A_fp32: gram_size * gram_size in FP32 (4 bytes) for accumulation
    // - partial_sums: kBlockSize in FP32 (4 bytes)
    const int tile_buffer_elements = kTileSize * (transposed ? dim : dstate);
    const int gram_size_sq = gram_size * gram_size;
    
    // We need enough space in tile_buffer for 2*gram_size² BF16 elements (A and A²)
    // tile_buffer has tile_buffer_elements BF16 slots
    // We need 2*gram_size² BF16 slots during polynomial computation
    const int required_tile_buffer_for_poly = 2 * gram_size_sq;
    const int actual_tile_buffer_size = max(tile_buffer_elements, required_tile_buffer_for_poly);
    
    const int smem_size = actual_tile_buffer_size * sizeof(__nv_bfloat16) + 
                          gram_size_sq * sizeof(float) +
                          kBlockSize * sizeof(float);
    
    // For dstate=64, dim=128 (transposed): tile=64*128=8192, gram=64², need 2*64²=8192 BF16
    //   smem = 8192*2 + 64*64*4 + 256*4 = 16384 + 16384 + 1024 = 34KB ✅
    
    if (smem_size > 48 * 1024) {
        // Configure extended shared memory for large dstate
        #ifndef USE_ROCM
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            newton_schulz_velocity_5step_kernel<input_t, weight_t, kBlockSize, kTileSize>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        ));
        #endif
    }
    
    newton_schulz_velocity_5step_kernel<input_t, weight_t, kBlockSize, kTileSize><<<grid, block, smem_size, stream>>>(
        u, delta, B, velocity_ortho, X_4_buffer,
        alpha, batch, dim, seqlen, dstate, t_start,
        u_batch_stride, u_d_stride,
        delta_batch_stride, delta_d_stride,
        B_batch_stride, B_group_stride,
        B_d_stride, B_dstate_stride,
        is_variable_B, n_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // Launch wrapper for 5-step NS backward
// ////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename input_t, typename weight_t>
// inline void launch_newton_schulz_velocity_5step_backward(
//     const float* grad_output,
//     const input_t* u, const input_t* delta, const weight_t* B,
//     float* grad_u, float* grad_delta, float* grad_B,
//     float alpha, int batch, int dim, int seqlen, int dstate,
//     int t_start, int t_end,
//     int u_batch_stride, int u_d_stride,
//     int delta_batch_stride, int delta_d_stride,
//     int B_batch_stride, int B_group_stride,
//     int B_d_stride, int B_dstate_stride,
//     bool is_variable_B, int n_groups,
//     cudaStream_t stream
// ) {
//     constexpr int kBlockSize = 256;
//     constexpr int kTileSize = 64;
    
//     const int num_timesteps = t_end - t_start;
//     if (num_timesteps <= 0) return;
    
//     dim3 grid(batch, num_timesteps);
//     dim3 block(kBlockSize);
    
//     // Shared memory calculation (similar to forward, but need extra space for gradients)
//     const bool transposed = (dim > dstate);
//     const int gram_size = transposed ? dstate : dim;
    
//     const int tile_buffer_elements = kTileSize * (transposed ? dim : dstate);
//     const int gram_size_sq = gram_size * gram_size;
    
//     const int required_tile_buffer_for_poly = 2 * gram_size_sq;
//     const int actual_tile_buffer_size = max(tile_buffer_elements, required_tile_buffer_for_poly);
    
//     // Additional space needed:
//     // - partial_sums: kBlockSize floats
//     // - dX_accumulator: needs space for dA_4 which is gram_size²
//     const int smem_size = actual_tile_buffer_size * sizeof(__nv_bfloat16) + 
//                           gram_size_sq * sizeof(float) +  // gram_A_fp32
//                           kBlockSize * sizeof(float) +     // partial_sums
//                           gram_size_sq * sizeof(float);    // dX_accumulator (for dA_4)
    
//     if (smem_size > 48 * 1024) {
//         #ifndef USE_ROCM
//         C10_CUDA_CHECK(cudaFuncSetAttribute(
//             newton_schulz_velocity_5step_backward_kernel<input_t, weight_t, kBlockSize, kTileSize>,
//             cudaFuncAttributeMaxDynamicSharedMemorySize,
//             smem_size
//         ));
//         #endif
//     }
    
//     newton_schulz_velocity_5step_backward_kernel<input_t, weight_t, kBlockSize, kTileSize><<<grid, block, smem_size, stream>>>(
//         grad_output, u, delta, B,
//         grad_u, grad_delta, grad_B,
//         alpha, batch, dim, seqlen, dstate, t_start,
//         u_batch_stride, u_d_stride,
//         delta_batch_stride, delta_d_stride,
//         B_batch_stride, B_group_stride,
//         B_d_stride, B_dstate_stride,
//         is_variable_B, n_groups
//     );
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
// }
