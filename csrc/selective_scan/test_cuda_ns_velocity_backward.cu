// Standalone CUDA test for Newton-Schulz Velocity 5-Step Backward Pass
// Tests CUDA implementation against PyTorch reference
// Compile: nvcc -o test_cuda_ns_velocity_backward test_cuda_ns_velocity_backward.cu -std=c++17 -arch=sm_80
// Run: ./test_cuda_ns_velocity_backward

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define C10_CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16_ns(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float bfloat16_to_float_ns(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ float to_float_ns(T x) {
    return float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16_reinterpret_ns(float f) {
    unsigned int f_bits = __float_as_uint(f);
    unsigned short bf16_raw = static_cast<unsigned short>(f_bits >> 16);
    unsigned int reconstructed = static_cast<unsigned int>(bf16_raw) << 16;
    float bf16_as_fp32 = __uint_as_float(reconstructed);
    return __float2bfloat16(bf16_as_fp32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Simplified Newton-Schulz Velocity Backward Kernel (for testing)
// Handles single timestep, small matrices (D <= 64, N <= 64)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kBlockSize = 256>
__global__ void newton_schulz_velocity_backward_simple(
    const float* __restrict__ grad_output,  // [D, N] - gradient from forward
    const float* __restrict__ G_input,      // [D, N] - original input (G = alpha * delta * B * u)
    float* __restrict__ grad_G,             // [D, N] - output gradient
    float* __restrict__ X_temp,             // [D, N] - temporary buffer for X_4
    float* __restrict__ dX_4_temp,          // [D, N] - temporary buffer for dX_4
    int D, int N
) {
    const int tid = threadIdx.x;
    
    // Newton-Schulz coefficients
    constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    constexpr float eps = 1e-7f;
    
    // No transpose for small test (assume D <= N)
    const bool transposed = (D > N);
    const int gram_size = transposed ? N : D;
    
    // Shared memory layout
    extern __shared__ float smem[];
    float* gram_A_fp32 = smem;                      // [gram_size, gram_size]
    float* partial_sums = gram_A_fp32 + gram_size * gram_size;  // [kBlockSize]
    float* dA_4_accum = partial_sums + kBlockSize;  // [gram_size, gram_size]
    
    // ========== PHASE 1: Recompute X_0 → X_4 (Detached, 4 iterations) ==========
    
    // Step 1: Compute b_t (here it's just G_input), convert to BF16, compute norm
    float norm_sq_local = 0.0f;
    
    for (int idx = tid; idx < D * N; idx += kBlockSize) {
        const int d = idx / N;
        const int n = idx % N;
        
        float g_val = G_input[d * N + n];
        __nv_bfloat16 g_bf16 = __float2bfloat16(g_val);
        float g_rounded = __bfloat162float(g_bf16);
        
        norm_sq_local += g_rounded * g_rounded;
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
    
    float norm = sqrtf(partial_sums[0] + eps);
    __syncthreads();
    
    // Step 2: Normalize to get X_0, store in X_temp
    for (int idx = tid; idx < D * N; idx += kBlockSize) {
        const int d = idx / N;
        const int n = idx % N;
        
        float g_val = G_input[d * N + n];
        __nv_bfloat16 g_bf16 = __float2bfloat16(g_val);
        float normalized = __bfloat162float(g_bf16) / norm;
        __nv_bfloat16 normalized_bf16 = __float2bfloat16(normalized);
        float normalized_as_float = __bfloat162float(normalized_bf16);
        
        X_temp[d * N + n] = normalized_as_float;
    }
    __syncthreads();
    
    // Step 3: Run 4 NS iterations (detached)
    for (int step = 0; step < 4; ++step) {
        // Compute A = X @ X.T
        for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
            gram_A_fp32[idx] = 0.0f;
        }
        __syncthreads();
        
        if (!transposed) {
            // Fat: A[i,j] = sum_k X[i,k] * X[j,k]
            for (int i = 0; i < D; ++i) {
                for (int j = tid; j < D; j += kBlockSize) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; ++k) {
                        float x_ik = X_temp[i * N + k];
                        float x_jk = X_temp[j * N + k];
                        sum += x_ik * x_jk;
                    }
                    atomicAdd(&gram_A_fp32[i * gram_size + j], sum);
                }
            }
        } else {
            // Tall: A[i,j] = sum_k X[k,i] * X[k,j] (X is stored as [D,N])
            for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
                const int i = ij / gram_size;
                const int j = ij % gram_size;
                float sum = 0.0f;
                for (int k = 0; k < D; ++k) {
                    float x_ki = X_temp[k * N + i];
                    float x_kj = X_temp[k * N + j];
                    sum += x_ki * x_kj;
                }
                gram_A_fp32[ij] = sum;
            }
        }
        __syncthreads();
        
        // Convert A to BF16, compute A², then B = b*A + c*A²
        for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
            gram_A_fp32[idx] = __bfloat162float(__float2bfloat16(gram_A_fp32[idx]));
        }
        __syncthreads();
        
        // Compute A² in dA_4_accum
        for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
            const int i = ij / gram_size;
            const int j = ij % gram_size;
            
            float sum = 0.0f;
            for (int k = 0; k < gram_size; ++k) {
                float a_ik = gram_A_fp32[i * gram_size + k];
                float a_kj = gram_A_fp32[k * gram_size + j];
                sum += a_ik * a_kj;
            }
            dA_4_accum[ij] = __bfloat162float(__float2bfloat16(sum));
        }
        __syncthreads();
        
        // Compute B = b*A + c*A² (store back in gram_A_fp32)
        for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
            float B_val = b * gram_A_fp32[idx] + c * dA_4_accum[idx];
            gram_A_fp32[idx] = __bfloat162float(__float2bfloat16(B_val));
        }
        __syncthreads();
        
        // Apply X = a*X + B@X
        if (!transposed) {
            for (int d = 0; d < D; ++d) {
                for (int n = tid; n < N; n += kBlockSize) {
                    float x_val = X_temp[d * N + n];
                    float sum = 0.0f;
                    for (int k = 0; k < gram_size; ++k) {
                        sum += gram_A_fp32[d * gram_size + k] * X_temp[k * N + n];
                    }
                    float x_new = a * x_val + sum;
                    X_temp[d * N + n] = __bfloat162float(__float2bfloat16(x_new));
                }
            }
        } else {
            for (int d = 0; d < D; ++d) {
                for (int n = tid; n < N; n += kBlockSize) {
                    float x_val = X_temp[d * N + n];
                    float sum = 0.0f;
                    for (int k = 0; k < gram_size; ++k) {
                        sum += X_temp[d * N + k] * gram_A_fp32[k * gram_size + n];
                    }
                    float x_new = a * x_val + sum;
                    X_temp[d * N + n] = __bfloat162float(__float2bfloat16(x_new));
                }
            }
        }
        __syncthreads();
    }
    
    // Now X_temp contains X_4
    
    // DEBUG: Check X_temp after 4 iterations
    if (tid == 0) {
        printf("[DEBUG] After 4 NS iterations:\n");
        printf("  X_temp[0] = %.6f, X_temp[1] = %.6f\n", X_temp[0], X_temp[1]);
        printf("  X_temp[%d] = %.6f\n", D*N-1, X_temp[D*N-1]);
        
        // Compute norm of X_temp
        float x_norm = 0.0f;
        for (int i = 0; i < D * N; ++i) {
            x_norm += X_temp[i] * X_temp[i];
        }
        printf("  X_temp norm² = %.6f, norm = %.6f\n", x_norm, sqrtf(x_norm));
    }
    __syncthreads();
    
    // ========== PHASE 2: Backward Through 5th Iteration ==========
    
    // Compute A_4 = X_4 @ X_4.T
    for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
        gram_A_fp32[idx] = 0.0f;
    }
    __syncthreads();
    
    if (!transposed) {
        for (int i = 0; i < D; ++i) {
            for (int j = tid; j < D; j += kBlockSize) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += X_temp[i * N + k] * X_temp[j * N + k];
                }
                atomicAdd(&gram_A_fp32[i * gram_size + j], sum);
            }
        }
    } else {
        for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
            const int i = ij / gram_size;
            const int j = ij % gram_size;
            float sum = 0.0f;
            for (int k = 0; k < D; ++k) {
                sum += X_temp[k * N + i] * X_temp[k * N + j];
            }
            gram_A_fp32[ij] = sum;
        }
    }
    __syncthreads();
    
    // Convert A_4 to BF16, compute A_4², then B_4 = b*A_4 + c*A_4²
    for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
        gram_A_fp32[idx] = __bfloat162float(__float2bfloat16(gram_A_fp32[idx]));
    }
    __syncthreads();
    
    float* A_4 = gram_A_fp32;  // A_4 is in gram_A_fp32 (BF16 rounded, stored as FP32)
    
    // Compute A_4² in dA_4_accum
    for (int ij = tid; ij < gram_size * gram_size; ij += kBlockSize) {
        const int i = ij / gram_size;
        const int j = ij % gram_size;
        float sum = 0.0f;
        for (int k = 0; k < gram_size; ++k) {
            sum += A_4[i * gram_size + k] * A_4[k * gram_size + j];
        }
        dA_4_accum[ij] = __bfloat162float(__float2bfloat16(sum));
    }
    __syncthreads();
    
    // Now we need B_4 = b*A_4 + c*A_4² 
    // But we need to keep A_4 for later, so we'll recompute or use smem carefully
    // For simplicity, store B_4 inline when computing gradients
    
    // DEBUG: Check A_4 values
    if (tid == 0) {
        printf("[DEBUG] A_4[0,0] = %.6f, A_4²[0,0] = %.6f\n", A_4[0], dA_4_accum[0]);
    }
    __syncthreads();
    
    // Initialize dX_4 = a * grad_output
    for (int idx = tid; idx < D * N; idx += kBlockSize) {
        dX_4_temp[idx] = a * grad_output[idx];
    }
    __syncthreads();
    
    // DEBUG: Check dX_4 after initialization
    if (tid == 0) {
        printf("[DEBUG] dX_4[0] after init = %.6f (grad_output[0] = %.6f)\n", dX_4_temp[0], grad_output[0]);
    }
    __syncthreads();
    
    // Skip backward through B_4@X_4 for now to isolate the issue
    // This will give us a*grad_output only
    /*
    // Add gradient through B_4@X_4: dX_4 += B_4.T @ grad_output
    if (!transposed) {
        for (int i = 0; i < D; ++i) {
            for (int j = tid; j < N; j += kBlockSize) {
                float sum = 0.0f;
                for (int k = 0; k < gram_size; ++k) {
                    float B_4_ki = b * A_4[k * gram_size + i] + c * dA_4_accum[k * gram_size + i];
                    sum += B_4_ki * grad_output[k * N + j];
                }
                dX_4_temp[i * N + j] += sum;
            }
        }
    }
    */
    __syncthreads();
    
    // Backward through normalization
    // d(b_t) = (dX_4 - X_4 * <dX_4, X_4>) / norm
    float dot_local = 0.0f;
    for (int idx = tid; idx < D * N; idx += kBlockSize) {
        dot_local += dX_4_temp[idx] * X_temp[idx];
    }
    
    partial_sums[tid] = dot_local;
    __syncthreads();
    
    for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    float dot_product = partial_sums[0];
    __syncthreads();
    
    // Compute final gradient
    for (int idx = tid; idx < D * N; idx += kBlockSize) {
        float dX_4_val = dX_4_temp[idx];
        float X_4_val = X_temp[idx];
        grad_G[idx] = (dX_4_val - X_4_val * dot_product) / norm;
    }
    __syncthreads();
    
    if (tid == 0) {
        printf("[CUDA] Backward kernel completed: D=%d, N=%d, norm=%.6f\n", D, N, norm);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host functions
////////////////////////////////////////////////////////////////////////////////////////////////////

void compare_results(const char* name, const float* cuda_result, const float* torch_result, int size, float tolerance = 1e-4) {
    float max_diff = 0.0f;
    float max_rel_error = 0.0f;
    int max_diff_idx = 0;
    
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(cuda_result[i] - torch_result[i]);
        float rel_error = diff / (fabsf(torch_result[i]) + 1e-8f);
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
        }
    }
    
    printf("\n%s Comparison:\n", name);
    printf("  Max absolute difference: %.6e (at index %d)\n", max_diff, max_diff_idx);
    printf("  Max relative error: %.6e\n", max_rel_error);
    printf("  Sample values:\n");
    printf("    CUDA  [0]: %.6f, Torch [0]: %.6f\n", cuda_result[0], torch_result[0]);
    printf("    CUDA [%2d]: %.6f, Torch [%2d]: %.6f\n", size/2, cuda_result[size/2], size/2, torch_result[size/2]);
    
    if (max_rel_error < tolerance) {
        printf("  ✅ PASS (rel_error < %.2e)\n", tolerance);
    } else {
        printf("  ❌ FAIL (rel_error >= %.2e)\n", tolerance);
    }
}

int main() {
    printf("================================================================================\n");
    printf("CUDA Newton-Schulz Velocity 5-Step Backward Pass Test\n");
    printf("Compares CUDA kernel vs PyTorch reference (load from file)\n");
    printf("================================================================================\n\n");
    
    // Test parameters
    const int D = 16;
    const int N = 32;
    const int size = D * N;
    
    printf("Test configuration:\n");
    printf("  Matrix shape: [%d, %d]\n", D, N);
    printf("  Total elements: %d\n\n", size);
    
    // Allocate host memory
    float *h_G_input = (float*)malloc(size * sizeof(float));
    float *h_grad_output = (float*)malloc(size * sizeof(float));
    float *h_grad_G_cuda = (float*)malloc(size * sizeof(float));
    float *h_grad_G_torch = (float*)malloc(size * sizeof(float));
    
    // Load test data from Python-generated file
    FILE *f = fopen("/tmp/ns_test_data.bin", "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open test data file. Please run Python script first.\n");
        fprintf(stderr, "Run: python3 -c \"import torch; ...; save test data to /tmp/ns_test_data.bin\"\n");
        return 1;
    }
    
    fread(h_G_input, sizeof(float), size, f);
    fread(h_grad_output, sizeof(float), size, f);
    fread(h_grad_G_torch, sizeof(float), size, f);
    fclose(f);
    
    printf("✓ Loaded test data from /tmp/ns_test_data.bin\n\n");
    
    // Allocate device memory
    float *d_G_input, *d_grad_output, *d_grad_G, *d_X_temp, *d_dX_4_temp;
    C10_CUDA_CHECK(cudaMalloc(&d_G_input, size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_output, size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_G, size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_X_temp, size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_dX_4_temp, size * sizeof(float)));
    
    // Copy inputs to device
    C10_CUDA_CHECK(cudaMemcpy(d_G_input, h_G_input, size * sizeof(float), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch CUDA kernel
    constexpr int kBlockSize = 256;
    const int gram_size = (D > N) ? N : D;
    int smem_size = (gram_size * gram_size * 2 + kBlockSize) * sizeof(float);
    
    printf("Launching CUDA kernel...\n");
    printf("  Block size: %d threads\n", kBlockSize);
    printf("  Shared memory: %d bytes\n", smem_size);
    printf("\n");
    
    newton_schulz_velocity_backward_simple<kBlockSize><<<1, kBlockSize, smem_size>>>(
        d_grad_output, d_G_input, d_grad_G, d_X_temp, d_dX_4_temp, D, N
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n");
    
    // Copy result back
    C10_CUDA_CHECK(cudaMemcpy(h_grad_G_cuda, d_grad_G, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compare results
    compare_results("Gradient", h_grad_G_cuda, h_grad_G_torch, size, 1e-3);
    
    printf("\n================================================================================\n");
    printf("Test completed!\n");
    printf("================================================================================\n");
    
    // Cleanup
    free(h_G_input);
    free(h_grad_output);
    free(h_grad_G_cuda);
    free(h_grad_G_torch);
    cudaFree(d_G_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_G);
    cudaFree(d_X_temp);
    cudaFree(d_dX_4_temp);
    
    return 0;
}

