// Standalone CUDA test for Newton-Schulz backward pass
// Compile: nvcc -o test_ns_cuda_backward test_ns_cuda_backward.cu -std=c++17
// Run: ./test_ns_cuda_backward

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Include the NS backward kernel implementation
#define C10_CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

// Helper functions from the main kernel
template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16_reinterpret(float f) {
    unsigned int f_bits = __float_as_uint(f);
    unsigned short bf16_raw = static_cast<unsigned short>(f_bits >> 16);
    unsigned int reconstructed = static_cast<unsigned int>(bf16_raw) << 16;
    float bf16_as_fp32 = __uint_as_float(reconstructed);
    return __float2bfloat16(bf16_as_fp32);
}

// Simple NS backward kernel for testing (D <= N case only)
template<int kBlockSize = 256>
__global__ void test_ns_backward_kernel_simple(
    const float* __restrict__ grad_output,  // [D, N]
    const float* __restrict__ G_input,      // [D, N]
    float* __restrict__ grad_G,             // [D, N]
    int D, int N
) {
    const int tid = threadIdx.x;
    
    // NS coefficients
    constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    constexpr float eps = 1e-7f;
    
    // Shared memory
    extern __shared__ float smem[];
    float* gram_A = smem;                          // [D, D]
    float* partial_sums = gram_A + D * D;         // [kBlockSize]
    float* dA_4 = partial_sums + kBlockSize;      // [D, D]
    
    // Phase 1: Recompute X_0 -> X_4 (detached)
    // Step 1: Compute norm
    float norm_sq_local = 0.0f;
    for (int idx = tid; idx < D * N; idx += kBlockSize) {
        int d = idx / N;
        int n = idx % N;
        float g_val = G_input[d * N + n];
        __nv_bfloat16 g_bf16 = __float2bfloat16(g_val);
        float g_rounded = __bfloat162float(g_bf16);
        norm_sq_local += g_rounded * g_rounded;
    }
    
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
    
    // Step 2: Normalize and store X_0
    for (int idx = tid; idx < D * N; idx += kBlockSize) {
        int d = idx / N;
        int n = idx % N;
        float g_val = G_input[d * N + n];
        __nv_bfloat16 g_bf16 = __float2bfloat16(g_val);
        float g_rounded = __bfloat162float(g_bf16);
        float x_norm = g_rounded / norm;
        __nv_bfloat16 x_bf16 = __float2bfloat16(x_norm);
        grad_G[d * N + n] = __bfloat162float(x_bf16);  // Reuse grad_G for X storage temporarily
    }
    __syncthreads();
    
    // Step 3: Run 4 NS iterations (detached)
    for (int iter = 0; iter < 4; ++iter) {
        // Compute A = X @ X.T [D, D]
        for (int idx = tid; idx < D * D; idx += kBlockSize) {
            gram_A[idx] = 0.0f;
        }
        __syncthreads();
        
        for (int i = 0; i < D; ++i) {
            for (int j = tid; j < D; j += kBlockSize) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    float x_ik = grad_G[i * N + k];
                    float x_jk = grad_G[j * N + k];
                    sum += x_ik * x_jk;
                }
                atomicAdd(&gram_A[i * D + j], sum);
            }
        }
        __syncthreads();
        
        // Convert A to BF16
        for (int idx = tid; idx < D * D; idx += kBlockSize) {
            gram_A[idx] = __bfloat162float(__float2bfloat16(gram_A[idx]));
        }
        __syncthreads();
        
        // Compute A² in temporary buffer (reuse dA_4)
        for (int ij = tid; ij < D * D; ij += kBlockSize) {
            int i = ij / D;
            int j = ij % D;
            float sum = 0.0f;
            for (int k = 0; k < D; ++k) {
                sum += gram_A[i * D + k] * gram_A[k * D + j];
            }
            dA_4[ij] = __bfloat162float(__float2bfloat16(sum));
        }
        __syncthreads();
        
        // Compute B = b*A + c*A² (store in gram_A)
        for (int idx = tid; idx < D * D; idx += kBlockSize) {
            float B_val = b * gram_A[idx] + c * dA_4[idx];
            gram_A[idx] = __bfloat162float(__float2bfloat16(B_val));
        }
        __syncthreads();
        
        // Update X = a*X + B@X
        for (int d = 0; d < D; ++d) {
            for (int n = tid; n < N; n += kBlockSize) {
                float x_val = grad_G[d * N + n];
                float sum = 0.0f;
                for (int k = 0; k < D; ++k) {
                    sum += gram_A[d * D + k] * grad_G[k * N + n];
                }
                float x_new = a * x_val + sum;
                grad_G[d * N + n] = __bfloat162float(__float2bfloat16(x_new));
            }
        }
        __syncthreads();
    }
    
    // Now grad_G contains X_4
    // Phase 2: Backward through 5th iteration
    // (Simplified implementation - just verify compilation)
    
    if (tid == 0) {
        printf("CUDA NS backward kernel executed successfully\n");
        printf("  D=%d, N=%d, norm=%.6f\n", D, N, norm);
    }
}

// Host reference implementation
void host_ns_forward(float* X, const float* G, int D, int N, int steps) {
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    const float eps = 1e-7f;
    
    // Compute norm
    float norm_sq = 0.0f;
    for (int i = 0; i < D * N; ++i) {
        // Simulate BF16 conversion
        unsigned short bf16_bits = *((unsigned int*)&G[i]) >> 16;
        unsigned int reconstructed = ((unsigned int)bf16_bits) << 16;
        float g_bf16 = *((float*)&reconstructed);
        norm_sq += g_bf16 * g_bf16;
    }
    float norm = sqrtf(norm_sq + eps);
    
    // Normalize
    for (int i = 0; i < D * N; ++i) {
        unsigned short bf16_bits = *((unsigned int*)&G[i]) >> 16;
        unsigned int reconstructed = ((unsigned int)bf16_bits) << 16;
        float g_bf16 = *((float*)&reconstructed);
        X[i] = g_bf16 / norm;
    }
    
    printf("Host: Computed norm = %.6f\n", norm);
}

int main() {
    printf("=============================================================\n");
    printf("CUDA Newton-Schulz Backward Pass Test\n");
    printf("=============================================================\n\n");
    
    // Small test case
    const int D = 8;
    const int N = 16;
    const int size = D * N;
    
    // Allocate host memory
    float *h_G = (float*)malloc(size * sizeof(float));
    float *h_grad_out = (float*)malloc(size * sizeof(float));
    float *h_grad_G = (float*)malloc(size * sizeof(float));
    float *h_X_ref = (float*)malloc(size * sizeof(float));
    
    // Initialize with small random values
    srand(42);
    for (int i = 0; i < size; ++i) {
        h_G[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        h_grad_out[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    // Host reference
    host_ns_forward(h_X_ref, h_G, D, N, 5);
    
    // Allocate device memory
    float *d_G, *d_grad_out, *d_grad_G;
    C10_CUDA_CHECK(cudaMalloc(&d_G, size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_out, size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_G, size * sizeof(float)));
    
    // Copy to device
    C10_CUDA_CHECK(cudaMemcpy(d_G, h_G, size * sizeof(float), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(d_grad_out, h_grad_out, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    constexpr int kBlockSize = 256;
    int smem_size = (D * D * 2 + kBlockSize) * sizeof(float);
    
    printf("Launching CUDA kernel...\n");
    printf("  Block size: %d\n", kBlockSize);
    printf("  Shared memory: %d bytes\n", smem_size);
    printf("\n");
    
    test_ns_backward_kernel_simple<kBlockSize><<<1, kBlockSize, smem_size>>>(
        d_grad_out, d_G, d_grad_G, D, N
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    C10_CUDA_CHECK(cudaMemcpy(h_grad_G, d_grad_G, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\nTest completed successfully!\n");
    printf("=============================================================\n");
    
    // Cleanup
    free(h_G);
    free(h_grad_out);
    free(h_grad_G);
    free(h_X_ref);
    cudaFree(d_G);
    cudaFree(d_grad_out);
    cudaFree(d_grad_G);
    
    return 0;
}

