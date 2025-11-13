// Test actual Newton-Schulz Velocity 5-Step Backward Pass from newton_schulz_bwd_kernel.cuh
// Compile: nvcc -o test_real_ns_backward test_real_ns_backward.cu -std=c++17 -arch=sm_80 -I/home/khanhnt/.conda/envs/LinOSS/lib/python3.10/site-packages/torch/include -I/home/khanhnt/.conda/envs/LinOSS/lib/python3.10/site-packages/torch/include/torch/csrc/api/include
// Run: ./test_real_ns_backward

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// PyTorch includes for C10
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

// Include the actual implementation
#include "newton_schulz_bwd_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host functions
////////////////////////////////////////////////////////////////////////////////////////////////////

void compare_results(const char* name, const float* cuda_result, const float* torch_result, int size, float tolerance = 1e-3) {
    float max_diff = 0.0f;
    float max_rel_error = 0.0f;
    int max_diff_idx = 0;
    int num_nans = 0;
    int num_infs = 0;
    
    for (int i = 0; i < size; ++i) {
        if (isnan(cuda_result[i])) num_nans++;
        if (isinf(cuda_result[i])) num_infs++;
        
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
    if (num_nans > 0) printf("  WARNING: %d NaN values in CUDA result\n", num_nans);
    if (num_infs > 0) printf("  WARNING: %d Inf values in CUDA result\n", num_infs);
    
    printf("  Sample values (first 5):\n");
    for (int i = 0; i < 5 && i < size; ++i) {
        printf("    [%d] CUDA: %.6f, Torch: %.6f, diff: %.6e\n", 
               i, cuda_result[i], torch_result[i], fabsf(cuda_result[i] - torch_result[i]));
    }
    
    if (max_rel_error < tolerance && num_nans == 0 && num_infs == 0) {
        printf("  ✅ PASS (rel_error < %.2e, no NaNs/Infs)\n", tolerance);
    } else {
        printf("  ❌ FAIL (rel_error >= %.2e or NaNs/Infs present)\n", tolerance);
    }
}

int main() {
    printf("================================================================================\n");
    printf("CUDA Newton-Schulz Velocity 5-Step Backward Pass Test\n");
    printf("Testing ACTUAL implementation from newton_schulz_bwd_kernel.cuh\n");
    printf("================================================================================\n\n");
    
    // Test parameters (small for testing)
    const int batch = 2;
    const int dim = 8;
    const int seqlen = 16;
    const int dstate = 16;
    const float alpha = 1.0f;
    
    printf("Test configuration:\n");
    printf("  Batch: %d, Dim: %d, Seqlen: %d, Dstate: %d\n", batch, dim, seqlen, dstate);
    printf("  Alpha: %.3f\n\n", alpha);
    
    // Calculate sizes
    const int u_size = batch * dim * seqlen;
    const int B_size = dim * dstate;  // Constant B
    const int grad_size = batch * dim * seqlen * dstate;
    
    // Allocate host memory
    float *h_grad_output = (float*)malloc(grad_size * sizeof(float));
    float *h_u = (float*)malloc(u_size * sizeof(float));
    float *h_delta = (float*)malloc(u_size * sizeof(float));
    float *h_B = (float*)malloc(B_size * sizeof(float));
    float *h_grad_u = (float*)calloc(u_size, sizeof(float));
    float *h_grad_delta = (float*)calloc(u_size, sizeof(float));
    float *h_grad_B = (float*)calloc(B_size, sizeof(float));
    float *h_grad_u_torch = (float*)malloc(u_size * sizeof(float));
    float *h_grad_delta_torch = (float*)malloc(u_size * sizeof(float));
    float *h_grad_B_torch = (float*)malloc(B_size * sizeof(float));
    
    // Load test data from Python-generated file
    FILE *f = fopen("/tmp/ns_velocity_test_data.bin", "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open test data file.\n");
        fprintf(stderr, "Please run: python generate_ns_velocity_test_data.py\n");
        return 1;
    }
    
    fread(h_grad_output, sizeof(float), grad_size, f);
    fread(h_u, sizeof(float), u_size, f);
    fread(h_delta, sizeof(float), u_size, f);
    fread(h_B, sizeof(float), B_size, f);
    fread(h_grad_u_torch, sizeof(float), u_size, f);
    fread(h_grad_delta_torch, sizeof(float), u_size, f);
    fread(h_grad_B_torch, sizeof(float), B_size, f);
    fclose(f);
    
    printf("✓ Loaded test data from /tmp/ns_velocity_test_data.bin\n\n");
    
    // Allocate device memory
    float *d_grad_output, *d_u, *d_delta, *d_B;
    float *d_grad_u, *d_grad_delta, *d_grad_B;
    
    C10_CUDA_CHECK(cudaMalloc(&d_grad_output, grad_size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_u, u_size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_delta, u_size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_B, B_size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_u, u_size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_delta, u_size * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_B, B_size * sizeof(float)));
    
    // Copy inputs to device
    C10_CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, grad_size * sizeof(float), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(d_u, h_u, u_size * sizeof(float), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(d_delta, h_delta, u_size * sizeof(float), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(d_B, h_B, B_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize gradient buffers to zero
    C10_CUDA_CHECK(cudaMemset(d_grad_u, 0, u_size * sizeof(float)));
    C10_CUDA_CHECK(cudaMemset(d_grad_delta, 0, u_size * sizeof(float)));
    C10_CUDA_CHECK(cudaMemset(d_grad_B, 0, B_size * sizeof(float)));
    
    printf("Launching Newton-Schulz velocity backward kernel...\n");
    
    // Launch the ACTUAL backward kernel from newton_schulz_bwd_kernel.cuh
    cudaStream_t stream = 0;
    
    launch_newton_schulz_velocity_5step_backward<float, float>(
        d_grad_output,
        d_u, d_delta, d_B,
        d_grad_u, d_grad_delta, d_grad_B,
        alpha,
        batch, dim, seqlen, dstate,
        0, seqlen,  // t_start, t_end
        dim * seqlen, seqlen,  // u_batch_stride, u_d_stride
        dim * seqlen, seqlen,  // delta_batch_stride, delta_d_stride
        0, 0,  // B_batch_stride, B_group_stride (constant B)
        dstate, 1,  // B_d_stride, B_dstate_stride
        false,  // is_variable_B
        1,  // n_groups
        stream
    );
    
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    printf("✓ Kernel completed\n\n");
    
    // Copy results back
    C10_CUDA_CHECK(cudaMemcpy(h_grad_u, d_grad_u, u_size * sizeof(float), cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemcpy(h_grad_delta, d_grad_delta, u_size * sizeof(float), cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemcpy(h_grad_B, d_grad_B, B_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compare results
    compare_results("grad_u", h_grad_u, h_grad_u_torch, u_size, 1e-3);
    compare_results("grad_delta", h_grad_delta, h_grad_delta_torch, u_size, 1e-3);
    compare_results("grad_B", h_grad_B, h_grad_B_torch, B_size, 1e-3);
    
    printf("\n================================================================================\n");
    printf("Test completed!\n");
    printf("================================================================================\n");
    
    // Cleanup
    free(h_grad_output);
    free(h_u);
    free(h_delta);
    free(h_B);
    free(h_grad_u);
    free(h_grad_delta);
    free(h_grad_B);
    free(h_grad_u_torch);
    free(h_grad_delta_torch);
    free(h_grad_B_torch);
    
    cudaFree(d_grad_output);
    cudaFree(d_u);
    cudaFree(d_delta);
    cudaFree(d_B);
    cudaFree(d_grad_u);
    cudaFree(d_grad_delta);
    cudaFree(d_grad_B);
    
    return 0;
}

