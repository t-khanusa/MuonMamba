/*
Minimal test for Newton-Schulz backward pass with 2x2 matrix
This will help us trace exactly where the bug is
*/

#include <stdio.h>
#include <c10/cuda/CUDAException.h>

// Include the backward kernel
#include "newton_schulz_bwd_kernel.cuh"

int main() {
    printf("=================================================================\n");
    printf("Minimal 2x2 Newton-Schulz Backward Test\n");
    printf("=================================================================\n");
    
    // Test configuration: 1 batch, 2 dims, 1 timestep, 2 dstate
    const int batch = 1;
    const int dim = 2;
    const int seqlen = 1;
    const int dstate = 2;
    const float alpha = 1.0f;
    
    printf("\nConfiguration:\n");
    printf("  batch=%d, dim=%d, seqlen=%d, dstate=%d, alpha=%.1f\n", 
           batch, dim, seqlen, dstate, alpha);
    
    // Host data - using same values as Python test
    // u = [0.2308, 0.1337]
    // delta = [0.5535, 0.5809]
    // B = [[0.3331, -0.5069], [-0.2967, 0.2874]]
    float h_u[2] = {0.2308f, 0.1337f};
    float h_delta[2] = {0.5535f, 0.5809f};
    float h_B[4] = {0.3331f, -0.5069f, -0.2967f, 0.2874f};  // Row-major
    
    // grad_output (computed by Python for reconstructed G)
    // Need to run Python to get this
    // For now, use simple gradient
    float h_grad_output[4] = {1.0f, 0.0f, 0.0f, 1.0f};  // Identity-like
    
    printf("\nInputs:\n");
    printf("  u = [%.4f, %.4f]\n", h_u[0], h_u[1]);
    printf("  delta = [%.4f, %.4f]\n", h_delta[0], h_delta[1]);
    printf("  B = [[%.4f, %.4f], [%.4f, %.4f]]\n", 
           h_B[0], h_B[1], h_B[2], h_B[3]);
    printf("  grad_output = [[%.4f, %.4f], [%.4f, %.4f]]\n",
           h_grad_output[0], h_grad_output[1], h_grad_output[2], h_grad_output[3]);
    
    // Compute G = alpha * delta * B * u
    printf("\nComputed G (element-wise):\n");
    for (int d = 0; d < 2; d++) {
        printf("  G[%d,:] = %.6f * %.6f * [%.6f, %.6f] * %.6f\n",
               d, alpha, h_delta[d], h_B[d*2], h_B[d*2+1], h_u[d]);
        float g0 = alpha * h_delta[d] * h_B[d*2] * h_u[d];
        float g1 = alpha * h_delta[d] * h_B[d*2+1] * h_u[d];
        printf("          = [%.6f, %.6f]\n", g0, g1);
    }
    
    // Allocate device memory
    float *d_u, *d_delta, *d_B, *d_grad_output;
    float *d_grad_u, *d_grad_delta, *d_grad_B;
    
    C10_CUDA_CHECK(cudaMalloc(&d_u, 2 * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_delta, 2 * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_B, 4 * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_output, 4 * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_u, 2 * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_delta, 2 * sizeof(float)));
    C10_CUDA_CHECK(cudaMalloc(&d_grad_B, 4 * sizeof(float)));
    
    // Copy to device
    C10_CUDA_CHECK(cudaMemcpy(d_u, h_u, 2 * sizeof(float), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(d_delta, h_delta, 2 * sizeof(float), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(d_B, h_B, 4 * sizeof(float), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, 4 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Zero gradient buffers
    C10_CUDA_CHECK(cudaMemset(d_grad_u, 0, 2 * sizeof(float)));
    C10_CUDA_CHECK(cudaMemset(d_grad_delta, 0, 2 * sizeof(float)));
    C10_CUDA_CHECK(cudaMemset(d_grad_B, 0, 4 * sizeof(float)));
    
    printf("\nLaunching Newton-Schulz velocity backward kernel...\n");
    
    // Launch kernel with simplified strides
    launch_newton_schulz_velocity_5step_backward<float, float>(
        d_grad_output,       // grad_output [1, 2, 1, 2]
        d_u,                 // u [1, 2, 1]
        d_delta,             // delta [1, 2, 1]
        d_B,                 // B [2, 2]
        d_grad_u,            // grad_u [1, 2, 1]
        d_grad_delta,        // grad_delta [1, 2, 1]
        d_grad_B,            // grad_B [2, 2]
        alpha,
        batch, dim, seqlen, dstate,
        0, 1,                // t_start, t_end
        2, 1,                // u_batch_stride, u_d_stride
        2, 1,                // delta_batch_stride, delta_d_stride
        0, 0,                // B_batch_stride, B_group_stride (not used)
        2, 1,                // B_d_stride, B_dstate_stride
        false, 1,            // is_variable_B, n_groups
        0                    // stream
    );
    
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel completed!\n");
    
    // Copy results back
    float h_grad_u[2], h_grad_delta[2], h_grad_B[4];
    C10_CUDA_CHECK(cudaMemcpy(h_grad_u, d_grad_u, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemcpy(h_grad_delta, d_grad_delta, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemcpy(h_grad_B, d_grad_B, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\nCUDA Results:\n");
    printf("  grad_u = [%.6f, %.6f]\n", h_grad_u[0], h_grad_u[1]);
    printf("  grad_delta = [%.6f, %.6f]\n", h_grad_delta[0], h_grad_delta[1]);
    printf("  grad_B = [[%.6f, %.6f], [%.6f, %.6f]]\n",
           h_grad_B[0], h_grad_B[1], h_grad_B[2], h_grad_B[3]);
    
    printf("\nExpected from Python:\n");
    printf("  grad_u = [0.518968, -0.920257]\n");
    printf("  grad_delta = [0.216401, -0.211806]\n");
    printf("  grad_B = [[-1.124957, -0.975539], [-0.477024, -0.920569]]\n");
    
    printf("\nDifferences:\n");
    printf("  grad_u[0]: %.6f (CUDA) vs 0.518968 (Python), diff = %.6f\n", 
           h_grad_u[0], h_grad_u[0] - 0.518968f);
    printf("  grad_u[1]: %.6f (CUDA) vs -0.920257 (Python), diff = %.6f\n",
           h_grad_u[1], h_grad_u[1] - (-0.920257f));
    printf("  grad_delta[0]: %.6f (CUDA) vs 0.216401 (Python), diff = %.6f\n",
           h_grad_delta[0], h_grad_delta[0] - 0.216401f);
    printf("  grad_delta[1]: %.6f (CUDA) vs -0.211806 (Python), diff = %.6f\n",
           h_grad_delta[1], h_grad_delta[1] - (-0.211806f));
    printf("  grad_B[0,0]: %.6f (CUDA) vs -1.124957 (Python), diff = %.6f\n",
           h_grad_B[0], h_grad_B[0] - (-1.124957f));
    printf("  grad_B[0,1]: %.6f (CUDA) vs -0.975539 (Python), diff = %.6f\n",
           h_grad_B[1], h_grad_B[1] - (-0.975539f));
    printf("  grad_B[1,0]: %.6f (CUDA) vs -0.477024 (Python), diff = %.6f\n",
           h_grad_B[2], h_grad_B[2] - (-0.477024f));
    printf("  grad_B[1,1]: %.6f (CUDA) vs -0.920569 (Python), diff = %.6f\n",
           h_grad_B[3], h_grad_B[3] - (-0.920569f));
    
    // Cleanup
    cudaFree(d_u);
    cudaFree(d_delta);
    cudaFree(d_B);
    cudaFree(d_grad_output);
    cudaFree(d_grad_u);
    cudaFree(d_grad_delta);
    cudaFree(d_grad_B);
    
    printf("\n=================================================================\n");
    return 0;
}

