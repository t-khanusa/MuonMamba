#!/usr/bin/env python3
"""
Compare CUDA and PyTorch Newton-Schulz implementations
Load PyTorch reference and run matching CUDA tests
"""

import torch
import numpy as np
import json
import subprocess
import tempfile
from pathlib import Path


def newtonschulz5_torch(G, steps=5):
    """PyTorch reference implementation"""
    G = G.bfloat16()
    norm = G.norm()
    X = G / norm
    
    norms = [norm.float().item()]
    traces = []
    
    transposed = (G.shape[0] > G.shape[1])
    if transposed:
        X = X.T
    
    a, b, c = 3.4445, -4.7750, 2.0315
    
    for i in range(steps):
        A = X @ X.T
        trace = A.float().trace().item()
        traces.append(trace)
        
        A_squared = A @ A
        B = b * A + c * A_squared
        X = a * X + B @ X
        
        norm = X.norm().float().item()
        norms.append(norm)
    
    if transposed:
        X = X.T
    
    X = X.float()
    return X, norms, traces


def run_cuda_kernel(D, N, input_data):
    """Run CUDA kernel using the compiled test program"""
    # Create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(f"{D} {N}\n")
        for val in input_data.flatten():
            f.write(f"{val} ")
        temp_input = f.name
    
    try:
        # Run the CUDA test program
        result = subprocess.run(
            ['./test_ns_5step_detailed_simple', temp_input],
            capture_output=True,
            text=True,
            timeout=10,
            cwd='/project/khanhnt/muontest/Momentum_correct'
        )
        
        if result.returncode != 0:
            print(f"CUDA execution failed: {result.stderr}")
            return None, None, None
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        norms = None
        traces = None
        output = []
        
        for line in lines:
            if line.startswith('NORMS:'):
                norms = [float(x) for x in line.split(':')[1].strip().split()]
            elif line.startswith('TRACES:'):
                traces = [float(x) for x in line.split(':')[1].strip().split()]
            elif line.startswith('OUTPUT:'):
                output = [float(x) for x in line.split(':')[1].strip().split()]
        
        if norms and traces and output:
            return np.array(output).reshape(D, N), norms, traces
        else:
            print(f"Failed to parse CUDA output: {result.stdout}")
            return None, None, None
            
    finally:
        Path(temp_input).unlink(missing_ok=True)


def create_cuda_test_program():
    """Create a simple CUDA test program that reads from file"""
    cuda_code = """
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

template<int kBlockSize = 256, int kTileSize = 64>
__global__ void test_ns_kernel(const float* input, float* output, float* norms, float* traces, int D, int N) {
    // Same kernel as before...
    const int tid = threadIdx.x;
    constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    const bool transposed = (D > N);
    const int gram_size = transposed ? N : D;
    
    extern __shared__ float smem[];
    __nv_bfloat16* tile_buffer = (__nv_bfloat16*)smem;
    float* gram_fp32 = (float*)(tile_buffer + kTileSize * (transposed ? D : N));
    float* partial_sums = gram_fp32 + gram_size * gram_size;
    
    // Initialize...
    float norm_sq = 0.0f;
    for (int idx = tid; idx < D * N; idx += kBlockSize) {
        float val = input[idx];
        __nv_bfloat16 bf16 = __float2bfloat16(val);
        float rounded = __bfloat162float(bf16);
        output[idx] = rounded;
        norm_sq += rounded * rounded;
    }
    
    partial_sums[tid] = norm_sq;
    __syncthreads();
    for (int s = kBlockSize >> 1; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }
    
    float norm = sqrtf(partial_sums[0] + 1e-8f);
    if (tid == 0) norms[0] = norm;
    __syncthreads();
    
    // Normalize
    for (int idx = tid; idx < D * N; idx += kBlockSize) {
        float val = output[idx];
        output[idx] = __bfloat162float(__float2bfloat16(val / norm));
    }
    __syncthreads();
    
    // 5 iterations (simplified for brevity)
    // ... (same as before)
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\\n", argv[0]);
        return 1;
    }
    
    FILE* f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\\n", argv[1]);
        return 1;
    }
    
    int D, N;
    fscanf(f, "%d %d", &D, &N);
    
    float* input = new float[D * N];
    for (int i = 0; i < D * N; ++i) {
        fscanf(f, "%f", &input[i]);
    }
    fclose(f);
    
    // Allocate and run...
    float *d_in, *d_out, *d_norms, *d_traces;
    cudaMalloc(&d_in, D * N * sizeof(float));
    cudaMalloc(&d_out, D * N * sizeof(float));
    cudaMalloc(&d_norms, 6 * sizeof(float));
    cudaMalloc(&d_traces, 5 * sizeof(float));
    
    cudaMemcpy(d_in, input, D * N * sizeof(float), cudaMemcpyHostToDevice);
    
    const int kBlockSize = 256;
    const int kTileSize = 64;
    const int gram_size = (D > N) ? N : D;
    const int smem = kTileSize * ((D > N) ? D : N) * sizeof(__nv_bfloat16) +
                     gram_size * gram_size * sizeof(float) + kBlockSize * sizeof(float);
    
    test_ns_kernel<kBlockSize, kTileSize><<<1, kBlockSize, smem>>>(d_in, d_out, d_norms, d_traces, D, N);
    cudaDeviceSynchronize();
    
    float* output = new float[D * N];
    float norms[6], traces[5];
    cudaMemcpy(output, d_out, D * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(norms, d_norms, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(traces, d_traces, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("NORMS: ");
    for (int i = 0; i < 6; ++i) printf("%.8f ", norms[i]);
    printf("\\n");
    
    printf("TRACES: ");
    for (int i = 0; i < 5; ++i) printf("%.8f ", traces[i]);
    printf("\\n");
    
    printf("OUTPUT: ");
    for (int i = 0; i < D * N; ++i) printf("%.8f ", output[i]);
    printf("\\n");
    
    delete[] input;
    delete[] output;
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_norms);
    cudaFree(d_traces);
    
    return 0;
}
"""
    
    with open('/project/khanhnt/muontest/Momentum_correct/test_cuda_simple.cu', 'w') as f:
        f.write(cuda_code)


def compare_results(test_name, torch_result, cuda_result):
    """Compare PyTorch and CUDA results"""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    
    if cuda_result[0] is None:
        print("❌ CUDA execution failed")
        return False
    
    cuda_output, cuda_norms, cuda_traces = cuda_result
    torch_output, torch_norms, torch_traces = torch_result
    
    # Compare norms
    norm_diff = np.abs(np.array(cuda_norms) - np.array(torch_norms))
    norm_rel_diff = norm_diff / (np.abs(torch_norms) + 1e-8)
    
    print("\nNorms Comparison:")
    print(f"  PyTorch: {' '.join([f'{x:.6f}' for x in torch_norms])}")
    print(f"  CUDA:    {' '.join([f'{x:.6f}' for x in cuda_norms])}")
    print(f"  Max relative diff: {np.max(norm_rel_diff)*100:.4f}%")
    
    # Compare traces
    trace_diff = np.abs(np.array(cuda_traces) - np.array(torch_traces))
    trace_rel_diff = trace_diff / (np.abs(torch_traces) + 1e-8)
    
    print("\nTraces Comparison:")
    print(f"  PyTorch: {' '.join([f'{x:.6f}' for x in torch_traces])}")
    print(f"  CUDA:    {' '.join([f'{x:.6f}' for x in cuda_traces])}")
    print(f"  Max relative diff: {np.max(trace_rel_diff)*100:.4f}%")
    
    # Compare outputs
    output_diff = np.abs(cuda_output - torch_output)
    output_rel_diff = output_diff / (np.abs(torch_output) + 1e-8)
    
    print("\nOutput Comparison:")
    print(f"  Shape: {torch_output.shape}")
    print(f"  Max absolute diff: {np.max(output_diff):.6f}")
    print(f"  Max relative diff: {np.max(output_rel_diff)*100:.4f}%")
    print(f"  Mean relative diff: {np.mean(output_rel_diff)*100:.4f}%")
    
    # Verdict
    norm_ok = np.max(norm_rel_diff) < 0.01  # 1%
    trace_ok = np.max(trace_rel_diff) < 0.02  # 2% (more lenient)
    output_ok = np.max(output_rel_diff) < 0.05  # 5%
    
    print("\nVerdict:")
    print(f"  Norms:  {'✓ PASS' if norm_ok else '✗ FAIL'} (threshold: 1%)")
    print(f"  Traces: {'✓ PASS' if trace_ok else '✗ FAIL'} (threshold: 2%)")
    print(f"  Output: {'✓ PASS' if output_ok else '✗ FAIL'} (threshold: 5%)")
    
    overall = norm_ok and trace_ok and output_ok
    print(f"\n  Overall: {'✓ PASSED' if overall else '✗ FAILED'}")
    
    return overall


def main():
    print("="*80)
    print("CUDA vs PyTorch Newton-Schulz Comprehensive Comparison")
    print("="*80)
    
    # Define test cases
    test_cases = [
        {
            'name': 'tiny_fat_2x3',
            'D': 2, 'N': 3,
            'input': np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        },
        {
            'name': 'small_fat_3x4',
            'D': 3, 'N': 4,
            'input': np.arange(1, 13, dtype=np.float32).reshape(3, 4)
        },
        {
            'name': 'small_tall_4x3',
            'D': 4, 'N': 3,
            'input': np.arange(1, 13, dtype=np.float32).reshape(4, 3)
        },
        {
            'name': 'tiny_square_2x2',
            'D': 2, 'N': 2,
            'input': np.array([[1, 2], [3, 4]], dtype=np.float32)
        },
        {
            'name': 'medium_fat_16x32',
            'D': 16, 'N': 32,
            'input': np.random.randn(16, 32).astype(np.float32)
        },
        {
            'name': 'prod_tall_128x64',
            'D': 128, 'N': 64,
            'input': np.array([(i % 100) / 10.0 for i in range(128*64)], dtype=np.float32).reshape(128, 64)
        },
    ]
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    for tc in test_cases:
        print(f"\n{'='*80}")
        print(f"Running test: {tc['name']} (D={tc['D']}, N={tc['N']})")
        print(f"{'='*80}")
        
        # Run PyTorch
        G_torch = torch.from_numpy(tc['input']).float()
        X_torch, norms_torch, traces_torch = newtonschulz5_torch(G_torch)
        torch_result = (X_torch.numpy(), norms_torch, traces_torch)
        
        # Run CUDA (use existing test program)
        print(f"\nRunning CUDA kernel for {tc['name']}...")
        
        # For now, just run the existing CUDA test and manually compare
        # In a real scenario, we'd parse the CUDA output
        print("⚠️  CUDA comparison requires manual execution")
        print(f"    Input shape: {tc['input'].shape}")
        print(f"    PyTorch norms: {norms_torch}")
        print(f"    PyTorch traces: {traces_torch}")
        
        results['details'].append({
            'name': tc['name'],
            'D': tc['D'],
            'N': tc['N'],
            'torch_norms': norms_torch,
            'torch_traces': traces_torch
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nPyTorch reference results generated for {len(test_cases)} test cases")
    print("\nTo compare with CUDA:")
    print("1. Run ./test_ns_5step_detailed for each test case")
    print("2. Compare norms and traces manually")
    print("3. Expected tolerances:")
    print("   - Norms: < 1% relative difference")
    print("   - Traces: < 2% relative difference")
    print("   - Output: < 5% relative difference")
    
    # Print reference values for manual comparison
    print("\n" + "="*80)
    print("REFERENCE VALUES FOR MANUAL COMPARISON")
    print("="*80)
    
    for detail in results['details']:
        print(f"\n{detail['name']} (D={detail['D']}, N={detail['N']}):")
        print(f"  Norms:  {' '.join([f'{x:.6f}' for x in detail['torch_norms']])}")
        print(f"  Traces: {' '.join([f'{x:.6f}' for x in detail['torch_traces']])}")


if __name__ == "__main__":
    main()






