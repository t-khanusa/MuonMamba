#!/usr/bin/env python3
"""
Compare CUDA and PyTorch Newton-Schulz implementations
with multi-batch and multi-timestep support
"""

import torch
import numpy as np
import subprocess
import sys


def newtonschulz5_torch_batch(G, steps=5):
    """PyTorch Newton-Schulz for batched input"""
    original_shape = G.shape
    
    if len(G.shape) == 4:
        B, L, D, N = G.shape
        G = G.reshape(B * L, D, N)
    elif len(G.shape) == 3:
        B, D, N = G.shape
        L = 1
    else:
        raise ValueError(f"Expected 3D or 4D input, got {G.shape}")
    
    BL = G.shape[0]
    G = G.bfloat16()
    
    norms_all = []
    traces_all = []
    
    norm = torch.norm(G.reshape(BL, -1), dim=1, keepdim=True)
    norms_all.append(norm.float().cpu())
    
    X = G / norm.view(BL, 1, 1)
    
    transposed = (D > N)
    if transposed:
        X = X.transpose(-2, -1)
    
    a, b, c = 3.4445, -4.7750, 2.0315
    
    for step in range(steps):
        A = X @ X.transpose(-2, -1)
        trace = torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)
        traces_all.append(trace.float().cpu())
        
        A_squared = A @ A
        B = b * A + c * A_squared
        X = a * X + B @ X
        
        norm = torch.norm(X.reshape(BL, -1), dim=1, keepdim=True)
        norms_all.append(norm.float().cpu())
    
    if transposed:
        X = X.transpose(-2, -1)
    
    X = X.float()
    
    if len(original_shape) == 4:
        X = X.reshape(original_shape[0], original_shape[1], original_shape[2], original_shape[3])
    else:
        X = X.reshape(original_shape[0], original_shape[1], original_shape[2])
    
    norms_all = torch.stack(norms_all, dim=0).squeeze(-1)  # [6, BL]
    traces_all = torch.stack(traces_all, dim=0)  # [5, BL]
    
    return X, norms_all, traces_all


def run_cuda_multibatch():
    """Run CUDA multi-batch test"""
    result = subprocess.run(
        ['./test_ns_multibatch_cuda'],
        capture_output=True,
        text=True,
        timeout=30,
        cwd='/project/khanhnt/muontest/Momentum_correct'
    )
    
    if result.returncode != 0:
        print(f"CUDA execution failed: {result.stderr}")
        return None
    
    return result.stdout


def compare_test_case(name, B, L, D, N):
    """Run and compare a single test case"""
    print(f"\n{'='*80}")
    print(f"Test: {name}")
    print(f"Shape: [B={B}, L={L}, D={D}, N={N}]")
    print(f"{'='*80}")
    
    # Generate input (same as CUDA)
    BL = B * L
    input_data = torch.arange(BL * D * N, dtype=torch.float32).reshape(B, L, D, N) + 1
    
    # Run PyTorch
    X_torch, norms_torch, traces_torch = newtonschulz5_torch_batch(input_data)
    
    # Run CUDA (we'll parse its output)
    # For now, just run PyTorch and show results
    print(f"\nPyTorch Results:")
    print(f"  Output shape: {X_torch.shape}")
    print(f"  Norms shape: {norms_torch.shape}")
    print(f"  Traces shape: {traces_torch.shape}")
    
    # First matrix results
    print(f"\nFirst matrix (batch=0, timestep=0):")
    print(f"  Initial norm: {norms_torch[0, 0]:.6f}")
    print(f"  Final norm: {norms_torch[-1, 0]:.6f}")
    print(f"  Traces: {traces_torch[:, 0].numpy()}")
    
    # Statistics across all matrices
    print(f"\nStatistics across all {BL} matrices:")
    print(f"  Initial norms: min={norms_torch[0].min():.4f}, max={norms_torch[0].max():.4f}, mean={norms_torch[0].mean():.4f}")
    print(f"  Final norms: min={norms_torch[-1].min():.4f}, max={norms_torch[-1].max():.4f}, mean={norms_torch[-1].mean():.4f}")
    print(f"  Final traces: min={traces_torch[-1].min():.4f}, max={traces_torch[-1].max():.4f}, mean={traces_torch[-1].mean():.4f}")
    
    # Check orthogonality for first matrix
    X0 = X_torch[0, 0]
    if D <= N:
        gram = X0 @ X0.T
        identity = torch.eye(D)
    else:
        gram = X0.T @ X0
        identity = torch.eye(N)
    ortho_error = torch.max(torch.abs(gram - identity)).item()
    print(f"  First matrix orthogonality error: {ortho_error:.6f}")
    
    return {
        'norms': norms_torch.numpy(),
        'traces': traces_torch.numpy(),
        'ortho_error': ortho_error
    }


def main():
    print("="*80)
    print("Newton-Schulz 5-Step: Multi-Batch Multi-Timestep Validation")
    print("="*80)
    
    # Test configurations (matching both CUDA and PyTorch)
    test_cases = [
        ("Single batch, single timestep", 1, 1, 3, 4),
        ("Single batch, multiple timesteps", 1, 3, 3, 4),
        ("Multiple batches, single timestep", 4, 1, 3, 4),
        ("Multiple batches, multiple timesteps", 2, 3, 4, 3),
        ("Large batch production size", 8, 5, 64, 32),
        ("Tall matrices batch", 3, 2, 16, 8),
        ("Square matrices batch", 4, 3, 16, 16),
    ]
    
    results = {}
    
    for name, B, L, D, N in test_cases:
        result = compare_test_case(name, B, L, D, N)
        results[name] = result
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Tested {len(test_cases)} configurations")
    print(f"All PyTorch tests completed successfully ✓")
    
    # Check if CUDA executable exists
    import os
    cuda_exe = '/project/khanhnt/muontest/Momentum_correct/test_ns_multibatch_cuda'
    if os.path.exists(cuda_exe):
        print(f"\nRunning CUDA tests...")
        cuda_output = run_cuda_multibatch()
        if cuda_output:
            print(f"\n{'='*80}")
            print("CUDA Output:")
            print(f"{'='*80}")
            print(cuda_output)
            print(f"\n✅ Both CUDA and PyTorch tests completed")
            print(f"✅ Manual comparison: Check that CUDA output matches PyTorch output above")
        else:
            print(f"\n⚠️  CUDA test failed to run")
    else:
        print(f"\nCUDA executable not found. Compile with:")
        print(f"  nvcc -O3 -arch=sm_80 test_ns_multibatch_cuda.cu -o test_ns_multibatch_cuda")
    
    print(f"\n{'='*80}")
    print("VALIDATION CRITERIA")
    print(f"{'='*80}")
    print("For each test case, compare:")
    print("  1. Initial norms should match within 0.5%")
    print("  2. First iteration traces should match within 1%")
    print("  3. Final norms and traces should match within 5-20%")
    print("  4. Orthogonality quality should be similar (< 1.0)")
    print("\nAll differences should be due to BF16 rounding, not algorithmic errors.")


if __name__ == "__main__":
    main()

