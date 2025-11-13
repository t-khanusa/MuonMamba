#!/usr/bin/env python3
"""
Generate test data for CUDA Newton-Schulz backward test
Saves inputs and PyTorch reference outputs to binary file
"""

import torch
import struct

# Same implementation as in our test
def newtonschulz5_velocity_detached_backward(G, grad_output, alpha=1.0, steps=5, eps=1e-7):
    """
    Newton-Schulz 5-step with backward through LAST step only
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # === PHASE 1: Recompute first 4 iterations (DETACHED) ===
    with torch.no_grad():
        # Compute b_t = alpha * G
        b_t = alpha * G
        
        # Convert to bfloat16 and normalize
        X = b_t.bfloat16().float()
        norm = X.norm() + eps
        X = X / norm
        
        # Transpose if needed
        transposed = (X.size(0) > X.size(1))
        if transposed:
            X = X.T
        
        # Run 4 NS iterations (detached)
        for _ in range(4):
            A = X @ X.T
            A_bf16 = A.bfloat16().float()
            A2 = A_bf16 @ A_bf16
            A2_bf16 = A2.bfloat16().float()
            B_mat = b * A_bf16 + c * A2_bf16
            X = a * X + B_mat @ X
    
    # === PHASE 2: 5th iteration WITH gradients ===
    X_4 = X.detach().requires_grad_(True)
    
    # Compute 5th iteration
    A_4 = X_4 @ X_4.T
    A_4_bf16 = A_4.bfloat16().float()
    A_4_squared = A_4_bf16 @ A_4_bf16
    A_4_squared_bf16 = A_4_squared.bfloat16().float()
    B_4 = b * A_4_bf16 + c * A_4_squared_bf16
    X_5 = a * X_4 + B_4 @ X_4
    
    # Backward pass through 5th iteration
    if transposed:
        X_5.backward(grad_output.T)
    else:
        X_5.backward(grad_output)
    
    dX_4 = X_4.grad
    
    # === PHASE 3: Backward through normalization ===
    if transposed:
        dX_4 = dX_4.T
        X_4_for_norm = X_4.detach().T
    else:
        X_4_for_norm = X_4.detach()
    
    dot_product = (dX_4 * X_4_for_norm).sum()
    d_b_t_bf16 = (dX_4 - X_4_for_norm * dot_product) / norm
    d_b_t = d_b_t_bf16
    d_G = alpha * d_b_t
    
    return d_G.float()


def main():
    print("="*80)
    print("Generating Newton-Schulz Test Data for CUDA")
    print("="*80)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters (must match CUDA test)
    D, N = 16, 32
    alpha = 1.0
    
    print(f"\nTest configuration:")
    print(f"  Matrix shape: [{D}, {N}]")
    print(f"  Alpha: {alpha}")
    
    # Generate test inputs
    G_input = torch.randn(D, N, dtype=torch.float32)
    grad_output = torch.randn(D, N, dtype=torch.float32)
    
    print(f"\nInput statistics:")
    print(f"  G_input: mean={G_input.mean():.4f}, std={G_input.std():.4f}")
    print(f"  grad_output: mean={grad_output.mean():.4f}, std={grad_output.std():.4f}")
    
    # Compute PyTorch reference gradient
    print(f"\nComputing PyTorch reference gradient...")
    grad_G_torch = newtonschulz5_velocity_detached_backward(G_input, grad_output, alpha)
    
    print(f"  grad_G: mean={grad_G_torch.mean():.4f}, std={grad_G_torch.std():.4f}")
    print(f"  grad_G: min={grad_G_torch.min():.4f}, max={grad_G_torch.max():.4f}")
    
    # Save to binary file
    output_file = "/tmp/ns_test_data.bin"
    with open(output_file, "wb") as f:
        # Write in order: G_input, grad_output, grad_G_torch
        f.write(G_input.numpy().tobytes())
        f.write(grad_output.numpy().tobytes())
        f.write(grad_G_torch.numpy().tobytes())
    
    print(f"\n✅ Test data saved to: {output_file}")
    print(f"   Total size: {(D * N * 3 * 4) / 1024:.2f} KB")
    
    print("\n" + "="*80)
    print("Now run the CUDA test:")
    print("  cd /project/khanhnt/muontest/Momentum_correct/csrc/selective_scan")
    print("  nvcc -o test_cuda_ns_velocity_backward test_cuda_ns_velocity_backward.cu -std=c++17")
    print("  ./test_cuda_ns_velocity_backward")
    print("="*80)


if __name__ == "__main__":
    main()

