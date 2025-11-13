#!/usr/bin/env python3
"""
Generate test data for Newton-Schulz velocity backward pass testing.
Saves data to /tmp/ns_velocity_test_data.bin for CUDA test to load.
"""

import torch
import numpy as np
import struct

def newtonschulz5(G, steps=5, eps=1e-7):
    """PyTorch reference implementation (no in-place ops for autograd)"""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)  # Not in-place
    transposed = (G.size(0) > G.size(1))
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X

def newtonschulz5_velocity_detached_backward(G_input, grad_output):
    """
    Backward pass: recompute first 4 iterations (detached), then backprop through last iteration only.
    Returns gradients for G_input.
    This matches the CUDA kernel's behavior EXACTLY.
    """
    assert G_input.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    eps = 1e-8  # Match CUDA kernel (not 1e-7!)
    
    # Phase 1: Recompute X_0 → X_4 (detached, 4 iterations)
    with torch.no_grad():
        # Convert to BF16 and compute norm in FP32 (matching CUDA)
        G_bf16 = G_input.bfloat16()
        G_bf16_fp32 = G_bf16.float()  # BF16 values in FP32 container
        norm = torch.sqrt((G_bf16_fp32 ** 2).sum() + eps)
        
        # Normalize in FP32, then round to BF16 (matching CUDA line 634-636)
        X_fp32 = G_bf16_fp32 / norm
        X_bf16 = X_fp32.bfloat16()
        X = X_bf16.float()  # BF16 values in FP32 container
        
        transposed = (G_input.size(0) > G_input.size(1))
        if transposed:
            X = X.T
        
        for _ in range(4):  # Only 4 iterations
            # Compute A in FP32, round to BF16
            A_fp32 = X @ X.T
            A_bf16 = A_fp32.bfloat16()
            A = A_bf16.float()
            
            # Compute A^2 in FP32, round to BF16
            A2_fp32 = A @ A
            A2_bf16 = A2_fp32.bfloat16()
            A2 = A2_bf16.float()
            
            # Compute B = b*A + c*A^2, round to BF16
            B_fp32 = b * A + c * A2
            B_bf16 = B_fp32.bfloat16()
            B = B_bf16.float()
            
            # Compute X_new = a*X + B@X, round to BF16
            X_new_fp32 = a * X + B @ X
            X_new_bf16 = X_new_fp32.bfloat16()
            X = X_new_bf16.float()
        
        # Store X_4 for later
        X_4_detached = X.clone()
    
    # DEBUG: Print X_4 stats for first call
    import sys
    if not hasattr(sys, '_x4_printed'):
        print(f"[Python DEBUG] X_4 stats: mean={X_4_detached.mean():.6f}, std={X_4_detached.std():.6f}")
        print(f"  X_4[0,0]={X_4_detached[0,0]:.6f}, X_4[0,1]={X_4_detached[0,1]:.6f}")
        sys._x4_printed = True
    
    # Phase 2: 5th iteration WITH gradients
    # IMPORTANT: Compute A_4 and B_4 DETACHED (no gradients through their computation)
    # Only backprop through the APPLICATION of B_4 to X_4
    X_4 = X_4_detached.requires_grad_(True)
    
    # Compute A_4 and B_4 WITHOUT gradients (detached)
    A_4_fp32 = X_4.detach() @ X_4.detach().T
    A_4 = A_4_fp32.bfloat16().float()  # Round to BF16, store as FP32
    
    A_4_sq_fp32 = A_4 @ A_4
    A_4_sq = A_4_sq_fp32.bfloat16().float()  # Round to BF16
    
    B_4_fp32 = b * A_4 + c * A_4_sq
    B_4 = B_4_fp32.bfloat16().float()  # Round to BF16 (now B_4 is detached)
    
    # Apply B_4 (as a constant) to X_4 (with gradients)
    X_5 = a * X_4 + B_4 @ X_4
    
    # Adjust grad_output for transpose if needed
    if transposed:
        grad_output_for_backward = grad_output.T
    else:
        grad_output_for_backward = grad_output
    
    # Backward through 5th iteration
    X_5.backward(grad_output_for_backward)
    dX_4 = X_4.grad
    
    # Phase 3: Backward through initial normalization
    with torch.no_grad():
        # Recompute X_0 (normalized input) exactly as in forward
        G_bf16 = G_input.bfloat16()
        G_bf16_fp32 = G_bf16.float()
        norm = torch.sqrt((G_bf16_fp32 ** 2).sum() + eps)
        X_fp32 = G_bf16_fp32 / norm
        X_bf16 = X_fp32.bfloat16()
        X_0 = X_bf16.float()
        
        if transposed:
            X_0 = X_0.T
            dX_4 = dX_4.T
        
        # Gradient through normalization: d(G/||G||) = (dX - X * <dX, X>) / ||G||
        dot_product = (dX_4 * X_0).sum()
        grad_G_normalized = (dX_4 - X_0 * dot_product) / norm
        
        if transposed:
            grad_G_normalized = grad_G_normalized.T
    
    return grad_G_normalized

def main():
    print("=" * 80)
    print("Generating Newton-Schulz Velocity Backward Pass Test Data")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    # Test configuration
    batch = 2
    dim = 8
    seqlen = 16
    dstate = 16
    alpha = 1.0
    beta = 0.9
    
    print(f"\nConfiguration:")
    print(f"  Batch: {batch}, Dim: {dim}, Seqlen: {seqlen}, Dstate: {dstate}")
    print(f"  Alpha: {alpha}, Beta: {beta}")
    
    # Generate random inputs
    u = torch.randn(batch, dim, seqlen, dtype=torch.float32) * 0.5
    delta = torch.randn(batch, dim, seqlen, dtype=torch.float32) * 0.1 + 0.5
    B = torch.randn(dim, dstate, dtype=torch.float32) * 0.3
    
    # Generate random grad_output (gradient from forward pass)
    grad_output = torch.randn(batch, dim, seqlen, dstate, dtype=torch.float32) * 0.1
    
    print("\nInput statistics:")
    print(f"  u: mean={u.mean():.4f}, std={u.std():.4f}")
    print(f"  delta: mean={delta.mean():.4f}, std={delta.std():.4f}")
    print(f"  B: mean={B.mean():.4f}, std={B.std():.4f}")
    print(f"  grad_output: mean={grad_output.mean():.4f}, std={grad_output.std():.4f}")
    
    # Compute gradients using detached backward (last iteration only)
    print("\nComputing reference gradients with detached backward (last iteration only)...")
    
    # Initialize gradient accumulators
    grad_u_torch = torch.zeros_like(u)
    grad_delta_torch = torch.zeros_like(delta)
    grad_B_torch = torch.zeros_like(B)
    
    # Forward pass and backward for each (batch, timestep)
    # Newton-Schulz is applied to [dim, dstate] matrix for each (batch, timestep)
    for b in range(batch):
        for t in range(seqlen):
            # Compute G = alpha * delta * B * u for this timestep
            # G shape: [dim, dstate]
            # Broadcasting: delta[b,:,t] is [dim], u[b,:,t] is [dim], B is [dim, dstate]
            G_bt = alpha * delta[b, :, t].unsqueeze(1) * B * u[b, :, t].unsqueeze(1)
            # G_bt shape: [dim, dstate]
            
            # Get gradient from output
            grad_V_bt = grad_output[b, :, t, :]  # [dim, dstate]
            
            # Compute gradient using detached backward (only through last iteration)
            grad_G_bt = newtonschulz5_velocity_detached_backward(G_bt.detach(), grad_V_bt)
            # grad_G_bt shape: [dim, dstate]
            
            # DEBUG: Print first call
            if b == 0 and t == 0:
                print(f"\n[DEBUG] First (b=0, t=0) call:")
                print(f"  G_bt[0,:4] = {G_bt[0,:4]}")
                print(f"  grad_V_bt[0,:4] = {grad_V_bt[0,:4]}")
                print(f"  grad_G_bt[0,:4] = {grad_G_bt[0,:4]}")
            
            # Accumulate gradients for u, delta, B
            # G_bt[d, n] = alpha * delta[b, d, t] * B[d, n] * u[b, d, t]
            # dG/du = alpha * delta * B
            # dG/ddelta = alpha * B * u
            # dG/dB = alpha * delta * u
            for d in range(dim):
                for n in range(dstate):
                    grad_u_torch[b, d, t] += grad_G_bt[d, n] * alpha * delta[b, d, t] * B[d, n]
                    grad_delta_torch[b, d, t] += grad_G_bt[d, n] * alpha * B[d, n] * u[b, d, t]
                    grad_B_torch[d, n] += grad_G_bt[d, n] * alpha * delta[b, d, t] * u[b, d, t]
    
    print("✓ Reference gradients computed")
    print(f"\nGradient statistics:")
    print(f"  grad_u: mean={grad_u_torch.mean():.6e}, std={grad_u_torch.std():.6e}")
    print(f"  grad_delta: mean={grad_delta_torch.mean():.6e}, std={grad_delta_torch.std():.6e}")
    print(f"  grad_B: mean={grad_B_torch.mean():.6e}, std={grad_B_torch.std():.6e}")
    
    # Save to binary file
    output_file = "/tmp/ns_velocity_test_data.bin"
    print(f"\nSaving data to {output_file}...")
    
    with open(output_file, "wb") as f:
        # Write in order: grad_output, u, delta, B, grad_u, grad_delta, grad_B
        grad_output.numpy().astype(np.float32).tofile(f)
        u.detach().numpy().astype(np.float32).tofile(f)
        delta.detach().numpy().astype(np.float32).tofile(f)
        B.detach().numpy().astype(np.float32).tofile(f)
        grad_u_torch.numpy().astype(np.float32).tofile(f)
        grad_delta_torch.numpy().astype(np.float32).tofile(f)
        grad_B_torch.numpy().astype(np.float32).tofile(f)
    
    print("✓ Data saved successfully")
    
    # Verify file size
    import os
    expected_size = (
        batch * dim * seqlen * dstate +  # grad_output
        batch * dim * seqlen +           # u
        batch * dim * seqlen +           # delta
        dim * dstate +                   # B
        batch * dim * seqlen +           # grad_u
        batch * dim * seqlen +           # grad_delta
        dim * dstate                     # grad_B
    ) * 4  # float32
    
    actual_size = os.path.getsize(output_file)
    print(f"\nFile verification:")
    print(f"  Expected size: {expected_size} bytes")
    print(f"  Actual size: {actual_size} bytes")
    print(f"  {'✓ Match' if expected_size == actual_size else '✗ Mismatch'}")
    
    print("\n" + "=" * 80)
    print("Data generation complete! Run CUDA test with:")
    print("  nvcc -o test_real_ns_backward test_real_ns_backward.cu -std=c++17 -arch=sm_80 \\")
    print("    -I$CONDA_PREFIX/lib/python3.10/site-packages/torch/include \\")
    print("    -I$CONDA_PREFIX/lib/python3.10/site-packages/torch/include/torch/csrc/api/include")
    print("  ./test_real_ns_backward")
    print("=" * 80)

if __name__ == "__main__":
    main()

