#!/usr/bin/env python3
"""
Detailed Python backward pass with clear step-by-step computation
This implements: detach first 4 steps, gradient only in last step
"""

import torch

def newtonschulz5_backward_detailed(G_input, grad_output, print_steps=False):
    """
    Backward pass for Newton-Schulz with detached first 4 iterations.
    Returns gradients for G_input.
    
    Args:
        G_input: [M, N] input matrix
        grad_output: [M, N] gradient from forward (dL/dX_5)
    
    Returns:
        grad_G: [M, N] gradient w.r.t. G_input
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    eps = 1e-8
    
    M, N = G_input.shape
    
    if print_steps:
        print("=" * 80)
        print("PYTHON BACKWARD PASS (Detached first 4 steps)")
        print("=" * 80)
    
    # ========== PHASE 1: Recompute X_0 → X_4 (Detached, 4 iterations) ==========
    with torch.no_grad():
        # Step 1.1: Convert to BF16
        G_bf16 = G_input.bfloat16()
        G_bf16_fp32 = G_bf16.float()
        
        # Step 1.2: Compute norm from BF16 values
        norm = torch.sqrt((G_bf16_fp32 ** 2).sum() + eps)
        
        if print_steps:
            print(f"\nStep 1: Normalization")
            print(f"  norm = {norm:.6f}")
        
        # Step 1.3: Normalize and round to BF16
        X_fp32 = G_bf16_fp32 / norm
        X_bf16 = X_fp32.bfloat16()
        X = X_bf16.float()
        
        transposed = (M > N)
        if transposed:
            X = X.T
        
        if print_steps:
            print(f"  X_0[0,0] = {X[0,0]:.6f}")
            print(f"  transposed = {transposed}")
        
        # Step 1.4: Run 4 NS iterations (detached)
        for step in range(4):
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
            
            if print_steps and step == 3:
                print(f"\nStep 2: After 4 NS iterations")
                print(f"  X_4[0,0] = {X[0,0]:.6f}")
        
        X_4_detached = X.clone()
    
    # ========== PHASE 2: 5th iteration WITH gradients ==========
    if print_steps:
        print(f"\nStep 3: 5th iteration (with gradients)")
    
    X_4 = X_4_detached.requires_grad_(True)
    
    # Compute A_4 (round to BF16 for consistency)
    A_4_fp32 = X_4 @ X_4.T
    A_4 = A_4_fp32.bfloat16().float()
    
    # Compute A_4^2 (round to BF16)
    A_4_sq_fp32 = A_4 @ A_4
    A_4_sq = A_4_sq_fp32.bfloat16().float()
    
    # Compute B_4 = b*A_4 + c*A_4^2 (round to BF16)
    B_4_fp32 = b * A_4 + c * A_4_sq
    B_4 = B_4_fp32.bfloat16().float()
    
    # Compute X_5 = a*X_4 + B_4@X_4
    X_5 = a * X_4 + B_4 @ X_4
    
    if print_steps:
        print(f"  A_4[0,0] = {A_4[0,0]:.6f}")
        print(f"  B_4[0,0] = {B_4[0,0]:.6f}")
        print(f"  X_5[0,0] = {X_5[0,0]:.6f}")
    
    # Adjust grad_output for transpose if needed
    if transposed:
        grad_output_for_backward = grad_output.T
    else:
        grad_output_for_backward = grad_output
    
    # ========== PHASE 3: Backward through 5th iteration ==========
    if print_steps:
        print(f"\nStep 4: Backward through 5th iteration")
    
    X_5.backward(grad_output_for_backward)
    dX_4 = X_4.grad
    
    if print_steps:
        print(f"  dX_4[0,0] = {dX_4[0,0]:.6f}")
    
    # ========== PHASE 4: Backward through normalization ==========
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
        
        if print_steps:
            print(f"\nStep 5: Backward through normalization")
            print(f"  X_0[0,0] = {X_0[0,0]:.6f}")
        
        # Gradient through normalization: d(G/||G||) = (dX - X * <dX, X>) / ||G||
        dot_product = (dX_4 * X_0).sum()
        grad_G_normalized = (dX_4 - X_0 * dot_product) / norm
        
        if print_steps:
            print(f"  dot_product = {dot_product:.6f}")
            print(f"  grad_G[0,0] = {grad_G_normalized[0,0]:.6f}")
        
        if transposed:
            grad_G_normalized = grad_G_normalized.T
    
    return grad_G_normalized


def test_with_simple_case():
    """Test with 2x2 case and print all intermediate values"""
    torch.manual_seed(42)
    G = torch.randn(2, 2, dtype=torch.float32)
    grad_output = torch.randn(2, 2, dtype=torch.float32)
    
    print("\nInput:")
    print(f"G =\n{G}")
    print(f"\ngrad_output =\n{grad_output}")
    
    grad_G = newtonschulz5_backward_detailed(G, grad_output, print_steps=True)
    
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(f"grad_G =\n{grad_G}")
    
    # Now test the gradient accumulation formula
    print("\n" + "=" * 80)
    print("GRADIENT ACCUMULATION (for u, delta, B)")
    print("=" * 80)
    
    # Assume G = alpha * delta * B * u (element-wise for each d)
    # For testing, let's say:
    alpha = 1.0
    u = torch.randn(2, dtype=torch.float32) * 0.5
    delta = torch.randn(2, dtype=torch.float32) * 0.1 + 0.5
    B = torch.randn(2, 2, dtype=torch.float32) * 0.3
    
    # Reconstruct G from u, delta, B
    G_reconstructed = alpha * delta.unsqueeze(1) * B * u.unsqueeze(1)
    
    print(f"\nInputs:")
    print(f"  u = {u}")
    print(f"  delta = {delta}")
    print(f"  B =\n{B}")
    print(f"\nReconstructed G =\n{G_reconstructed}")
    
    # Compute grad_G for reconstructed input
    grad_G_recon = newtonschulz5_backward_detailed(G_reconstructed, grad_output, print_steps=False)
    
    # Now accumulate gradients for u, delta, B
    # grad_u[d] = sum_n alpha * delta[d] * B[d,n] * grad_G[d,n]
    # grad_delta[d] = sum_n alpha * B[d,n] * u[d] * grad_G[d,n]
    # grad_B[d,n] = alpha * delta[d] * u[d] * grad_G[d,n]
    
    grad_u = torch.zeros_like(u)
    grad_delta = torch.zeros_like(delta)
    grad_B = torch.zeros_like(B)
    
    for d in range(2):
        for n in range(2):
            grad_u[d] += alpha * delta[d] * B[d, n] * grad_G_recon[d, n]
            grad_delta[d] += alpha * B[d, n] * u[d] * grad_G_recon[d, n]
            grad_B[d, n] = alpha * delta[d] * u[d] * grad_G_recon[d, n]
    
    print(f"\nGradients:")
    print(f"  grad_u = {grad_u}")
    print(f"  grad_delta = {grad_delta}")
    print(f"  grad_B =\n{grad_B}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_with_simple_case()

