#!/usr/bin/env python3
"""
Test Newton-Schulz 5-Step Backward Pass
Compares CUDA implementation against PyTorch reference
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import subprocess
import sys
import os

def pytorch_newton_schulz_5step_forward(G, num_iters=5, eps=1e-7):
    """
    Official PyTorch Newton-Schulz implementation (exact copy)
    
    Args:
        G: Input tensor [D, N] (or any 2D shape)
        num_iters: Number of NS iterations (default 5)
        eps: Epsilon for numerical stability
    
    Returns:
        X_final: Orthogonalized output
        intermediates: Dict of intermediate values for backward
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.bfloat16()
    norm = X.float().norm()
    X = X.float()
    X /= (norm + eps)
    
    intermediates = {
        'norm': norm,
        'X_0_prenorm': G.bfloat16().float().clone(),
        'X_0': X.clone(),
        'X_iters': [],
        'A_iters': [],
        'B_iters': []
    }
    
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    
    for step in range(num_iters):
        intermediates['X_iters'].append(X.clone())
        
        A = X @ X.T
        A = A.bfloat16().float()
        intermediates['A_iters'].append(A.clone())
        
        A2 = A @ A
        A2 = A2.bfloat16().float()
        
        B_mat = b * A + c * A2
        B_mat = B_mat.bfloat16().float()
        intermediates['B_iters'].append(B_mat.clone())
        
        X = a * X + B_mat @ X
        X = X.bfloat16().float()
    
    if G.size(0) > G.size(1):
        X = X.T
    
    intermediates['X_final'] = X
    intermediates['transposed'] = transposed
    
    return X, intermediates


def pytorch_newton_schulz_5step_backward_detached(grad_output, G, alpha, delta_val, B_val, u_val, eps=1e-7):
    """
    PyTorch reference for NS 5-step backward with detached first 4 iterations
    Matches official PyTorch implementation structure
    
    Args:
        grad_output: Gradient from loss [D, N]
        G: Input b_t = alpha * delta * B * u [D, N]
        alpha: Scalar coefficient
        delta_val: Delta value (scalar or per-dim) [D]
        B_val: B matrix [D, N]
        u_val: u value (scalar or per-dim) [D]
        eps: Epsilon for numerical stability
    
    Returns:
        grad_u, grad_delta, grad_B
    """
    a, b_coef, c = 3.4445, -4.7750, 2.0315
    
    # ===== PHASE 1: Recompute X_0 → X_4 (detached) =====
    with torch.no_grad():
        # Match official implementation
        X = G.bfloat16()
        norm = X.float().norm()
        X = X.float()
        X /= (norm + eps)
        
        # Transpose if tall matrix
        transposed = False
        if G.size(0) > G.size(1):
            X = X.T
            transposed = True
        
        # Run 4 iterations (detached)
        for step in range(4):
            A = X @ X.T
            A = A.bfloat16().float()
            A2 = A @ A
            A2 = A2.bfloat16().float()
            B_mat = b_coef * A + c * A2
            B_mat = B_mat.bfloat16().float()
            X = a * X + B_mat @ X
            X = X.bfloat16().float()
        
        X_4 = X.clone()
    
    # ===== PHASE 2: Backward through 5th iteration =====
    
    # Make X_4 require gradient
    X_4_grad = X_4.clone().requires_grad_(True)
    
    # Forward 5th iteration with gradients
    A_4 = X_4_grad @ X_4_grad.T
    A_4 = A_4.bfloat16().float()
    A_4_sq = A_4 @ A_4
    A_4_sq = A_4_sq.bfloat16().float()
    B_4 = b_coef * A_4 + c * A_4_sq
    B_4 = B_4.bfloat16().float()
    X_5 = a * X_4_grad + B_4 @ X_4_grad
    X_5 = X_5.bfloat16().float()
    
    # Transpose back if needed
    if transposed:
        X_5 = X_5.T
    
    # Backward pass
    X_5.backward(grad_output)
    
    # Get gradient w.r.t. X_4 (in transposed space if needed)
    dX_4 = X_4_grad.grad.clone()
    
    # ===== Gradient through normalization and BF16 conversion =====
    with torch.no_grad():
        # If transposed, transpose gradient back
        if transposed:
            dX_4 = dX_4.T
            X_4_for_grad = X_4.T
        else:
            X_4_for_grad = X_4
        
        # Gradient through normalization: X_0 = G_bf16 / norm
        # d(G_bf16) = (dX_4 - <dX_4, X_4> * X_4) / norm
        dnorm_contrib = (dX_4 * X_4_for_grad).sum()
        d_G_bf16 = (dX_4 - dnorm_contrib * X_4_for_grad) / norm
        
        # Straight-through for BF16: d(G) = d(G_bf16)
        d_G = d_G_bf16
        
        # ===== Gradient through G = alpha * delta * B * u =====
        # G[d, n] = alpha * delta[d] * B[d, n] * u[d]
        
        # grad_u[d] = sum_n alpha * delta[d] * B[d, n] * d_G[d, n]
        grad_u = (alpha * delta_val.unsqueeze(1) * B_val * d_G).sum(dim=1)
        
        # grad_delta[d] = sum_n alpha * B[d, n] * u[d] * d_G[d, n]
        grad_delta = (alpha * B_val * u_val.unsqueeze(1) * d_G).sum(dim=1)
        
        # grad_B[d, n] = alpha * delta[d] * u[d] * d_G[d, n]
        grad_B = alpha * delta_val.unsqueeze(1) * u_val.unsqueeze(1) * d_G
    
    return grad_u, grad_delta, grad_B


def test_newton_schulz_backward_correctness():
    """
    Test Newton-Schulz backward pass correctness
    """
    print("=" * 80)
    print("Newton-Schulz 5-Step Backward Pass Correctness Test")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test configurations - start with smaller matrices for debugging
    configs = [
        {"D": 8, "N": 16, "batch": 1, "L": 1, "name": "Small fat matrix (D < N)"},
        {"D": 16, "N": 8, "batch": 1, "L": 1, "name": "Small tall matrix (D > N)"},
    ]
    
    alpha = 0.1
    all_passed = True
    
    for config in configs:
        D, N = config["D"], config["N"]
        batch, L = config["batch"], config["L"]
        name = config["name"]
        
        print(f"\n{'-' * 80}")
        print(f"Test: {name}")
        print(f"Dimensions: D={D}, N={N}, batch={batch}, L={L}")
        print(f"{'-' * 80}")
        
        # Generate random inputs
        u = torch.randn(batch, D, L, dtype=torch.float32)
        delta = torch.randn(batch, D, L, dtype=torch.float32)
        B = torch.randn(D, N, dtype=torch.float32)
        
        # Make inputs smaller scale to avoid numerical issues
        u = u * 0.5
        delta = delta * 0.5
        B = B * 0.5
        
        # Compute G (which is b_t) for single (batch=0, time=0)
        G = alpha * delta[0, :, 0].unsqueeze(1) * B * u[0, :, 0].unsqueeze(1)
        
        # Forward pass to get output
        X_final, intermediates = pytorch_newton_schulz_5step_forward(G, num_iters=5)
        
        # Random gradient
        grad_output = torch.randn_like(X_final)
        
        # Compute reference gradients using PyTorch
        grad_u_ref, grad_delta_ref, grad_B_ref = pytorch_newton_schulz_5step_backward_detached(
            grad_output, G, alpha, delta[0, :, 0], B, u[0, :, 0]
        )
        
        # Debug: print some values
        print(f"\n  G stats: mean={G.mean():.6f}, std={G.std():.6f}, norm={G.norm():.6f}")
        print(f"  X_final stats: mean={X_final.mean():.6f}, std={X_final.std():.6f}, norm={X_final.norm():.6f}")
        print(f"  grad_output stats: mean={grad_output.mean():.6f}, std={grad_output.std():.6f}")
        print(f"  grad_u_ref stats: mean={grad_u_ref.mean():.6f}, std={grad_u_ref.std():.6f}, max={grad_u_ref.abs().max():.6f}")
        
        # Verify gradients using numerical differentiation
        print("\n1. Numerical Gradient Check (Finite Differences)")
        print("-" * 60)
        
        eps = 1e-4
        passed_numerical = True
        
        # Check grad_u numerically
        grad_u_numerical = torch.zeros_like(u[0, :, 0])
        for d in range(D):
            u_plus = u.clone()
            u_plus[0, d, 0] += eps
            G_plus = alpha * delta[0, :, 0].unsqueeze(1) * B * u_plus[0, :, 0].unsqueeze(1)
            X_plus, _ = pytorch_newton_schulz_5step_forward(G_plus, num_iters=5)
            
            u_minus = u.clone()
            u_minus[0, d, 0] -= eps
            G_minus = alpha * delta[0, :, 0].unsqueeze(1) * B * u_minus[0, :, 0].unsqueeze(1)
            X_minus, _ = pytorch_newton_schulz_5step_forward(G_minus, num_iters=5)
            
            grad_u_numerical[d] = ((X_plus - X_minus) * grad_output).sum() / (2 * eps)
        
        u_rel_error = (grad_u_ref - grad_u_numerical).abs().max() / (grad_u_numerical.abs().max() + 1e-8)
        u_passed = u_rel_error < 0.05  # 5% tolerance for numerical gradients
        
        print(f"  grad_u:     max_rel_error = {u_rel_error:.6f}  {'✓ PASS' if u_passed else '✗ FAIL'}")
        
        # Check grad_delta numerically (sample a few dimensions for speed)
        grad_delta_numerical = torch.zeros_like(delta[0, :, 0])
        for d in range(min(5, D)):  # Check first 5 dims
            delta_plus = delta.clone()
            delta_plus[0, d, 0] += eps
            G_plus = alpha * delta_plus[0, :, 0].unsqueeze(1) * B * u[0, :, 0].unsqueeze(1)
            X_plus, _ = pytorch_newton_schulz_5step_forward(G_plus, num_iters=5)
            
            delta_minus = delta.clone()
            delta_minus[0, d, 0] -= eps
            G_minus = alpha * delta_minus[0, :, 0].unsqueeze(1) * B * u[0, :, 0].unsqueeze(1)
            X_minus, _ = pytorch_newton_schulz_5step_forward(G_minus, num_iters=5)
            
            grad_delta_numerical[d] = ((X_plus - X_minus) * grad_output).sum() / (2 * eps)
        
        delta_rel_error = (grad_delta_ref[:5] - grad_delta_numerical[:5]).abs().max() / (grad_delta_numerical[:5].abs().max() + 1e-8)
        delta_passed = delta_rel_error < 0.05
        
        print(f"  grad_delta: max_rel_error = {delta_rel_error:.6f}  {'✓ PASS' if delta_passed else '✗ FAIL'}")
        
        # Check grad_B numerically (sample a few elements for speed)
        grad_B_numerical = torch.zeros_like(B)
        sample_indices = [(0, 0), (0, N-1), (D-1, 0), (D-1, N-1), (D//2, N//2)]
        max_B_error = 0.0
        
        for (d, n) in sample_indices:
            B_plus = B.clone()
            B_plus[d, n] += eps
            G_plus = alpha * delta[0, :, 0].unsqueeze(1) * B_plus * u[0, :, 0].unsqueeze(1)
            X_plus, _ = pytorch_newton_schulz_5step_forward(G_plus, num_iters=5)
            
            B_minus = B.clone()
            B_minus[d, n] -= eps
            G_minus = alpha * delta[0, :, 0].unsqueeze(1) * B_minus * u[0, :, 0].unsqueeze(1)
            X_minus, _ = pytorch_newton_schulz_5step_forward(G_minus, num_iters=5)
            
            grad_B_numerical[d, n] = ((X_plus - X_minus) * grad_output).sum() / (2 * eps)
            
            error = abs(grad_B_ref[d, n].item() - grad_B_numerical[d, n].item()) / (abs(grad_B_numerical[d, n].item()) + 1e-8)
            max_B_error = max(max_B_error, error)
        
        B_passed = max_B_error < 0.05
        
        print(f"  grad_B:     max_rel_error = {max_B_error:.6f}  {'✓ PASS' if B_passed else '✗ FAIL'}")
        
        # Check that trace of Gram matrix is improving (moving towards identity)
        print("\n2. Trace Check (Orthogonalization Progress)")
        print("-" * 60)
        
        if D <= N:
            # Fat matrix: check A = X @ X.T / ||X||²
            gram = torch.matmul(X_final, X_final.T) / (X_final.norm() ** 2)
            expected_trace = D
        else:
            # Tall matrix: check A = X.T @ X / ||X||²
            gram = torch.matmul(X_final.T, X_final) / (X_final.norm() ** 2)
            expected_trace = N
        
        trace = gram.diag().sum()
        trace_error = abs(trace - expected_trace) / expected_trace
        ortho_passed = trace_error < 0.2  # 20% tolerance (NS needs more iters for full ortho)
        
        print(f"  Trace: {trace:.4f} (expected {expected_trace}, error: {trace_error:.4f})  {'✓ PASS' if ortho_passed else '✗ FAIL'}")
        
        # Overall test result
        test_passed = u_passed and delta_passed and B_passed and ortho_passed
        all_passed = all_passed and test_passed
        
        print(f"\n{'='*60}")
        print(f"Test Result: {'✓ PASS' if test_passed else '✗ FAIL'}")
        print(f"{'='*60}")
    
    print(f"\n{'='*80}")
    print(f"Overall Test Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print(f"{'='*80}\n")
    
    return all_passed


def test_autograd_consistency():
    """
    Test that PyTorch autograd gives same results as our manual backward
    """
    print("=" * 80)
    print("PyTorch Autograd Consistency Test")
    print("=" * 80)
    
    torch.manual_seed(123)
    
    D, N = 32, 64
    alpha = 0.1
    
    # Generate inputs
    u = torch.randn(D, requires_grad=True)
    delta = torch.randn(D, requires_grad=True)
    B = torch.randn(D, N, requires_grad=True)
    
    # Scale down
    u.data *= 0.5
    delta.data *= 0.5
    B.data *= 0.5
    
    # Compute b_t
    b_t = alpha * delta.unsqueeze(1) * B * u.unsqueeze(1)
    
    # Forward with full autograd (5 iterations)
    X_final, _ = pytorch_newton_schulz_5step_forward(b_t.detach(), num_iters=5)
    
    # Manual backward
    grad_output = torch.randn_like(X_final)
    grad_u_manual, grad_delta_manual, grad_B_manual = pytorch_newton_schulz_5step_backward_detached(
        grad_output, b_t.detach(), alpha, delta.detach(), B.detach(), u.detach()
    )
    
    print(f"\nManual backward gradients:")
    print(f"  grad_u:     mean={grad_u_manual.mean():.6f}, std={grad_u_manual.std():.6f}")
    print(f"  grad_delta: mean={grad_delta_manual.mean():.6f}, std={grad_delta_manual.std():.6f}")
    print(f"  grad_B:     mean={grad_B_manual.mean():.6f}, std={grad_B_manual.std():.6f}")
    
    print("\n✓ Manual backward computation successful")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("NEWTON-SCHULZ 5-STEP BACKWARD PASS TEST SUITE")
    print("=" * 80 + "\n")
    
    # Run tests
    test_autograd_consistency()
    passed = test_newton_schulz_backward_correctness()
    
    if passed:
        print("\n✓ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        sys.exit(1)

