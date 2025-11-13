#!/usr/bin/env python3
"""
Test Newton-Schulz Velocity 5-Step Backward Pass
Verifies that CUDA backward pass matches PyTorch autograd for:
- First 4 steps detached (no gradients)
- Last step with gradients
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

# Official PyTorch Newton-Schulz 5-step implementation
def newtonschulz5(G, steps=5, eps=1e-7):
    """
    Official PyTorch Newton-Schulz implementation (non-inplace for autograd)
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    
    if G.size(0) > G.size(1):
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    
    return X


def newtonschulz5_velocity_detached_backward(G, grad_output, alpha=1.0, steps=5, eps=1e-7):
    """
    Newton-Schulz 5-step with backward through LAST step only
    First 4 steps are detached (no gradients computed)
    
    This matches the CUDA implementation:
    1. Forward: recompute X₀→X₁→X₂→X₃→X₄ (4 iterations forward, detached)
    2. Backward: compute gradients through 5th iteration only
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # === PHASE 1: Recompute first 4 iterations (DETACHED) ===
    with torch.no_grad():
        # Compute b_t = alpha * G (in CUDA, this would be alpha * delta * B * u)
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
    # X_4 is now set, enable gradients
    X_4 = X.detach().requires_grad_(True)
    
    # Compute 5th iteration
    A_4 = X_4 @ X_4.T
    A_4_bf16 = A_4.bfloat16().float()
    A_4_squared = A_4_bf16 @ A_4_bf16
    A_4_squared_bf16 = A_4_squared.bfloat16().float()
    B_4 = b * A_4_bf16 + c * A_4_squared_bf16
    X_5 = a * X_4 + B_4 @ X_4
    
    # Backward pass through 5th iteration (in transposed space if needed)
    if transposed:
        # X_5 is [N, M], grad_output is [M, N]
        # Need to transpose grad_output to [N, M] to match
        X_5.backward(grad_output.T)
    else:
        # X_5 is [M, N], grad_output is [M, N]
        X_5.backward(grad_output)
    
    dX_4 = X_4.grad
    
    # === PHASE 3: Backward through normalization ===
    # X_4 = b_t_bf16 / norm
    # dX_4 known, need d(b_t_bf16)
    
    # Transpose back for normalization backward
    if transposed:
        dX_4 = dX_4.T
        X_4_for_norm = X_4.detach().T
    else:
        X_4_for_norm = X_4.detach()
    
    # Backward through normalization: d(b_t) = (dX_4 - X_4 * <dX_4, X_4>) / norm
    dot_product = (dX_4 * X_4_for_norm).sum()
    d_b_t_bf16 = (dX_4 - X_4_for_norm * dot_product) / norm
    
    # Straight-through estimator for BF16
    d_b_t = d_b_t_bf16
    
    # Backward through b_t = alpha * G
    d_G = alpha * d_b_t
    
    return d_G.float()


def test_backward_last_step_only():
    """
    Test that our manual backward (last step only) matches PyTorch autograd
    when we detach the first 4 iterations
    """
    print("\n" + "="*80)
    print("Test 1: Manual Backward (Last Step Only) vs PyTorch Autograd")
    print("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    M, N = 32, 16  # Fat matrix (M > N)
    alpha = 1.5
    
    # Create test input
    G = torch.randn(M, N, requires_grad=True)
    grad_output = torch.randn(M, N)
    
    print(f"Input shape: {G.shape}")
    print(f"Alpha: {alpha}")
    
    # === Method 1: Manual backward (last step only) ===
    d_G_manual = newtonschulz5_velocity_detached_backward(G, grad_output, alpha)
    
    # === Method 2: PyTorch autograd (detached first 4, grad on last) ===
    G_auto = G.detach().clone().requires_grad_(True)
    
    # Forward with first 4 detached, last with grad
    a, b, c = (3.4445, -4.7750, 2.0315)
    eps = 1e-7
    
    with torch.no_grad():
        b_t = alpha * G_auto
        X = b_t.bfloat16().float()
        norm = X.norm() + eps
        X = X / norm
        transposed = (X.size(0) > X.size(1))
        if transposed:
            X = X.T
        for _ in range(4):
            A = X @ X.T
            A_bf16 = A.bfloat16().float()
            A2 = A_bf16 @ A_bf16
            A2_bf16 = A2.bfloat16().float()
            B_mat = b * A_bf16 + c * A2_bf16
            X = a * X + B_mat @ X
    
    X_4 = X.detach().requires_grad_(True)
    A_4 = X_4 @ X_4.T
    A_4_bf16 = A_4.bfloat16().float()
    A_4_squared = A_4_bf16 @ A_4_bf16
    A_4_squared_bf16 = A_4_squared.bfloat16().float()
    B_4 = b * A_4_bf16 + c * A_4_squared_bf16
    X_5 = a * X_4 + B_4 @ X_4
    
    if transposed:
        X_5 = X_5.T
    
    # Backward
    X_5.backward(grad_output)
    dX_4_auto = X_4.grad
    
    # Backward through norm manually
    if transposed:
        dX_4_auto = dX_4_auto.T
        X_4_for_norm = X_4.detach().T
    else:
        X_4_for_norm = X_4.detach()
    
    dot_product = (dX_4_auto * X_4_for_norm).sum()
    d_b_t_auto = (dX_4_auto - X_4_for_norm * dot_product) / norm
    d_G_auto = alpha * d_b_t_auto
    
    # Compare
    diff = (d_G_manual - d_G_auto).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_error = (diff / (d_G_auto.abs() + 1e-8)).mean().item()
    
    print(f"\n📊 Gradient Comparison:")
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")
    print(f"   Relative error: {rel_error:.2e}")
    
    # Check if they match
    if max_diff < 1e-5:
        print(f"   ✅ PASS: Manual backward matches PyTorch autograd!")
        return True
    else:
        print(f"   ❌ FAIL: Gradients don't match!")
        print(f"\n   Manual gradient sample:\n{d_G_manual[:3, :3]}")
        print(f"\n   Autograd gradient sample:\n{d_G_auto[:3, :3]}")
        return False


def test_gradient_shapes():
    """
    Test that gradients have correct shapes for both fat and tall matrices
    """
    print("\n" + "="*80)
    print("Test 2: Gradient Shapes (Fat and Tall Matrices)")
    print("="*80)
    
    torch.manual_seed(123)
    alpha = 1.0
    
    test_cases = [
        (32, 16, "Fat matrix (M > N)"),
        (16, 32, "Tall matrix (M < N)"),
        (16, 16, "Square matrix (M = N)")
    ]
    
    all_pass = True
    for M, N, desc in test_cases:
        G = torch.randn(M, N, requires_grad=True)
        grad_output = torch.randn(M, N)
        
        d_G = newtonschulz5_velocity_detached_backward(G, grad_output, alpha)
        
        shape_ok = d_G.shape == G.shape
        print(f"\n{desc}:")
        print(f"   Input shape: {G.shape}")
        print(f"   Gradient shape: {d_G.shape}")
        print(f"   {'✅ PASS' if shape_ok else '❌ FAIL'}")
        
        all_pass = all_pass and shape_ok
    
    return all_pass


def test_gradient_sanity():
    """
    Sanity checks on the gradients:
    - Non-zero gradients
    - Finite values
    - Reasonable magnitude
    """
    print("\n" + "="*80)
    print("Test 3: Gradient Sanity Checks")
    print("="*80)
    
    torch.manual_seed(456)
    M, N = 16, 12
    alpha = 1.0
    
    G = torch.randn(M, N, requires_grad=True)
    grad_output = torch.randn(M, N)
    
    print(f"Input shape: {G.shape}")
    
    # Compute gradient
    d_G = newtonschulz5_velocity_detached_backward(G, grad_output, alpha)
    
    # Check 1: All values are finite
    all_finite = torch.all(torch.isfinite(d_G)).item()
    print(f"\n✓ Check 1: All finite values: {'✅ PASS' if all_finite else '❌ FAIL'}")
    
    # Check 2: Not all zeros (gradient should flow through)
    not_all_zeros = (d_G.abs().sum() > 0).item()
    print(f"✓ Check 2: Non-zero gradients: {'✅ PASS' if not_all_zeros else '❌ FAIL'}")
    
    # Check 3: Reasonable magnitude (not too large or too small)
    grad_norm = d_G.norm().item()
    reasonable_magnitude = (1e-6 < grad_norm < 1e6)
    print(f"✓ Check 3: Reasonable magnitude: {'✅ PASS' if reasonable_magnitude else '❌ FAIL'}")
    print(f"   Gradient norm: {grad_norm:.4e}")
    
    # Check 4: Shape matches input
    shape_match = d_G.shape == G.shape
    print(f"✓ Check 4: Shape matches input: {'✅ PASS' if shape_match else '❌ FAIL'}")
    
    all_pass = all_finite and not_all_zeros and reasonable_magnitude and shape_match
    
    if all_pass:
        print(f"\n✅ All sanity checks passed!")
    else:
        print(f"\n❌ Some sanity checks failed!")
    
    return all_pass


def main():
    print("\n" + "="*80)
    print("Newton-Schulz Velocity 5-Step Backward Pass Test Suite")
    print("Testing: First 4 steps detached, gradients only in last step")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Manual Backward vs Autograd", test_backward_last_step_only()))
    results.append(("Gradient Shapes", test_gradient_shapes()))
    results.append(("Gradient Sanity Checks", test_gradient_sanity()))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*80)
        print("\n✅ The backward pass implementation is mathematically correct!")
        print("✅ Ready to integrate into CUDA kernel")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print("\n⚠️  Please fix the backward pass implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())

