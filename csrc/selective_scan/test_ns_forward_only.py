#!/usr/bin/env python3
"""
Simple test to verify NS forward pass matches official implementation
"""

import torch

def newtonschulz5_official(G, steps=5, eps=1e-7):
    """Official PyTorch implementation"""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for step in range(steps):
        A = X @ X.T
        B_mat = b * A + c * A @ A
        X = a * X + B_mat @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

def newtonschulz5_test(G, steps=5, eps=1e-7):
    """Our implementation with debug"""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.bfloat16()
    print(f"After BF16: norm={X.float().norm():.6f}")
    X /= (X.norm() + eps)
    print(f"After norm: norm={X.norm():.6f}")
    
    if G.size(0) > G.size(1):
        X = X.T
        print("Transposed (tall matrix)")
    
    for step in range(steps):
        A = X @ X.T
        print(f"Step {step}: A.shape={A.shape}, A.trace()={A.diag().sum():.6f}")
        B_mat = b * A + c * A @ A
        X = a * X + B_mat @ X
        print(f"Step {step}: X.norm()={X.norm():.6f}")
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

# Test with small matrix
torch.manual_seed(42)
G = torch.randn(8, 16) * 0.01  # Small values to match test

print("=" * 60)
print("Testing Newton-Schulz Forward Pass")
print("=" * 60)
print(f"\nInput G: shape={G.shape}, norm={G.norm():.6f}")
print(f"mean={G.mean():.6f}, std={G.std():.6f}")

print("\n" + "-" * 60)
print("Official Implementation:")
print("-" * 60)
X_official = newtonschulz5_official(G)
print(f"\nFinal: shape={X_official.shape}, norm={X_official.norm():.6f}")

# Check orthogonality
A_final = X_official @ X_official.T
print(f"Gram matrix trace: {A_final.diag().sum():.6f} (expected {min(G.size(0), G.size(1))})")
print(f"Gram matrix off-diagonal max: {(A_final - torch.eye(A_final.size(0))).abs().max():.6f}")

print("\n" + "-" * 60)
print("Test Implementation (with debug):")
print("-" * 60)
X_test = newtonschulz5_test(G)
print(f"\nFinal: shape={X_test.shape}, norm={X_test.norm():.6f}")

# Check orthogonality
A_final_test = X_test @ X_test.T
print(f"Gram matrix trace: {A_final_test.diag().sum():.6f} (expected {min(G.size(0), G.size(1))})")

print("\n" + "=" * 60)
print(f"Match: {torch.allclose(X_official, X_test, atol=1e-5)}")

