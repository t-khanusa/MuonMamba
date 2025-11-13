#!/usr/bin/env python3
"""
Debug Newton-Schulz formula - test different interpretations
"""

import torch

def ns_version1(G, steps=5, eps=1e-7):
    """Official version from Muon"""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def ns_version2_wrong_interpretation(G, steps=5, eps=1e-7):
    """Wrong: X_new = a*X + (b*A + c*A²) @ X"""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B_mat = b * A + c * (A @ A)
        X = a * X + B_mat @ X  # This is what we're doing
    if G.size(0) > G.size(1):
        X = X.T
    return X


# Test both
torch.manual_seed(42)
G = torch.randn(4, 3)

print("Testing Official Formula:")
print("="*60)
result1 = ns_version1(G, steps=5)
result1_float = result1.float()
gram1 = result1_float.T @ result1_float
identity = torch.eye(3)
error1 = torch.norm(gram1 - identity).item()
print(f"G^T @ G:\n{gram1}")
print(f"Error: {error1:.6f}")
print("✅ PASS" if error1 < 0.01 else "❌ FAIL")

print("\n" + "="*60)
print("\nLet's trace through step by step:")
print("="*60)

# Detailed trace
a, b, c = (3.4445, -4.7750, 2.0315)
X = G.bfloat16()
norm = X.norm()
print(f"Initial norm: {norm:.6f}")
X = X / (norm + 1e-7)
print(f"After normalization: ||X|| = {X.float().norm():.6f}")

# Transpose
if G.size(0) > G.size(1):
    X = X.T
    print(f"Transposed X shape: {X.shape}")

for step in range(5):
    A = X @ X.T
    print(f"\nStep {step}:")
    print(f"  ||A - I||_F = {torch.norm(A.float() - torch.eye(A.shape[0])).item():.6f}")
    
    # Official formula: B = b * A + c * A @ A
    B = b * A + c * A @ A
    
    # Update: X = a * X + B @ X
    X_new = a * X + B @ X
    
    print(f"  ||X||_F = {X.float().norm():.6f}")
    print(f"  ||X_new||_F = {X_new.float().norm():.6f}")
    
    X = X_new

# Transpose back
if G.size(0) > G.size(1):
    X = X.T

X_float = X.float()
gram = X_float.T @ X_float
error = torch.norm(gram - torch.eye(3)).item()
print(f"\nFinal Error: {error:.6f}")
print("✅ PASS" if error < 0.01 else "❌ FAIL")

print("\n" + "="*60)
print("Checking formula interpretation:")
print("="*60)
print("Official code: B = b * A + c * A @ A")
print("              X = a * X + B @ X")
print("")
print("This means: X_new = a*X + (b*A + c*A²) @ X")
print("          = a*X + b*A@X + c*A²@X")
print("")
print("This is a matrix polynomial applied to X!")





