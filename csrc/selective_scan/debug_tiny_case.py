#!/usr/bin/env python3
"""
Minimal 2x2 test case to debug the backward pass
"""

import torch

print("=" * 80)
print("Minimal 2x2 Test Case for NS Backward")
print("=" * 80)

# Tiny test
torch.manual_seed(42)
G = torch.randn(2, 2, dtype=torch.float32)
grad_output = torch.randn(2, 2, dtype=torch.float32)

a, b, c = (3.4445, -4.7750, 2.0315)
eps = 1e-8

print(f"\nInput G:\n{G}")
print(f"\nGrad output:\n{grad_output}")

# Forward (4 detached + 1 with grad)
with torch.no_grad():
    X = G.bfloat16().float()
    X = X / (torch.sqrt((X**2).sum()) + eps)
    X = X.bfloat16().float()
    
    print(f"\nX_0 (normalized):\n{X}")
    
    for i in range(4):
        A = X @ X.T
        A = A.bfloat16().float()
        A2 = A @ A
        A2 = A2.bfloat16().float()
        B = b * A + c * A2
        B = B.bfloat16().float()
        X = a * X + B @ X
        X = X.bfloat16().float()
        print(f"\nX_{i+1}:\n{X}")
    
    X_4_detached = X.clone()

# 5th iteration with gradients
X_4 = X_4_detached.requires_grad_(True)
A_4 = X_4 @ X_4.T
A_4 = A_4.bfloat16().float()
A_4_sq = A_4 @ A_4
A_4_sq = A_4_sq.bfloat16().float()
B_4 = b * A_4 + c * A_4_sq
B_4 = B_4.bfloat16().float()
X_5 = a * X_4 + B_4 @ X_4

print(f"\nA_4:\n{A_4}")
print(f"\nB_4:\n{B_4}")
print(f"\nX_5:\n{X_5}")

# Backward
X_5.backward(grad_output)
dX_4 = X_4.grad

print(f"\ndX_4 (from autograd):\n{dX_4}")

# Manual backward through normalization
with torch.no_grad():
    X_0 = G.bfloat16().float()
    norm = torch.sqrt((X_0**2).sum()) + eps
    X_0 = X_0 / norm
    X_0 = X_0.bfloat16().float()
    
    dot_product = (dX_4 * X_0).sum()
    grad_G = (dX_4 - X_0 * dot_product) / norm
    
    print(f"\ngrad_G (manual):\n{grad_G}")
    print(f"\n  norm: {norm:.6f}")
    print(f"  dot: {dot_product:.6f}")

print("\n" + "=" * 80)
print("Now check if CUDA computes the same dX_4 and grad_G")
print("=" * 80)

