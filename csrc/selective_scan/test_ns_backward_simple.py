#!/usr/bin/env python3
"""
Simple test: Compare manual NS backward with PyTorch autograd
"""

import torch

def newtonschulz5(G, steps=5, eps=1e-7):
    """Official implementation (non-inplace for autograd)"""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)  # Non-inplace for autograd compatibility
    if G.size(0) > G.size(1):
        X = X.T
    for step in range(steps):
        A = X @ X.T
        B_mat = b * A + c * A @ A
        X = a * X + B_mat @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

def manual_backward_last_iter_only(grad_output, G, eps=1e-7):
    """
    Backward through only the last iteration (iterations 0-3 detached)
    """
    a, b_coef, c = 3.4445, -4.7750, 2.0315
    
    # Recompute X_0 → X_4 (detached)
    with torch.no_grad():
        X = G.bfloat16()
        norm = X.float().norm()
        X = X.float()
        X = X / (norm + eps)  # Non-inplace
        
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
    
    # Forward 5th iteration with gradients
    X_4_grad = X_4.clone().requires_grad_(True)
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
    
    # Backward
    X_5.backward(grad_output)
    
    dX_4 = X_4_grad.grad.clone()
    
    # Gradient through transpose (if needed), normalization, BF16
    with torch.no_grad():
        if transposed:
            dX_4 = dX_4.T
            X_4_for_grad = X_4.T
        else:
            X_4_for_grad = X_4
        
        # Gradient through normalization
        dnorm_contrib = (dX_4 * X_4_for_grad).sum()
        d_G = (dX_4 - dnorm_contrib * X_4_for_grad) / norm
    
    return d_G

# Test
torch.manual_seed(42)
D, N = 8, 16
G = torch.randn(D, N, requires_grad=True) * 0.01

print("=" * 70)
print("Newton-Schulz Backward Pass Test")
print("=" * 70)
print(f"\nInput G: shape={G.shape}, norm={G.detach().norm():.6f}")

# Forward pass
X_out = newtonschulz5(G.detach())
print(f"Output X: shape={X_out.shape}, norm={X_out.norm():.6f}")

# Random gradient
grad_output = torch.randn_like(X_out)
print(f"grad_output: norm={grad_output.norm():.6f}")

# Manual backward
d_G_manual = manual_backward_last_iter_only(grad_output, G.detach())
print(f"\nManual grad: mean={d_G_manual.mean():.6f}, std={d_G_manual.std():.6f}, norm={d_G_manual.norm():.6f}")

# PyTorch autograd backward (ALL 5 iterations - for comparison only)
# NOTE: This will be different from manual because manual only backprops through last iter
G_auto_full = G.detach().requires_grad_(True)
X_auto_full = newtonschulz5(G_auto_full)
X_auto_full.backward(grad_output)
d_G_auto_full = G_auto_full.grad
print(f"Auto grad (full 5 iters): mean={d_G_auto_full.mean():.6f}, std={d_G_auto_full.std():.6f}, norm={d_G_auto_full.norm():.6f}")

# PyTorch autograd backward (ONLY last iteration - should match manual)
with torch.no_grad():
    G_bf16 = G.detach().bfloat16()
    norm = G_bf16.float().norm()
    X_0 = G_bf16.float() / (norm + 1e-7)
    
    transposed = False
    if G.size(0) > G.size(1):
        X_0 = X_0.T
        transposed = True
    
    # Run 4 iterations detached
    X = X_0
    a, b_coef, c = 3.4445, -4.7750, 2.0315
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

# Now run 5th iteration with gradients
X_4_for_grad = X_4.clone().requires_grad_(True)
A_4 = X_4_for_grad @ X_4_for_grad.T
A_4 = A_4.bfloat16().float()
A_4_sq = A_4 @ A_4
A_4_sq = A_4_sq.bfloat16().float()
B_4 = b_coef * A_4 + c * A_4_sq
B_4 = B_4.bfloat16().float()
X_5 = a * X_4_for_grad + B_4 @ X_4_for_grad
X_5 = X_5.bfloat16().float()

if transposed:
    X_5 = X_5.T

# Backward through 5th iteration only
X_5.backward(grad_output)
dX_4_auto = X_4_for_grad.grad

# Now backprop through normalization manually
with torch.no_grad():
    if transposed:
        dX_4_auto = dX_4_auto.T
        X_4_for_norm = X_4.T
    else:
        X_4_for_norm = X_4
    
    dnorm_contrib = (dX_4_auto * X_4_for_norm).sum()
    d_G_auto = (dX_4_auto - dnorm_contrib * X_4_for_norm) / norm

print(f"Auto grad (last iter only): mean={d_G_auto.mean():.6f}, std={d_G_auto.std():.6f}, norm={d_G_auto.norm():.6f}")

# Compare
diff = (d_G_manual - d_G_auto).abs()
rel_error = diff.max() / (d_G_auto.abs().max() + 1e-8)
print(f"\nDifference: max_abs={diff.max():.6f}, max_rel={rel_error:.6f}")
print(f"Match: {torch.allclose(d_G_manual, d_G_auto, rtol=0.01, atol=1e-6)}")

# Print some sample values
print(f"\nSample gradients (first 3x3):")
print("Manual:", d_G_manual[:3, :3])
print("Auto:  ", d_G_auto[:3, :3])

