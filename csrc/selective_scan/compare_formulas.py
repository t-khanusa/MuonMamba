#!/usr/bin/env python3
"""
Compare normalization backward formulas
"""

import torch

# Test setup
torch.manual_seed(42)
x = torch.randn(4, 4, dtype=torch.float32)
grad_y = torch.randn(4, 4, dtype=torch.float32)

eps = 1e-8

# Convert to BF16 and back
x_bf16 = x.bfloat16().float()

# Method 1: Current CUDA implementation
norm = torch.sqrt((x_bf16 ** 2).sum() + eps)
y1 = x_bf16 / norm
y1.requires_grad_(True)

# Backward
y1.backward(grad_y)
grad_x1 = y1.grad

# Manual with current formula
with torch.no_grad():
    X_0 = x_bf16 / norm
    dot = (grad_y * X_0).sum()
    grad_x1_manual = (grad_y - X_0 * dot) / norm
    
print("Method 1 (current CUDA formula):")
print(f"  Formula: (grad_y - X_0 * <grad_y, X_0>) / norm")
print(f"  Match: {torch.allclose(grad_x1, grad_x1_manual, rtol=1e-4)}")
print(f"  grad_x[0,0] = {grad_x1_manual[0,0]:.6f}")

# Method 2: Pseudo code formula
with torch.no_grad():
    s_val = torch.sqrt((x_bf16 ** 2).sum() + 1e-12)
    norm2 = s_val  # pseudo code adds eps again but that seems wrong
    dot2 = (grad_y * x_bf16).sum()
    
    # Pseudo formula: grad_b = g4 / norm - u * (dot / (s_val * norm^2))
    grad_x2_manual = grad_y / norm2 - x_bf16 * (dot2 / (s_val * norm2 * norm2))
    
print("\nMethod 2 (pseudo code formula):")
print(f"  Formula: grad_y / norm - x * (dot / (s_val * norm^2))")
print(f"  Match with Method 1: {torch.allclose(grad_x1_manual, grad_x2_manual, rtol=1e-4)}")
print(f"  grad_x[0,0] = {grad_x2_manual[0,0]:.6f}")

# Method 3: Standard normalization gradient
with torch.no_grad():
    norm3 = torch.sqrt((x_bf16 ** 2).sum() + eps)
    dot3 = (grad_y * x_bf16).sum()
    # Standard: grad_x = (grad_y - x * <grad_y, x> / ||x||^2) / ||x||
    grad_x3_manual = (grad_y - x_bf16 * dot3 / (norm3 * norm3)) / norm3
    
print("\nMethod 3 (standard formula):")
print(f"  Formula: (grad_y - x * <grad_y, x> / norm^2) / norm")
print(f"  Match with Method 1: {torch.allclose(grad_x1_manual, grad_x3_manual, rtol=1e-6)}")
print(f"  grad_x[0,0] = {grad_x3_manual[0,0]:.6f}")

# Check if all three are equivalent
print("\n" + "=" * 60)
print("CONCLUSION:")
if torch.allclose(grad_x1_manual, grad_x2_manual, rtol=1e-6) and torch.allclose(grad_x1_manual, grad_x3_manual, rtol=1e-6):
    print("✅ All three formulas are mathematically equivalent!")
else:
    print("❌ Formulas produce different results")
    print(f"Max diff 1 vs 2: {(grad_x1_manual - grad_x2_manual).abs().max():.6e}")
    print(f"Max diff 1 vs 3: {(grad_x1_manual - grad_x3_manual).abs().max():.6e}")

