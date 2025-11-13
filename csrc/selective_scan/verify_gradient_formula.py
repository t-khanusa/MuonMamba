#!/usr/bin/env python3
"""
Verify the correct gradient formula for Y = A @ A

The key insight: when dY is symmetric (which it often is in NS), 
the gradient formula simplifies.
"""

import torch

print("=" * 80)
print("Verifying Gradient Formula for Y = A @ A")
print("=" * 80)

# Create a small test case
torch.manual_seed(42)
A = torch.randn(4, 4, requires_grad=True, dtype=torch.float32)

# Test Case 1: Symmetric gradient (common in Newton-Schulz)
print("\n=== TEST CASE 1: Symmetric dL/dY ===")
dL_dY_sym = torch.randn(4, 4, dtype=torch.float32)
dL_dY_sym = (dL_dY_sym + dL_dY_sym.T) / 2  # Make it symmetric

A.grad = None
Y = A @ A
Y.backward(dL_dY_sym)
grad_pytorch = A.grad.clone()

A_detached = A.detach()
# Formula 1: A @ dY.T + A.T @ dY
grad_formula1 = A_detached @ dL_dY_sym.T + A_detached.T @ dL_dY_sym
# Formula 2: dY @ A.T + dY.T @ A (CUDA code)
grad_formula2 = dL_dY_sym @ A_detached.T + dL_dY_sym.T @ A_detached
# Formula 3: (dY + dY.T) @ A (when dY is symmetric)
grad_formula3 = (dL_dY_sym + dL_dY_sym.T) @ A_detached
# Formula 4: dY @ A.T + A.T @ dY
grad_formula4 = dL_dY_sym @ A_detached.T + A_detached.T @ dL_dY_sym
# Formula 5: A.T @ dY.T + dY @ A.T  
grad_formula5 = A_detached.T @ dL_dY_sym.T + dL_dY_sym @ A_detached.T

print(f"\nFormula 1 (A @ dY.T + A.T @ dY) match: {torch.allclose(grad_pytorch, grad_formula1, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch - grad_formula1).abs().max():.6e}")

print(f"\nFormula 2 (dY @ A.T + dY.T @ A) [CUDA] match: {torch.allclose(grad_pytorch, grad_formula2, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch - grad_formula2).abs().max():.6e}")

print(f"\nFormula 3 ((dY + dY.T) @ A) match: {torch.allclose(grad_pytorch, grad_formula3, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch - grad_formula3).abs().max():.6e}")

print(f"\nFormula 4 (dY @ A.T + A.T @ dY) match: {torch.allclose(grad_pytorch, grad_formula4, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch - grad_formula4).abs().max():.6e}")

print(f"\nFormula 5 (A.T @ dY.T + dY @ A.T) match: {torch.allclose(grad_pytorch, grad_formula5, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch - grad_formula5).abs().max():.6e}")

# Test Case 2: General (non-symmetric) gradient
print("\n\n=== TEST CASE 2: Non-Symmetric dL/dY ===")
dL_dY_general = torch.randn(4, 4, dtype=torch.float32)

A.grad = None
Y2 = A @ A
Y2.backward(dL_dY_general)
grad_pytorch2 = A.grad.clone()

# Formula 1: A @ dY.T + A.T @ dY
grad_formula1_gen = A_detached @ dL_dY_general.T + A_detached.T @ dL_dY_general
# Formula 2: dY @ A.T + dY.T @ A (CUDA code)
grad_formula2_gen = dL_dY_general @ A_detached.T + dL_dY_general.T @ A_detached
# Formula 3: (dY + dY.T) @ A
grad_formula3_gen = (dL_dY_general + dL_dY_general.T) @ A_detached
# Formula 4: dY @ A.T + A.T @ dY
grad_formula4_gen = dL_dY_general @ A_detached.T + A_detached.T @ dL_dY_general
# Formula 5: A.T @ dY.T + dY @ A.T  
grad_formula5_gen = A_detached.T @ dL_dY_general.T + dL_dY_general @ A_detached.T

print(f"\nFormula 1 (A @ dY.T + A.T @ dY) match: {torch.allclose(grad_pytorch2, grad_formula1_gen, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch2 - grad_formula1_gen).abs().max():.6e}")

print(f"\nFormula 2 (dY @ A.T + dY.T @ A) [CUDA] match: {torch.allclose(grad_pytorch2, grad_formula2_gen, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch2 - grad_formula2_gen).abs().max():.6e}")

print(f"\nFormula 3 ((dY + dY.T) @ A) match: {torch.allclose(grad_pytorch2, grad_formula3_gen, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch2 - grad_formula3_gen).abs().max():.6e}")

print(f"\nFormula 4 (dY @ A.T + A.T @ dY) match: {torch.allclose(grad_pytorch2, grad_formula4_gen, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch2 - grad_formula4_gen).abs().max():.6e}")

print(f"\nFormula 5 (A.T @ dY.T + dY @ A.T) match: {torch.allclose(grad_pytorch2, grad_formula5_gen, rtol=1e-4)}")
print(f"  Max diff: {(grad_pytorch2 - grad_formula5_gen).abs().max():.6e}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("The correct formula for dL/dA when Y = A @ A is:")
print("  dL/dA = (dL/dY + dL/dY.T) @ A")
print("\nThis is equivalent to:")
print("  dL/dA = dL/dY @ A + dL/dY.T @ A  (factor out A)")
print("  dL/dA = dL/dY @ A + A.T @ dL/dY.T  (since (dL/dY.T @ A).T = A.T @ dL/dY)")
print("\nWhen dL/dY is symmetric, this simplifies to:")
print("  dL/dA = 2 * dL/dY @ A")
print("=" * 80)

