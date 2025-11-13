#!/usr/bin/env python3
"""
Compute reference values for minimal 2x2 CUDA test
"""

import torch
import sys
sys.path.insert(0, '/project/khanhnt/muontest/Momentum_correct/csrc/selective_scan')
from python_backward_detailed import newtonschulz5_backward_detailed

# Same values as CUDA test
u = torch.tensor([0.2308, 0.1337], dtype=torch.float32)
delta = torch.tensor([0.5535, 0.5809], dtype=torch.float32)
B = torch.tensor([[0.3331, -0.5069], [-0.2967, 0.2874]], dtype=torch.float32)
alpha = 1.0

# Compute G = alpha * delta * B * u (element-wise)
G = alpha * delta.unsqueeze(1) * B * u.unsqueeze(1)

print("=" * 80)
print("Python Reference for Minimal 2x2 Test")
print("=" * 80)
print(f"\nInputs:")
print(f"  u = {u}")
print(f"  delta = {delta}")
print(f"  B =\n{B}")
print(f"\nG = alpha * delta * B * u =\n{G}")

# Use identity-like grad_output for simplicity
grad_output = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
print(f"\ngrad_output (identity-like for testing) =\n{grad_output}")

# Compute grad_G using our backward function
print("\nComputing grad_G...")
grad_G = newtonschulz5_backward_detailed(G.detach(), grad_output, print_steps=True)

print("\n" + "=" * 80)
print("GRADIENT ACCUMULATION")
print("=" * 80)

# Now accumulate gradients
grad_u = torch.zeros_like(u)
grad_delta = torch.zeros_like(delta)
grad_B = torch.zeros_like(B)

for d in range(2):
    for n in range(2):
        grad_u[d] += alpha * delta[d] * B[d, n] * grad_G[d, n]
        grad_delta[d] += alpha * B[d, n] * u[d] * grad_G[d, n]
        grad_B[d, n] = alpha * delta[d] * u[d] * grad_G[d, n]

print(f"\nFinal gradients:")
print(f"  grad_u = {grad_u}")
print(f"  grad_delta = {grad_delta}")
print(f"  grad_B =\n{grad_B}")

print("\n" + "=" * 80)
print("EXPECTED VALUES FOR CUDA TEST")
print("=" * 80)
print(f"  grad_u = [{grad_u[0]:.6f}, {grad_u[1]:.6f}]")
print(f"  grad_delta = [{grad_delta[0]:.6f}, {grad_delta[1]:.6f}]")
print(f"  grad_B = [[{grad_B[0,0]:.6f}, {grad_B[0,1]:.6f}], [{grad_B[1,0]:.6f}, {grad_B[1,1]:.6f}]]")
print("=" * 80)

