#!/usr/bin/env python3
"""Minimal test to verify NS backward input/output"""

import torch
from test_comprehensive_ns_backward_accurate import pytorch_ns_backward_ref_accurate

# Test with very small input (like what we see in actual run)
torch.manual_seed(42)
dim, dstate = 2, 2
alpha = 1.0

# Small input values (similar to what we see in debug)
delta_val = torch.tensor([0.1, 0.2], dtype=torch.float32, device='cuda') * 0.1
B = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32, device='cuda') * 0.1
u_val = torch.tensor([0.5, 0.6], dtype=torch.float32, device='cuda') * 0.1

# Compute G_input = alpha * delta * B * u
G_input = alpha * (delta_val.unsqueeze(1) * B * u_val.unsqueeze(1))
print(f"G_input:\n{G_input}")
print(f"G_input mean: {G_input.mean().item():.6f}, max: {G_input.max().item():.6f}")

# Small grad_output (similar to db_t_ortho from debug)
grad_output = torch.tensor([[0.001, 0.002], [0.003, 0.004]], dtype=torch.float32, device='cuda')
print(f"\ngrad_output:\n{grad_output}")
print(f"grad_output mean: {grad_output.mean().item():.6f}")

# Test NS backward
grad_u, grad_delta, grad_B = pytorch_ns_backward_ref_accurate(
    grad_output, G_input, alpha, delta_val, B, u_val
)

print(f"\nNS Backward Results:")
print(f"  grad_u: {grad_u}")
print(f"  grad_delta: {grad_delta}")
print(f"  grad_B:\n{grad_B}")

# Check if results are reasonable
print(f"\nMagnitude check:")
print(f"  |grad_u|: {torch.abs(grad_u).mean().item():.6f}")
print(f"  |grad_delta|: {torch.abs(grad_delta).mean().item():.6f}")
print(f"  |grad_B|: {torch.abs(grad_B).mean().item():.6f}")

# Compare with expected: if G_input is small, gradients might be large due to normalization
print(f"\nNote: If G_input is very small, normalization can amplify gradients")





