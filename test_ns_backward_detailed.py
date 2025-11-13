#!/usr/bin/env python3
"""Test NS backward step-by-step"""

import torch
from test_comprehensive_ns_backward_accurate import pytorch_ns_backward_ref_accurate

# Small test
torch.manual_seed(42)
dim, dstate = 2, 2
alpha = 1.0

# Create simple inputs
delta_val = torch.tensor([0.1, 0.2], dtype=torch.float32, device='cuda')
B = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32, device='cuda')
u_val = torch.tensor([0.5, 0.6], dtype=torch.float32, device='cuda')

# Compute G_input = alpha * delta * B * u
G_input = alpha * (delta_val.unsqueeze(1) * B * u_val.unsqueeze(1))
print(f"G_input:\n{G_input}")

# Create a simple grad_output
grad_output = torch.ones(dim, dstate, dtype=torch.float32, device='cuda')
print(f"\ngrad_output:\n{grad_output}")

# Test NS backward
grad_u, grad_delta, grad_B = pytorch_ns_backward_ref_accurate(
    grad_output, G_input, alpha, delta_val, B, u_val
)

print(f"\nResults:")
print(f"  grad_u: {grad_u}")
print(f"  grad_delta: {grad_delta}")
print(f"  grad_B:\n{grad_B}")

# Manual check: G_input = alpha * delta * B * u
# So G_input[d, n] = alpha * delta[d] * B[d, n] * u[d]
# 
# ∂G/∂u[d] = sum_n (alpha * delta[d] * B[d, n] * d_G[d, n])
#          = alpha * delta[d] * sum_n (B[d, n] * d_G[d, n])
#
# ∂G/∂delta[d] = sum_n (alpha * B[d, n] * u[d] * d_G[d, n])
#               = alpha * u[d] * sum_n (B[d, n] * d_G[d, n])
#
# But d_G comes from NS backward, which is complex...

print(f"\nManual check (if NS backward worked correctly):")
# If NS backward produced d_G (gradient w.r.t. G_input):
# (This is what we should get after NS backward)
print("  Need to compute d_G from NS backward first...")





