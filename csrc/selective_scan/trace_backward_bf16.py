#!/usr/bin/env python3
"""
Trace backward pass with BF16 rounding to match CUDA
"""

import torch
import numpy as np

# Load test data
with open('/tmp/ns_velocity_test_data.bin', 'rb') as f:
    batch, dim, seqlen, dstate = 2, 8, 16, 16
    
    grad_output = torch.from_numpy(
        np.fromfile(f, dtype=np.float32, count=batch*dim*seqlen*dstate)
    ).reshape(batch, dim, seqlen, dstate)
    
    u = torch.from_numpy(
        np.fromfile(f, dtype=np.float32, count=batch*dim*seqlen)
    ).reshape(batch, dim, seqlen)
    
    delta = torch.from_numpy(
        np.fromfile(f, dtype=np.float32, count=batch*dim*seqlen)
    ).reshape(batch, dim, seqlen)
    
    B = torch.from_numpy(
        np.fromfile(f, dtype=np.float32, count=dim*dstate)
    ).reshape(dim, dstate)

print("=" * 80)
print("Backward Pass with BF16 rounding (matching CUDA)")
print("=" * 80)

alpha = 1.0
b, t = 0, 0
eps = 1e-8

G_bt = alpha * delta[b, :, t].unsqueeze(1) * B * u[b, :, t].unsqueeze(1)
grad_V_bt = grad_output[b, :, t, :]

def bf16_round(x):
    return x.bfloat16().float()

a, b_coef, c = (3.4445, -4.7750, 2.0315)

# Forward: compute X_4 with BF16
G_bf16 = bf16_round(G_bt)
norm_fp32 = (G_bf16 * G_bf16).sum().sqrt().item() + eps
X_0 = bf16_round(G_bf16 / norm_fp32)

X = X_0.detach()
for iteration in range(4):
    A = bf16_round(X @ X.T)
    B_mat = bf16_round(b_coef * A + c * bf16_round(A @ A))
    X = bf16_round(a * X + B_mat @ X)

X_4 = X
print(f"X_4[0,:4] = {X_4[0,:4]}")

# 5th iteration with BF16 rounding
X_4_grad = X_4.clone().requires_grad_(True)

A_4 = bf16_round(X_4_grad @ X_4_grad.T)
A_4_sq = bf16_round(A_4 @ A_4)
B_4 = bf16_round(b_coef * A_4 + c * A_4_sq)
X_5 = bf16_round(a * X_4_grad + B_4 @ X_4_grad)

print(f"B_4[0,:4] = {B_4[0,:4]}")
print(f"X_5[0,:4] = {X_5[0,:4]}")

# Backward
X_5.backward(grad_V_bt)

dX_4 = X_4_grad.grad

print(f"\ndX_4 (with BF16):")
print(f"dX_4[0,:4] = {dX_4[0,:4]}")

print(f"\nCUDA dX_4:")
print(f"dX_4[0,:4] = [-0.152872, 0.023401, 0.221965, 0.019959]")

print(f"\nDifferences:")
for i in range(4):
    cuda = [-0.152872, 0.023401, 0.221965, 0.019959][i]
    py = dX_4[0,i].item()
    print(f"  dX_4[0,{i}]: Python={py:.6f}, CUDA={cuda:.6f}, diff={abs(py-cuda):.6f}")

print("=" * 80)

