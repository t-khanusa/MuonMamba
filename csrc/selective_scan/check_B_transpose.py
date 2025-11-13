#!/usr/bin/env python3
"""
Manually compute B.T @ dX_5 to compare with CUDA
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
    
    B_data = torch.from_numpy(
        np.fromfile(f, dtype=np.float32, count=dim*dstate)
    ).reshape(dim, dstate)

alpha = 1.0
b, t = 0, 0

G_bt = alpha * delta[b, :, t].unsqueeze(1) * B_data * u[b, :, t].unsqueeze(1)
grad_V_bt = grad_output[b, :, t, :]

def bf16_round(x):
    return x.bfloat16().float()

a, b_coef, c = (3.4445, -4.7750, 2.0315)

# Forward: compute X_4 and B_4
G_bf16 = bf16_round(G_bt)
norm_fp32 = (G_bf16 * G_bf16).sum().sqrt().item() + 1e-8
X_0 = bf16_round(G_bf16 / norm_fp32)

X = X_0.detach()
for iteration in range(4):
    A = bf16_round(X @ X.T)
    B_mat = bf16_round(b_coef * A + c * bf16_round(A @ A))
    X = bf16_round(a * X + B_mat @ X)

X_4 = X

# Compute A_4 and B_4
A_4 = bf16_round(X_4 @ X_4.T)
A_4_sq = bf16_round(A_4 @ A_4)
B_4 = bf16_round(b_coef * A_4 + c * A_4_sq)

print("=" * 80)
print("Manual Computation of B.T @ dX_5")
print("=" * 80)

print(f"\nB_4[0,:4] = {B_4[0,:4]}")
print(f"grad_V[0,:4] = {grad_V_bt[0,:4]}")

# Manually compute (B_4.T @ grad_V_bt)[0,:]
result = B_4.T @ grad_V_bt
print(f"\n(B_4.T @ grad_V)[0,:4] = {result[0,:4]}")

print(f"\nCUDA B.T@dX_5 values:")
print(f"[0.288270, 0.036618, -0.438321, -0.096635]")

print(f"\nComparison:")
for i in range(4):
    cuda = [0.288270, 0.036618, -0.438321, -0.096635][i]
    py = result[0,i].item()
    print(f"  [0,{i}]: Python={py:.6f}, CUDA={cuda:.6f}, diff={abs(py-cuda):.6f}")

print("=" * 80)

