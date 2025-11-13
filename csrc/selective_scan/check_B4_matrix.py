#!/usr/bin/env python3
"""
Check B_4 matrix values from Python
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
eps = 1e-8

G_bt = alpha * delta[b, :, t].unsqueeze(1) * B_data * u[b, :, t].unsqueeze(1)

def bf16_round(x):
    return x.bfloat16().float()

a, b_coef, c = (3.4445, -4.7750, 2.0315)

# Forward: compute X_4
G_bf16 = bf16_round(G_bt)
norm_fp32 = (G_bf16 * G_bf16).sum().sqrt().item() + eps
X_0 = bf16_round(G_bf16 / norm_fp32)

X = X_0.detach()
for iteration in range(4):
    A = bf16_round(X @ X.T)
    B_mat = bf16_round(b_coef * A + c * bf16_round(A @ A))
    X = bf16_round(a * X + B_mat @ X)

X_4 = X

# Compute B_4
A_4 = bf16_round(X_4 @ X_4.T)
A_4_sq = bf16_round(A_4 @ A_4)
B_4 = bf16_round(b_coef * A_4 + c * A_4_sq)

print("Python B_4[0,:8]:")
print(B_4[0,:8])

print("\nCUDA B_4[0,:8] (from debug output):")
print("[-2.343750, -0.064453, 0.014038, -0.151367, -0.330078, 0.009155, 0.062988, -0.055664]")

print("\nDo they match?")
cuda_b4 = torch.tensor([-2.343750, -0.064453, 0.014038, -0.151367, -0.330078, 0.009155, 0.062988, -0.055664])
print(f"Match: {torch.allclose(B_4[0,:8], cuda_b4, atol=1e-5)}")

