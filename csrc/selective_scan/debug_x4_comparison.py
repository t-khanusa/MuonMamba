#!/usr/bin/env python3
"""
Check what Python computes for X_4 and compare with CUDA
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/project/khanhnt/muontest/Momentum_correct/csrc/selective_scan')

# Load the test data
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
print("Python X_4 Computation")
print("=" * 80)

# Compute G for first (batch=0, time=0)
alpha = 1.0
b, t = 0, 0
eps = 1e-8

G_bt = alpha * delta[b, :, t].unsqueeze(1) * B * u[b, :, t].unsqueeze(1)

print(f"\nG (batch={b}, time={t}) shape: {G_bt.shape}")
print(f"G[0,:4] = {G_bt[0,:4]}")

# Forward pass: compute X_4 (4 detached NS iterations)
def bf16_round(x):
    return x.bfloat16().float()

a, b_coef, c = (3.4445, -4.7750, 2.0315)

# Step 1: Normalize
G_bf16 = bf16_round(G_bt)
norm_fp32 = (G_bf16 * G_bf16).sum().sqrt().item() + eps
X_0 = bf16_round(G_bf16 / norm_fp32)

print(f"norm = {norm_fp32}")
print(f"X_0[0,0] = {X_0[0,0]}")

# Step 2: 4 detached NS iterations
X = X_0.detach()
for iteration in range(4):
    A = bf16_round(X @ X.T)
    B_mat = bf16_round(b_coef * A + c * bf16_round(A @ A))
    X = bf16_round(a * X + B_mat @ X)
    if iteration == 3:
        print(f"After iteration {iteration+1}, X[0,0] = {X[0,0]}")

X_4 = X
norm = norm_fp32

print(f"\nX_4 (after 4 NS iterations):")
print(f"X_4[0,:4] = {X_4[0,:4]}")

print(f"\nCUDA X_4 values:")
print(f"X_4[0,:4] = [-0.357422, 0.029053, -0.064941, -0.056885]")

print(f"\nDo they match?")
for n in range(4):
    cuda_val = [-0.357422, 0.029053, -0.064941, -0.056885][n]
    py_val = X_4[0,n].item()
    diff = abs(cuda_val - py_val)
    print(f"  X_4[0,{n}]: Python={py_val:.6f}, CUDA={cuda_val:.6f}, diff={diff:.6f}")

print(f"\ngrad_output[0,:4] = {grad_output[b, 0, t, :4]}")
print("=" * 80)

