#!/usr/bin/env python3
"""
Trace the backward pass step-by-step to find where dX_4 comes from
"""

import torch
import numpy as np

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
print("Tracing Backward Pass for (batch=0, time=0)")
print("=" * 80)

alpha = 1.0
b, t = 0, 0
eps = 1e-8

G_bt = alpha * delta[b, :, t].unsqueeze(1) * B * u[b, :, t].unsqueeze(1)
grad_V_bt = grad_output[b, :, t, :]

print(f"\nG shape: {G_bt.shape}")
print(f"grad_V (grad_output) shape: {grad_V_bt.shape}")
print(f"grad_V[0,:4] = {grad_V_bt[0,:4]}")

# Forward pass to compute X_4
def bf16_round(x):
    return x.bfloat16().float()

a, b_coef, c = (3.4445, -4.7750, 2.0315)

G_bf16 = bf16_round(G_bt)
norm_fp32 = (G_bf16 * G_bf16).sum().sqrt().item() + eps
X_0 = bf16_round(G_bf16 / norm_fp32)

X = X_0.detach()
for iteration in range(4):
    A = bf16_round(X @ X.T)
    B_mat = bf16_round(b_coef * A + c * bf16_round(A @ A))
    X = bf16_round(a * X + B_mat @ X)

X_4 = X

print(f"\nX_4[0,:4] = {X_4[0,:4]}")

# Now we need to do the 5th iteration WITH GRADIENTS
# X_5 = a * X_4 + B_4 @ X_4
# where B_4 = b * A_4 + c * A_4²
# and A_4 = X_4 @ X_4.T

# Enable gradients for the 5th iteration
X_4_grad = X_4.clone().requires_grad_(True)

A_4 = X_4_grad @ X_4_grad.T
A_4_sq = A_4 @ A_4
B_4 = b_coef * A_4 + c * A_4_sq
X_5 = a * X_4_grad + B_4 @ X_4_grad

print(f"\nX_5[0,:4] = {X_5[0,:4]}")

# Now backward pass from X_5
X_5.backward(grad_V_bt)

dX_4 = X_4_grad.grad

print(f"\ndX_4 (gradient wrt X_4):")
print(f"dX_4[0,:4] = {dX_4[0,:4]}")

print(f"\nCUDA dX_4 values:")
print(f"dX_4[0,:4] = [0.146780, 0.240263, 0.048034, 0.265362]")

print(f"\nDo they match?")
for n in range(4):
    cuda_val = [0.146780, 0.240263, 0.048034, 0.265362][n]
    py_val = dX_4[0,n].item()
    diff = abs(cuda_val - py_val)
    print(f"  dX_4[0,{n}]: Python={py_val:.6f}, CUDA={cuda_val:.6f}, diff={diff:.6f}")

print("=" * 80)

