#!/usr/bin/env python3
"""
Debug the first element (batch=0, time=0) to find the exact bug
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/project/khanhnt/muontest/Momentum_correct/csrc/selective_scan')
from python_backward_detailed import newtonschulz5_backward_detailed

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
print("Debugging First Element (batch=0, time=0)")
print("=" * 80)

# Compute G for first (batch=0, time=0)
alpha = 1.0
b, t = 0, 0

G_bt = alpha * delta[b, :, t].unsqueeze(1) * B * u[b, :, t].unsqueeze(1)
grad_V_bt = grad_output[b, :, t, :]

print(f"\nG (batch={b}, time={t}) shape: {G_bt.shape}")
print(f"G[0,0] = {G_bt[0,0]:.6f}")
print(f"\ngrad_output shape: {grad_V_bt.shape}")
print(f"grad_output[0,0] = {grad_V_bt[0,0]:.6f}")

# Compute grad_G using detached backward
print("\nComputing grad_G...")
grad_G_bt = newtonschulz5_backward_detailed(G_bt.detach(), grad_V_bt, print_steps=False)

print(f"\ngrad_G shape: {grad_G_bt.shape}")
print(f"grad_G[0,0] = {grad_G_bt[0,0]:.6f}")

# Now accumulate gradients for u, delta, B
grad_u_elem = torch.zeros_like(u[b, :, t])
grad_delta_elem = torch.zeros_like(delta[b, :, t])
grad_B_elem = torch.zeros_like(B)

for d in range(dim):
    for n in range(dstate):
        grad_u_elem[d] += alpha * delta[b, d, t] * B[d, n] * grad_G_bt[d, n]
        grad_delta_elem[d] += alpha * B[d, n] * u[b, d, t] * grad_G_bt[d, n]
        grad_B_elem[d, n] = alpha * delta[b, d, t] * u[b, d, t] * grad_G_bt[d, n]

print("\n" + "=" * 80)
print("EXPECTED GRADIENTS for (batch=0, time=0)")
print("=" * 80)
print(f"grad_u[0] = {grad_u_elem[0]:.6f}  (CUDA computed: 0.016855)")
print(f"Ratio: {0.016855 / grad_u_elem[0]:.4f}x")

print(f"\nAll grad_u values:")
for d in range(dim):
    print(f"  grad_u[{d}] = {grad_u_elem[d]:.6f}")

print(f"\nDetailed computation for grad_u[0]:")
print(f"  grad_u[0] = sum_n alpha * delta[0] * B[0,n] * grad_G[0,n]")
print(f"  alpha={alpha}, delta[0]={delta[b,0,t]:.6f}, u[0]={u[b,0,t]:.6f}")
total = 0
for n in range(dstate):
    contrib = alpha * delta[b, 0, t] * B[0, n] * grad_G_bt[0, n]
    total += contrib
    if n < 5:  # Print first 5
        print(f"    n={n}: B[0,{n}]={B[0,n]:.6f}, grad_G[0,{n}]={grad_G_bt[0,n]:.6f}, contrib={contrib:.6f}")
print(f"  Total: {total:.6f}")

print("\n" + "=" * 80)

