#!/usr/bin/env python3
"""
Verify the test data matches what CUDA is computing
"""

import torch
import numpy as np

# Load the test data that CUDA is using
with open('/tmp/ns_velocity_test_data.bin', 'rb') as f:
    # Read in order: grad_output, u, delta, B, grad_u, grad_delta, grad_B
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

print("First elements from test data:")
print(f"u[0,0,0] = {u[0,0,0]:.6f}")
print(f"delta[0,0,0] = {delta[0,0,0]:.6f}")
print(f"B[0,0] = {B[0,0]:.6f}")
print(f"B[0,:] = {B[0,:]}")

print(f"\nCUDA debug showed:")
print(f"u=0.963458, delta=0.431452, B=-0.409131")

print(f"\nDo they match?")
print(f"u match: {abs(u[0,0,0] - 0.963458) < 1e-5}")
print(f"delta match: {abs(delta[0,0,0] - 0.431452) < 1e-5}")
print(f"B match: {abs(B[0,0] - (-0.409131)) < 1e-5}")

# Compute G for first (batch=0, time=0)
alpha = 1.0
G_00 = alpha * delta[0, :, 0].unsqueeze(1) * B * u[0, :, 0].unsqueeze(1)
print(f"\nG[0,0,:,:] (batch=0, time=0) =\n{G_00}")

# Load the Python reference gradients
with open('/tmp/ns_velocity_test_data.bin', 'rb') as f:
    # Skip to gradients
    f.seek(batch*dim*seqlen*dstate*4 + batch*dim*seqlen*4 + batch*dim*seqlen*4 + dim*dstate*4)
    
    grad_u_ref = torch.from_numpy(
        np.fromfile(f, dtype=np.float32, count=batch*dim*seqlen)
    ).reshape(batch, dim, seqlen)

print(f"\nPython reference grad_u[0,0,0] = {grad_u_ref[0,0,0]:.6f}")
print(f"CUDA computed grad_u[0,0,0] = 0.016855")
print(f"Ratio: {0.016855 / grad_u_ref[0,0,0]:.3f}x")

