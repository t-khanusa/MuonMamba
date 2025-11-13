#!/usr/bin/env python3
"""
Verify the final gradient (grad_G) matches between standalone and main script
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/project/khanhnt/muontest/Momentum_correct/csrc/selective_scan')

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

# Compute G for first (batch=0, time=0) - same as main script
G_bt = alpha * delta[b, :, t].unsqueeze(1) * B_data * u[b, :, t].unsqueeze(1)
grad_V_bt = grad_output[b, :, t, :]

print("Using newtonschulz5_velocity_detached_backward:")
from generate_ns_velocity_test_data import newtonschulz5_velocity_detached_backward

grad_G_bt = newtonschulz5_velocity_detached_backward(G_bt.detach(), grad_V_bt)

print(f"G_bt[0,:4] = {G_bt[0,:4]}")
print(f"grad_V_bt[0,:4] = {grad_V_bt[0,:4]}")
print(f"grad_G_bt[0,:4] = {grad_G_bt[0,:4]}")

print(f"\nFrom test data generation:")
print(f"grad_G_bt[0,:4] = tensor([-0.1055,  0.0163,  0.1681,  0.0138])")

print(f"\nDo they match?")
expected = torch.tensor([-0.1055,  0.0163,  0.1681,  0.0138])
print(f"Match: {torch.allclose(grad_G_bt[0,:4], expected, atol=1e-3)}")

