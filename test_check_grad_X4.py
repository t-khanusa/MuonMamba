#!/usr/bin/env python3
"""
Test script to check grad_X_4_buffer values before NS backward
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda

# Minimal test case
batch, dim, seqlen, dstate = 1, 2, 4, 2
beta, alpha = 0.9, 1.0
device = 'cuda'
dtype = torch.float32

torch.manual_seed(42)
u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * 0.5
delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * 0.1 + 0.1
A = -torch.rand(dim, dstate, dtype=dtype, device=device) * 0.1
B = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
C = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
D = torch.randn(dim, dtype=dtype, device=device) * 0.1
dout = torch.ones(batch, dim, seqlen, dtype=dtype, device=device)

print("="*80)
print("Checking grad_X_4_buffer after main backward kernel")
print("="*80)

# Forward pass
fwd_result = selective_scan_cuda.fwd(
    u, delta, A, B, C, D, None, None, False, beta, alpha
)
X_4_buffer = fwd_result[2] if len(fwd_result) > 2 else None

# We need to hook into the backward pass to inspect grad_X_4_buffer
# Since we can't do that directly, let's manually check by creating a wrapper
# Actually, let's check if we can access grad_X_4_buffer after backward

# Create a custom backward that saves grad_X_4_buffer
class SelectiveScanBackwardHook:
    def __init__(self):
        self.grad_X_4_buffer = None
    
    def __call__(self, *args, **kwargs):
        # This won't work - we need to modify the C++ code
        pass

# Instead, let's modify the C++ code to save grad_X_4_buffer to a file or return it
# For now, let's check the logic by examining what should happen

print("Expected behavior:")
print("1. Main backward kernel accumulates dv into grad_X_4_buffer for all timesteps")
print("2. NS backward reads grad_X_4_buffer and computes gradients")
print("3. NS backward writes gradients to du_ns_temp, ddelta_ns_temp, dB_ns_temp")
print()

print("If timesteps 1-3 have zero NS contribution, possible causes:")
print("1. grad_X_4_buffer is zero for timesteps 1-3 (dv not accumulated)")
print("2. NS backward reads zeros for timesteps 1-3")
print("3. NS backward computes gradients but doesn't write them correctly")
print()

print("Let's check by running backward and comparing du with D*dout:")

# Run backward
bwd_result = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, fwd_result[1], None, None,
    False, False, beta, alpha, X_4_buffer
)
du_cuda = bwd_result[0]

# Compute D*dout
D_dout = D.unsqueeze(0).unsqueeze(-1) * dout.unsqueeze(-1)
D_dout = D_dout.squeeze(-1)

# NS contribution
ns_contrib = du_cuda - D_dout

print("\nNS contribution per timestep:")
for t in range(seqlen):
    print(f"Timestep {t}: {ns_contrib[0, :, t].cpu().numpy()}")
    if ns_contrib[0, :, t].abs().max() < 1e-6:
        print(f"  ⚠️  ZERO NS contribution for timestep {t}!")




