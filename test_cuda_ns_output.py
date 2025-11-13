#!/usr/bin/env python3
"""
Test to verify CUDA NS backward is actually writing to output buffers
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda

# Minimal test
batch, dim, seqlen, dstate = 1, 2, 4, 2
beta, alpha = 0.9, 1.0
device = 'cuda'
dtype = torch.float32

torch.manual_seed(42)
u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * 0.1
delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * 0.1
A = -torch.rand(dim, dstate, dtype=dtype, device=device) * 0.1
B = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
C = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
D = torch.randn(dim, dtype=dtype, device=device) * 0.1
dout = torch.ones(batch, dim, seqlen, dtype=dtype, device=device)

# Forward
fwd = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, None, False, beta, alpha)
out, x, X_4 = fwd[0], fwd[1], fwd[2] if len(fwd) > 2 else None

print("="*80)
print("CUDA Backward Analysis")
print("="*80)

# Backward
bwd = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, x, None, None,
    False, False, beta, alpha, X_4
)

du, ddelta, dA, dB, dC = bwd[0], bwd[1], bwd[2], bwd[3], bwd[4]

print(f"\nFinal Gradients:")
print(f"du:\n{du}")
print(f"ddelta:\n{ddelta}")
print(f"dB:\n{dB}")

# Check if du contains D*dout
du_d_dout = D.unsqueeze(0).unsqueeze(-1) * dout
print(f"\nD * dout (what should be in du from direct path):")
for t in range(seqlen):
    print(f"  Timestep {t}: {du_d_dout[:, :, t][0].cpu().numpy()}")

print(f"\nActual du values:")
for t in range(seqlen):
    print(f"  Timestep {t}: {du[0, :, t].cpu().numpy()}")

print(f"\nDifference (du - D*dout):")
for t in range(seqlen):
    diff = du[0, :, t] - du_d_dout[0, :, t]
    print(f"  Timestep {t}: {diff.cpu().numpy()} (this should be NS backward contribution)")




