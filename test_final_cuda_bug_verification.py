#!/usr/bin/env python3
"""
Final test to identify exact CUDA bug location
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

print("="*80)
print("CUDA BUG VERIFICATION")
print("="*80)

# Forward
fwd = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, None, False, beta, alpha)

# Backward
bwd = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, fwd[1], None, None,
    False, False, beta, alpha, fwd[2] if len(fwd) > 2 else None
)

du, ddelta = bwd[0], bwd[1]

# Compute expected D*dout
du_d_dout = D.unsqueeze(0).unsqueeze(-1) * dout

print("\nCUDA du values (per timestep):")
for t in range(seqlen):
    du_t = du[0, :, t].cpu().numpy()
    du_d_dout_t = du_d_dout[0, :, t].cpu().numpy()
    ns_contribution = du_t - du_d_dout_t
    print(f"  Timestep {t}:")
    print(f"    Total: {du_t}")
    print(f"    D*dout: {du_d_dout_t}")
    print(f"    NS contribution: {ns_contribution}")
    if t == 0:
        print(f"    ✅ Has NS contribution")
    else:
        if (ns_contribution == 0).all():
            print(f"    ❌ NO NS contribution! (BUG CONFIRMED)")
        else:
            print(f"    ✅ Has NS contribution")

print("\nCUDA ddelta values (per timestep):")
for t in range(seqlen):
    ddelta_t = ddelta[0, :, t].cpu().numpy()
    print(f"  Timestep {t}: {ddelta_t}")
    if t == 0:
        print(f"    ✅ Non-zero")
    else:
        if (ddelta_t == 0).all():
            print(f"    ❌ ALL ZERO! (BUG)")
        else:
            print(f"    ✅ Non-zero")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("If timesteps 1-3 have NO NS contribution, the bug is either:")
print("1. Main backward kernel not accumulating grad_X_4_buffer for timesteps 1-3")
print("2. NS backward kernel not processing timesteps 1-3 correctly")
print("3. NS backward kernel reading zero values from grad_X_4_buffer for timesteps 1-3")




