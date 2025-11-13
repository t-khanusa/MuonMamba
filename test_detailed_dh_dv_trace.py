#!/usr/bin/env python3
"""
Compare dh and dv values in detail to find exact divergence point
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
from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref
h = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
v = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
h_states = []
v_states = []

for t in range(seqlen):
    b_t = alpha * (delta[:, :, t].unsqueeze(-1) * B * u[:, :, t].unsqueeze(-1))
    b_t_ortho = torch.zeros_like(b_t)
    for b in range(batch):
        b_t_ortho[b] = newtonschulz5_ref(b_t[b], steps=5)
    v = beta * v + b_t_ortho
    v_states.append(v.clone())
    delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
    h = delta_A_t * h + v
    h_states.append(h.clone())

# Backward - trace dh and dv carefully
dh = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
dv = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)

print("="*80)
print("Backward Pass - Detailed dh/dv Tracing")
print("="*80)

for t in range(seqlen - 1, -1, -1):
    print(f"\n{'='*60}")
    print(f"Timestep {t} (backward iteration)")
    print(f"{'='*60}")
    
    h_t = h_states[t]
    v_t = v_states[t]
    
    # Local gradient from output
    dh_t_from_out = dout[:, :, t].unsqueeze(-1) * C.unsqueeze(0)
    print(f"\n1. Local gradient (dh_t_from_out):")
    print(f"   Mean: {dh_t_from_out.mean().item():.6f}, Max: {dh_t_from_out.max().item():.6f}")
    print(f"   Values[0,0,:]: {dh_t_from_out[0,0,:].cpu().numpy()}")
    
    # dh before update
    print(f"\n2. dh BEFORE update (from future timesteps):")
    print(f"   Mean: {dh.mean().item():.6f}, Max: {dh.max().item():.6f}")
    print(f"   Values[0,0,:]: {dh[0,0,:].cpu().numpy()}")
    
    # Update dh
    if t < seqlen - 1:
        delta_A_next = torch.exp(delta[:, :, t+1].unsqueeze(-1) * A.unsqueeze(0))
        print(f"\n3. Propagating from timestep {t+1}:")
        print(f"   exp(delta[{t+1}]*A) mean: {delta_A_next.mean().item():.6f}")
        print(f"   delta_A_next * dh mean: {(delta_A_next * dh).mean().item():.6f}")
        dh = dh_t_from_out + delta_A_next * dh
    else:
        print(f"\n3. Last timestep - no future propagation")
        dh = dh_t_from_out
    
    print(f"\n4. dh AFTER update:")
    print(f"   Mean: {dh.mean().item():.6f}, Max: {dh.max().item():.6f}")
    print(f"   Values[0,0,:]: {dh[0,0,:].cpu().numpy()}")
    
    # dv before update
    print(f"\n5. dv BEFORE update (from future timesteps):")
    print(f"   Mean: {dv.mean().item():.6f}, Max: {dv.max().item():.6f}")
    print(f"   Values[0,0,:]: {dv[0,0,:].cpu().numpy()}")
    
    # Update dv
    dv_t = dh + beta * dv
    print(f"\n6. Computing dv_t = dh + beta * dv:")
    print(f"   beta = {beta}")
    print(f"   beta * dv mean: {(beta * dv).mean().item():.6f}")
    print(f"   dv_t mean: {dv_t.mean().item():.6f}, Max: {dv_t.max().item():.6f}")
    print(f"   Values[0,0,:]: {dv_t[0,0,:].cpu().numpy()}")
    
    dv = dv_t
    
    # db_t_ortho
    db_t_ortho = dv_t
    print(f"\n7. db_t_ortho = dv_t:")
    print(f"   Mean: {db_t_ortho.mean().item():.6f}, Max: {db_t_ortho.max().item():.6f}")
    print(f"   Values[0]:\n{db_t_ortho[0].cpu().numpy()}")
    
    # Check h_t_minus_v_t
    h_t_minus_v_t = h_t - v_t
    print(f"\n8. h_t_minus_v_t:")
    print(f"   Mean: {h_t_minus_v_t.mean().item():.6f}")
    print(f"   Values[0]:\n{h_t_minus_v_t[0].cpu().numpy()}")

print("\n" + "="*80)
print("Summary")
print("="*80)
print("dh and dv values traced above. Check if they match expected CUDA behavior.")





