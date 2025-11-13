#!/usr/bin/env python3
"""
Test to verify the main backward kernel SHOULD accumulate dv into grad_X_4_buffer for all timesteps
This will help identify if there's a bug in CUDA's accumulation
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda
from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref

# Minimal test - verify what SHOULD be accumulated
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
print("VERIFYING: What SHOULD be in grad_X_4_buffer")
print("="*80)

# Forward
h = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
v = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
h_states = []

for t in range(seqlen):
    b_t = alpha * (delta[:, :, t].unsqueeze(-1) * B * u[:, :, t].unsqueeze(-1))
    b_t_ortho = torch.zeros_like(b_t)
    for b in range(batch):
        b_t_ortho[b] = newtonschulz5_ref(b_t[b], steps=5)
    v = beta * v + b_t_ortho
    delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
    h = delta_A_t * h + v
    h_states.append(h.clone())

# Backward - compute dv values that SHOULD be accumulated
dh = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
dv = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)

expected_grad_X4 = torch.zeros((batch, dim, seqlen, dstate), dtype=torch.float32, device=device)

for t in range(seqlen - 1, -1, -1):
    dh_t_from_out = dout[:, :, t].unsqueeze(-1) * C.unsqueeze(0)
    
    if t < seqlen - 1:
        delta_A_next = torch.exp(delta[:, :, t+1].unsqueeze(-1) * A.unsqueeze(0))
        dh = dh_t_from_out + delta_A_next * dh
    else:
        dh = dh_t_from_out
    
    dv_t = dh + beta * dv
    dv = dv_t
    
    # This SHOULD be accumulated into grad_X_4_buffer[batch, dim, timestep, dstate]
    expected_grad_X4[:, :, t, :] = dv_t
    
    print(f"\nTimestep {t} (backward):")
    print(f"  dv_t[0,0,:]: {dv_t[0,0,:].cpu().numpy()}")
    print(f"  dv_t[0,1,:]: {dv_t[0,1,:].cpu().numpy()}")

print("\n" + "="*80)
print("Expected grad_X_4_buffer summary:")
print("="*80)
print(f"Should have non-zero values for ALL timesteps:")
for t in range(seqlen):
    non_zero = (expected_grad_X4[:, :, t, :] != 0).sum().item()
    print(f"  Timestep {t}: {non_zero}/{expected_grad_X4[:, :, t, :].numel()} non-zero, mean={expected_grad_X4[:, :, t, :].mean().item():.6f}")

print("\nIf CUDA's NS backward only produces gradients for timestep 0,")
print("then either:")
print("1. grad_X_4_buffer is not accumulated correctly in main backward kernel")
print("2. NS backward kernel is not reading grad_X_4_buffer correctly")
print("3. NS backward kernel has a bug that skips timesteps 1-3")




