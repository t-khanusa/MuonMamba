#!/usr/bin/env python3
"""
Debug: Check if grad_X_4_buffer would have values for all timesteps
This will help identify if the CUDA bug is in accumulation or NS backward
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda
from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref

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

# Backward - compute what grad_X_4_buffer SHOULD contain
print("="*80)
print("Computing Expected grad_X_4_buffer Values")
print("="*80)

dh = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
dv = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)

# What should be in grad_X_4_buffer (dv values)
expected_grad_X4 = torch.zeros((batch, dim, seqlen, dstate), dtype=torch.float32, device=device)

for t in range(seqlen - 1, -1, -1):
    # Local gradient
    dh_t_from_out = dout[:, :, t].unsqueeze(-1) * C.unsqueeze(0)
    
    # Reverse scan hidden states
    if t < seqlen - 1:
        delta_A_next = torch.exp(delta[:, :, t+1].unsqueeze(-1) * A.unsqueeze(0))
        dh = dh_t_from_out + delta_A_next * dh
    else:
        dh = dh_t_from_out
    
    # Reverse scan velocity
    dv_t = dh + beta * dv
    dv = dv_t
    
    # This should be accumulated into grad_X_4_buffer[batch, dim, timestep, dstate]
    expected_grad_X4[:, :, t, :] = dv_t
    
    print(f"\nTimestep {t} (backward order):")
    print(f"  dv_t mean: {dv_t.mean().item():.6f}, max: {dv_t.max().item():.6f}")
    print(f"  dv_t[0,0,:]: {dv_t[0,0,:].cpu().numpy()}")

print("\n" + "="*80)
print("Expected grad_X_4_buffer summary:")
print("="*80)
print(f"Shape: {expected_grad_X4.shape}")
print(f"Non-zero elements: {(expected_grad_X4 != 0).sum().item()}/{expected_grad_X4.numel()}")
for t in range(seqlen):
    print(f"  Timestep {t}: mean={expected_grad_X4[:, :, t, :].mean().item():.6f}, max={expected_grad_X4[:, :, t, :].max().item():.6f}")

# Now if CUDA's NS backward processes this correctly, it should produce gradients for all timesteps
# But we saw that CUDA only produces gradients for timestep 0
# This suggests either:
# 1. grad_X_4_buffer is not accumulated correctly in CUDA (bug in main backward kernel)
# 2. NS backward kernel is not processing all timesteps correctly (bug in NS backward kernel)




