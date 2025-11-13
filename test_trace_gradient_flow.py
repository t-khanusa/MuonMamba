#!/usr/bin/env python3
"""Trace gradient flow to find exact bugs"""

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

print("Forward...")
fwd = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, None, False, beta, alpha)
out, x, X_4 = fwd[0], fwd[1], fwd[2] if len(fwd) > 2 else None

print(f"out mean: {out.mean().item():.6f}")
if X_4 is not None:
    print(f"X_4 mean: {X_4.mean().item():.6f}, shape: {X_4.shape}")

print("\nBackward...")
bwd = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, x, None, None,
    False, False, beta, alpha, X_4
)

du, ddelta, dA, dB, dC = bwd[0], bwd[1], bwd[2], bwd[3], bwd[4]

print(f"\nCUDA Gradients:")
print(f"  du: {du.flatten()}")
print(f"  ddelta: {ddelta.flatten()}")
print(f"  dB: {dB.flatten()}")
print(f"  dC: {dC.flatten()}")

# Now trace through reference manually for first timestep
print("\n" + "="*80)
print("Manual trace for timestep 0:")
print("="*80)

# Forward states
h_states = []
v_states = []
h = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
v = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)

for t in range(seqlen):
    b_t = alpha * (delta[:, :, t].unsqueeze(-1) * B * u[:, :, t].unsqueeze(-1))
    # Apply NS (simplified - just normalize for now)
    from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref
    b_t_ortho = torch.zeros_like(b_t)
    for b in range(batch):
        b_t_ortho[b] = newtonschulz5_ref(b_t[b], steps=5)
    v = beta * v + b_t_ortho
    delta_A = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
    h = delta_A * h + v
    h_states.append(h.clone())
    v_states.append(v.clone())

# Backward - trace last timestep (t=3)
t = seqlen - 1
print(f"\nTimestep {t} (last):")
h_t = h_states[t]
v_t = v_states[t]
print(f"  h_t mean: {h_t.mean().item():.6f}")
print(f"  v_t mean: {v_t.mean().item():.6f}")

dh_t_from_out = dout[:, :, t].unsqueeze(-1) * C.unsqueeze(0)
print(f"  dh_t_from_out mean: {dh_t_from_out.mean().item():.6f}")

dh = dh_t_from_out
dv_t = dh  # Last timestep: no future gradient
print(f"  dv_t (last timestep) mean: {dv_t.mean().item():.6f}")

db_t_ortho = dv_t
print(f"  db_t_ortho mean: {db_t_ortho.mean().item():.6f}")

# NS backward for last timestep
b_t_input = alpha * (delta[:, :, t].unsqueeze(-1) * B * u[:, :, t].unsqueeze(-1))
from test_comprehensive_ns_backward_accurate import pytorch_ns_backward_ref_accurate

# Check what db_t_ortho looks like
print(f"  db_t_ortho[0] shape: {db_t_ortho[0].shape}, values:\n{db_t_ortho[0]}")

# Also check b_t_input
print(f"  b_t_input[0] shape: {b_t_input[0].shape}, mean: {b_t_input[0].mean().item():.6f}")

grad_u_t, grad_delta_t, grad_B_t = pytorch_ns_backward_ref_accurate(
    db_t_ortho[0], b_t_input[0], alpha, delta[0, :, t], B, u[0, :, t]
)

print(f"\n  NS backward outputs:")
print(f"    grad_u_t: {grad_u_t}")
print(f"    grad_delta_t: {grad_delta_t}")
print(f"    grad_B_t:\n{grad_B_t}")

# Compare with CUDA
print(f"\n  CUDA for timestep {t}:")
print(f"    du[:, :, {t}]: {du[:, :, t]}")
print(f"    ddelta[:, :, {t}]: {ddelta[:, :, t]}")

