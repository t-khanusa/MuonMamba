#!/usr/bin/env python3
"""
Step-by-step fix: Compare db_t_ortho values and fix reverse scan logic
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda
from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref
from test_comprehensive_ns_backward_accurate import pytorch_ns_backward_ref_accurate

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
print("STEP-BY-STEP DEBUGGING")
print("="*80)

# Forward pass
h = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
v = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
h_states = []
v_states = []
b_t_states = []

for t in range(seqlen):
    b_t = alpha * (delta[:, :, t].unsqueeze(-1) * B * u[:, :, t].unsqueeze(-1))
    b_t_original = b_t.clone()
    
    b_t_ortho = torch.zeros_like(b_t)
    for b in range(batch):
        b_t_ortho[b] = newtonschulz5_ref(b_t[b], steps=5)
    b_t = b_t_ortho
    
    b_t_states.append(b_t_original)
    v = beta * v + b_t
    v_states.append(v.clone())
    
    delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
    h = delta_A_t * h + v
    h_states.append(h.clone())

# Backward pass - manual step-by-step
dh = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
dv = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)

du = torch.zeros_like(u)
ddelta = torch.zeros_like(delta)
dB = torch.zeros_like(B)
dC = torch.zeros_like(C)

# Initialize du
if D is not None:
    for t in range(seqlen):
        du[:, :, t] = D.unsqueeze(0) * dout[:, :, t]

print("\n" + "="*80)
print("BACKWARD PASS - Step by Step Analysis")
print("="*80)

# Store db_t_ortho values for comparison
db_t_ortho_values = {}

for t in range(seqlen - 1, -1, -1):
    print(f"\n{'='*60}")
    print(f"Timestep {t} (reverse order)")
    print(f"{'='*60}")
    
    h_t = h_states[t]
    v_t = v_states[t]
    
    # Step 1: Local gradient
    dh_t_from_out = dout[:, :, t].unsqueeze(-1) * C.unsqueeze(0)
    print(f"\n1. dh_t_from_out (local): mean={dh_t_from_out.mean().item():.6f}")
    
    # Step 2: Reverse scan for hidden states
    print(f"2. dh BEFORE: mean={dh.mean().item():.6f}")
    if t < seqlen - 1:
        delta_A_next = torch.exp(delta[:, :, t+1].unsqueeze(-1) * A.unsqueeze(0))
        print(f"   exp(delta[{t+1}]*A) mean: {delta_A_next.mean().item():.6f}")
        dh = dh_t_from_out + delta_A_next * dh
    else:
        dh = dh_t_from_out
    print(f"   dh AFTER: mean={dh.mean().item():.6f}")
    print(f"   dh[0,0,:]: {dh[0,0,:].cpu().numpy()}")
    
    # Step 3: Reverse scan for velocity
    print(f"\n3. dv BEFORE: mean={dv.mean().item():.6f}")
    print(f"   dv[0,0,:]: {dv[0,0,:].cpu().numpy()}")
    dv_t = dh + beta * dv
    print(f"   dv_t = dh + beta*dv: mean={dv_t.mean().item():.6f}")
    print(f"   dv_t[0,0,:]: {dv_t[0,0,:].cpu().numpy()}")
    dv = dv_t
    
    # Step 4: db_t_ortho
    db_t_ortho = dv_t
    db_t_ortho_values[t] = db_t_ortho[0].clone()
    print(f"\n4. db_t_ortho = dv_t:")
    print(f"   Mean: {db_t_ortho.mean().item():.6f}, Max: {db_t_ortho.max().item():.6f}")
    print(f"   db_t_ortho[0]:\n{db_t_ortho[0].cpu().numpy()}")
    
    # Step 5: NS backward
    b_t_input = alpha * (delta[:, :, t].unsqueeze(-1) * B * u[:, :, t].unsqueeze(-1))
    
    grad_u_t, grad_delta_t, grad_B_t = pytorch_ns_backward_ref_accurate(
        db_t_ortho[0], b_t_input[0], alpha, delta[0, :, t], B, u[0, :, t]
    )
    
    print(f"\n5. NS Backward Output:")
    print(f"   grad_u: {grad_u_t.cpu().numpy()}")
    print(f"   grad_delta: {grad_delta_t.cpu().numpy()}")
    print(f"   grad_B mean: {grad_B_t.mean().item():.6f}")
    
    du[0, :, t] += grad_u_t
    ddelta[0, :, t] += grad_delta_t
    dB += grad_B_t
    
    # Step 6: Exp path gradient
    h_t_minus_v_t = h_t - v_t
    ddelta_exp = (dh * A.unsqueeze(0) * h_t_minus_v_t).sum(dim=-1)
    print(f"\n6. Exp path gradient:")
    print(f"   ddelta_exp: {ddelta_exp[0].cpu().numpy()}")
    ddelta[:, :, t] += ddelta_exp
    
    # Step 7: C gradient
    dC += (dout[:, :, t].unsqueeze(-1) * h_t).sum(dim=(0, 1))
    
    print(f"\n7. Total gradients after timestep {t}:")
    print(f"   ddelta[{t}]: {ddelta[0, :, t].cpu().numpy()}")
    print(f"   du[{t}]: {du[0, :, t].cpu().numpy()}")

# CUDA comparison
fwd_cuda = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, None, False, beta, alpha)
out_cuda, x_cuda, X_4_cuda = fwd_cuda[0], fwd_cuda[1], fwd_cuda[2] if len(fwd_cuda) > 2 else None

bwd_cuda = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, x_cuda, None, None,
    False, False, beta, alpha, X_4_cuda
)
du_cuda, ddelta_cuda = bwd_cuda[0], bwd_cuda[1]

print("\n" + "="*80)
print("CUDA COMPARISON")
print("="*80)
for t in range(seqlen):
    print(f"\nTimestep {t}:")
    print(f"  PyTorch ddelta: {ddelta[0, :, t].cpu().numpy()}")
    print(f"  CUDA ddelta:    {ddelta_cuda[0, :, t].cpu().numpy()}")
    print(f"  Difference:     {(ddelta[0, :, t] - ddelta_cuda[0, :, t]).cpu().numpy()}")
    print(f"  PyTorch du:     {du[0, :, t].cpu().numpy()}")
    print(f"  CUDA du:        {du_cuda[0, :, t].cpu().numpy()}")
    print(f"  Difference:     {(du[0, :, t] - du_cuda[0, :, t]).cpu().numpy()}")




