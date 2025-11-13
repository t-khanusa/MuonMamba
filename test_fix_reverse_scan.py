#!/usr/bin/env python3
"""
Fix reverse scan logic step by step and verify with CUDA
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

# Backward - FIXED version
print("="*80)
print("BACKWARD WITH FIXED REVERSE SCAN")
print("="*80)

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

# Backward loop - CAREFULLY implementing reverse scan
for t in range(seqlen - 1, -1, -1):
    h_t = h_states[t]
    v_t = v_states[t]
    
    # Step 1: Local gradient from output
    # CUDA: thread_reverse_data[i].y = dout[i] * C
    dh_t_from_out = dout[:, :, t].unsqueeze(-1) * C.unsqueeze(0)
    
    # Step 2: Hidden state reverse scan
    # CUDA does inclusive reverse scan: dh[t] accumulates from t and all future
    # After scan: dh[t] = dout[t]*C + exp(delta[t+1]*A) * dout[t+1]*C + ...
    # Since we iterate backward, dh already contains accumulated gradient from future
    if t < seqlen - 1:
        # Propagate from future: dh[t] = local + exp(delta[t+1]*A) * dh[t+1]
        delta_A_next = torch.exp(delta[:, :, t+1].unsqueeze(-1) * A.unsqueeze(0))
        dh = dh_t_from_out + delta_A_next * dh
    else:
        # Last timestep: no future
        dh = dh_t_from_out
    
    # Step 3: Velocity reverse scan  
    # CUDA: dv_reverse_data[i] = (beta, dh[t])
    # After inclusive reverse scan: dv[t] = dh[t] + beta * dh[t+1] + beta^2 * dh[t+2] + ...
    # This can be computed recursively: dv[t] = dh[t] + beta * dv[t+1]
    # Since we iterate backward, dv already contains dv[t+1]
    dv_t = dh + beta * dv
    
    # Update for next iteration
    dv = dv_t
    
    # Step 4: db_t_ortho is the gradient w.r.t. b_t_ortho
    # This is what CUDA accumulates into grad_X_4_buffer
    db_t_ortho = dv_t
    
    # Step 5: NS backward
    b_t_input = alpha * (delta[:, :, t].unsqueeze(-1) * B * u[:, :, t].unsqueeze(-1))
    
    grad_u_t, grad_delta_t, grad_B_t = pytorch_ns_backward_ref_accurate(
        db_t_ortho[0], b_t_input[0], alpha, delta[0, :, t], B, u[0, :, t]
    )
    
    du[0, :, t] += grad_u_t
    ddelta[0, :, t] += grad_delta_t
    dB += grad_B_t
    
    # Step 6: Exp path gradient (only path when NS is enabled)
    h_t_minus_v_t = h_t - v_t
    ddelta_exp = (dh * A.unsqueeze(0) * h_t_minus_v_t).sum(dim=-1)
    ddelta[:, :, t] += ddelta_exp
    
    # Step 7: C gradient
    dC += (dout[:, :, t].unsqueeze(-1) * h_t).sum(dim=(0, 1))

# CUDA comparison
fwd_cuda = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, None, False, beta, alpha)
bwd_cuda = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, fwd_cuda[1], None, None,
    False, False, beta, alpha, fwd_cuda[2] if len(fwd_cuda) > 2 else None
)
du_cuda, ddelta_cuda, dB_cuda = bwd_cuda[0], bwd_cuda[1], bwd_cuda[3]

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"du diff: max={torch.abs(du_cuda - du).max().item():.6f}, mean={torch.abs(du_cuda - du).mean().item():.6f}")
print(f"ddelta diff: max={torch.abs(ddelta_cuda - ddelta).max().item():.6f}, mean={torch.abs(ddelta_cuda - ddelta).mean().item():.6f}")
print(f"dB diff: max={torch.abs(dB_cuda - dB).max().item():.6f}, mean={torch.abs(dB_cuda - dB).mean().item():.6f}")

print("\nPer-timestep comparison:")
for t in range(seqlen):
    print(f"\nTimestep {t}:")
    du_diff = torch.abs(du_cuda[0, :, t] - du[0, :, t]).max().item()
    ddelta_diff = torch.abs(ddelta_cuda[0, :, t] - ddelta[0, :, t]).max().item()
    print(f"  du max diff: {du_diff:.6f}")
    print(f"  ddelta max diff: {ddelta_diff:.6f}")
    if du_diff > 0.01 or ddelta_diff > 0.01:
        print(f"  ❌ LARGE DIFFERENCES:")
        print(f"    PyTorch du: {du[0, :, t].cpu().numpy()}")
        print(f"    CUDA du:    {du_cuda[0, :, t].cpu().numpy()}")
        print(f"    PyTorch ddelta: {ddelta[0, :, t].cpu().numpy()}")
        print(f"    CUDA ddelta:    {ddelta_cuda[0, :, t].cpu().numpy()}")




