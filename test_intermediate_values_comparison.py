#!/usr/bin/env python3
"""
Detailed intermediate value comparison between CUDA and PyTorch reference.
This will help identify exactly where gradients diverge.
"""

import torch
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda
from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref

# Minimal test case for detailed comparison
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
print("STEP 1: Forward Pass")
print("="*80)

# Forward pass - PyTorch reference
h_pytorch = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
v_pytorch = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
h_states = []
v_states = []
b_t_states = []

for t in range(seqlen):
    b_t = alpha * (delta[:, :, t].unsqueeze(-1) * B * u[:, :, t].unsqueeze(-1))
    b_t_original = b_t.clone()
    
    # Apply NS
    b_t_ortho = torch.zeros_like(b_t)
    for b in range(batch):
        b_t_matrix = b_t[b]
        b_t_ortho[b] = newtonschulz5_ref(b_t_matrix, steps=5)
    b_t = b_t_ortho
    
    b_t_states.append(b_t_original)
    v_pytorch = beta * v_pytorch + b_t
    v_states.append(v_pytorch.clone())
    
    delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
    h_pytorch = delta_A_t * h_pytorch + v_pytorch
    h_states.append(h_pytorch.clone())

print("Forward pass states computed")

# CUDA forward
fwd_cuda = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, None, False, beta, alpha)
out_cuda, x_cuda, X_4_cuda = fwd_cuda[0], fwd_cuda[1], fwd_cuda[2] if len(fwd_cuda) > 2 else None

print(f"CUDA forward output mean: {out_cuda.mean().item():.6f}")
if X_4_cuda is not None:
    print(f"X_4_buffer shape: {X_4_cuda.shape}, mean: {X_4_cuda.mean().item():.6f}")

print("\n" + "="*80)
print("STEP 2: Backward Pass - Tracing Intermediate Values")
print("="*80)

# PyTorch backward with detailed tracing
dh_pytorch = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
dv_pytorch = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)

du_pytorch = torch.zeros_like(u)
ddelta_pytorch = torch.zeros_like(delta)
dA_pytorch = torch.zeros_like(A)
dB_pytorch = torch.zeros_like(B)
dC_pytorch = torch.zeros_like(C)

# Initialize du with D*dout
if D is not None:
    for t in range(seqlen):
        du_pytorch[:, :, t] = D.unsqueeze(0) * dout[:, :, t]

pytorch_trace = []

# Backward loop
for t in range(seqlen - 1, -1, -1):
    h_t = h_states[t]
    v_t = v_states[t]
    
    # Gradient from output
    dh_t_from_out = dout[:, :, t].unsqueeze(-1) * C.unsqueeze(0)
    
    # Reverse scan for hidden states
    if t < seqlen - 1:
        delta_A_next = torch.exp(delta[:, :, t+1].unsqueeze(-1) * A.unsqueeze(0))
        dh_pytorch = dh_t_from_out + delta_A_next * dh_pytorch
    else:
        dh_pytorch = dh_t_from_out
    
    # Reverse scan for velocity
    dv_t_pytorch = dh_pytorch + beta * dv_pytorch
    dv_pytorch = dv_t_pytorch
    
    db_t_ortho_pytorch = dv_t_pytorch
    
    # Trace values
    trace_entry = {
        'timestep': t,
        'dh_mean': dh_pytorch.mean().item(),
        'dh_max': dh_pytorch.max().item(),
        'dv_t_mean': dv_t_pytorch.mean().item(),
        'dv_t_max': dv_t_pytorch.max().item(),
        'db_t_ortho_mean': db_t_ortho_pytorch.mean().item(),
        'db_t_ortho_max': db_t_ortho_pytorch.max().item(),
        'h_t_mean': h_t.mean().item(),
        'v_t_mean': v_t.mean().item(),
        'h_t_minus_v_t_mean': (h_t - v_t).mean().item(),
    }
    pytorch_trace.append(trace_entry)
    
    # NS backward
    b_t_input = alpha * (delta[:, :, t].unsqueeze(-1) * B * u[:, :, t].unsqueeze(-1))
    from test_comprehensive_ns_backward_accurate import pytorch_ns_backward_ref_accurate
    
    grad_u_t, grad_delta_t, grad_B_t = pytorch_ns_backward_ref_accurate(
        db_t_ortho_pytorch[0], b_t_input[0], alpha, delta[0, :, t], B, u[0, :, t]
    )
    
    trace_entry['ns_grad_u'] = grad_u_t.cpu().numpy()
    trace_entry['ns_grad_delta'] = grad_delta_t.cpu().numpy()
    trace_entry['ns_grad_B_mean'] = grad_B_t.mean().item()
    
    du_pytorch[0, :, t] += grad_u_t
    ddelta_pytorch[0, :, t] += grad_delta_t
    dB_pytorch += grad_B_t
    
    # Exp path gradient
    h_t_minus_v_t = h_t - v_t
    ddelta_exp = (dh_pytorch * A.unsqueeze(0) * h_t_minus_v_t).sum(dim=-1)
    trace_entry['ddelta_exp'] = ddelta_exp[0].cpu().numpy()
    ddelta_pytorch[:, :, t] += ddelta_exp
    
    # C gradient
    dC_pytorch += (dout[:, :, t].unsqueeze(-1) * h_t).sum(dim=(0, 1))
    
    trace_entry['du_after'] = du_pytorch[0, :, t].cpu().numpy()
    trace_entry['ddelta_after'] = ddelta_pytorch[0, :, t].cpu().numpy()

# CUDA backward
bwd_cuda = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, x_cuda, None, None,
    False, False, beta, alpha, X_4_cuda
)
du_cuda, ddelta_cuda, dA_cuda, dB_cuda, dC_cuda = bwd_cuda[0], bwd_cuda[1], bwd_cuda[2], bwd_cuda[3], bwd_cuda[4]

print("\n" + "="*80)
print("STEP 3: Comparison")
print("="*80)

print("\nIntermediate Values by Timestep (reverse order):")
for entry in pytorch_trace:
    print(f"\nTimestep {entry['timestep']}:")
    print(f"  dh: mean={entry['dh_mean']:.6f}, max={entry['dh_max']:.6f}")
    print(f"  dv_t: mean={entry['dv_t_mean']:.6f}, max={entry['dv_t_max']:.6f}")
    print(f"  db_t_ortho: mean={entry['db_t_ortho_mean']:.6f}, max={entry['db_t_ortho_max']:.6f}")
    print(f"  h_t_minus_v_t: mean={entry['h_t_minus_v_t_mean']:.6f}")
    print(f"  NS grad_delta: {entry['ns_grad_delta']}")
    print(f"  ddelta_exp: {entry['ddelta_exp']}")
    print(f"  Total ddelta (PyTorch): {entry['ddelta_after']}")
    print(f"  Total ddelta (CUDA): {ddelta_cuda[0, :, entry['timestep']].cpu().numpy()}")
    print(f"  Total du (PyTorch): {entry['du_after']}")
    print(f"  Total du (CUDA): {du_cuda[0, :, entry['timestep']].cpu().numpy()}")

print("\n" + "="*80)
print("Final Gradient Comparison")
print("="*80)
print(f"du diff: max={torch.abs(du_cuda - du_pytorch).max().item():.6f}")
print(f"ddelta diff: max={torch.abs(ddelta_cuda - ddelta_pytorch).max().item():.6f}")
print(f"dB diff: max={torch.abs(dB_cuda - dB_pytorch).max().item():.6f}")
print(f"dC diff: max={torch.abs(dC_cuda - dC_pytorch).max().item():.6f}")





