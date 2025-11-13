#!/usr/bin/env python3
"""Debug gradient flow to find bugs"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda
from test_comprehensive_ns_backward_accurate import selective_scan_backward_ref_accurate

# Small test case
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
print("CUDA Forward + Backward")
print("="*80)
fwd_cuda = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, None, False, beta, alpha)
out_cuda, x_cuda, X_4_cuda = fwd_cuda[0], fwd_cuda[1], fwd_cuda[2] if len(fwd_cuda) > 2 else None

print(f"out_cuda shape: {out_cuda.shape}, mean: {out_cuda.mean().item():.6f}")
if X_4_cuda is not None:
    print(f"X_4_cuda shape: {X_4_cuda.shape}, mean: {X_4_cuda.mean().item():.6f}")

bwd_cuda = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, x_cuda, None, None,
    False, False, beta, alpha, X_4_cuda
)
du_cuda, ddelta_cuda, dA_cuda, dB_cuda, dC_cuda = bwd_cuda[0], bwd_cuda[1], bwd_cuda[2], bwd_cuda[3], bwd_cuda[4]

print(f"\nCUDA Gradients:")
print(f"  du: sum={du_cuda.sum().item():.6f}, non-zero={du_cuda.nonzero().numel()}/{du_cuda.numel()}")
print(f"  ddelta: sum={ddelta_cuda.sum().item():.6f}, non-zero={ddelta_cuda.nonzero().numel()}/{ddelta_cuda.numel()}")
print(f"  dA: sum={dA_cuda.sum().item():.6f}, non-zero={dA_cuda.nonzero().numel()}/{dA_cuda.numel()}")
print(f"  dB: sum={dB_cuda.sum().item():.6f}, non-zero={dB_cuda.nonzero().numel()}/{dB_cuda.numel()}")
print(f"  dC: sum={dC_cuda.sum().item():.6f}, non-zero={dC_cuda.nonzero().numel()}/{dC_cuda.numel()}")

print("\n" + "="*80)
print("PyTorch Reference Forward + Backward")
print("="*80)
du_ref, ddelta_ref, dA_ref, dB_ref, dC_ref, dD_ref = selective_scan_backward_ref_accurate(
    u, delta, A, B, C, D, dout, beta=beta, alpha=alpha,
    delta_bias=None, delta_softplus=False
)

print(f"\nReference Gradients:")
print(f"  du: sum={du_ref.sum().item():.6f}, non-zero={du_ref.nonzero().numel()}/{du_ref.numel()}")
print(f"  ddelta: sum={ddelta_ref.sum().item():.6f}, non-zero={ddelta_ref.nonzero().numel()}/{ddelta_ref.numel()}")
print(f"  dA: sum={dA_ref.sum().item():.6f}, non-zero={dA_ref.nonzero().numel()}/{dA_ref.numel()}")
print(f"  dB: sum={dB_ref.sum().item():.6f}, non-zero={dB_ref.nonzero().numel()}/{dB_ref.numel()}")
print(f"  dC: sum={dC_ref.sum().item():.6f}, non-zero={dC_ref.nonzero().numel()}/{dC_ref.numel()}")

print("\n" + "="*80)
print("Differences:")
print("="*80)
print(f"  du diff: max={torch.abs(du_cuda - du_ref).max().item():.6f}, mean={torch.abs(du_cuda - du_ref).mean().item():.6f}")
print(f"  ddelta diff: max={torch.abs(ddelta_cuda - ddelta_ref).max().item():.6f}, mean={torch.abs(ddelta_cuda - ddelta_ref).mean().item():.6f}")
print(f"  dA diff: max={torch.abs(dA_cuda - dA_ref).max().item():.6f}, mean={torch.abs(dA_cuda - dA_ref).mean().item():.6f}")
print(f"  dB diff: max={torch.abs(dB_cuda - dB_ref).max().item():.6f}, mean={torch.abs(dB_cuda - dB_ref).mean().item():.6f}")
print(f"  dC diff: max={torch.abs(dC_cuda - dC_ref).max().item():.6f}, mean={torch.abs(dC_cuda - dC_ref).mean().item():.6f}")





