#!/usr/bin/env python3
"""Detailed tracing to find exact bugs"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda
from test_comprehensive_ns_backward_accurate import selective_scan_backward_ref_accurate

# Minimal test case for detailed tracing
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

bwd_cuda = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, x_cuda, None, None,
    False, False, beta, alpha, X_4_cuda
)
du_cuda, ddelta_cuda, dA_cuda, dB_cuda, dC_cuda = bwd_cuda[0], bwd_cuda[1], bwd_cuda[2], bwd_cuda[3], bwd_cuda[4]

print(f"\nCUDA Gradients:")
print(f"  du:\n{du_cuda}")
print(f"  ddelta:\n{ddelta_cuda}")
print(f"  dB:\n{dB_cuda}")

print("\n" + "="*80)
print("PyTorch Reference with Detailed Tracing")
print("="*80)

# Monkey-patch to add tracing
original_ns_backward = None
from test_comprehensive_ns_backward_accurate import pytorch_ns_backward_ref_accurate

def traced_ns_backward(grad_output, G_input, alpha, delta_val, B_val, u_val, eps=1e-8):
    """NS backward with tracing"""
    global trace_count
    if not hasattr(traced_ns_backward, 'count'):
        traced_ns_backward.count = 0
    traced_ns_backward.count += 1
    
    result = pytorch_ns_backward_ref_accurate(grad_output, G_input, alpha, delta_val, B_val, u_val, eps)
    
    if traced_ns_backward.count <= 4:  # Print first 4 calls
        print(f"\n  NS Backward call #{traced_ns_backward.count}:")
        print(f"    grad_output mean: {grad_output.mean().item():.6f}, shape: {grad_output.shape}")
        print(f"    G_input mean: {G_input.mean().item():.6f}, shape: {G_input.shape}")
        print(f"    grad_u: {result[0]}")
        print(f"    grad_delta: {result[1]}")
        print(f"    grad_B mean: {result[2].mean().item():.6f}")
    
    return result

# Replace NS backward with traced version
import test_comprehensive_ns_backward_accurate
test_comprehensive_ns_backward_accurate.pytorch_ns_backward_ref_accurate = traced_ns_backward

# Run backward with tracing
du_ref, ddelta_ref, dA_ref, dB_ref, dC_ref, dD_ref = selective_scan_backward_ref_accurate(
    u, delta, A, B, C, D, dout, beta=beta, alpha=alpha,
    delta_bias=None, delta_softplus=False
)

print(f"\nReference Gradients:")
print(f"  du:\n{du_ref}")
print(f"  ddelta:\n{ddelta_ref}")
print(f"  dB:\n{dB_ref}")

print(f"\nDifferences:")
print(f"  du diff:\n{du_cuda - du_ref}")
print(f"  ddelta diff:\n{ddelta_cuda - ddelta_ref}")
print(f"  dB diff:\n{dB_cuda - dB_ref}")





