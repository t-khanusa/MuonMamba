#!/usr/bin/env python3
"""
Test script to compare CUDA backward pass with PyTorch reference
Uses debug output from CUDA to trace exact bug location
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda
from test_comprehensive_ns_backward_accurate import selective_scan_backward_ref_accurate, pytorch_ns_backward_ref_accurate

def test_backward_debug_comparison():
    """Compare CUDA vs PyTorch backward with debug output"""
    
    # Minimal test case
    batch, dim, seqlen, dstate = 1, 2, 4, 2
    beta, alpha = 0.9, 1.0
    device = 'cuda'
    dtype = torch.float32
    
    torch.manual_seed(42)
    u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * 0.5
    delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * 0.1 + 0.1
    A = -torch.rand(dim, dstate, dtype=dtype, device=device) * 0.1
    B = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
    C = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
    D = torch.randn(dim, dtype=dtype, device=device) * 0.1
    dout = torch.ones(batch, dim, seqlen, dtype=dtype, device=device)
    
    print("=" * 80)
    print("Backward Pass Debug Comparison")
    print("=" * 80)
    print(f"Batch={batch}, Dim={dim}, SeqLen={seqlen}, DState={dstate}")
    print(f"alpha={alpha}, beta={beta}")
    print()
    
    # Forward pass (CUDA)
    print("Running CUDA forward pass...")
    fwd_result = selective_scan_cuda.fwd(
        u, delta, A, B, C, D, None, None, False, beta, alpha
    )
    y_cuda = fwd_result[0]
    X_4_buffer = fwd_result[2] if len(fwd_result) > 2 else None
    print("CUDA forward complete.")
    print()
    
    # Backward pass (CUDA with debug output)
    print("Running CUDA backward pass with debug output...")
    print("=" * 80)
    print("CUDA Debug Output:")
    print("=" * 80)
    
    bwd_result = selective_scan_cuda.bwd(
        u, delta, A, B, C, D, None, None, dout, fwd_result[1], None, None,
        False, False, beta, alpha, X_4_buffer
    )
    du_cuda = bwd_result[0]
    ddelta_cuda = bwd_result[1]
    dA_cuda = bwd_result[2]
    dB_cuda = bwd_result[3]
    dC_cuda = bwd_result[4]
    dD_cuda = bwd_result[5] if len(bwd_result) > 5 else None
    
    print("=" * 80)
    print("CUDA Debug Output Complete")
    print("=" * 80)
    print()
    
    # Backward pass (PyTorch reference)
    print("Running PyTorch reference backward pass...")
    du_ref, ddelta_ref, dA_ref, dB_ref, dC_ref, dD_ref = selective_scan_backward_ref_accurate(
        u, delta, A, B, C, D, dout, beta=beta, alpha=alpha
    )
    print("PyTorch reference complete.")
    print()
    
    # Compare gradients
    print("=" * 80)
    print("Gradient Comparison")
    print("=" * 80)
    
    print("\nGradient w.r.t. u (du):")
    print("CUDA:")
    print(du_cuda)
    print("\nPyTorch:")
    print(du_ref)
    print("\nDifference:")
    diff_du = du_cuda - du_ref
    print(diff_du)
    print(f"Max absolute difference: {diff_du.abs().max().item():.6f}")
    print(f"Max relative difference: {(diff_du.abs() / (du_ref.abs() + 1e-8)).max().item():.6f}")
    
    print("\n" + "=" * 80)
    print("\nGradient w.r.t. delta (ddelta):")
    print("CUDA:")
    print(ddelta_cuda)
    print("\nPyTorch:")
    print(ddelta_ref)
    print("\nDifference:")
    diff_ddelta = ddelta_cuda - ddelta_ref
    print(diff_ddelta)
    print(f"Max absolute difference: {diff_ddelta.abs().max().item():.6f}")
    print(f"Max relative difference: {(diff_ddelta.abs() / (ddelta_ref.abs() + 1e-8)).max().item():.6f}")
    
    print("\n" + "=" * 80)
    print("\nGradient w.r.t. B (dB):")
    print("CUDA:")
    print(dB_cuda)
    print("\nPyTorch:")
    print(dB_ref)
    print("\nDifference:")
    diff_dB = dB_cuda - dB_ref
    print(diff_dB)
    print(f"Max absolute difference: {diff_dB.abs().max().item():.6f}")
    print(f"Max relative difference: {(diff_dB.abs() / (dB_ref.abs() + 1e-8)).max().item():.6f}")
    
    # Analyze du per timestep
    print("\n" + "=" * 80)
    print("Analysis: du per timestep")
    print("=" * 80)
    print("D * dout (direct feedthrough):")
    # D is [dim], dout is [batch, dim, seqlen]
    D_dout = D.unsqueeze(0).unsqueeze(-1) * dout  # [batch, dim, seqlen]
    print(D_dout)
    
    print("\nCUDA du (should be D*dout + NS_grad):")
    print(du_cuda)
    
    print("\nNS contribution (du - D*dout):")
    ns_contrib = du_cuda - D_dout
    print(ns_contrib)
    
    print("\nPyTorch NS contribution (du_ref - D*dout):")
    ns_contrib_ref = du_ref - D_dout
    print(ns_contrib_ref)
    
    print("\nDifference in NS contribution:")
    diff_ns = ns_contrib - ns_contrib_ref
    print(diff_ns)
    
    # Check if timesteps 1-3 have zero NS contribution in CUDA
    print("\n" + "=" * 80)
    print("Bug Check: NS contribution per timestep")
    print("=" * 80)
    for t in range(seqlen):
        ns_contrib_t = ns_contrib[0, :, t]
        ns_contrib_ref_t = ns_contrib_ref[0, :, t]
        print(f"Timestep {t}:")
        print(f"  CUDA NS contrib: {ns_contrib_t.tolist()}")
        print(f"  PyTorch NS contrib: {ns_contrib_ref_t.tolist()}")
        print(f"  Difference: {(ns_contrib_t - ns_contrib_ref_t).abs().max().item():.6f}")
        if ns_contrib_t.abs().max() < 1e-6 and ns_contrib_ref_t.abs().max() > 1e-6:
            print(f"  ⚠️  BUG: CUDA has zero NS contribution but PyTorch has non-zero!")
        elif ns_contrib_t.abs().max() > 1e-6 and ns_contrib_ref_t.abs().max() < 1e-6:
            print(f"  ⚠️  PyTorch has zero NS contribution but CUDA has non-zero!")
        print()

if __name__ == "__main__":
    test_backward_debug_comparison()

