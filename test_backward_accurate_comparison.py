#!/usr/bin/env python3
"""
Compare CUDA backward with accurate PyTorch reference
Uses the new accurate NS backward implementation
"""

import torch
import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    import selective_scan_cuda
except ImportError as e:
    print(f"ERROR: Cannot import selective_scan_cuda: {e}")
    sys.exit(1)

from test_comprehensive_ns_backward_accurate import (
    selective_scan_backward_ref_accurate,
    pytorch_ns_backward_ref_accurate
)


def compare_gradients(grad_cuda, grad_ref, name, tol_abs=1e-3, tol_rel=1e-2, verbose=True):
    """Compare CUDA and reference gradients with detailed statistics"""
    if grad_cuda.shape != grad_ref.shape:
        print(f"\n❌ {name}: Shape mismatch!")
        print(f"  CUDA: {grad_cuda.shape}, Reference: {grad_ref.shape}")
        return False
    
    grad_cuda_flat = grad_cuda.flatten().float()
    grad_ref_flat = grad_ref.flatten().float()
    
    abs_diff = (grad_cuda_flat - grad_ref_flat).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    max_magnitude = torch.maximum(grad_cuda_flat.abs(), grad_ref_flat.abs()) + 1e-8
    rel_diff = abs_diff / max_magnitude
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    has_nan = grad_cuda_flat.isnan().any().item() or grad_ref_flat.isnan().any().item()
    has_inf = grad_cuda_flat.isinf().any().item() or grad_ref_flat.isinf().any().item()
    
    ref_max = grad_ref_flat.abs().max().item()
    adaptive_tol_abs = max(tol_abs, ref_max * tol_rel)
    
    exceed_tol_count = (rel_diff > tol_rel).sum().item()
    exceed_tol_ratio = exceed_tol_count / len(rel_diff) if len(rel_diff) > 0 else 0.0
    
    # More lenient pass criteria for accurate comparison
    passed = (exceed_tol_ratio < 0.10 and  # Allow up to 10% to exceed tolerance
              max_rel_diff < tol_rel * 5.0 and  # Allow 5x max relative error
              not has_nan and not has_inf)
    
    if verbose:
        status = "✅" if passed else "❌"
        print(f"\n{status} {name}:")
        print(f"  Max abs diff: {max_abs_diff:.6e} (adaptive tol: {adaptive_tol_abs:.6e})")
        print(f"  Mean abs diff: {mean_abs_diff:.6e}")
        print(f"  Max rel diff: {max_rel_diff:.6e} (tol: {tol_rel:.6e})")
        print(f"  Mean rel diff: {mean_rel_diff:.6e}")
        print(f"  Ref max magnitude: {ref_max:.6e}")
        print(f"  Exceed tolerance: {exceed_tol_count}/{len(rel_diff)} ({exceed_tol_ratio*100:.2f}%)")
        if has_nan:
            print(f"  ⚠️  NaNs detected!")
        if has_inf:
            print(f"  ⚠️  Infs detected!")
        
        if not passed and max_abs_diff > 0:
            worst_idx = abs_diff.argmax().item()
            print(f"  Worst mismatch at idx {worst_idx}:")
            print(f"    CUDA: {grad_cuda_flat[worst_idx]:.8e}, Ref: {grad_ref_flat[worst_idx]:.8e}")
            print(f"    Rel error: {rel_diff[worst_idx].item():.6e}")
    
    return passed


def test_backward_accurate(batch=2, dim=8, seqlen=32, dstate=8, beta=0.9, alpha=1.0,
                          is_variable_B=False, is_variable_C=False,
                          use_d=False, dtype=torch.float32, seed=42):
    """Test backward with accurate reference"""
    print(f"\n{'='*80}")
    print(f"Testing: B={batch}, D={dim}, L={seqlen}, N={dstate}")
    print(f"  beta={beta}, alpha={alpha}, variable_B={is_variable_B}, variable_C={is_variable_C}")
    print(f"{'='*80}")
    
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate inputs
    scale = 0.1
    u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * scale
    delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * scale
    
    A = -torch.rand(dim, dstate, dtype=dtype, device=device) * 0.1
    
    if is_variable_B:
        n_groups = 1
        B = torch.randn(batch, n_groups, dstate, seqlen, dtype=dtype, device=device) * scale
    else:
        B = torch.randn(dim, dstate, dtype=dtype, device=device) * scale
    
    if is_variable_C:
        n_groups = 1
        C = torch.randn(batch, n_groups, dstate, seqlen, dtype=dtype, device=device) * scale
    else:
        C = torch.randn(dim, dstate, dtype=dtype, device=device) * scale
    
    D = torch.randn(dim, dtype=dtype, device=device) * 0.1 if use_d else None
    
    dout = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * scale
    
    # CUDA forward + backward
    try:
        fwd_result = selective_scan_cuda.fwd(
            u, delta, A, B, C, D, None, None, False, beta, alpha
        )
        out_cuda = fwd_result[0]
        x_cuda = fwd_result[1]
        X_4_buffer = fwd_result[2] if len(fwd_result) > 2 and beta != 0.0 else None
        
        bwd_result = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, None, dout, x_cuda, None, None,
            False, False, beta, alpha, X_4_buffer
        )
        du_cuda = bwd_result[0]
        ddelta_cuda = bwd_result[1]
        dA_cuda = bwd_result[2]
        dB_cuda = bwd_result[3]
        dC_cuda = bwd_result[4]
        dD_cuda = bwd_result[5] if len(bwd_result) > 5 else None
    except Exception as e:
        print(f"❌ CUDA failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Accurate PyTorch reference
    try:
        du_ref, ddelta_ref, dA_ref, dB_ref, dC_ref, dD_ref = selective_scan_backward_ref_accurate(
            u, delta, A, B, C, D, dout, beta=beta, alpha=alpha,
            delta_bias=None, delta_softplus=False
        )
    except Exception as e:
        print(f"❌ Reference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare gradients
    all_passed = True
    all_passed &= compare_gradients(du_cuda, du_ref, "du", tol_abs=1e-3, tol_rel=2e-2, verbose=True)
    all_passed &= compare_gradients(ddelta_cuda, ddelta_ref, "ddelta", tol_abs=1e-3, tol_rel=2e-2, verbose=True)
    all_passed &= compare_gradients(dA_cuda, dA_ref, "dA", tol_abs=1e-3, tol_rel=2e-2, verbose=True)
    all_passed &= compare_gradients(dB_cuda, dB_ref, "dB", tol_abs=1e-3, tol_rel=2e-2, verbose=True)
    all_passed &= compare_gradients(dC_cuda, dC_ref, "dC", tol_abs=1e-3, tol_rel=2e-2, verbose=True)
    if use_d:
        all_passed &= compare_gradients(dD_cuda, dD_ref, "dD", tol_abs=1e-3, tol_rel=2e-2, verbose=True)
    
    return all_passed


def main():
    print("="*80)
    print("Accurate CUDA vs PyTorch Reference Comparison")
    print("Using Newton-Schulz with detached first 4 steps")
    print("="*80)
    
    test_cases = [
        ("Small Basic", {'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
                        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False}),
        ("Small Var B", {'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
                        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': True, 'is_variable_C': False}),
        ("Medium Basic", {'batch': 4, 'dim': 32, 'seqlen': 128, 'dstate': 16,
                         'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False}),
    ]
    
    results = []
    for name, kwargs in test_cases:
        print(f"\n[{len(results)+1}/{len(test_cases)}] Testing: {name}")
        passed = test_backward_accurate(**kwargs)
        results.append((name, passed))
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    passed_count = sum(1 for _, p in results if p)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    print(f"\nResults: {passed_count}/{len(results)} passed")
    
    return passed_count == len(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)





