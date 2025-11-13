#!/usr/bin/env python3
"""
Comprehensive MuonMamba (Momentum + Newton-Schulz) Test Suite
Tests both forward and backward passes with multiple scenarios
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

try:
    import selective_scan_cuda
except ImportError as e:
    print(f"ERROR: Cannot import selective_scan_cuda: {e}")
    print("Please make sure the CUDA extension is built.")
    sys.exit(1)

from mamba_ssm.ops.selective_scan_interface import selective_scan_ref, newtonschulz5_ref, selective_scan_fn


def compare_tensors(t1, t2, name, tol_abs=1e-4, tol_rel=1e-3, verbose=True):
    """Compare two tensors and return True if they match within tolerances"""
    if t1.shape != t2.shape:
        print(f"\n❌ {name}: Shape mismatch! CUDA: {t1.shape}, Reference: {t2.shape}")
        return False
    
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    
    abs_diff = (t1_flat - t2_flat).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    # Relative error
    denom = t2_flat.abs() + 1e-8
    rel_error = abs_diff / denom
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    # Check for NaNs/Infs
    t1_nan = t1_flat.isnan().any().item()
    t2_nan = t2_flat.isnan().any().item()
    t1_inf = t1_flat.isinf().any().item()
    t2_inf = t2_flat.isinf().any().item()
    
    # Pass criteria
    pass_test = (max_abs_diff < tol_abs and max_rel_error < tol_rel and 
                not t1_nan and not t2_nan and not t1_inf and not t2_inf)
    
    if verbose:
        status = "✅" if pass_test else "❌"
        print(f"\n{status} {name}:")
        print(f"  Max abs diff: {max_abs_diff:.6e} (tol: {tol_abs:.6e})")
        print(f"  Mean abs diff: {mean_abs_diff:.6e}")
        print(f"  Max rel error: {max_rel_error:.6e} (tol: {tol_rel:.6e})")
        print(f"  Mean rel error: {mean_rel_error:.6e}")
        if t1_nan or t2_nan:
            print(f"  ⚠️  NaNs detected! (CUDA: {t1_nan}, Ref: {t2_nan})")
        if t1_inf or t2_inf:
            print(f"  ⚠️  Infs detected! (CUDA: {t1_inf}, Ref: {t2_inf})")
        
        if not pass_test and max_abs_diff > 0:
            # Show worst mismatches
            worst_idx = abs_diff.argmax().item()
            print(f"  Worst mismatch at idx {worst_idx}:")
            print(f"    CUDA: {t1_flat[worst_idx]:.8e}, Ref: {t2_flat[worst_idx]:.8e}")
    
    return pass_test


def test_case(name, batch, dim, seqlen, dstate, beta, alpha, device='cuda', 
              dtype=torch.float32, is_complex=False, use_variable_B=False, 
              use_variable_C=False, use_skip=False, seed=42):
    """Test a single case"""
    print("\n" + "=" * 80)
    print(f"Test Case: {name}")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  batch={batch}, dim={dim}, seqlen={seqlen}, dstate={dstate}")
    print(f"  beta={beta}, alpha={alpha}")
    print(f"  dtype={dtype}, complex={is_complex}")
    print(f"  variable_B={use_variable_B}, variable_C={use_variable_C}, skip={use_skip}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate inputs with requires_grad for backward pass
    u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device, requires_grad=True)
    delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device, requires_grad=True)
    
    if is_complex:
        A = torch.randn(dim, dstate, dtype=torch.complex64, device=device, requires_grad=True)
    else:
        A = torch.randn(dim, dstate, dtype=dtype, device=device, requires_grad=True)
    
    if use_variable_B:
        B = torch.randn(batch, 1, dstate, seqlen, dtype=dtype, device=device, requires_grad=True)
    else:
        B = torch.randn(dim, dstate, dtype=dtype, device=device, requires_grad=True)
    
    if use_variable_C:
        C = torch.randn(batch, 1, dstate, seqlen, dtype=dtype, device=device, requires_grad=True)
    else:
        C = torch.randn(dim, dstate, dtype=dtype, device=device, requires_grad=True)
    
    D = None
    if use_skip:
        D = torch.randn(dim, dtype=dtype, device=device, requires_grad=True)
    
    # Create dummy gradient
    dout = torch.randn(batch, dim, seqlen, dtype=dtype, device=device)
    
    print("\n" + "-" * 80)
    print("Forward Pass Test")
    print("-" * 80)
    
    # CUDA forward
    u_cuda = u.detach().clone()
    delta_cuda = delta.detach().clone()
    A_cuda = A.detach().clone()
    B_cuda = B.detach().clone()
    C_cuda = C.detach().clone()
    D_cuda = D.detach().clone() if D is not None else None
    
    out_cuda = selective_scan_cuda.fwd(
        u_cuda, delta_cuda, A_cuda, B_cuda, C_cuda, D_cuda,
        None, None, False, beta, alpha
    )[0]
    
    print(f"CUDA output: shape={out_cuda.shape}, mean={out_cuda.mean():.6f}, std={out_cuda.std():.6f}")
    
    # PyTorch reference forward
    out_ref = selective_scan_ref(u, delta, A, B, C, D, None, None, False, False, beta, alpha)
    
    print(f"Reference output: shape={out_ref.shape}, mean={out_ref.mean():.6f}, std={out_ref.std():.6f}")
    
    # Compare forward
    forward_ok = compare_tensors(out_cuda.cpu(), out_ref.detach().cpu(), "Forward Output", 
                                  tol_abs=1e-3, tol_rel=5e-2)
    
    if not forward_ok:
        print("\n❌ Forward pass failed! Skipping backward test for this case.")
        return False
    
    print("\n" + "-" * 80)
    print("Backward Pass Test")
    print("-" * 80)
    
    # CUDA backward
    du_cuda, ddelta_cuda, dA_cuda, dB_cuda, dC_cuda, dD_cuda, ddelta_bias_cuda = selective_scan_cuda.bwd(
        u_cuda, delta_cuda, A_cuda, B_cuda, C_cuda, D_cuda,
        None, None, dout, None, None, None, False, False, beta, alpha
    )
    
    print(f"\nCUDA gradients:")
    print(f"  grad_u: mean={du_cuda.mean():.6e}, std={du_cuda.std():.6e}")
    print(f"  grad_delta: mean={ddelta_cuda.mean():.6e}, std={ddelta_cuda.std():.6e}")
    print(f"  grad_A: mean={dA_cuda.mean():.6e}, std={dA_cuda.std():.6e}")
    print(f"  grad_B: mean={dB_cuda.mean():.6e}, std={dB_cuda.std():.6e}")
    print(f"  grad_C: mean={dC_cuda.mean():.6e}, std={dC_cuda.std():.6e}")
    
    # PyTorch backward using autograd on the forward reference
    out_ref.backward(dout)
    
    print(f"\nPyTorch gradients:")
    print(f"  grad_u: mean={u.grad.mean():.6e}, std={u.grad.std():.6e}")
    print(f"  grad_delta: mean={delta.grad.mean():.6e}, std={delta.grad.std():.6e}")
    print(f"  grad_A: mean={A.grad.mean():.6e}, std={A.grad.std():.6e}")
    print(f"  grad_B: mean={B.grad.mean():.6e}, std={B.grad.std():.6e}")
    print(f"  grad_C: mean={C.grad.mean():.6e}, std={C.grad.std():.6e}")
    
    # Compare gradients (relaxed tolerances for NS)
    grads_ok = True
    grads_ok = grads_ok and compare_tensors(du_cuda.cpu(), u.grad.cpu(), "grad_u", 
                                            tol_abs=1e-3, tol_rel=5e-2, verbose=True)
    grads_ok = grads_ok and compare_tensors(ddelta_cuda.cpu(), delta.grad.cpu(), "grad_delta",
                                            tol_abs=1e-3, tol_rel=5e-2, verbose=True)
    grads_ok = grads_ok and compare_tensors(dA_cuda.cpu(), A.grad.cpu(), "grad_A",
                                            tol_abs=1e-3, tol_rel=5e-2, verbose=True)
    grads_ok = grads_ok and compare_tensors(dB_cuda.cpu(), B.grad.cpu(), "grad_B",
                                            tol_abs=1e-3, tol_rel=5e-2, verbose=True)
    grads_ok = grads_ok and compare_tensors(dC_cuda.cpu(), C.grad.cpu(), "grad_C",
                                            tol_abs=1e-3, tol_rel=5e-2, verbose=True)
    
    if D is not None:
        grads_ok = grads_ok and compare_tensors(dD_cuda.cpu(), D.grad.cpu(), "grad_D",
                                                tol_abs=1e-3, tol_rel=5e-2, verbose=True)
    
    return forward_ok and grads_ok


def main():
    """Run comprehensive test suite"""
    print("=" * 80)
    print("MuonMamba (Momentum + Newton-Schulz5) Comprehensive Test Suite")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("ERROR: CUDA not available!")
        return False
    
    results = []
    
    # Test 1: Basic MuonMamba (momentum with NS)
    results.append(test_case(
        "Basic MuonMamba", 
        batch=1, dim=8, seqlen=16, dstate=8,
        beta=0.9, alpha=1.0, is_complex=False
    ))
    
    # Test 2: Larger dimensions
    results.append(test_case(
        "Large Dimensions",
        batch=2, dim=16, seqlen=32, dstate=16,
        beta=0.9, alpha=1.0
    ))
    
    # Test 3: Tall matrix case (dim > dstate)
    results.append(test_case(
        "Tall Matrix (D > N)",
        batch=1, dim=16, seqlen=8, dstate=8,
        beta=0.9, alpha=1.0
    ))
    
    # Test 4: Fat matrix case (dim < dstate)
    results.append(test_case(
        "Fat Matrix (D < N)",
        batch=1, dim=4, seqlen=16, dstate=8,
        beta=0.9, alpha=1.0
    ))
    
    # Test 5: Variable B
    results.append(test_case(
        "Variable B",
        batch=2, dim=8, seqlen=16, dstate=8,
        beta=0.9, alpha=1.0, use_variable_B=True
    ))
    
    # Test 6: Variable C
    results.append(test_case(
        "Variable C",
        batch=2, dim=8, seqlen=16, dstate=8,
        beta=0.9, alpha=1.0, use_variable_C=True
    ))
    
    # Test 7: Variable B and C
    results.append(test_case(
        "Variable B and C",
        batch=1, dim=8, seqlen=16, dstate=8,
        beta=0.9, alpha=1.0, use_variable_B=True, use_variable_C=True
    ))
    
    # Test 8: With skip connection
    results.append(test_case(
        "With Skip Connection",
        batch=2, dim=8, seqlen=16, dstate=8,
        beta=0.9, alpha=1.0, use_skip=True
    ))
    
    # Test 9: Complex A
    results.append(test_case(
        "Complex A",
        batch=1, dim=8, seqlen=16, dstate=8,
        beta=0.9, alpha=1.0, is_complex=True
    ))
    
    # Test 10: Very long sequence
    results.append(test_case(
        "Long Sequence",
        batch=1, dim=8, seqlen=128, dstate=8,
        beta=0.9, alpha=1.0
    ))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if all(results):
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        for i, result in enumerate(results):
            status = "✅" if result else "❌"
            print(f"  Test {i+1}: {status}")
    
    print("=" * 80)
    
    return all(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

