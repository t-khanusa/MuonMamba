#!/usr/bin/env python3
"""
Comprehensive Newton-Schulz Backward Pass Test Suite
Tests many production configurations and compares CUDA with PyTorch reference
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path
from einops import rearrange

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    import selective_scan_cuda
except ImportError as e:
    print(f"ERROR: Cannot import selective_scan_cuda: {e}")
    print("Please make sure the CUDA extension is built.")
    sys.exit(1)

from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref, selective_scan_fn


def pytorch_ns_backward_ref(grad_output, G, alpha, delta_val, B_val, u_val, eps=1e-7):
    """
    PyTorch reference for NS 5-step backward with detached first 4 iterations
    Matches the CUDA implementation: first 4 steps detached, backward through 5th step only
    """
    a, b_coef, c = 3.4445, -4.7750, 2.0315
    
    # ===== PHASE 1: Recompute X_0 → X_4 (detached) =====
    with torch.no_grad():
        # Match official implementation
        X = G.bfloat16()
        norm = X.float().norm()
        X = X.float()
        X /= (norm + eps)
        
        # Transpose if tall matrix
        transposed = False
        if G.size(0) > G.size(1):
            X = X.T
            transposed = True
        
        # Run 4 iterations (detached)
        for step in range(4):
            A = X @ X.T
            A = A.bfloat16().float()
            A2 = A @ A
            A2 = A2.bfloat16().float()
            B_mat = b_coef * A + c * A2
            B_mat = B_mat.bfloat16().float()
            X = a * X + B_mat @ X
            X = X.bfloat16().float()
        
        X_4 = X.clone()
    
    # ===== PHASE 2: Backward through 5th iteration =====
    X_4_grad = X_4.clone().requires_grad_(True)
    
    # Forward 5th iteration with gradients
    A_4 = X_4_grad @ X_4_grad.T
    A_4 = A_4.bfloat16().float()
    A_4_sq = A_4 @ A_4
    A_4_sq = A_4_sq.bfloat16().float()
    B_4 = b_coef * A_4 + c * A_4_sq
    B_4 = B_4.bfloat16().float()
    X_5 = a * X_4_grad + B_4 @ X_4_grad
    X_5 = X_5.bfloat16().float()
    
    # Transpose back if needed
    if transposed:
        X_5 = X_5.T
    
    # Backward pass
    X_5.backward(grad_output)
    dX_4 = X_4_grad.grad.clone()
    
    # ===== Gradient through normalization and BF16 conversion =====
    with torch.no_grad():
        # If transposed, transpose gradient back
        if transposed:
            dX_4 = dX_4.T
            X_4_for_grad = X_4.T
        else:
            X_4_for_grad = X_4
        
        # Gradient through normalization: X_0 = G_bf16 / norm
        dnorm_contrib = (dX_4 * X_4_for_grad).sum()
        d_G_bf16 = (dX_4 - dnorm_contrib * X_4_for_grad) / norm
        
        # Straight-through for BF16: d(G) = d(G_bf16)
        d_G = d_G_bf16
        
        # ===== Gradient through G = alpha * delta * B * u =====
        grad_u = (alpha * delta_val.unsqueeze(1) * B_val * d_G).sum(dim=1)
        grad_delta = (alpha * B_val * u_val.unsqueeze(1) * d_G).sum(dim=1)
        grad_B = alpha * delta_val.unsqueeze(1) * u_val.unsqueeze(1) * d_G
    
    return grad_u, grad_delta, grad_B


def selective_scan_backward_ref(u, delta, A, B, C, D, dout, beta=0.0, alpha=1.0,
                                delta_bias=None, delta_softplus=False):
    """Manual backward pass reference matching CUDA implementation"""
    device = u.device
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    dout = dout.float()
    
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = delta.clamp(max=20.0)
        delta = torch.where(delta <= 20.0, torch.log1p(torch.exp(delta)), delta)
    
    batch, dim, seqlen = u.shape
    dstate = A.shape[-1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    is_complex = A.is_complex()
    use_newton_schulz = (beta != 0.0)
    
    if is_complex:
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
        A = A.float()
    
    h_dtype = torch.complex64 if is_complex else torch.float32
    
    # Forward pass to get states
    h = torch.zeros((batch, dim, dstate), dtype=h_dtype, device=device)
    v = torch.zeros((batch, dim, dstate), dtype=h_dtype, device=device)
    h_states = []
    v_states = []
    b_t_states = []
    
    B_const = None
    C_const = None
    if not is_variable_B:
        B_const = B
    if not is_variable_C:
        C_const = C
    
    n_groups = None
    group_size = None
    if is_variable_B:
        n_groups = B.shape[1]
        group_size = (dim + n_groups - 1) // n_groups
    
    # Forward pass
    for t in range(seqlen):
        # Compute b_t
        if not is_variable_B:
            b_t = alpha * (delta[:, :, t].unsqueeze(-1) * B_const * u[:, :, t].unsqueeze(-1))
            b_t = b_t.to(h_dtype)
        else:
            b_t = torch.zeros((batch, dim, dstate), dtype=h_dtype, device=device)
            for d in range(dim):
                group_id = min(d // group_size, n_groups - 1)
                B_gt = B[:, group_id, :, t]
                b_t[:, d, :] = alpha * delta[:, d, t].unsqueeze(-1).to(h_dtype) * B_gt * u[:, d, t].unsqueeze(-1).to(h_dtype)
        
        b_t_original = b_t.clone()
        
        # Apply NS if enabled (detached for first 4 steps in backward)
        if use_newton_schulz:
            b_t_ortho = torch.zeros_like(b_t)
            for b in range(batch):
                b_t_matrix = b_t[b]
                if is_complex:
                    b_t_real = torch.stack([b_t_matrix.real, b_t_matrix.imag], dim=-1)
                    b_t_flat = b_t_real.view(b_t_matrix.shape[0], -1)
                    b_t_ortho_flat = newtonschulz5_ref(b_t_flat, steps=5)
                    b_t_ortho_real = b_t_ortho_flat.view(b_t_matrix.shape[0], -1, 2)
                    b_t_ortho[b] = torch.complex(b_t_ortho_real[..., 0], b_t_ortho_real[..., 1])
                else:
                    b_t_ortho[b] = newtonschulz5_ref(b_t_matrix, steps=5)
            b_t = b_t_ortho
        else:
            b_t_original = b_t
        
        b_t_states.append(b_t_original)
        v = beta * v + b_t
        v_states.append(v.clone())
        
        if is_complex:
            delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1).to(delta.dtype) * A.unsqueeze(0))
        else:
            delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
        h = delta_A_t * h + v
        h_states.append(h.clone())
    
    # Backward pass - reverse through time
    du = torch.zeros_like(u)
    ddelta = torch.zeros_like(delta)
    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)
    dC = torch.zeros_like(C)
    dD = torch.zeros_like(D) if D is not None else None
    
    dh = torch.zeros_like(h)
    dv = torch.zeros_like(v)
    
    # Process in reverse order
    for t in range(seqlen - 1, -1, -1):
        h_t = h_states[t]
        v_t = v_states[t]
        b_t_original = b_t_states[t]
        
        # Gradient from output
        if not is_variable_C:
            if use_newton_schulz or beta != 1.0:
                dh_t_from_out = dout[:, :, t].unsqueeze(-1) * C_const.unsqueeze(0)
            else:
                if not is_variable_B:
                    BC = B_const * C_const
                    dh_t_from_out = dout[:, :, t].unsqueeze(-1) * BC.unsqueeze(0)
                else:
                    dh_t_from_out = dout[:, :, t].unsqueeze(-1) * C_const.unsqueeze(0)
        else:
            n_groups_C = C.shape[1]
            group_size_C = (dim + n_groups_C - 1) // n_groups_C
            dh_t_from_out = torch.zeros((batch, dim, dstate), dtype=h_dtype, device=device)
            for d in range(dim):
                group_id_C = min(d // group_size_C, n_groups_C - 1)
                C_gt = C[:, group_id_C, :, t]
                if use_newton_schulz or beta != 1.0:
                    dh_t_from_out[:, d, :] = dout[:, d, t].unsqueeze(-1) * C_gt
                else:
                    if not is_variable_B:
                        BC = B_const[d] * C_gt
                        dh_t_from_out[:, d, :] = dout[:, d, t].unsqueeze(-1) * BC
                    else:
                        dh_t_from_out[:, d, :] = dout[:, d, t].unsqueeze(-1) * C_gt
        
        dh = dh + dh_t_from_out
        dv_t = dh
        dv = dv + dv_t
        
        # For NS backward: gradient flows through NS backward kernel
        # For reference, compute NS backward if needed
        db_t = dv
        
        if use_newton_schulz:
            # Compute b_t from inputs
            if not is_variable_B:
                b_t_input = alpha * (delta[:, :, t].unsqueeze(-1) * B_const * u[:, :, t].unsqueeze(-1))
            else:
                b_t_input = torch.zeros((batch, dim, dstate), dtype=h_dtype, device=device)
                for d in range(dim):
                    group_id = min(d // group_size, n_groups - 1)
                    B_gt = B[:, group_id, :, t]
                    b_t_input[:, d, :] = alpha * delta[:, d, t].unsqueeze(-1).to(h_dtype) * B_gt * u[:, d, t].unsqueeze(-1).to(h_dtype)
            
            # Compute NS backward for each batch element
            for b in range(batch):
                if is_complex:
                    b_t_matrix = b_t_input[b]
                    b_t_real = torch.stack([b_t_matrix.real, b_t_matrix.imag], dim=-1)
                    b_t_flat = b_t_real.view(b_t_matrix.shape[0], -1)
                    grad_b_t_flat = db_t[b]
                    grad_b_t_real = grad_b_t_flat.view(b_t_matrix.shape[0], -1, 2)
                    grad_b_t_complex = torch.complex(grad_b_t_real[..., 0], grad_b_t_real[..., 1])
                    # Transpose to match expected shape for NS backward
                    grad_b_t_matrix = grad_b_t_complex.T if b_t_matrix.shape[0] > b_t_matrix.shape[1] else grad_b_t_complex
                    grad_b_t_flat = torch.stack([grad_b_t_matrix.real, grad_b_t_matrix.imag], dim=-1).view(b_t_matrix.shape[0], -1)
                    
                    grad_u_b, grad_delta_b, grad_B_b = pytorch_ns_backward_ref(
                        grad_b_t_flat, b_t_flat, alpha,
                        delta[b, :, t], B_const if not is_variable_B else B[b, :, :, t],
                        u[b, :, t]
                    )
                else:
                    b_t_matrix = b_t_input[b]
                    grad_b_t_matrix = db_t[b]
                    
                    grad_u_b, grad_delta_b, grad_B_b = pytorch_ns_backward_ref(
                        grad_b_t_matrix, b_t_matrix, alpha,
                        delta[b, :, t], B_const if not is_variable_B else B[b, :, :, t],
                        u[b, :, t]
                    )
                
                du[b, :, t] += grad_u_b
                ddelta[b, :, t] += grad_delta_b
                if not is_variable_B:
                    dB += grad_B_b.sum(dim=0) if grad_B_b.dim() > 2 else grad_B_b
                else:
                    for d in range(dim):
                        group_id = min(d // group_size, n_groups - 1)
                        dB[b, group_id, :, t] += grad_B_b[d]
        else:
            # Normal backward without NS
            if not is_variable_B:
                if is_complex:
                    ddelta_v = (db_t * alpha * B_const.unsqueeze(0) * u[:, :, t].unsqueeze(-1)).real.sum(dim=-1)
                    du_v = (db_t * alpha * delta[:, :, t].unsqueeze(-1) * B_const.unsqueeze(0)).real.sum(dim=-1)
                    dB_v = (db_t * alpha * delta[:, :, t].unsqueeze(-1) * u[:, :, t].unsqueeze(-1)).real
                    dB += dB_v.sum(dim=(0, 1))
                else:
                    ddelta_v = (db_t * alpha * B_const.unsqueeze(0) * u[:, :, t].unsqueeze(-1)).sum(dim=-1)
                    du_v = (db_t * alpha * delta[:, :, t].unsqueeze(-1) * B_const.unsqueeze(0)).sum(dim=-1)
                    dB_v = db_t * alpha * delta[:, :, t].unsqueeze(-1) * u[:, :, t].unsqueeze(-1)
                    dB += dB_v.sum(dim=(0, 1))
                ddelta[:, :, t] += ddelta_v
                du[:, :, t] += du_v
        
        dv = beta * dv
        
        # Gradient w.r.t. delta (through exp)
        h_t_minus_v_t = h_t - v_t
        if is_complex:
            ddelta_exp = (dh * torch.conj(A).unsqueeze(0) * torch.conj(h_t_minus_v_t)).real.sum(dim=-1)
        else:
            ddelta_exp = (dh * A.unsqueeze(0) * h_t_minus_v_t).sum(dim=-1)
        ddelta[:, :, t] += ddelta_exp
        
        # Gradient w.r.t. A
        if is_complex:
            dA_t = delta[:, :, t].unsqueeze(-1) * dh * torch.conj(h_t_minus_v_t)
            dA += dA_t.sum(dim=(0, 1))
        else:
            dA_t = delta[:, :, t].unsqueeze(-1) * dh * h_t_minus_v_t
            dA += dA_t.sum(dim=(0, 1))
        
        # Gradient w.r.t. C
        if not is_variable_C:
            if is_complex:
                dC += (dout[:, :, t].unsqueeze(-1) * torch.conj(h_t)).sum(dim=(0, 1))
            else:
                dC += (dout[:, :, t].unsqueeze(-1) * h_t).sum(dim=(0, 1))
        else:
            n_groups_C = C.shape[1]
            group_size_C = (dim + n_groups_C - 1) // n_groups_C
            for d in range(dim):
                group_id_C = min(d // group_size_C, n_groups_C - 1)
                if is_complex:
                    dC[:, group_id_C, :, t] += dout[:, d, t].unsqueeze(-1) * torch.conj(h_t[:, d, :])
                else:
                    dC[:, group_id_C, :, t] += dout[:, d, t].unsqueeze(-1) * h_t[:, d, :]
        
        # Gradient w.r.t. D
        if D is not None:
            dD += (dout[:, :, t] * u[:, :, t]).sum(dim=0)
        
        # Update dh for next iteration
        if t > 0:
            if is_complex:
                delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1).to(delta.dtype) * A.unsqueeze(0))
                dh = dh * torch.conj(delta_A_t)
            else:
                delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
                dh = dh * delta_A_t
    
    # Convert back to original dtype
    du = du.to(dtype_in)
    ddelta = ddelta.to(dtype_in)
    dA = dA.to(dtype_in)
    dB = dB.to(dtype_in)
    dC = dC.to(dtype_in)
    if dD is not None:
        dD = dD.to(dtype_in)
    
    return du, ddelta, dA, dB, dC, dD


def compare_gradients(grad_cuda, grad_ref, name, tol_abs=1e-3, tol_rel=1e-2, verbose=True):
    """Compare CUDA and reference gradients"""
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
    
    passed = (exceed_tol_ratio < 0.05 and
              max_rel_diff < tol_rel * 3.0 and
              not has_nan and not has_inf)
    
    if verbose:
        status = "✅" if passed else "❌"
        print(f"\n{status} {name}:")
        print(f"  Max abs diff: {max_abs_diff:.6e} (tol: {adaptive_tol_abs:.6e})")
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


def test_backward_case(name, batch, dim, seqlen, dstate, beta, alpha,
                      is_variable_B=False, is_variable_C=False,
                      use_d=False, dtype=torch.float32, seed=42,
                      tol_abs=1e-3, tol_rel=1e-2, scale_inputs=True):
    """Test a single backward pass configuration"""
    print("\n" + "=" * 80)
    print(f"Test: {name}")
    print("=" * 80)
    print(f"  Config: B={batch}, D={dim}, L={seqlen}, N={dstate}")
    print(f"  beta={beta}, alpha={alpha}")
    print(f"  variable_B={is_variable_B}, variable_C={is_variable_C}")
    print(f"  dtype={dtype}, D={use_d}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate inputs with smaller scale for stability
    scale = 0.1 if scale_inputs else 1.0
    u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * scale
    delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * scale
    delta_bias = None
    
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
    
    # Run CUDA forward + backward
    try:
        fwd_result = selective_scan_cuda.fwd(
            u, delta, A, B, C, D, None, delta_bias, False, beta, alpha
        )
        out_cuda = fwd_result[0]
        x_cuda = fwd_result[1]
        
        # Check for NaN/Inf in forward
        if torch.isnan(out_cuda).any() or torch.isinf(out_cuda).any():
            print(f"  ⚠️  Warning: NaN/Inf in CUDA forward output")
        
        bwd_result = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x_cuda, None, None,
            False, False, beta, alpha
        )
        du_cuda = bwd_result[0]
        ddelta_cuda = bwd_result[1]
        dA_cuda = bwd_result[2]
        dB_cuda = bwd_result[3]
        dC_cuda = bwd_result[4]
        dD_cuda = bwd_result[5] if len(bwd_result) > 5 else None
        
        # Debug: Print gradient statistics
        print(f"  CUDA grad stats:")
        print(f"    du: mean={du_cuda.abs().mean():.6e}, max={du_cuda.abs().max():.6e}")
        print(f"    ddelta: mean={ddelta_cuda.abs().mean():.6e}, max={ddelta_cuda.abs().max():.6e}")
        print(f"    dA: mean={dA_cuda.abs().mean():.6e}, max={dA_cuda.abs().max():.6e}")
        print(f"    dB: mean={dB_cuda.abs().mean():.6e}, max={dB_cuda.abs().max():.6e}")
        
    except Exception as e:
        print(f"\n❌ CUDA backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run PyTorch reference backward
    try:
        du_ref, ddelta_ref, dA_ref, dB_ref, dC_ref, dD_ref = selective_scan_backward_ref(
            u, delta, A, B, C, D, dout, beta=beta, alpha=alpha,
            delta_bias=delta_bias, delta_softplus=False
        )
    except Exception as e:
        print(f"\n❌ Reference backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare all gradients
    all_passed = True
    all_passed &= compare_gradients(du_cuda, du_ref, "du", tol_abs, tol_rel, verbose=True)
    all_passed &= compare_gradients(ddelta_cuda, ddelta_ref, "ddelta", tol_abs, tol_rel, verbose=True)
    all_passed &= compare_gradients(dA_cuda, dA_ref, "dA", tol_abs, tol_rel, verbose=True)
    all_passed &= compare_gradients(dB_cuda, dB_ref, "dB", tol_abs, tol_rel, verbose=True)
    all_passed &= compare_gradients(dC_cuda, dC_ref, "dC", tol_abs, tol_rel, verbose=True)
    if use_d:
        all_passed &= compare_gradients(dD_cuda, dD_ref, "dD", tol_abs, tol_rel, verbose=True)
    
    return all_passed


def main():
    """Run comprehensive backward test suite with many production cases"""
    print("=" * 80)
    print("Comprehensive NS Backward Pass Test Suite")
    print("CUDA vs PyTorch Reference - Production Configurations")
    print("=" * 80)
    
    results = []
    test_cases = []
    
    # ========== SMALL CONFIGURATIONS (Quick verification) ==========
    
    # Test 1: Basic momentum (const B, C)
    test_cases.append(("Small: Basic Momentum (const B, C)", {
        'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False,
        'tol_rel': 2e-2  # Slightly relaxed for small configs
    }))
    
    # Test 2: Momentum (var B, const C)
    test_cases.append(("Small: Momentum (var B, const C)", {
        'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': True, 'is_variable_C': False,
        'tol_rel': 2e-2
    }))
    
    # Test 3: Momentum (const B, var C)
    test_cases.append(("Small: Momentum (const B, var C)", {
        'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': True,
        'tol_rel': 2e-2
    }))
    
    # Test 4: Momentum (var B, var C)
    test_cases.append(("Small: Momentum (var B, var C)", {
        'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': True, 'is_variable_C': True,
        'tol_rel': 2e-2
    }))
    
    # ========== MEDIUM CONFIGURATIONS ==========
    
    # Test 5: Medium basic momentum
    test_cases.append(("Medium: Basic Momentum", {
        'batch': 4, 'dim': 32, 'seqlen': 128, 'dstate': 16,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False
    }))
    
    # Test 6: Medium with different beta
    test_cases.append(("Medium: High Beta (0.95)", {
        'batch': 4, 'dim': 32, 'seqlen': 128, 'dstate': 16,
        'beta': 0.95, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False
    }))
    
    # Test 7: Medium with different alpha
    test_cases.append(("Medium: Alpha 0.5", {
        'batch': 4, 'dim': 32, 'seqlen': 128, 'dstate': 16,
        'beta': 0.9, 'alpha': 0.5, 'is_variable_B': False, 'is_variable_C': False
    }))
    
    # Test 8: Medium variable B
    test_cases.append(("Medium: Variable B", {
        'batch': 4, 'dim': 32, 'seqlen': 128, 'dstate': 16,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': True, 'is_variable_C': False
    }))
    
    # ========== LARGE CONFIGURATIONS (Production-like) ==========
    
    # Test 9: Large basic momentum
    test_cases.append(("Large: Basic Momentum", {
        'batch': 8, 'dim': 64, 'seqlen': 256, 'dstate': 32,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False
    }))
    
    # Test 10: Large with skip connection
    test_cases.append(("Large: With Skip Connection", {
        'batch': 8, 'dim': 64, 'seqlen': 256, 'dstate': 32,
        'beta': 0.9, 'alpha': 1.0, 'use_d': True, 'is_variable_B': False, 'is_variable_C': False
    }))
    
    # ========== PRODUCTION CONFIGURATIONS ==========
    
    # Test 11: Production size - small batch
    test_cases.append(("Production: Small Batch", {
        'batch': 4, 'dim': 128, 'seqlen': 512, 'dstate': 64,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False,
        'tol_rel': 5e-2  # More relaxed for production sizes
    }))
    
    # Test 12: Production size - medium batch
    test_cases.append(("Production: Medium Batch", {
        'batch': 8, 'dim': 128, 'seqlen': 512, 'dstate': 64,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False,
        'tol_rel': 5e-2
    }))
    
    # Test 13: Production size - full batch
    test_cases.append(("Production: Full Batch", {
        'batch': 16, 'dim': 128, 'seqlen': 512, 'dstate': 64,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False,
        'tol_rel': 5e-2
    }))
    
    # Test 14: Production with variable B
    test_cases.append(("Production: Variable B", {
        'batch': 16, 'dim': 128, 'seqlen': 512, 'dstate': 64,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': True, 'is_variable_C': False,
        'tol_rel': 5e-2
    }))
    
    # Test 15: Production with different beta values
    test_cases.append(("Production: Beta 0.95", {
        'batch': 16, 'dim': 128, 'seqlen': 512, 'dstate': 64,
        'beta': 0.95, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False,
        'tol_rel': 5e-2
    }))
    
    # Test 16: Production with different alpha
    test_cases.append(("Production: Alpha 0.8", {
        'batch': 16, 'dim': 128, 'seqlen': 512, 'dstate': 64,
        'beta': 0.9, 'alpha': 0.8, 'is_variable_B': False, 'is_variable_C': False,
        'tol_rel': 5e-2
    }))
    
    # Test 17: Production with skip connection
    test_cases.append(("Production: With Skip Connection", {
        'batch': 16, 'dim': 128, 'seqlen': 512, 'dstate': 64,
        'beta': 0.9, 'alpha': 1.0, 'use_d': True, 'is_variable_B': False, 'is_variable_C': False,
        'tol_rel': 5e-2
    }))
    
    # Test 18: Production - Tall matrix case (D > N is handled internally)
    test_cases.append(("Production: Tall Matrix (D=128, N=64)", {
        'batch': 16, 'dim': 128, 'seqlen': 512, 'dstate': 64,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False,
        'tol_rel': 5e-2
    }))
    
    # Test 19: Edge case - Very high beta
    test_cases.append(("Edge Case: Very High Beta (0.99)", {
        'batch': 4, 'dim': 32, 'seqlen': 64, 'dstate': 16,
        'beta': 0.99, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False
    }))
    
    # Test 20: Edge case - Very low alpha
    test_cases.append(("Edge Case: Very Low Alpha (0.1)", {
        'batch': 4, 'dim': 32, 'seqlen': 64, 'dstate': 16,
        'beta': 0.9, 'alpha': 0.1, 'is_variable_B': False, 'is_variable_C': False
    }))
    
    # Run all tests
    print(f"\nRunning {len(test_cases)} test cases...\n")
    for i, (name, kwargs) in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] ", end="")
        passed = test_backward_case(name, **kwargs)
        results.append((name, passed))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for i, (name, passed) in enumerate(results, 1):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{i:2d}. {status} - {name}")
    
    print(f"\nResults: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")
    
    # Group by category
    print("\n" + "=" * 80)
    print("Results by Category")
    print("=" * 80)
    
    categories = {
        'Small': [r for r in results if 'Small' in r[0]],
        'Medium': [r for r in results if 'Medium' in r[0]],
        'Large': [r for r in results if 'Large' in r[0]],
        'Production': [r for r in results if 'Production' in r[0]],
        'Edge Case': [r for r in results if 'Edge Case' in r[0]],
    }
    
    for category, cat_results in categories.items():
        if cat_results:
            cat_passed = sum(1 for _, p in cat_results if p)
            print(f"{category}: {cat_passed}/{len(cat_results)} passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

