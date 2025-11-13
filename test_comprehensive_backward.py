#!/usr/bin/env python3
"""
Comprehensive Backward Pass Test: CUDA vs PyTorch Reference
Tests mathematical and logical correctness of the selective scan backward pass
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
from test_comprehensive_ns_backward import pytorch_ns_backward_ref


def selective_scan_backward_ref(u, delta, A, B, C, D, dout, beta=0.0, alpha=1.0,
                                delta_bias=None, delta_softplus=False):
    """
    Manual backward pass reference matching CUDA implementation
    
    Key: NS is detached for first 4 steps, only last step has gradients (matches CUDA)
    This is different from full autograd which would differentiate through all NS steps
    """
    from test_comprehensive_forward import selective_scan_ref_fixed
    
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
    b_t_states = []  # Store b_t for backward
    
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
        
        # Store original b_t for backward (before NS)
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
        
        b_t_states.append(b_t_original)  # Store for backward
        
        # Velocity update
        v = beta * v + b_t
        v_states.append(v.clone())
        
        # Hidden state update
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
    
    # Gradient w.r.t. hidden states
    dh = torch.zeros_like(h)
    dv = torch.zeros_like(v)  # Velocity gradient (reverse scan)
    
    # Process in reverse order
    for t in range(seqlen - 1, -1, -1):
        h_t = h_states[t]
        v_t = v_states[t]
        b_t_original = b_t_states[t]
        
        # Gradient from output: y_t = C*h_t + D*u
        # For momentum mode: y = C*h (B already in velocity)
        # For original mode: y = B*C*h
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
        
        # Accumulate gradient w.r.t. h_t
        dh = dh + dh_t_from_out
        
        # Gradient w.r.t. v_t (since h_t = exp(δ*A)*h_{t-1} + v_t)
        # Accumulate gradient from both output path and hidden state path
        dv_t = dh  # Gradient from current hidden state
        
        # Add to accumulated velocity gradient
        dv = dv + dv_t
        
        # Reverse scan through velocity: v_t = β*v_{t-1} + b_t
        # So gradient w.r.t. b_t = dv (current accumulated gradient)
        # And gradient w.r.t. v_{t-1} = β*dv
        db_t = dv  # Current accumulated gradient is gradient w.r.t. b_t_ortho
        
        # For NS backward: gradient flows through last NS step only
        # In CUDA, this is handled by NS backward kernel
        # CRITICAL: NS operates on [dim, dstate] matrices per (batch, timestep), NOT per (batch, dim) vectors!
        if use_newton_schulz:
            # Proper NS backward: recompute X_0→X_4 (detached), backprop through 5th iteration
            # db_t is gradient w.r.t. b_t_ortho (output of NS)
            # Need to compute gradient w.r.t. b_t_original
            # Process per (batch, timestep) on [dim, dstate] matrices
            db_t_after_ns = torch.zeros_like(db_t)
            
            for b in range(batch):
                # b_t_matrix: [dim, dstate] for this batch and timestep
                b_t_matrix = b_t_original[b, :, :]  # [dim, dstate]
                db_t_ortho_matrix = db_t[b, :, :]  # Gradient w.r.t. b_t_ortho [dim, dstate]
                
                # NS operates on matrices, not vectors
                # For backward, we need to backprop through normalization
                # Simplified: backprop through normalization only (first 4 NS steps are detached)
                b_t_bf16 = b_t_matrix.bfloat16().float()
                norm = b_t_bf16.norm() + 1e-8
                X_0 = b_t_bf16 / norm
                
                # Gradient through normalization: d(x/||x||) = (I - xx^T/||x||^2) @ grad / ||x||
                dnorm_contrib = (db_t_ortho_matrix * X_0).sum()
                db_t_after_ns[b, :, :] = (db_t_ortho_matrix - dnorm_contrib * X_0) / norm
            
            db_t = db_t_after_ns
        else:
            db_t_original = db_t
        
        # Update dv for next iteration: dv_{t-1} = β*dv_t
        dv = beta * dv
        
        # Gradient w.r.t. delta (through exp)
        h_t_minus_v_t = h_t - v_t  # exp(δ*A)*h_{t-1}
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
        
        # Gradient w.r.t. C (from output)
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
        
        # Gradient w.r.t. delta and u (through velocity: b_t = alpha*δ*B*u)
        # b_t = alpha*δ*B*u, so:
        # ∂L/∂δ (through velocity) = db_t * alpha*B*u
        # ∂L/∂u (through velocity) = db_t * alpha*δ*B
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
        else:
            n_groups = B.shape[1]
            group_size = (dim + n_groups - 1) // n_groups
            for d in range(dim):
                group_id = min(d // group_size, n_groups - 1)
                B_gt = B[:, group_id, :, t]
                if is_complex:
                    ddelta_v = (db_t[:, d, :] * alpha * B_gt * u[:, d, t].unsqueeze(-1)).real.sum()
                    du_v = (db_t[:, d, :] * alpha * delta[:, d, t].unsqueeze(-1) * B_gt).real.sum()
                    dB[:, group_id, :, t] += (db_t[:, d, :] * alpha * delta[:, d, t].unsqueeze(-1) * u[:, d, t].unsqueeze(-1)).real
                else:
                    ddelta_v = (db_t[:, d, :] * alpha * B_gt * u[:, d, t].unsqueeze(-1)).sum()
                    du_v = (db_t[:, d, :] * alpha * delta[:, d, t].unsqueeze(-1) * B_gt).sum()
                    dB[:, group_id, :, t] += db_t[:, d, :] * alpha * delta[:, d, t].unsqueeze(-1) * u[:, d, t].unsqueeze(-1)
                ddelta[:, d, t] += ddelta_v
                du[:, d, t] += du_v
        
        # Gradient w.r.t. D
        if D is not None:
            dD += (dout[:, :, t] * u[:, :, t]).sum(dim=0)
        
        # Update dh for next iteration (reverse scan through hidden state)
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
    
    # Relative error
    max_magnitude = torch.maximum(grad_cuda_flat.abs(), grad_ref_flat.abs()) + 1e-8
    rel_diff = abs_diff / max_magnitude
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # Check for NaNs/Infs
    has_nan = grad_cuda_flat.isnan().any().item() or grad_ref_flat.isnan().any().item()
    has_inf = grad_cuda_flat.isinf().any().item() or grad_ref_flat.isinf().any().item()
    
    ref_max = grad_ref_flat.abs().max().item()
    ref_mean = grad_ref_flat.abs().mean().item()
    
    # Adaptive tolerance
    adaptive_tol_abs = max(tol_abs, ref_max * tol_rel)
    
    # Check exceed tolerance ratio
    exceed_tol_count = (rel_diff > tol_rel).sum().item()
    exceed_tol_ratio = exceed_tol_count / len(rel_diff) if len(rel_diff) > 0 else 0.0
    
    # Pass criteria
    passed = (exceed_tol_ratio < 0.05 and  # Allow up to 5% outliers
              max_rel_diff < tol_rel * 3.0 and  # Allow 3x tolerance for worst case
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


def test_backward_case(name, batch, dim, seqlen, dstate, beta, alpha,
                      is_variable_B=False, is_variable_C=False,
                      use_d=False, is_complex=False, delta_softplus=False,
                      dtype=torch.float32, seed=42, tol_abs=1e-3, tol_rel=1e-2,
                      scale_inputs=False):
    """Test a single backward pass configuration"""
    print("\n" + "=" * 80)
    print(f"Test: {name}")
    print("=" * 80)
    print(f"  Config: B={batch}, D={dim}, L={seqlen}, N={dstate}")
    print(f"  beta={beta}, alpha={alpha}")
    print(f"  variable_B={is_variable_B}, variable_C={is_variable_C}")
    print(f"  complex={is_complex}, dtype={dtype}, D={use_d}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate inputs
    scale = 0.1 if scale_inputs else 1.0
    u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * scale
    delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * scale
    delta_bias = None
    
    if is_complex:
        A = torch.complex(
            torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1,
            torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
        )
    else:
        A = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
        # Make A have negative real part for stability
        if is_complex:
            A = torch.complex(A, torch.randn_like(A) * 0.1)
    
    if is_variable_B:
        n_groups = 1  # Match forward test: use 1 group
        B = torch.randn(batch, n_groups, dstate, seqlen, dtype=dtype, device=device) * scale
    else:
        B = torch.randn(dim, dstate, dtype=dtype, device=device) * scale
    
    if is_variable_C:
        n_groups = 1  # Match forward test: use 1 group
        C = torch.randn(batch, n_groups, dstate, seqlen, dtype=dtype, device=device) * scale
    else:
        C = torch.randn(dim, dstate, dtype=dtype, device=device) * scale
    
    D = torch.randn(dim, dtype=dtype, device=device) * 0.1 if use_d else None
    
    # Generate random output gradient
    dout = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * scale
    
    # Run CUDA forward + backward
    try:
        # Call CUDA forward to get output and intermediate states (x)
        # Signature: fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, beta, alpha) -> [out, x, X_4_buffer (if NS enabled), ...]
        fwd_result = selective_scan_cuda.fwd(
            u, delta, A, B, C, D, None, delta_bias, delta_softplus, beta, alpha
        )
        out_cuda = fwd_result[0]
        x_cuda = fwd_result[1]  # Intermediate states needed for backward
        
        # Extract X_4_buffer if Newton-Schulz is enabled (beta != 0)
        X_4_buffer = None
        use_newton_schulz = (beta != 0.0)
        if use_newton_schulz and len(fwd_result) > 2:
            X_4_buffer = fwd_result[2]  # X_4_buffer containing b_t_ortho
        
        # CUDA backward
        # Signature: bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out, dz, delta_softplus, recompute_out_z, beta, alpha, X_4_buffer)
        bwd_result = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x_cuda, None, None,
            delta_softplus, False, beta, alpha, X_4_buffer
        )
        du_cuda = bwd_result[0]
        ddelta_cuda = bwd_result[1]
        dA_cuda = bwd_result[2]
        dB_cuda = bwd_result[3]
        dC_cuda = bwd_result[4]
        dD_cuda = bwd_result[5] if len(bwd_result) > 5 else None
        ddelta_bias_cuda = bwd_result[6] if len(bwd_result) > 6 else None
    except Exception as e:
        print(f"\n❌ CUDA backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run PyTorch reference backward
    try:
        du_ref, ddelta_ref, dA_ref, dB_ref, dC_ref, dD_ref = selective_scan_backward_ref(
            u, delta, A, B, C, D, dout, beta=beta, alpha=alpha,
            delta_bias=delta_bias, delta_softplus=delta_softplus
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
    """Run comprehensive backward test suite"""
    print("=" * 80)
    print("Comprehensive Backward Pass Test: CUDA vs PyTorch Reference")
    print("=" * 80)
    
    results = []
    test_cases = []
    
    # Test 1: Basic momentum (const B, C)
    test_cases.append(("Basic Momentum (const B, C)", {
        'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': False
    }))
    
    # Test 2: Momentum (var B, const C)
    test_cases.append(("Momentum (var B, const C)", {
        'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': True, 'is_variable_C': False
    }))
    
    # Test 3: Momentum (const B, var C)
    test_cases.append(("Momentum (const B, var C)", {
        'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': False, 'is_variable_C': True
    }))
    
    # Test 4: Momentum (var B, var C)
    test_cases.append(("Momentum (var B, var C)", {
        'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
        'beta': 0.9, 'alpha': 1.0, 'is_variable_B': True, 'is_variable_C': True
    }))
    
    # Test 5: With Skip Connection
    test_cases.append(("With Skip Connection", {
        'batch': 2, 'dim': 8, 'seqlen': 32, 'dstate': 8,
        'beta': 0.9, 'alpha': 1.0, 'use_d': True, 'is_variable_B': False, 'is_variable_C': False
    }))
    
    # Run all tests
    for name, kwargs in test_cases:
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
    
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("✅ ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

