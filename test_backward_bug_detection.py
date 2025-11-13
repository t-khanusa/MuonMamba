#!/usr/bin/env python3
"""
Bug Detection Test for Backward Pass
Creates comprehensive test cases to identify exact issues in backward pass
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

from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref


def pytorch_ns_backward_detailed(grad_output, G, alpha, delta_val, B_val, u_val, eps=1e-7):
    """
    Detailed PyTorch reference for NS backward with detached first 4 iterations
    Matches CUDA: first 4 steps detached, backward through 5th step only
    """
    a, b_coef, c = 3.4445, -4.7750, 2.0315
    
    # ===== PHASE 1: Recompute X_0 → X_4 (detached) =====
    with torch.no_grad():
        # Match CUDA: convert to bfloat16 then back to float
        X = G.bfloat16()
        norm = X.float().norm() + eps
        X = X.float() / norm
        
        # Transpose if tall matrix
        transposed = False
        if G.size(0) > G.size(1):
            X = X.T
            transposed = True
        
        # Run 4 iterations (detached) - match CUDA exactly
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
        # d(G_bf16) = (dX_4 - <dX_4, X_4> * X_4) / norm
        dnorm_contrib = (dX_4 * X_4_for_grad).sum()
        d_G_bf16 = (dX_4 - dnorm_contrib * X_4_for_grad) / norm
        
        # Straight-through for BF16: d(G) = d(G_bf16)
        d_G = d_G_bf16
        
        # ===== Gradient through G = alpha * delta * B * u =====
        # G is [dim, dstate], delta_val is [dim], B_val is [dim, dstate], u_val is [dim]
        # G[b, d, n] = alpha * delta[d] * B[d, n] * u[d]
        grad_u = (alpha * delta_val.unsqueeze(1) * B_val * d_G).sum(dim=1)  # [dim]
        grad_delta = (alpha * B_val * u_val.unsqueeze(1) * d_G).sum(dim=1)  # [dim]
        grad_B = alpha * delta_val.unsqueeze(1) * u_val.unsqueeze(1) * d_G  # [dim, dstate]
    
    return grad_u, grad_delta, grad_B


def selective_scan_backward_ref_detailed(u, delta, A, B, C, D, dout, beta=0.0, alpha=1.0,
                                        delta_bias=None, delta_softplus=False):
    """
    Detailed manual backward pass reference matching CUDA implementation
    Handles NS backward correctly
    """
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
    b_t_states = []  # Store original b_t (before NS) for backward
    
    B_const = None
    C_const = None
    if not is_variable_B:
        B_const = B  # [dim, dstate]
    if not is_variable_C:
        C_const = C  # [dim, dstate]
    
    n_groups = None
    group_size = None
    if is_variable_B:
        n_groups = B.shape[1]
        group_size = (dim + n_groups - 1) // n_groups
    
    # Forward pass
    for t in range(seqlen):
        # Compute b_t = alpha * delta * B * u
        if not is_variable_B:
            b_t = alpha * (delta[:, :, t].unsqueeze(-1) * B_const * u[:, :, t].unsqueeze(-1))
            b_t = b_t.to(h_dtype)
        else:
            b_t = torch.zeros((batch, dim, dstate), dtype=h_dtype, device=device)
            for d in range(dim):
                group_id = min(d // group_size, n_groups - 1)
                B_gt = B[:, group_id, :, t]
                b_t[:, d, :] = alpha * delta[:, d, t].unsqueeze(-1).to(h_dtype) * B_gt * u[:, d, t].unsqueeze(-1).to(h_dtype)
        
        # Store original b_t (before NS)
        b_t_original = b_t.clone()
        
        # Apply NS if enabled
        if use_newton_schulz:
            b_t_ortho = torch.zeros_like(b_t)
            for b in range(batch):
                b_t_matrix = b_t[b]  # [dim, dstate]
                if is_complex:
                    b_t_real = torch.stack([b_t_matrix.real, b_t_matrix.imag], dim=-1)
                    b_t_flat = b_t_real.view(b_t_matrix.shape[0], -1)
                    b_t_ortho_flat = newtonschulz5_ref(b_t_flat, steps=5)
                    b_t_ortho_real = b_t_ortho_flat.view(b_t_matrix.shape[0], -1, 2)
                    b_t_ortho[b] = torch.complex(b_t_ortho_real[..., 0], b_t_ortho_real[..., 1])
                else:
                    b_t_ortho[b] = newtonschulz5_ref(b_t_matrix, steps=5)
            b_t = b_t_ortho
        
        b_t_states.append(b_t_original)  # Store original for NS backward
        
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
    
    # For NS backward: accumulate gradients w.r.t. b_t_ortho, then compute backward through NS
    db_t_ortho_accumulated = torch.zeros((batch, dim, seqlen, dstate), dtype=h_dtype, device=device)
    
    # Process in reverse order
    for t in range(seqlen - 1, -1, -1):
        h_t = h_states[t]
        v_t = v_states[t]
        b_t_original = b_t_states[t]
        
        # Gradient from output: y_t = C*h_t + D*u
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
        dv_t = dh  # Gradient from current hidden state
        
        # Add to accumulated velocity gradient
        dv = dv + dv_t
        
        # Reverse scan through velocity: v_t = β*v_{t-1} + b_t_ortho
        # So gradient w.r.t. b_t_ortho = dv (current accumulated gradient)
        db_t_ortho = dv  # Current accumulated gradient is gradient w.r.t. b_t_ortho
        db_t_ortho_accumulated[:, :, t, :] = db_t_ortho
        
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
        
        # Update dh for next iteration (reverse scan through hidden state)
        if t > 0:
            if is_complex:
                delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1).to(delta.dtype) * A.unsqueeze(0))
                dh = dh * torch.conj(delta_A_t)
            else:
                delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
                dh = dh * delta_A_t
    
    # Now compute NS backward for all timesteps
    if use_newton_schulz:
        for t in range(seqlen):
            db_t_ortho = db_t_ortho_accumulated[:, :, t, :]  # [batch, dim, dstate]
            b_t_original = b_t_states[t]
            
            for b in range(batch):
                for d in range(dim):
                    # Get gradients w.r.t. b_t_ortho for this batch, dim, timestep
                    grad_b_t_ortho = db_t_ortho[b, d, :]  # [dstate]
                    
                    if is_complex:
                        # For complex, convert to real representation
                        b_t_matrix = b_t_original[b, d, :]  # [dstate] complex
                        b_t_real = torch.stack([b_t_matrix.real, b_t_matrix.imag], dim=-1)  # [dstate, 2]
                        b_t_flat = b_t_real.view(-1)  # [2*dstate]
                        
                        grad_b_t_ortho_real = torch.stack([grad_b_t_ortho.real, grad_b_t_ortho.imag], dim=-1)  # [dstate, 2]
                        grad_b_t_ortho_flat = grad_b_t_ortho_real.view(-1)  # [2*dstate]
                        
                        # NS backward
                        if not is_variable_B:
                            delta_val = delta[b, d, t].item()
                            B_val_d = B_const[d, :]  # [dstate]
                            u_val = u[b, d, t].item()
                            
                            # Convert to real representation for NS
                            B_val_real = torch.stack([B_val_d.real, B_val_d.imag], dim=-1)  # [dstate, 2]
                            B_val_flat = B_val_real.view(-1)  # [2*dstate]
                            B_val_matrix = B_val_flat.view(1, -1)  # [1, 2*dstate]
                            
                            grad_u_bd, grad_delta_bd, grad_B_bd_flat = pytorch_ns_backward_detailed(
                                grad_b_t_ortho_flat.unsqueeze(0),  # [1, 2*dstate]
                                b_t_flat.unsqueeze(0),  # [1, 2*dstate]
                                alpha,
                                torch.tensor([delta_val], device=device),
                                B_val_matrix,  # [1, 2*dstate]
                                torch.tensor([u_val], device=device)
                            )
                            
                            du[b, d, t] += grad_u_bd[0]
                            ddelta[b, d, t] += grad_delta_bd[0]
                            
                            # Convert grad_B back to complex
                            grad_B_bd_flat = grad_B_bd_flat.view(1, -1)  # [1, 2*dstate]
                            grad_B_bd_real = grad_B_bd_flat.view(-1, 2)  # [dstate, 2]
                            grad_B_bd = torch.complex(grad_B_bd_real[:, 0], grad_B_bd_real[:, 1])  # [dstate]
                            dB[d, :] += grad_B_bd
                        else:
                            # Variable B case
                            group_id = min(d // group_size, n_groups - 1)
                            B_gt = B[b, group_id, :, t]  # [dstate]
                            # Similar processing for variable B...
                            pass  # Simplified for now
                    else:
                        # Real case
                        grad_b_t_ortho_vec = grad_b_t_ortho.unsqueeze(0)  # [1, dstate]
                        b_t_vec = b_t_original[b, d, :].unsqueeze(0)  # [1, dstate]
                        
                        if not is_variable_B:
                            delta_val = delta[b, d, t].item()
                            B_val = B_const[d, :]  # [dstate]
                            u_val = u[b, d, t].item()
                            
                            grad_u_bd, grad_delta_bd, grad_B_bd = pytorch_ns_backward_detailed(
                                grad_b_t_ortho_vec,
                                b_t_vec,
                                alpha,
                                torch.tensor([delta_val], device=device),
                                B_val.unsqueeze(0),  # [1, dstate]
                                torch.tensor([u_val], device=device)
                            )
                            
                            du[b, d, t] += grad_u_bd[0]
                            ddelta[b, d, t] += grad_delta_bd[0]
                            dB[d, :] += grad_B_bd[0]
                        else:
                            # Variable B case - similar processing
                            group_id = min(d // group_size, n_groups - 1)
                            B_gt = B[b, group_id, :, t]  # [dstate]
                            # Process similarly...
                            pass
    
    # Convert back to original dtype
    du = du.to(dtype_in)
    ddelta = ddelta.to(dtype_in)
    dA = dA.to(dtype_in)
    dB = dB.to(dtype_in)
    dC = dC.to(dtype_in)
    if dD is not None:
        dD = dD.to(dtype_in)
    
    return du, ddelta, dA, dB, dC, dD


def test_backward_simple():
    """Simple test case to identify bugs"""
    print("=" * 80)
    print("Simple Backward Test: Beta=0.9, Const B, Const C")
    print("=" * 80)
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch, dim, seqlen, dstate = 2, 4, 8, 4
    beta, alpha = 0.9, 1.0
    
    # Generate inputs
    u = torch.randn(batch, dim, seqlen, dtype=torch.float32, device=device) * 0.1
    delta = torch.randn(batch, dim, seqlen, dtype=torch.float32, device=device) * 0.1
    A = torch.randn(dim, dstate, dtype=torch.float32, device=device) * 0.01 - 2.0
    B = torch.randn(dim, dstate, dtype=torch.float32, device=device) * 0.1
    C = torch.randn(dim, dstate, dtype=torch.float32, device=device) * 0.1
    D = None
    dout = torch.randn(batch, dim, seqlen, dtype=torch.float32, device=device) * 0.1
    
    # CUDA forward + backward
    try:
        fwd_result = selective_scan_cuda.fwd(
            u, delta, A, B, C, D, None, None, False, beta, alpha
        )
        out_cuda = fwd_result[0]
        x_cuda = fwd_result[1]
        
        bwd_result = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, None, dout, x_cuda, None, None,
            False, False, beta, alpha
        )
        du_cuda, ddelta_cuda, dA_cuda, dB_cuda, dC_cuda, dD_cuda = bwd_result[:6]
    except Exception as e:
        print(f"❌ CUDA failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Reference backward
    try:
        du_ref, ddelta_ref, dA_ref, dB_ref, dC_ref, dD_ref = selective_scan_backward_ref_detailed(
            u, delta, A, B, C, D, dout, beta=beta, alpha=alpha
        )
    except Exception as e:
        print(f"❌ Reference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare gradients
    def compare_grad(name, cuda_grad, ref_grad):
        if cuda_grad.shape != ref_grad.shape:
            print(f"\n❌ {name}: Shape mismatch! CUDA: {cuda_grad.shape}, Ref: {ref_grad.shape}")
            return False
        
        cuda_flat = cuda_grad.flatten().float()
        ref_flat = ref_grad.flatten().float()
        
        abs_diff = (cuda_flat - ref_flat).abs()
        max_abs_diff = abs_diff.max().item()
        max_rel_diff = (abs_diff / (ref_flat.abs() + 1e-8)).max().item()
        
        print(f"\n{name}:")
        print(f"  Max abs diff: {max_abs_diff:.6e}")
        print(f"  Max rel diff: {max_rel_diff:.6e}")
        
        if max_rel_diff > 1e-2:
            worst_idx = abs_diff.argmax().item()
            print(f"  Worst at idx {worst_idx}: CUDA={cuda_flat[worst_idx]:.6e}, Ref={ref_flat[worst_idx]:.6e}")
            return False
        
        return True
    
    all_passed = True
    all_passed &= compare_grad("du", du_cuda, du_ref)
    all_passed &= compare_grad("ddelta", ddelta_cuda, ddelta_ref)
    all_passed &= compare_grad("dA", dA_cuda, dA_ref)
    all_passed &= compare_grad("dB", dB_cuda, dB_ref)
    all_passed &= compare_grad("dC", dC_cuda, dC_ref)
    
    if all_passed:
        print("\n✅ All gradients match!")
    else:
        print("\n❌ Gradient mismatches found!")
    
    return all_passed


if __name__ == '__main__':
    success = test_backward_simple()
    sys.exit(0 if success else 1)







