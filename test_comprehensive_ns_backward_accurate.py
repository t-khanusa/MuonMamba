#!/usr/bin/env python3
"""
Accurate PyTorch Reference for Newton-Schulz 5-Step Backward
EXACTLY matches CUDA implementation with detached first 4 steps, gradient only in last step
"""

import torch
import torch.nn.functional as F

def pytorch_ns_backward_ref_accurate(grad_output, G_input, alpha, delta_val, B_val, u_val, eps=1e-8):
    """
    Accurate PyTorch reference for NS 5-step backward with detached first 4 iterations
    
    Args:
        grad_output: [D, N] - gradient w.r.t. NS output (X_5)
        G_input: [D, N] - original input matrix (b_t = alpha * delta * B * u, before NS)
        alpha: scalar
        delta_val: [D] - delta values
        B_val: [D, N] - B matrix
        u_val: [D] - u values
        eps: epsilon (1e-8 to match CUDA)
    
    Returns:
        grad_u: [D] - gradient w.r.t. u
        grad_delta: [D] - gradient w.r.t. delta
        grad_B: [D, N] - gradient w.r.t. B
    """
    a, b_coef, c = 3.4445, -4.7750, 2.0315
    D, N = G_input.shape
    
    # ===== PHASE 1: Recompute X_0 → X_4 (Detached, 4 iterations) =====
    # This exactly matches CUDA lines 569-979
    with torch.no_grad():
        # Step 1: Compute b_t, convert to BF16, compute norm (CUDA lines 600-622)
        b_t_bf16 = G_input.bfloat16().float()  # Convert to BF16, store as float
        norm_sq = (b_t_bf16 ** 2).sum()
        norm = torch.sqrt(norm_sq + eps)  # eps = 1e-8
        
        # Step 2: Normalize to get X_0 (CUDA lines 625-645)
        X_0_fp32 = b_t_bf16 / norm
        X_0_bf16 = X_0_fp32.bfloat16().float()  # Round to BF16
        X = X_0_bf16.clone()
        
        # Determine transpose (CUDA line 554)
        transposed = (D > N)
        if transposed:
            X = X.T
        
        # Step 3: Run 4 NS iterations (CUDA lines 648-979)
        # Each iteration matches CUDA exactly with BF16 rounding
        for step in range(4):
            # Compute A = X @ X.T (round to BF16)
            A_fp32 = X @ X.T
            A_bf16 = A_fp32.bfloat16().float()
            
            # Compute A^2 (round to BF16)
            A2_fp32 = A_bf16 @ A_bf16
            A2_bf16 = A2_fp32.bfloat16().float()
            
            # Compute B = b*A + c*A^2 (round to BF16)
            B_fp32 = b_coef * A_bf16 + c * A2_bf16
            B_bf16 = B_fp32.bfloat16().float()
            
            # Compute X_new = a*X + B@X (round to BF16)
            X_new_fp32 = a * X + B_bf16 @ X
            X_new_bf16 = X_new_fp32.bfloat16().float()
            X = X_new_bf16
        
        # After 4 iterations, X is in transposed space if transposed
        # CUDA stores X_4 in transposed space, then transposes back at end
        # For backward, we need X_4 in the space it was after 4 iterations
        X_4_detached = X.clone()
        # Note: X is already in transposed space if transposed was True
        # We'll handle transpose back when needed
    
    # ===== PHASE 2: Backward through 5th iteration only =====
    # This matches CUDA lines 997-1138
    X_4 = X_4_detached.clone().requires_grad_(True)
    
    # Forward 5th iteration (with gradients enabled)
    # CRITICAL: B_4 must be computed from DETACHED X_4 (not from X_4 with gradients)
    # This matches CUDA: B_4 is treated as a constant matrix in backward
    X_4_detached_for_B = X_4.detach()  # Detach X_4 for computing B_4
    
    # Compute A_4 and B_4 from detached X_4 (no gradients through their computation)
    A_4_fp32 = X_4_detached_for_B @ X_4_detached_for_B.T
    A_4_bf16 = A_4_fp32.bfloat16().float()
    
    A_4_sq_fp32 = A_4_bf16 @ A_4_bf16
    A_4_sq_bf16 = A_4_sq_fp32.bfloat16().float()
    
    B_4_fp32 = b_coef * A_4_bf16 + c * A_4_sq_bf16
    B_4_bf16 = B_4_fp32.bfloat16().float()
    # B_4 is already detached (computed from detached X_4)
    
    # X_5 = a*X_4 + B_4@X_4 (X_4 has gradients, B_4 is detached constant)
    X_5_fp32 = a * X_4 + B_4_bf16 @ X_4
    X_5_bf16 = X_5_fp32.bfloat16().float()
    
    # Prepare grad_output - if transposed, X_5 is logically transposed too
    # CUDA: grad_output comes in as [D, N] always, but if transposed, X_5 is [N, D]
    # So we need to transpose grad_output to match X_5's orientation
    if transposed:
        # X_5 is [N, D] (logically [D, N] transposed), grad_output is [D, N]
        # For backward, grad_output needs to be [N, D]
        grad_output_T = grad_output.T
        X_5_for_backward = X_5_bf16  # Already in [N, D] space
    else:
        grad_output_T = grad_output  # [D, N]
        X_5_for_backward = X_5_bf16  # [D, N]
    
    # Backward pass through 5th iteration
    # dX_4 = a*dX_5 + B_4.T @ dX_5 (B_4 is detached)
    X_5_for_backward.backward(grad_output_T)
    dX_4 = X_4.grad.clone()
    
    # dX_4 is now in the same space as X_4 (transposed if X_4 was transposed)
    
    # ===== PHASE 3: Backward through normalization =====
    # This matches CUDA lines 1150-1254
    with torch.no_grad():
        # Recompute X_0 (normalized input) for dot product (CUDA lines 1180-1185)
        # X_0 is always in [D, N] space (original G_input orientation)
        b_t_bf16_recomp = G_input.bfloat16().float()
        X_0_fp32_recomp = b_t_bf16_recomp / norm
        X_0_bf16_recomp = X_0_fp32_recomp.bfloat16().float()
        
        # dX_4 is in the same space as X_4
        # If transposed, X_4 is [N, D], so dX_4 is [N, D]
        # X_0 is [D, N], so we need to transpose dX_4 to [D, N] to match
        if transposed:
            # dX_4 is [N, D], transpose to [D, N] to match X_0
            dX_4_in_DN_space = dX_4.T
        else:
            # dX_4 is [D, N], matches X_0
            dX_4_in_DN_space = dX_4
        
        # Compute dnorm = sum(dX_4 * X_0) (CUDA line 1192)
        # Both should be in [D, N] orientation now
        dnorm_from_loss = (dX_4_in_DN_space * X_0_bf16_recomp).sum()
        
        # Gradient through normalization: d(b_t_bf16) = (dX_4 - dnorm * X_0) / norm
        # This matches CUDA line 1301
        # Result is in [D, N] space
        d_b_t_bf16 = (dX_4_in_DN_space - dnorm_from_loss * X_0_bf16_recomp) / norm
        
        # Straight-through: d(G_input) = d(b_t_bf16) (CUDA treats BF16 conversion as straight-through)
        d_G_input = d_b_t_bf16
    
    # ===== PHASE 4: Compute gradients w.r.t. u, delta, B =====
    # This matches CUDA lines 1303-1323
    with torch.no_grad():
        # G_input = alpha * delta * B * u
        # where:
        #   delta_val: [D]
        #   B_val: [D, N]
        #   u_val: [D]
        #   G_input: [D, N]
        
        # ∂G/∂u = alpha * delta * B (sum over N dimension)
        grad_u = (alpha * delta_val.unsqueeze(1) * B_val * d_G_input).sum(dim=1)
        
        # ∂G/∂delta = alpha * B * u (sum over N dimension)
        grad_delta = (alpha * B_val * u_val.unsqueeze(1) * d_G_input).sum(dim=1)
        
        # ∂G/∂B = alpha * delta * u
        grad_B = alpha * delta_val.unsqueeze(1) * u_val.unsqueeze(1) * d_G_input
    
    return grad_u, grad_delta, grad_B


def selective_scan_backward_ref_accurate(u, delta, A, B, C, D, dout, beta=0.0, alpha=1.0,
                                        delta_bias=None, delta_softplus=False, verify_cuda=True):
    """
    Accurate PyTorch reference backward pass matching CUDA implementation exactly
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
        # For complex, NS is applied to real representation
        # But for now, let's handle it differently - need to check CUDA behavior
        raise NotImplementedError("Complex case needs special handling")
    
    B = B.float()
    C = C.float()
    A = A.float()
    
    # Forward pass to get states (same as before)
    h = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
    v = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
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
    
    # Import newtonschulz5_ref for forward pass
    from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref
    
    # Forward pass
    for t in range(seqlen):
        # Compute b_t
        if not is_variable_B:
            b_t = alpha * (delta[:, :, t].unsqueeze(-1) * B_const * u[:, :, t].unsqueeze(-1))
        else:
            b_t = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
            for d in range(dim):
                group_id = min(d // group_size, n_groups - 1)
                B_gt = B[:, group_id, :, t]
                b_t[:, d, :] = alpha * delta[:, d, t].unsqueeze(-1) * B_gt * u[:, d, t].unsqueeze(-1)
        
        b_t_original = b_t.clone()
        
        # Apply NS if enabled (forward pass - full 5 steps)
        if use_newton_schulz:
            b_t_ortho = torch.zeros_like(b_t)
            for b in range(batch):
                # NS operates per (batch, timestep) on [dim, dstate] matrix
                b_t_matrix = b_t[b]  # [dim, dstate]
                # Apply NS forward (5 steps, all differentiable for forward)
                b_t_ortho[b] = newtonschulz5_ref(b_t_matrix, steps=5)
            b_t = b_t_ortho
        else:
            b_t_original = b_t
        
        b_t_states.append(b_t_original)  # Store original for NS backward recomputation
        
        v = beta * v + b_t
        v_states.append(v.clone())
        
        delta_A_t = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
        h = delta_A_t * h + v
        h_states.append(h.clone())
    
    # Backward pass - reverse through time
    du = torch.zeros_like(u)
    ddelta = torch.zeros_like(delta)
    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)
    dC = torch.zeros_like(C)
    dD_grad = torch.zeros_like(D) if D is not None else None
    
    dh = torch.zeros_like(h)
    dv = torch.zeros_like(v)
    
    # CRITICAL: Initialize du with D*dout gradient (direct feedthrough from y = C*h + D*u)
    # This matches CUDA line 219: du_vals[i] = D_val * dout_vals[i]
    if D is not None:
        for t in range(seqlen):
            du[:, :, t] = D.unsqueeze(0) * dout[:, :, t]
    
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
            dh_t_from_out = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
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
        
        # CUDA does inclusive reverse scan on hidden states:
        # 1. Initialize: thread_reverse_data[i].y = dout[i] * C
        # 2. Inclusive reverse scan accumulates: dh[t] = dout[t]*C + exp(delta[t+1]*A) * dout[t+1]*C + ...
        # 
        # Since we iterate backward (t from seqlen-1 to 0):
        #   - At start of iteration t: dh contains accumulated gradient from t+1, t+2, ... (future)
        #   - Add local: dh = dout[t]*C + exp(delta[t+1]*A) * dh (propagate from future)
        #   - After computing dh[t], propagate it backward for next iteration using exp(delta[t]*A)
        
        # Add local contribution and propagate from future
        if t < seqlen - 1:
            # Propagate from future timestep: dh[t] = local + exp(delta[t+1]*A) * dh[t+1]
            delta_A_next = torch.exp(delta[:, :, t+1].unsqueeze(-1) * A.unsqueeze(0))
            dh = dh_t_from_out + delta_A_next * dh
        else:
            # Last timestep: no future, just local contribution
            dh = dh_t_from_out
        
        # Reverse scan for velocity:
        # CUDA: dv_reverse_data[i] = (beta, dx) where dx = dh[t]
        # After inclusive reverse scan with SSMScanOp:
        #   dv[t] = dh[t] + beta * dh[t+1] + beta^2 * dh[t+2] + ...
        # This is accumulated BACKWARD (from future to past)
        #
        # Since we iterate backward (t from seqlen-1 to 0):
        #   - At start of iteration t: dv contains accumulated gradient from t+1, t+2, ... (future)
        #   - The reverse scan formula: dv[t] = dh[t] + beta * dv[t+1]
        #   - But dv[t+1] already includes: dv[t+1] = dh[t+1] + beta * dv[t+2] = dh[t+1] + beta * (dh[t+2] + beta * dv[t+3]) = ...
        #   - So: dv_t = dh + beta * dv gives: dv[t] = dh[t] + beta * (dh[t+1] + beta * dv[t+2]) = dh[t] + beta * dh[t+1] + beta^2 * dv[t+2]
        #   - This correctly accumulates: dv[t] = dh[t] + beta * dh[t+1] + beta^2 * dh[t+2] + ...
        dv_t = dh + beta * dv
        
        # Update dv for next iteration
        # For iteration t-1, dv will contain dv[t] which has accumulated from t and all future
        dv = dv_t
        
        # Gradient w.r.t. b_t_ortho (output of NS)
        # This is the accumulated velocity gradient at this timestep
        # CUDA accumulates dv into grad_X_4_buffer, which is [batch, dim, seqlen, dstate]
        # For each (batch, dim, timestep), grad_X_4_buffer contains gradient w.r.t. b_t_ortho[dstate]
        # NS backward processes per (batch, timestep) on [dim, dstate] matrices
        db_t_ortho = dv_t
        
        # NS backward if enabled
        # CRITICAL: In CUDA, dv is accumulated into grad_X_4_buffer, then NS backward is called ONCE at the end
        # For accuracy, we'll also accumulate and call NS backward per timestep (equivalent mathematically)
        # But we need to make sure db_t_ortho matches what CUDA accumulates
        if use_newton_schulz:
            # Accumulate db_t_ortho into a buffer (similar to CUDA's grad_X_4_buffer)
            # For each (batch, dim, timestep), db_t_ortho[b, d, :] is [dstate] vector
            # This will be processed by NS backward per (batch, timestep) on [dim, dstate] matrices
            
            # For now, call NS backward per timestep (should be equivalent to calling once at end)
            # Recompute b_t from inputs for NS backward
            if not is_variable_B:
                b_t_input = alpha * (delta[:, :, t].unsqueeze(-1) * B_const * u[:, :, t].unsqueeze(-1))
            else:
                b_t_input = torch.zeros((batch, dim, dstate), dtype=torch.float32, device=device)
                for d in range(dim):
                    group_id = min(d // group_size, n_groups - 1)
                    B_gt = B[:, group_id, :, t]
                    b_t_input[:, d, :] = alpha * delta[:, d, t].unsqueeze(-1) * B_gt * u[:, d, t].unsqueeze(-1)
            
            # Apply NS backward for each batch element
            # db_t_ortho[b] is [dim, dstate] - gradient w.r.t. b_t_ortho for batch b at timestep t
            # This matches what CUDA accumulates: grad_X_4_buffer[batch, :, timestep, :]
            for b in range(batch):
                # Get B matrix for this (batch, timestep)
                if not is_variable_B:
                    B_matrix = B_const  # [dim, dstate]
                else:
                    # For variable B: construct [dim, dstate] matrix from groups
                    B_matrix = torch.zeros((dim, dstate), dtype=torch.float32, device=device)
                    for d in range(dim):
                        group_id = min(d // group_size, n_groups - 1)
                        B_matrix[d, :] = B[b, group_id, :, t]  # [dstate]
                
                # Call NS backward with db_t_ortho[b] which is [dim, dstate]
                grad_u_b, grad_delta_b, grad_B_b = pytorch_ns_backward_ref_accurate(
                    db_t_ortho[b],  # grad_output: [dim, dstate] - gradient w.r.t. NS output
                    b_t_input[b],   # G_input: [dim, dstate] - original b_t before NS
                    alpha,
                    delta[b, :, t],  # [dim]
                    B_matrix,  # [dim, dstate]
                    u[b, :, t]  # [dim]
                )
                
                # Accumulate gradients
                du[b, :, t] += grad_u_b
                ddelta[b, :, t] += grad_delta_b
                if not is_variable_B:
                    # grad_B_b is [dim, dstate], accumulate over batches and timesteps
                    dB += grad_B_b
                else:
                    # For variable B, accumulate per (batch, group, timestep)
                    for d in range(dim):
                        group_id = min(d // group_size, n_groups - 1)
                        dB[b, group_id, :, t] += grad_B_b[d, :]  # grad_B_b[d] is [dstate]
        else:
            # Normal backward without NS
            if not is_variable_B:
                ddelta_v = (db_t_ortho * alpha * B_const.unsqueeze(0) * u[:, :, t].unsqueeze(-1)).sum(dim=-1)
                du_v = (db_t_ortho * alpha * delta[:, :, t].unsqueeze(-1) * B_const.unsqueeze(0)).sum(dim=-1)
                dB_v = db_t_ortho * alpha * delta[:, :, t].unsqueeze(-1) * u[:, :, t].unsqueeze(-1)
                dB += dB_v.sum(dim=(0, 1))
                ddelta[:, :, t] += ddelta_v
                du[:, :, t] += du_v
            else:
                for d in range(dim):
                    group_id = min(d // group_size, n_groups - 1)
                    B_gt = B[:, group_id, :, t]
                    ddelta_v_d = (db_t_ortho[:, d, :] * alpha * B_gt * u[:, d, t].unsqueeze(-1)).sum()
                    du_v_d = (db_t_ortho[:, d, :] * alpha * delta[:, d, t].unsqueeze(-1) * B_gt).sum()
                    dB[:, group_id, :, t] += db_t_ortho[:, d, :] * alpha * delta[:, d, t].unsqueeze(-1) * u[:, d, t].unsqueeze(-1)
                    ddelta[:, d, t] += ddelta_v_d
                    du[:, d, t] += du_v_d
        
        # Note: dv is already updated above in the reverse scan, don't multiply by beta again
        
        # Gradient w.r.t. delta (through exp)
        h_t_minus_v_t = h_t - v_t
        ddelta_exp = (dh * A.unsqueeze(0) * h_t_minus_v_t).sum(dim=-1)
        ddelta[:, :, t] += ddelta_exp
        
        # Gradient w.r.t. A
        dA_t = delta[:, :, t].unsqueeze(-1) * dh * h_t_minus_v_t
        dA += dA_t.sum(dim=(0, 1))
        
        # Gradient w.r.t. C
        if not is_variable_C:
            dC += (dout[:, :, t].unsqueeze(-1) * h_t).sum(dim=(0, 1))
        else:
            n_groups_C = C.shape[1]
            group_size_C = (dim + n_groups_C - 1) // n_groups_C
            for d in range(dim):
                group_id_C = min(d // group_size_C, n_groups_C - 1)
                dC[:, group_id_C, :, t] += dout[:, d, t].unsqueeze(-1) * h_t[:, d, :]
        
        # Gradient w.r.t. D
        if D is not None:
            dD_grad += (dout[:, :, t] * u[:, :, t]).sum(dim=0)
        
        # Note: dh is already updated above in the reverse scan, no need to update again
    
    # Convert back to original dtype
    du = du.to(dtype_in)
    ddelta = ddelta.to(dtype_in)
    dA = dA.to(dtype_in)
    dB = dB.to(dtype_in)
    dC = dC.to(dtype_in)
    if dD_grad is not None:
        dD_grad = dD_grad.to(dtype_in)
    
    return du, ddelta, dA, dB, dC, dD_grad

