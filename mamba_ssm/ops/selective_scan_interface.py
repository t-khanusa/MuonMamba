# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from mamba_ssm.utils.torch import custom_bwd, custom_fwd

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.cpp_functions import causal_conv1d_fwd_function, causal_conv1d_bwd_function, causal_conv1d_update_function
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None
    causal_conv1d_update_function = None

from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd

import selective_scan_cuda


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, beta=None, alpha=None):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        print(f"Hiii, I'm here, in SelectiveScanFn.forward")
        # Set default values for momentum parameters
        if beta is None:
            beta = 0.0  # No momentum by default
        if alpha is None:
            alpha = 1.0
        
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, 
                                                 float(beta), float(alpha))
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        ctx.beta = beta
        ctx.alpha = alpha
        ctx.use_newton_schulz = (beta != 0.0)
        
        # Extract states from x tensor
        # x has shape (batch, dim, n_chunks, dstate * 4) of floats
        # This represents (batch, dim, n_chunks, dstate * 2) of float2 values
        # Reshape to separate float2 components (a, b)
        batch, dim, n_chunks, _ = x.shape
        dstate = A.shape[-1]
        x_reshaped = x.view(batch, dim, n_chunks, dstate * 2, 2)  # (batch, dim, n_chunks, dstate*2, 2)
        
        # Take the 'b' component (index 1) which contains the state values
        states = x_reshaped[:, :, -1, :, 1]  # (batch, dim, dstate * 2)
        
        # Even indices: velocity states, Odd indices: hidden states
        last_velocity = states[:, :, 0::2]  # (batch, dim, dstate)
        last_state = states[:, :, 1::2]  # (batch, dim, dstate)
        
        # Extract X_4_buffer if Newton-Schulz was used
        X_4_buffer = None
        if ctx.use_newton_schulz:
            # X_4_buffer is the last item in rest (after out_z if has_z)
            if ctx.has_z:
                out_z = rest[0]
                X_4_buffer = rest[1] if len(rest) > 1 else None
            else:
                X_4_buffer = rest[0] if len(rest) > 0 else None
        elif ctx.has_z:
            out_z = rest[0]
        
        if not ctx.has_z:
            if ctx.use_newton_schulz:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, out, x, X_4_buffer)
            else:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, out, x)
            return out if not return_last_state else (out, last_state, last_velocity)
        else:
            if ctx.use_newton_schulz:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, z, out, x, X_4_buffer)
            else:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, z, out, x)
            return (out, out_z) if not return_last_state else (out, out_z, last_state, last_velocity)

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, *rest = ctx.saved_tensors
        
        # Extract z, out, x, and X_4_buffer from rest
        # Saved order depends on ctx.has_z and ctx.use_newton_schulz:
        # no z, no NS: (out, x)
        # no z, with NS: (out, x, X_4_buffer)
        # with z, no NS: (z, out, x)
        # with z, with NS: (z, out, x, X_4_buffer)
        
        z = None
        out = None
        x = None
        X_4_buffer = None
        
        if ctx.has_z:
            z = rest.pop(0)
        
        out = rest.pop(0)  # out is always after z (if present)
        x = rest.pop(0)    # x is always after out
        
        if ctx.use_newton_schulz:
            X_4_buffer = rest.pop(0) if len(rest) > 0 else None
        
        dout_z = None
        if ctx.has_z:
            dout_z = args[0]
            dout, dout_z = dout.contiguous(), dout_z.contiguous()
        
        # Pass X_4_buffer to backward (it will be used to load b_t_ortho)
        # C++ returns: [du, ddelta, dA, dB, dC, dD, ddelta_bias] + optional [dz] if has_z
        bwd_results = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, dout_z, ctx.delta_softplus, False, ctx.beta, ctx.alpha, X_4_buffer
        )
        
        # Unpack based on whether z was passed
        if ctx.has_z:
            # 8 values: du, ddelta, dA, dB, dC, dD, ddelta_bias, dz
            du, ddelta, dA, dB, dC, dD, ddelta_bias, dz_result = bwd_results
        else:
            # 7 values: du, ddelta, dA, dB, dC, dD, ddelta_bias
            du, ddelta, dA, dB, dC, dD, ddelta_bias = bwd_results
            dz_result = None
        
        dB = dB.squeeze(1) if hasattr(ctx, "squeeze_B") and ctx.squeeze_B else dB
        dC = dC.squeeze(1) if hasattr(ctx, "squeeze_C") and ctx.squeeze_C else dC
        
        # Backward return order must match forward input order:
        # forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, beta=None, alpha=None)
        # So backward returns: [du, ddelta, dA, dB, dC, dD, dz, ddelta_bias, None, None, None, None, None]
        dD_result = dD if D is not None else None
        result = [du, ddelta, dA, dB, dC, dD_result, dz_result, ddelta_bias, None, None, None, None, None]
        return tuple(result)


def rms_norm_forward(
    x,
    weight,
    bias,
    eps=1e-6,
    is_rms_norm=True,
):
    # x (b l) d
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = _layer_norm_fwd(
        x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm
    )[0]
    # y (b l) d
    return y


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False, beta=None, alpha=None):
    """if return_last_state is True, returns (out, last_state, last_velocity)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    beta: momentum decay parameter (scalar)
    alpha: momentum scale parameter (scalar)
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, beta, alpha)


def newtonschulz5_ref(G, steps=5, eps=1e-7):
    """
    Official Newton-Schulz 5-step orthogonalization
    G: input matrix [D, N]
    Returns: orthogonalized matrix [D, N]
    
    This is the OFFICIAL implementation from the Muon optimizer paper.
    Operates directly on bfloat16 tensors (PyTorch handles precision naturally).
    """
    assert G.ndim == 2, "Input must be 2D matrix"
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Convert to bfloat16 and normalize
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    
    # Handle tall matrices: transpose to make it fat (rows <= cols)
    if G.size(0) > G.size(1):
        X = X.T
    
    # 5 Newton-Schulz iterations (operates on bfloat16 tensors)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    # Transpose back if needed
    if G.size(0) > G.size(1):
        X = X.T
    
    return X


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False, beta=0.0, alpha=1.0):
    """
    MuonMamba reference implementation (Momentum + Newton-Schulz5)
    This implements the EXACT forward pass logic matching the CUDA kernel
    
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    beta: momentum decay (scalar)
    alpha: momentum scale (scalar)

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    last_velocity (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    is_complex = A.is_complex()
    
    # Convert complex
    if is_complex:
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    
    h = torch.zeros((batch, dim, dstate), dtype=A.dtype, device=A.device)  # Hidden state
    v = torch.zeros((batch, dim, dstate), dtype=A.dtype, device=A.device)  # Velocity state
    ys = []
    
    # Determine if we use Newton-Schulz
    use_newton_schulz = (beta != 0.0)
    
    for i in range(u.shape[2]):
        # Compute b_t = alpha * delta * B * u for this timestep
        if not is_variable_B:
            # B shape: [dim, dstate]
            b_t = alpha * delta[:, :, i].unsqueeze(-1) * B * u[:, :, i].unsqueeze(-1)
            # b_t shape: [batch, dim, dstate]
        else:
            # B shape: [batch, n_groups, dstate, seqlen]
            n_groups = B.shape[1]
            group_size = dim // n_groups
            b_t = torch.zeros((batch, dim, dstate), dtype=A.dtype, device=A.device)
            for g in range(n_groups):
                d_start = g * group_size
                d_end = min(d_start + group_size, dim)
                B_gt = B[:, g, :, i]  # [batch, dstate]
                for d in range(d_start, d_end):
                    b_t[:, d, :] = alpha * delta[:, d, i].unsqueeze(-1) * B_gt * u[:, d, i].unsqueeze(-1)
        
        # Apply Newton-Schulz orthogonalization
        if use_newton_schulz:
            b_t_ortho = torch.zeros_like(b_t)
            for b in range(batch):
                # Apply NS to [dim, dstate] matrix for each (batch, timestep)
                b_t_bf16 = newtonschulz5_ref(b_t[b], steps=5)
                b_t_ortho[b] = b_t_bf16
            b_t = b_t_ortho
        
        # Momentum: v_t = beta * v_{t-1} + b_t_ortho
        v = beta * v + b_t
        
        # Hidden state: h_t = exp(delta * A) * h_{t-1} + v_t
        delta_A_t = torch.exp(torch.einsum('bd,dn->bdn', delta[:, :, i], A))
        h = delta_A_t * h + v
        
        # Output: y_t = C @ h_t
        if not is_variable_C:
            if is_complex:
                y = (torch.einsum('bdn,dn->bd', h, C) * 2).real
            else:
                y = torch.einsum('bdn,dn->bd', h, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', h, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', h, C[:, :, :, i])
            if is_complex:
                y = (y * 2).real
        
        # Skip connection
        if D is not None:
            y = y + u[:, :, i] * D
        
        ys.append(y)
    
    out = torch.stack(ys, dim=2)  # [batch, dim, L]
    
    # Apply gating if provided
    if z is not None:
        out = out * F.silu(z)
    
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, h, v)


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight=None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_fwd_function(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
            
        if b_rms_weight is not None:
            B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if c_rms_weight is not None:
            C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if dt_rms_weight is not None:
            delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
        
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_fwd_function(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
            if dt_rms_weight is not None:
                delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
                delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
                delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
            if b_rms_weight is not None:
                # Recompute & RMSNorm B
                B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                B = rms_norm_forward(
                    B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            if c_rms_weight is not None:
                # Recompute & RMSNorm C
                C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                C = rms_norm_forward(
                    C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                # 6-None are delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
                dB_proj_bias, dC_proj_bias, None, None, None, None, None, None)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight= None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps)


def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


class MuonMambaInnerFn(torch.autograd.Function):
    """
    MuonMambaInnerFn: Fused kernel for MuonMamba with Momentum + Newton-Schulz5
    Based on MambaInnerFn but adds support for beta (momentum decay) and alpha (momentum scaling)
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, 
                b_rms_weight=None, c_rms_weight=None, dt_rms_weight=None, b_c_dt_rms_eps=1e-6,
                beta=0.0, alpha=1.0):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_fwd_function(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
            
        # Apply RMS normalization with learnable weights (same as MambaInnerFn)
        if b_rms_weight is not None:
            B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if c_rms_weight is not None:
            C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if dt_rms_weight is not None:
            delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
        
        # CRITICAL: Call selective_scan_cuda.fwd with beta and alpha for momentum support
        # This enables Newton-Schulz5 orthogonalization when beta > 0
        # When beta > 0, the return signature may include X_4_buffer for backward pass
        use_newton_schulz = (beta != 0.0)
        if use_newton_schulz:
            # With momentum, may return additional values (X_4_buffer)
            result = selective_scan_cuda.fwd(
                conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus,
                float(beta), float(alpha)
            )
            # Unpack: could be (out, scan_intermediates, out_z, X_4_buffer) or (out, scan_intermediates, out_z)
            if len(result) == 4:
                out, scan_intermediates, out_z, X_4_buffer = result
            else:
                out, scan_intermediates, out_z = result
                X_4_buffer = None
        else:
            # Standard Mamba: returns (out, scan_intermediates, out_z)
            out, scan_intermediates, out_z = selective_scan_cuda.fwd(
                conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus,
                float(beta), float(alpha)
            )
            X_4_buffer = None
        
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
        ctx.beta = beta
        ctx.alpha = alpha
        ctx.use_newton_schulz = use_newton_schulz
        
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        
        # Save tensors for backward pass (matching MambaInnerFn structure)
        # Order: [xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
        #         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, 
        #         b_rms_weight, c_rms_weight, dt_rms_weight, out, X_4_buffer?]
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out, X_4_buffer if use_newton_schulz else None)
        
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        
        # Unpack saved tensors (matching MambaInnerFn structure)
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out, X_4_buffer) = ctx.saved_tensors
        
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # Ensure z is contiguous (chunk creates views that might not be contiguous)
        if z.stride(-1) != 1:
            z = z.contiguous()
        
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_fwd_function(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
            # if dt_rms_weight is not None:
            #     print("Recompute & RMSNorm delta")
            #     delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            #     delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
            #     delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
            # if b_rms_weight is not None:
            #     # Recompute & RMSNorm B
            #     print("Recompute & RMSNorm B")
            #     B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            #     B = rms_norm_forward(
            #         B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps
            #     )
            #     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            # if c_rms_weight is not None:
            #     # Recompute & RMSNorm C
            #     print("Recompute & RMSNorm C")
            #     C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            #     C = rms_norm_forward(
            #         C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps
            #     )
            #     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        
        # Ensure all input tensors are contiguous before calling CUDA kernel
        # (required by CUDA kernels for performance and correctness)
        if conv1d_out is not None and conv1d_out.stride(-1) != 1:
            conv1d_out = conv1d_out.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if A.stride(-1) != 1:
            A = A.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if dout_y.stride(-1) != 1:
            dout_y = dout_y.contiguous()
        if isinstance(scan_intermediates, torch.Tensor) and scan_intermediates.stride(-1) != 1:
            scan_intermediates = scan_intermediates.contiguous()
        if out.stride(-1) != 1:
            out = out.contiguous()
        if dz.stride(-1) != 1:
            dz = dz.contiguous()
        # Ensure X_4_buffer is contiguous if it's a tensor (needed for momentum backward pass)
        if X_4_buffer is not None and isinstance(X_4_buffer, torch.Tensor) and X_4_buffer.stride(-1) != 1:
            X_4_buffer = X_4_buffer.contiguous()
        
        # CRITICAL: Call selective_scan_cuda.bwd with beta, alpha, and X_4_buffer for momentum support
        # When recompute_out_z=True, bwd always returns out_z as the last value (9 values total)
        # Signature: bwd(..., recompute_out_z, beta, alpha, X_4_buffer)
        bwd_results = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus, True,  # recompute_out_z
            ctx.beta, ctx.alpha, X_4_buffer
        )
        
        # Unpack backward results: always 9 values when recompute_out_z=True
        # [dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z]
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz_result, out_z = bwd_results
        
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                # 6-None are delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
                dB_proj_bias, dC_proj_bias, None, None, None, None, None, None,
                # beta and alpha are not learnable (buffers), so gradients are None
                None, None)


def muon_mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, 
    b_rms_weight=None, c_rms_weight=None, dt_rms_weight=None, b_c_dt_rms_eps=1e-6,
    beta=0.0, alpha=1.0
):
    """
    MuonMambaInnerFn wrapper: Fused kernel for MuonMamba with Momentum + Newton-Schulz5
    
    This is the fast path equivalent of selective_scan_fn but with fused convolution,
    projection, and scan operations for better performance.
    """
    return MuonMambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                                  out_proj_weight, out_proj_bias,
                                  A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 
                                  checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps,
                                  beta, alpha)
