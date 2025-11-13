# Copyright (c) 2023, Tri Dao, Albert Gu.
# MuonMamba: Mamba with Momentum and Newton-Schulz5 Orthogonalization

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, muon_mamba_inner_fn, newtonschulz5_ref, rms_norm_forward

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm as TritonRMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    TritonRMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


@dataclass
class MuonMambaConfig:
    """
    Configuration for MuonMamba: Mamba with Momentum and Newton-Schulz5 Orthogonalization
    
    Key parameters:
    - momentum_beta: β ∈ [0, 1] - Controls velocity decay (0 = no momentum, 0.9 = high momentum)
    - momentum_alpha: α > 0 - Scales the velocity contribution (typically 0.5-1.5)
    
    When beta > 0, Newton-Schulz5 orthogonalization is automatically applied to b_t
    to stabilize momentum accumulation.
    """
    d_model: int  # D - model dimension
    n_layers: int  # Number of layers
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  # N - SSM state dimension
    expand_factor: int = 2  # E - expansion factor
    d_conv: int = 4  # Convolution kernel size
    
    # SSM discretization parameters
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    
    # MuonMamba: Momentum + Newton-Schulz5 parameters
    momentum_beta: float = 0.9  # β - momentum decay factor (0 = no momentum, 0.9 = high momentum)
    momentum_alpha: float = 1.0  # α - momentum input scaling (typically 0.5-1.5)
    
    # Note: When beta > 0, Newton-Schulz5 orthogonalization is automatically applied
    # to b_t = alpha * delta * B * u at each timestep to ensure numerical stability
    
    # Normalization
    rms_norm_eps: float = 1e-5
    base_std: float = 0.02
    
    # Architecture options
    bias: bool = False
    conv_bias: bool = True
    use_fast_path: bool = True  # Use fused CUDA kernels
    
    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        
        # Validate momentum parameters
        if not (0.0 <= self.momentum_beta <= 1.0):
            raise ValueError(f"momentum_beta must be in [0, 1], got {self.momentum_beta}")
        
        if self.momentum_alpha <= 0:
            raise ValueError(f"momentum_alpha must be positive, got {self.momentum_alpha}")
        
        # Warn about high beta values
        if self.momentum_beta > 0.95:
            import warnings
            warnings.warn(
                f"High momentum beta={self.momentum_beta} may cause numerical instability. "
                f"Recommended: beta ∈ [0.5, 0.9] for most applications. "
                f"Newton-Schulz5 orthogonalization helps but very high beta can still be unstable.",
                UserWarning
            )


class RMSNorm(nn.Module):
    """RMSNorm implementation - fallback if Triton version not available"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class MuonMamba(nn.Module):
    """
    Multi-layer MuonMamba model with residual connections
    
    MuonMamba = Mamba + Momentum + Newton-Schulz5 Orthogonalization
    
    Architecture:
    - Each layer: RMSNorm → MuonMambaBlock → Residual connection
    - MuonMambaBlock: Uses momentum-based SSM with NS5 orthogonalization
    """
    def __init__(self, config: MuonMambaConfig):
        super().__init__()
        
        self.config = config
        self.layers = nn.ModuleList([
            ResidualBlock(config, layer_idx=i) 
            for i in range(config.n_layers)
        ])
    
    def forward(self, x):
        """
        Forward pass through all layers
        
        Args:
            x: (B, L, D) - input sequence
        
        Returns:
            (B, L, D) - output sequence
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def step(self, x, caches):
        """
        Autoregressive generation step (inference mode)
        
        Args:
            x: (B, D) - single token input (matching Mamba.step signature)
            caches: [cache(layer) for all layers]
                    Each cache = (conv_state, ssm_state, velocity_state)
        
        Returns:
            (B, D) - output token
            updated caches
        """
        # Pass through each layer sequentially
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches


class ResidualBlock(nn.Module):
    """
    Residual block: x + MuonMambaBlock(RMSNorm(x))
    """
    def __init__(self, config: MuonMambaConfig, layer_idx: int = None):
        super().__init__()
        
        self.mixer = MuonMambaBlock(config, layer_idx=layer_idx)
        # Use Triton RMSNorm if available, otherwise use our implementation
        if TritonRMSNorm is not None:
            self.norm = TritonRMSNorm(config.d_model, eps=config.rms_norm_eps)
        else:
            self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        """
        Inference step with caching
        
        Args:
            x: (B, D) - single token
            cache: (conv_state, ssm_state, velocity_state)
        
        Returns:
            (B, D) - output token
            updated cache
        """
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MuonMambaBlock(nn.Module):
    """
    MuonMamba block wrapper - delegates to Mamba with momentum parameters
    """
    def __init__(self, config: MuonMambaConfig, layer_idx: int = None):
        super().__init__()
        
        self.config = config
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand_factor,
            dt_rank=config.dt_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            dt_scale=config.dt_scale,
            dt_init_floor=config.dt_init_floor,
            conv_bias=config.conv_bias,
            bias=config.bias,
            use_fast_path=config.use_fast_path,
            layer_idx=layer_idx,
            beta=config.momentum_beta,  # Momentum decay (enables NS5 when > 0)
            alpha=config.momentum_alpha,  # Momentum scaling
        )
    
    def forward(self, x, inference_params=None):
        """
        Args:
            x: (B, L, D)
            inference_params: Optional inference parameters for caching
        
        Returns:
            (B, L, D)
        """
        return self.mamba(x, inference_params)
    
    def step(self, x, cache):
        """
        Single-token inference step
        
        Args:
            x: (B, D) - single token
            cache: (conv_state, ssm_state, velocity_state)
        
        Returns:
            (B, D) - output token
            updated cache: (conv_state, ssm_state, velocity_state)
        """
        conv_state, ssm_state, velocity_state = cache
        out, conv_state, ssm_state, velocity_state = self.mamba.step(
            x.unsqueeze(1), conv_state, ssm_state, velocity_state
        )
        return out.squeeze(1), (conv_state, ssm_state, velocity_state)


class Mamba(nn.Module):
    """
    Core Mamba SSM with Momentum and Newton-Schulz5 Orthogonalization
    
    MuonMamba Equations:
    1. b_t = alpha * delta * B * u_t
    2. b_t_ortho = NewtonSchulz5(b_t)  [when beta > 0]
    3. v_t = beta * v_{t-1} + b_t_ortho
    4. h_t = exp(delta * A) * h_{t-1} + v_t
    5. y_t = C_t * h_t + D * u_t
    
    Newton-Schulz5 applies 5-step orthogonalization to b_t per timestep
    to stabilize momentum accumulation.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        beta=0.9,  # Momentum decay (β) - enables NS5 when > 0
        alpha=1.0,  # Momentum scaling (α)
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # Input projection: [D] → [2E*D] (for x and z gating)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # SSM parameter projections
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt projection to preserve variance
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias: F.softplus(dt_bias) ∈ [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization for A
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # Momentum parameters (non-learnable buffers)
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32, device=device))
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32, device=device))
        
        # Numerical stability warning
        if beta > 0.95:
            import warnings
            warnings.warn(
                f"MuonMamba: High momentum beta={beta} may cause numerical instability. "
                f"Recommended: beta ∈ [0.5, 0.9] for most applications. "
                f"Newton-Schulz5 orthogonalization is enabled (beta > 0) which helps stability, "
                f"but very high beta can still lead to gradient issues.",
                UserWarning
            )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.b_c_dt_rms_eps = 1e-6 # Epsilon for RMS normalization
        self.b_rms_weight = nn.Parameter(torch.ones(self.d_state, device=device))
        self.c_rms_weight = nn.Parameter(torch.ones(self.d_state, device=device))
        self.dt_rms_weight = nn.Parameter(torch.ones(self.d_inner, device=device))

    def forward(self, hidden_states, inference_params=None):
        """
        Forward pass through MuonMamba SSM
        
        Args:
            hidden_states: (B, L, D) - input sequence
            inference_params: Optional caching parameters for generation
        
        Returns:
            (B, L, D) - output sequence
        """
        batch, seqlen, dim = hidden_states.shape

        # Check for cached states (inference mode)
        conv_state, ssm_state, velocity_state = None, None, None
        if inference_params is not None:
            conv_state, ssm_state, velocity_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # Incremental decoding: update states inplace
                out, _, _, _ = self.step(hidden_states, conv_state, ssm_state, velocity_state)
                return out

        # Input projection: [B, L, D] → [B, 2E*D, L]
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state) - negative for stability

        # CRITICAL: Fast path (mamba_inner_fn) does NOT support momentum/Newton-Schulz!
        # mamba_inner_fn is the original Mamba's fused kernel and doesn't accept beta/alpha.
        # When momentum is enabled (beta > 0), we MUST use selective_scan_fn instead.
        # Fast path: use fused CUDA kernel ONLY if momentum is disabled (beta == 0)
        print(f"use_fast_path: {self.use_fast_path}, causal_conv1d_fn: {causal_conv1d_fn is not None}, inference_params: {inference_params is None}")
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:
            print("Use the fast path")
            out = muon_mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                beta=self.beta,
                alpha=self.alpha,
                b_rms_weight=self.b_rms_weight,
                c_rms_weight=self.c_rms_weight,
                dt_rms_weight=self.dt_rms_weight,
                b_c_dt_rms_eps=self.b_c_dt_rms_eps
            )
        else:
            print("Use the standard path")
            # Standard path: step-by-step computation
            x, z = xz.chunk(2, dim=1)
            
            # Causal convolution
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
            
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # Project to SSM parameters: dt, B, C
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            # dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            # B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            # C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            


            # B = rearrange(B, "b dstate l -> (b l) dstate", l=seqlen).contiguous()
            B = rms_norm_forward(B, self.b_rms_weight, bias=None, eps=self.b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            # C = rearrange(C, "b dstate l -> (b l) dstate", l=seqlen).contiguous()
            C = rms_norm_forward(C, self.c_rms_weight, bias=None, eps=self.b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            # dt = rearrange(dt, "b d l -> (b l) d", l=seqlen).contiguous()
            dt = dt.t()  # (b*l, d_inner) - transpose from (d_inner, b*l) 
            dt = rms_norm_forward(dt, self.dt_rms_weight, bias=None, eps=self.b_c_dt_rms_eps)
            dt = rearrange(dt, "(b l) d -> b d l", l=seqlen).contiguous()
            # CRITICAL: Apply RMS normalization with learnable weights (matching MambaInnerFn)
            # This is ALWAYS applied (not conditional on beta) to maintain consistency with fast path
            # The learnable weights allow the model to learn the optimal scale for normalized values

            # Selective scan with momentum + Newton-Schulz5
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
                beta=self.beta,  # Momentum decay (enables NS5 when > 0)
                alpha=self.alpha,  # Momentum scaling
            )
            
            # Handle return signature: when z is passed, returns (out, out_z, ...) tuple
            if ssm_state is not None:
                # With states: (out, out_z, last_state, last_velocity)
                y, out_z, last_state, last_velocity = y
                ssm_state.copy_(last_state)
                velocity_state.copy_(last_velocity)
            else:
                # No states: (out, out_z) - unpack the tuple
                y, out_z = y
            
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        
        return out

    def step(self, hidden_states, conv_state, ssm_state, velocity_state):
        """
        Single-token autoregressive step (for generation)
        
        Args:
            hidden_states: (B, 1, D) - single token input
            conv_state: (B, E*D, d_conv) - convolution cache
            ssm_state: (B, E*D, N) - SSM hidden state cache
            velocity_state: (B, E*D, N) - momentum velocity cache
        
        Returns:
            (B, 1, D) - output token
            updated states: (conv_state, ssm_state, velocity_state)
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        
        # Input projection
        xz = self.in_proj(hidden_states.squeeze(1))  # (B, 2E*D)
        x, z = xz.chunk(2, dim=-1)  # (B, E*D) each

        # Convolution step
        if causal_conv1d_update is None:
            # Manual conv state update
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        # Project to SSM parameters
        x_db = self.x_proj(x)  # (B, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)  # (B, d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step with momentum
        # Use manual path if Triton is unavailable OR if momentum is enabled (Triton doesn't support momentum)
        if selective_state_update is None or self.beta != 0.0:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            
            # MuonMamba Equations (per timestep):
            # 1. b_t = alpha * delta * B * u_t (where u_t = x here)
            # 2. b_t_ortho = NewtonSchulz5(b_t)  [when beta > 0]
            # 3. v_t = beta * v_{t-1} + b_t_ortho
            # 4. h_t = exp(delta * A) * h_{t-1} + v_t
            
            # Compute b_t = alpha * delta * B * x
            # Shape: (B, d_inner, d_state) where each (d_inner, d_state) is a matrix
            b_t = self.alpha * rearrange(x, "b d -> b d 1") * dB  # (B, d_inner, d_state)
            
            # Apply Newton-Schulz5 orthogonalization when momentum is enabled
            if self.beta != 0.0:
                # Apply NS5 to each batch's b_t matrix: (d_inner, d_state) -> (d_inner, d_state)_ortho
                b_t_ortho = torch.zeros_like(b_t)
                for b in range(b_t.shape[0]):
                    b_t_ortho[b] = newtonschulz5_ref(b_t[b], steps=5)
                b_t = b_t_ortho
            
            # Momentum: v_t = beta * v_{t-1} + b_t_ortho (or b_t if beta == 0)
            velocity_state.copy_(self.beta * velocity_state + b_t)
            
            # Hidden state: h_t = A_t * h_{t-1} + v_t
            ssm_state.copy_(ssm_state * dA + velocity_state)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            # Triton path (only when beta=0, i.e., no momentum)
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, 
                dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state, velocity_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocate KV cache for autoregressive generation
        
        Args:
            batch_size: int
            max_seqlen: int (not used, kept for API compatibility)
            dtype: torch.dtype (optional)
        
        Returns:
            (conv_state, ssm_state, velocity_state) tuple
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        
        conv_state = torch.zeros(
            batch_size, self.d_inner, self.d_conv, 
            device=device, dtype=conv_dtype
        )
        ssm_state = torch.zeros(
            batch_size, self.d_inner, self.d_state, 
            device=device, dtype=ssm_dtype
        )
        velocity_state = torch.zeros(
            batch_size, self.d_inner, self.d_state, 
            device=device, dtype=ssm_dtype
        )
        
        return conv_state, ssm_state, velocity_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        """
        Get or initialize states from inference cache
        
        Args:
            inference_params: object with key_value_memory_dict
            batch_size: int
            initialize_states: bool - whether to zero out states
        
        Returns:
            (conv_state, ssm_state, velocity_state) from cache
        """
        assert self.layer_idx is not None
        
        if self.layer_idx not in inference_params.key_value_memory_dict:
            # Initialize new cache
            conv_state = torch.zeros(
                batch_size, self.d_inner, self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size, self.d_inner, self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            velocity_state = torch.zeros(
                batch_size, self.d_inner, self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state, ssm_state, velocity_state
            )
        else:
            # Retrieve existing cache
            conv_state, ssm_state, velocity_state = \
                inference_params.key_value_memory_dict[self.layer_idx]
            
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
                velocity_state.zero_()
        
        return conv_state, ssm_state, velocity_state

