import torch
import selective_scan_cuda

# Simple test to verify X_4_buffer is passed correctly
batch, dim, seqlen, dstate = 2, 4, 8, 4
beta, alpha = 0.9, 1.0

device = 'cuda'
dtype = torch.float32

# Create simple inputs
u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device)
delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * 0.1
A = -torch.rand(dim, dstate, dtype=dtype, device=device) - 1.0
B = torch.randn(batch, 1, dstate, seqlen, dtype=dtype, device=device) * 0.1
C = torch.randn(batch, 1, dstate, seqlen, dtype=dtype, device=device) * 0.1
D = torch.randn(dim, dtype=dtype, device=device)
delta_bias = None
delta_softplus = False

# Forward pass
print("Running forward pass...")
fwd_result = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus, beta, alpha)
out = fwd_result[0]
x = fwd_result[1]

print(f"Forward result length: {len(fwd_result)}")
print(f"out shape: {out.shape}")
print(f"x shape: {x.shape}")

if len(fwd_result) > 2:
    X_4_buffer = fwd_result[2]
    print(f"X_4_buffer shape: {X_4_buffer.shape}")
    print(f"X_4_buffer dtype: {X_4_buffer.dtype}")
    print(f"X_4_buffer mean: {X_4_buffer.mean().item():.6f}")
    print(f"X_4_buffer std: {X_4_buffer.std().item():.6f}")
    print(f"X_4_buffer has NaN: {torch.isnan(X_4_buffer).any().item()}")
    print(f"X_4_buffer has Inf: {torch.isinf(X_4_buffer).any().item()}")
else:
    print("ERROR: X_4_buffer not returned from forward!")
    X_4_buffer = None

# Backward pass
print("\nRunning backward pass...")
dout = torch.randn_like(out)
try:
    bwd_result = selective_scan_cuda.bwd(
        u, delta, A, B, C, D, None, delta_bias, dout, x, None, None,
        delta_softplus, False, beta, alpha, X_4_buffer
    )
    print("Backward pass succeeded!")
    du = bwd_result[0]
    print(f"du shape: {du.shape}")
    print(f"du mean: {du.mean().item():.6f}")
    print(f"du std: {du.std().item():.6f}")
except Exception as e:
    print(f"Backward pass failed: {e}")





