import torch
import selective_scan_cuda

# Test with NON-UNIFORM inputs  
batch, dim, seqlen, dstate = 1, 2, 4, 2
beta, alpha = 0.9, 1.0
device = 'cuda'
dtype = torch.float32

# NON-UNIFORM inputs
u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device).abs() + 0.1
delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device).abs() * 0.1 + 0.05  
A = -torch.rand(dim, dstate, dtype=dtype, device=device) - 0.5
B = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1 + 0.1
C = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1 + 0.1
D = torch.randn(dim, dtype=dtype, device=device) + 1.0

print("Testing with non-uniform inputs...")
print(f"u range: [{u.min().item():.3f}, {u.max().item():.3f}]")
print(f"delta range: [{delta.min().item():.3f}, {delta.max().item():.3f}]")

fwd = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, None, False, beta, alpha)
out, x, X_4 = fwd[0], fwd[1], fwd[2]

dout = torch.randn_like(out)  # Non-uniform gradient
bwd = selective_scan_cuda.bwd(u, delta, A, B, C, D, None, None, dout, x, None, None, False, False, beta, alpha, X_4)
du, ddelta, dA, dB, dC, dD, ddelta_bias = bwd

print(f"\nGradient outputs:")
print(f"du: sum={du.sum().item():.6f}, std={du.std().item():.6f}")
print(f"ddelta: sum={ddelta.sum().item():.6f}, std={ddelta.std().item():.6f}")
print(f"dA: sum={dA.sum().item():.6f}")
print(f"dB: sum={dB.sum().item():.6f}")
print(f"dC: sum={dC.sum().item():.6f}")
print(f"dD: sum={dD.sum().item():.6f}")

print(f"\nStatus:")
for name, grad in [('du', du), ('ddelta', ddelta), ('dA', dA), ('dB', dB), ('dC', dC), ('dD', dD)]:
    if grad.abs().sum().item() < 1e-6:
        print(f"  ❌ {name} is ~zero")
    else:
        print(f"  ✓ {name} has gradients")

