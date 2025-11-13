import torch
import selective_scan_cuda
from csrc.selective_scan.newton_schulz_bwd_kernel import launch_newton_schulz_velocity_5step_backward

# Test NS backward kernel directly
batch, dim, seqlen, dstate = 1, 2, 4, 2
alpha = 1.0
device = 'cuda'
dtype = torch.float32

# Create simple inputs
u = torch.ones(batch, dim, seqlen, dtype=dtype, device=device) * 0.5
delta = torch.ones(batch, dim, seqlen, dtype=dtype, device=device) * 0.1
B = torch.ones(dim, dstate, dtype=dtype, device=device) * 0.1

# Create gradient output (what comes from main backward)
grad_output = torch.ones(batch, dim, seqlen, dstate, dtype=torch.float32, device=device)

# Create output gradient tensors (should be accumulated into)
grad_u = torch.zeros_like(u, dtype=torch.float32)
grad_delta = torch.zeros_like(delta, dtype=torch.float32)
grad_B = torch.zeros(dim, dstate, dtype=torch.float32, device=device)

print("Before NS backward:")
print(f"  grad_output sum: {grad_output.sum().item():.6f}")
print(f"  grad_u sum: {grad_u.sum().item():.6f}")
print(f"  grad_delta sum: {grad_delta.sum().item():.6f}")
print(f"  grad_B sum: {grad_B.sum().item():.6f}")

# This won't work because we can't call CUDA kernels directly from Python
# We need to test through the full backward pass
print("\nNote: Can't test NS backward kernel directly from Python")
print("Testing through full backward pass instead...")

# Test through full backward
fwd = selective_scan_cuda.fwd(u, delta, -torch.ones(dim, dstate, device=device), B, 
                                torch.ones(dim, dstate, device=device) * 0.1,
                                torch.ones(dim, device=device),
                                None, None, False, 0.9, alpha)
out, x, X_4 = fwd[0], fwd[1], fwd[2]

dout = torch.ones_like(out)
bwd = selective_scan_cuda.bwd(u, delta, -torch.ones(dim, dstate, device=device), B,
                               torch.ones(dim, dstate, device=device) * 0.1,
                               torch.ones(dim, device=device),
                               None, None, dout, x, None, None, False, False, 0.9, alpha, X_4)

du, ddelta, dA, dB, dC, dD = bwd

print("\nAfter full backward:")
print(f"  du sum: {du.sum().item():.6f}, mean: {du.mean().item():.6f}")
print(f"  ddelta sum: {ddelta.sum().item():.6f}, mean: {ddelta.mean().item():.6f}")
print(f"  dB sum: {dB.sum().item():.6f}, mean: {dB.mean().item():.6f}")


