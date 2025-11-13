import torch
import selective_scan_cuda
from mamba_ssm.ops.selective_scan_interface import newtonschulz5_ref

# Test CUDA backward against PyTorch autograd
batch, dim, seqlen, dstate = 2, 4, 8, 4
beta, alpha = 0.9, 1.0
device = 'cuda'
dtype = torch.float32

# Create inputs with requires_grad
u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device, requires_grad=True)
delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device, requires_grad=True) * 0.1
A = -torch.rand(dim, dstate, dtype=dtype, device=device, requires_grad=True) - 1.0
B = torch.randn(dim, dstate, dtype=dtype, device=device, requires_grad=True) * 0.1
C = torch.randn(dim, dstate, dtype=dtype, device=device, requires_grad=True) * 0.1
D = torch.randn(dim, dtype=dtype, device=device, requires_grad=True)

# PyTorch reference implementation
def pytorch_forward(u, delta, A, B, C, D, beta, alpha):
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    
    out = torch.zeros(batch, dim, seqlen, dtype=u.dtype, device=u.device)
    h = torch.zeros(batch, dim, dstate, dtype=torch.float32, device=u.device)
    v = torch.zeros(batch, dim, dstate, dtype=torch.float32, device=u.device)
    
    for t in range(seqlen):
        # Compute b_t = alpha * delta * B * u
        b_t = alpha * delta[:, :, t].unsqueeze(-1) * B.unsqueeze(0) * u[:, :, t].unsqueeze(-1)
        
        # Apply NS orthogonalization per (batch, timestep) on [dim, dstate] matrices
        b_t_ortho = torch.zeros_like(b_t)
        for b in range(batch):
            b_t_ortho[b] = newtonschulz5_ref(b_t[b].T, steps=5).T
        
        # Velocity update: v_t = beta * v_{t-1} + b_t_ortho
        v = beta * v + b_t_ortho
        
        # Hidden state update: h_t = exp(delta*A) * h_{t-1} + v_t
        dA = delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0)
        h = torch.exp(dA) * h + v
        
        # Output: y_t = C * h_t + D * u_t
        out[:, :, t] = (C.unsqueeze(0) * h).sum(dim=-1) + D.unsqueeze(0) * u[:, :, t]
    
    return out

# Test forward pass
print("Testing forward pass...")
out_pytorch = pytorch_forward(u, delta, A, B, C, D, beta, alpha)
fwd_result = selective_scan_cuda.fwd(u, delta, A, B, C, None, None, None, False, beta, alpha)
out_cuda = fwd_result[0]

print(f"Forward output diff: {(out_cuda - out_pytorch).abs().max().item():.6e}")

# Test backward pass
print("\nTesting backward pass...")
dout = torch.randn_like(out_cuda)

# PyTorch autograd
u_auto = u.clone().detach().requires_grad_(True)
delta_auto = delta.clone().detach().requires_grad_(True)
A_auto = A.clone().detach().requires_grad_(True)
B_auto = B.clone().detach().requires_grad_(True)
C_auto = C.clone().detach().requires_grad_(True)
D_auto = D.clone().detach().requires_grad_(True)

out_auto = pytorch_forward(u_auto, delta_auto, A_auto, B_auto, C_auto, D_auto, beta, alpha)
out_auto.backward(dout)

du_auto = u_auto.grad
ddelta_auto = delta_auto.grad
dA_auto = A_auto.grad
dB_auto = B_auto.grad
dC_auto = C_auto.grad
dD_auto = D_auto.grad

# CUDA backward
x_cuda = fwd_result[1]
X_4_buffer = fwd_result[2] if len(fwd_result) > 2 else None

bwd_result = selective_scan_cuda.bwd(
    u, delta, A, B, C, D, None, None, dout, x_cuda, None, None,
    False, False, beta, alpha, X_4_buffer
)

du_cuda = bwd_result[0]
ddelta_cuda = bwd_result[1]
dA_cuda = bwd_result[2]
dB_cuda = bwd_result[3]
dC_cuda = bwd_result[4]
dD_cuda = bwd_result[5]

# Compare
def compare_grads(name, cuda_grad, auto_grad):
    diff = (cuda_grad - auto_grad).abs()
    rel_diff = diff / (auto_grad.abs() + 1e-8)
    print(f"\n{name}:")
    print(f"  Max abs diff: {diff.max().item():.6e}")
    print(f"  Mean abs diff: {diff.mean().item():.6e}")
    print(f"  Max rel diff: {rel_diff.max().item():.6e}")
    print(f"  Mean rel diff: {rel_diff.mean().item():.6e}")
    exceed = (rel_diff > 1e-2).sum().item()
    total = rel_diff.numel()
    print(f"  Exceed 1% tolerance: {exceed}/{total} ({100*exceed/total:.2f}%)")
    if rel_diff.max().item() < 1e-2:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL")

compare_grads("du", du_cuda, du_auto)
compare_grads("ddelta", ddelta_cuda, ddelta_auto)
compare_grads("dA", dA_cuda, dA_auto)
compare_grads("dB", dB_cuda, dB_auto)
compare_grads("dC", dC_cuda, dC_auto)
compare_grads("dD", dD_cuda, dD_auto)





