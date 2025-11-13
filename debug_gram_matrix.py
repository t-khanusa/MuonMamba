#!/usr/bin/env python3
"""
Verify Gram matrix computation by comparing step-by-step with PyTorch
"""
import torch

# Create same matrix as CUDA test
torch.manual_seed(42)
dim = 128
dstate = 64

u = torch.randn(1, dim, 1, dtype=torch.float16)
delta = torch.ones(1, dim, 1, dtype=torch.float16) 
B_mat = torch.randn(dim, dstate, dtype=torch.float32)
alpha = 1.0

# Compute b_t
delta_t = delta[:, :, 0].float()
u_t = u[:, :, 0].float()
b_t = alpha * delta_t.unsqueeze(2) * B_mat.unsqueeze(0) * u_t.unsqueeze(2)  # [1, 128, 64]

G = b_t[0]  # [128, 64]

print("="*80)
print("Step-by-step Newton-Schulz verification")
print("="*80)

# Step 1: Convert to BF16
X = G.bfloat16()
print(f"\n1. After BF16 conversion:")
print(f"   X.shape = {X.shape}")
print(f"   X norm = {X.norm():.6f}")

# Step 2: Normalize  
X = X / X.norm()
print(f"\n2. After normalization:")
print(f"   X norm = {X.norm():.6f}")
print(f"   X[0,0:3] = {X[0, :3]}")

# Step 3: Transpose (D > N case)
print(f"\n3. Transpose (D={dim} > N={dstate}):")
X_T = X.T  # [64, 128]
print(f"   X_T.shape = {X_T.shape}")
print(f"   X_T[0,0:3] = {X_T[0, :3]}")

# Step 4: Compute Gram matrix A = X_T @ X_T.T
A = X_T @ X_T.T  # [64, 64]
print(f"\n4. Gram matrix A = X_T @ X_T.T:")
print(f"   A.shape = {A.shape}")
print(f"   A.dtype = {A.dtype}")
print(f"   A diagonal (should be close to 1 for normalized X):")
print(f"     A[0,0] = {A[0,0]:.6f}")
print(f"     A[1,1] = {A[1,1]:.6f}")
print(f"     A[2,2] = {A[2,2]:.6f}")
print(f"   A off-diagonal (should be small):")
print(f"     A[0,1] = {A[0,1]:.6f}")
print(f"     A[0,2] = {A[0,2]:.6f}")
print(f"   A min/max: [{A.min():.6f}, {A.max():.6f}]")
print(f"   A trace (sum of diagonal): {A.diag().sum():.6f} (should be ~{A.shape[0]})")

# Step 5: Check if A is reasonable for NS
trace = A.diag().sum()
expected_trace = A.shape[0]  # Should be ~64 for identity-like matrix
print(f"\n5. Sanity check:")
if abs(trace - expected_trace) > expected_trace * 0.5:
    print(f"   ⚠️  WARNING: A trace {trace:.2f} is far from expected {expected_trace}")
    print(f"   This suggests X is not properly normalized or Gram is wrong!")
else:
    print(f"   ✅ A trace {trace:.2f} is reasonable (expected ~{expected_trace})")

# Step 6: Compute A²
A2 = A @ A
print(f"\n6. Compute A²:")
print(f"   A²[0,0] = {A2[0,0]:.6f}")
print(f"   A²[1,1] = {A2[1,1]:.6f}")

# Step 7: Compute B = b*A + c*A²
a, b, c = 3.4445, -4.7750, 2.0315
B_poly = b * A + c * A2
print(f"\n7. Compute B = b*A + c*A²:")
print(f"   B[0,0] = {B_poly[0,0]:.6f}")
print(f"   B[1,1] = {B_poly[1,1]:.6f}")
print(f"   B min/max: [{B_poly.min():.6f}, {B_poly.max():.6f}]")

# Step 8: Apply X_new = a*X + B@X
X_T_new = a * X_T + B_poly @ X_T
print(f"\n8. Update X_new = a*X + B@X:")
print(f"   X_T_new[0,0:3] = {X_T_new[0, :3]}")
print(f"   X_T_new norm = {X_T_new.norm():.6f}")

print("\n" + "="*80)







