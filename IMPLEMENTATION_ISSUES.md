# Implementation Issues Found During Testing

## Current Status

✅ **Implemented**:
- Parameter structures updated
- 5-step NS forward kernel written
- Scan kernel integration updated
- Buffer allocation in C++
- Reference implementation in Python

❌ **Issues Found**:
1. Newton-Schulz orthogonality error is high (~7.88 instead of <1e-4)
2. CUDA kernel produces NaN values when Newton-Schulz is enabled
3. Reference implementation produces inf/nan values

## Issue 1: Newton-Schulz Orthogonality

### Problem
The orthogonality error after 5 NS iterations is ~7.88, which suggests the algorithm isn't converging properly.

### Possible Causes
1. **Coefficient mismatch**: The coefficients (a=3.4445, b=-4.7750, c=2.0315) might be incorrect
2. **Normalization issue**: The initial normalization might not be appropriate
3. **Iteration formula**: The update `X = a*X + B@X` might have wrong matrix dimensions

### Debug Steps
```python
# Test NS on a simple 4x4 matrix
G = torch.randn(4, 4)
X = G / torch.sqrt((G*G).sum())
for i in range(5):
    A = X @ X.T
    print(f"Iteration {i}: ||A - I|| = {torch.norm(A - torch.eye(4))}")
    A2 = A @ A
    B = b * A + c * A2
    X = a * X + B @ X
```

### Expected Behavior
After 5 iterations, `X @ X.T` should be close to identity (error < 1e-4).

## Issue 2: CUDA Kernel NaN Values

### Problem
When `beta > 0`, the CUDA kernel produces NaN values in the output.

### Possible Causes
1. **Division by zero**: In norm computation
2. **Buffer addressing**: Wrong indices in X_4_buffer
3. **Atomic operations**: Race conditions in gram matrix accumulation
4. **Uninitialized memory**: velocity_ortho buffer not properly initialized

### Debug Steps in CUDA Kernel

```cuda
// Add debug prints in newton_schulz_velocity_5step_kernel
if (tid == 0 && batch_idx == 0 && time_idx == 0) {
    printf("Norm: %f\n", norm);
    printf("First b_t element: %f\n", velocity_ortho[0]);
}

// Check for NaN after each step
for (int step = 0; step < 5; ++step) {
    // ... NS iteration ...
    
    if (tid == 0) {
        float test_val = velocity_ortho[batch_idx * D * L * dstate];
        if (isnan(test_val) || isinf(test_val)) {
            printf("NaN/Inf detected at step %d\n", step);
        }
    }
}
```

###Fix Checklist
- [ ] Check norm computation doesn't produce zero
- [ ] Verify buffer indices are correct
- [ ] Check atomicAdd for race conditions
- [ ] Initialize velocity_ortho buffer properly
- [ ] Verify gram_A accumulation

## Issue 3: Reference Implementation inf/nan

### Problem
Reference implementation produces inf/nan when beta > 0.

### Root Cause
Likely the Newton-Schulz isn't converging, leading to exploding values in the momentum accumulation.

### Fix
Need to ensure NS produces properly orthogonalized matrices before integrating into the scan loop.

## Recommended Next Steps

### Step 1: Fix Newton-Schulz Algorithm
Focus on getting NS to work correctly first, independent of the full scan:

```python
def test_ns_basic():
    # Test on known matrix
    G = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.float32)  # Already orthogonal
    G_ortho = newtonschulz5_ref(G)
    
    norm = torch.sqrt((G_ortho * G_ortho).sum())
    G_norm = G_ortho / norm
    gram = G_norm.T @ G_norm
    print(gram)  # Should be close to identity
```

### Step 2: Verify CUDA Kernel Logic
Once Python NS works:
- Compare CUDA output to Python step-by-step
- Add extensive debug prints
- Test with small matrices (4x4) first

### Step 3: Integration Testing
After both work individually:
- Test full forward pass with small inputs
- Gradually increase size
- Monitor for numerical issues

## Newton-Schulz Coefficient Verification

The coefficients should satisfy:
```
For f(A) = a*I + b*A + c*A^2
Applied as: X_{k+1} = f(A) @ X_k where A = X_k @ X_k^T

Optimal for fast convergence: a ≈ 3.4445, b ≈ -4.7750, c ≈ 2.0315
```

These are derived from minimizing the error `||f(A) @ X - X_ortho||` for matrices with eigenvalues in [0, 2].

### Verification Test
```python
# Test convergence on known eigenvalue spectrum
A = torch.diag(torch.tensor([0.5, 1.0, 1.5]))  # Eigenvalues in [0, 2]
f_A = a * torch.eye(3) + b * A + c * (A @ A)
print(torch.eig(f_A))  # Should have eigenvalues close to 1
```

## Files Needing Attention

1. **mamba_ssm/ops/selective_scan_interface.py**
   - `newtonschulz5_ref()` - Fix algorithm
   - `selective_scan_ref()` - Handle NaN gracefully

2. **csrc/selective_scan/newton_schulz_fwd_kernel.cuh**
   - `newton_schulz_velocity_5step_kernel` - Add debug prints
   - Fix norm computation
   - Verify buffer addressing

3. **test_5step_ns_forward.py**
   - Add more granular tests
   - Test NS in isolation
   - Compare step-by-step with reference

## Temporary Workarounds

Until issues are fixed, you can:
1. Use beta=0 (disable momentum) for testing other parts
2. Test without NS first to verify scan logic
3. Use 1-step NS temporarily (less accurate but faster to debug)

## References

- Newton-Schulz iteration: https://en.wikipedia.org/wiki/Iterative_refinement#Newton_iteration
- Optimal coefficients: "Fast Computation of Matrix Functions" (Higham, 2008)
- MuonMamba paper: Section on orthogonalization






