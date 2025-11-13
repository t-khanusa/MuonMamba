# Mathematical Derivation: Newton-Schulz Backward Pass

## The Bug: Incorrect Gradient Formula for A²

### Forward Pass (5th iteration)
```
A_4 = X_4 @ X_4.T
B_4 = b*A_4 + c*A_4²  (where A_4² = A_4 @ A_4)
X_5 = a*X_4 + B_4 @ X_4
```

### Backward Pass

Given `dL/dX_5` (grad_output), we need to compute `dL/dX_4`.

#### Step 1: Gradient through X_5 = a*X_4 + B_4@X_4
```
dL/dX_4 = a*dL/dX_5  (from first term)
dL/dX_4 += B_4.T @ dL/dX_5  (from second term, gradient through right operand)
dL/dB_4 = dL/dX_5 @ X_4.T  (gradient through left operand)
```

#### Step 2: Gradient through B_4 = b*A_4 + c*A_4²

For the linear term: `dL/dA_4 = b * dL/dB_4`

For the quadratic term `Y = c*A@A`, we need the gradient formula.

**Correct Mathematical Derivation:**

Let Y = A@A. We want dL/dA given dL/dY.

Using the chain rule:
```
dY = dA@A + A@dA  (product rule)
```

The gradient is:
```
tr(dL/dY.T @ dY) = tr(dL/dY.T @ (dA@A + A@dA))
                 = tr(dL/dY.T @ dA @ A) + tr(dL/dY.T @ A @ dA)
                 = tr(A @ dL/dY.T @ dA) + tr(dA @ A.T @ dL/dY)
                 = tr((A @ dL/dY.T + A.T @ dL/dY) @ dA)
```

Therefore:
```
dL/dA = A @ dL/dY.T + A.T @ dL/dY
```

For B_4 = c*A_4²:
```
dL/dA_4 (from quadratic) = c * (A_4 @ dL/dB_4.T + A_4.T @ dL/dB_4)
```

**Combined:**
```
dL/dA_4 = b*dL/dB_4 + c*(A_4 @ dL/dB_4.T + A_4.T @ dL/dB_4)
```

### The Bug in CUDA Code

**CUDA code (line 1158-1185):**
```cuda
dA_4 = b*dB_4 + c*(dB_4 @ A_4.T + dB_4.T @ A_4)
```

**Correct formula:**
```cuda
dA_4 = b*dB_4 + c*(A_4 @ dB_4.T + A_4.T @ dB_4)
```

**The Issue:** The CUDA code has `dB_4` and `A_4` in the wrong order! It's computing:
- `dB_4 @ A_4.T` instead of `A_4 @ dB_4.T`
- `dB_4.T @ A_4` instead of `A_4.T @ dB_4`

This is why the gradients have the wrong magnitudes!

## Verification with PyTorch

Let's verify with a simple example:
```python
import torch

A = torch.randn(3, 3, requires_grad=True)
Y = A @ A
dL_dY = torch.randn(3, 3)

Y.backward(dL_dY)
grad_pytorch = A.grad.clone()

# Manual computation (correct formula)
grad_manual = A @ dL_dY.T + A.T @ dL_dY

print("PyTorch grad:", grad_pytorch)
print("Manual grad:", grad_manual)
print("Match:", torch.allclose(grad_pytorch, grad_manual, rtol=1e-4))
```

This should confirm that the correct formula is `A @ dY.T + A.T @ dY`, not `dY @ A.T + dY.T @ A`.

