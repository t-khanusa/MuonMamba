# Newton-Schulz Backward Pass - BUG FOUND AND FIX

## The Bug

The CUDA backward kernel is computing gradients through BOTH:
1. **The APPLICATION of B_4**: `X_5 = a*X_4 + B_4 @ X_4` ✓ CORRECT
2. **The COMPUTATION of B_4**: `A_4 = X_4 @ X_4.T`, `B_4 = b*A_4 + c*A_4²` ✗ WRONG!

In lines 1082-1326 of `newton_schulz_bwd_kernel.cuh`:
- Step 2b: Compute dB_4 (accumulate gradients through B_4 computation)
- Step 3: Compute dA_4 from dB_4
- Step 4: Compute additional dX_4 from dA_4

These steps are computing gradients as if B_4 has gradients flowing through its computation!

## The Fix

For "detached first 4 steps, only gradient in last step":
- Treat B_4 as a **CONSTANT** (detached)
- Only backprop through the APPLICATION: `dX_4 = a*dX_5 + B_4.T @ dX_5`
- Do NOT backprop through B_4's computation

**Remove lines 1082-1326** (the dB_4, dA_4, and dX_4-from-dA_4 computations).

## Verification

After the fix:
1. INIT: dX_4[0,0] = -0.440447 (a * dX_5)
2. STORE: dX_4[0,0] = -0.152178 (after adding B.T @ dX_5)
3. LOAD: Should be -0.152178 (no additional modifications)

Currently LOAD shows 0.147474 because Step 4 overwrites it with gradients through B_4's computation.

## Python Reference

The Python reference correctly detaches B_4:
```python
A_4_fp32 = X_4.detach() @ X_4.detach().T  # Detached!
B_4 = bf16_round(b * A_4 + c * A_4_sq)     # B_4 has no gradients
X_5 = a * X_4 + B_4 @ X_4                  # Only X_4 has gradients
```

CUDA should match this by NOT computing dB_4 or dA_4.

