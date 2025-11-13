# Backward Pass Mathematical and Logical Issues

## Issue 1: Incorrect C gradient computation when both B and C are constant

**Location**: Lines 374, 776

**Problem**: When both B and C are constant:
- Line 374: `dBC_val += dout_vals[i] * h_t` (B constant case, C constant)
- Line 376: `dBC_val += dout_vals[i] * h_t` (C constant case, B variable)
- Line 774-777: C gradient uses `dBC_val`, which contains B gradient when B is constant

**Root Cause**: `smem_dbc` stores B gradients when B is constant, but then line 774-777 tries to use the same value for C gradients. The conversion formula `dBC_val * conj(B[...])` is mathematically incorrect.

**Correct Formula**:
- For constant C: `dC[d,n] = sum_{b,t} dout[b,d,t] * h[b,d,n,t]`
- The current `dBC_val` when B is constant contains: `sum_t dout * h_t` for each state
- But this is wrong because it should be `sum_t dout * h_t` aggregated across all timesteps for each C[d,n]

**Fix**: When both B and C are constant, we need separate accumulators:
- `dB_val` accumulates `sum_t dout * h_t` (through output) + velocity gradients
- `dC_val` accumulates `sum_t dout * h_t` (different from dB!)

## Issue 2: Incorrect dB_from_output formula for constant B with NS

**Location**: Line 388

**Problem**: 
```cuda
const float dB_from_output = dout_vals[i] * (!kIsVariableC ? h_t : h_t * C_vals[i]);
```

**Mathematical Analysis**:
- Forward: `y = C @ h` where `h` depends on B through velocity: `v_t = β*v_{t-1} + α*δ*B*u` (orthogonalized)
- `∂L/∂h = C^T @ dout` → `∂L/∂h[d,n,t] = dout[d,t] * C[d,n]`
- `∂L/∂B` through output = `∂L/∂h * ∂h/∂v * ∂v/∂B_ortho`

But `∂h/∂v = 1` (direct addition), and `∂v/∂B_ortho` comes through NS backward.

**Current formula issue**: 
- When C is constant: `dB_from_output = dout * h_t` 
- This should be: `dB_from_output = dout * C_val * h_t` because `∂L/∂h = dout * C`

**However**, `thread_reverse_data[i].y` already contains `dout * B_val * C_val` (line 294) when both are constant. This seems wrong for momentum mode.

**Fix**: For momentum mode (NS enabled), the output gradient formula at line 294 should be:
- `thread_reverse_data[i].y = dout_vals[i] * C_val` (not `B_val * C_val`)
- Because in momentum mode, B is already applied in velocity, so output is `y = C @ h`

## Issue 3: Inconsistent handling of constant B with NS

**Location**: Lines 382-402, 766-772

**Problem**: When B is constant and NS is enabled:
- Lines 382-402: B gradients (velocity + output) are accumulated per-timestep into `grad_X_4_buffer`
- Lines 766-772: B gradient accumulation is skipped (correct)
- BUT: `dBC_val` still accumulates output-path gradients in line 374, which are then ignored at line 766-772

**Issue**: This is wasteful but not wrong. However, the formula at line 374 might be wrong (see Issue 2).

## Issue 4: Missing C gradient computation when both B and C are constant and NS is enabled

**Location**: Lines 774-777

**Problem**: When both B and C are constant and NS is enabled:
- B gradients go to `grad_X_4_buffer` (correct)
- C gradients should be computed from `dout * h_t` aggregated across timesteps
- But `smem_dbc` contains B gradients, not C gradients
- Line 774-777 tries to convert: `dBC_val * conj(B[...])`, which is mathematically wrong

**Correct Fix**: When both are constant, we need:
1. Separate accumulator for C gradients: `dC_val += dout_vals[i] * h_t` (when C is constant)
2. Or compute C gradients directly without using `dBC_val`

## Issue 5: Complex case dB_from_output formula

**Location**: Line 620

**Problem**: 
```cuda
const float dB_from_output = ((2 * dout_vals[i]) * conj(!kIsVariableC ? h_t : h_t * C_vals[i])).real_;
```

**Analysis**: 
- The `2 *` factor is for complex case (line 516 shows `2 * dout_vals[i]`)
- But the formula `dout * h_t` vs `dout * h_t * C` is the same issue as Issue 2
- Should be `dout * C * h_t` when C is constant

## Recommended Fixes

1. **Fix Issue 1 & 4**: When both B and C are constant, maintain separate accumulators:
   - Add `weight_t *smem_dc` for C gradients (separate from `smem_dbc`)
   - Accumulate C gradients separately: `dC_val += dout_vals[i] * h_t` when C is constant

2. **Fix Issue 2**: Correct the output gradient formula:
   - In momentum mode (NS or beta != 1.0), `thread_reverse_data[i].y` should be `dout * C`, not `dout * B * C`
   - Update line 294 to check for momentum mode

3. **Fix Issue 3**: Ensure consistency - when NS is enabled and B is constant, skip `dBC_val` accumulation for B entirely.








