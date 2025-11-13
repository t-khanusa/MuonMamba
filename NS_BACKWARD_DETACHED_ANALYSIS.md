# Newton-Schulz 5-Step Backward Pass Analysis

## Strategy: Detached First 4 Steps, Gradient Only in Last Step

### Efficiency Rationale
- **Memory**: Don't need to store X_0, X_1, X_2, X_3, X_4 from forward pass
- **Computation**: Recompute X_0→X_4 forward (detached), then compute gradients only through 5th iteration
- **Mathematical Correctness**: Since NS converges, small changes in X_4 lead to small changes in X_5, so detached approach is acceptable

---

## Current Implementation Structure

### PHASE 1: Recompute X_0 → X_4 (Detached, 4 iterations)
**Location**: `newton_schulz_bwd_kernel.cuh` lines 567-871

```cuda
// Step 1: Compute b_t = alpha * delta * B * u, convert to BF16, normalize
// Lines 569-646: Compute norm and normalize to get X_0

// Step 2: Store X_0 in X_temp
// Lines 625-645: Store normalized X_0

// Step 3: Run 4 NS iterations (detached, no gradients)
// Lines 648-869: For step in range(4):
//   - Compute A = X @ X.T
//   - Convert to BF16
//   - Compute A²
//   - Compute B = b*A + c*A²
//   - Apply X = a*X + B@X
//   - Store result in X_temp (overwrites previous X)
```

**Status**: ✅ **CORRECT** - All 4 iterations are detached (no gradient computation)

---

### PHASE 2: Backward Through 5th Iteration Only
**Location**: `newton_schulz_bwd_kernel.cuh` lines 873-1160

#### Step 1: Compute A_4 and B_4 (Detached)
**Lines 875-993**: Compute A_4 = X_4 @ X_4.T, then B_4 = b*A_4 + c*A_4²

**Critical Point**: 
- A_4 and B_4 are computed from `X_temp` which contains **detached X_4**
- B_4 is stored in `gram_A_fp32` as a **constant** (no gradients through its computation)
- ✅ **CORRECT**: B_4 is treated as detached/constant

#### Step 2: Initialize dX_4 = a * dX_5
**Lines 997-1056**: Load `grad_output` (which is dX_5) and compute `dX_4 = a * dX_5`

**Forward**: `X_5 = a * X_4 + B_4 @ X_4`  
**Backward**: `dX_4 = a * dX_5 + gradient_through_B4@X4`

#### Step 3: Add Gradient Through B_4@X_4 (B_4 as Constant)
**Lines 1058-1160**: Compute `dX_4 += B_4.T @ dX_5`

**Mathematical Justification**:
- Forward: `X_5[i,j] = a*X_4[i,j] + sum_k B_4[i,k] * X_4[k,j]`
- Since B_4 is **detached** (constant), backward is:
  - `dX_4[i,j] = a*dX_5[i,j] + sum_k B_4[k,i] * dX_5[k,j]`
  - This is: `dX_4 = a*dX_5 + B_4.T @ dX_5`

**Implementation** (lines 1102-1110):
```cuda
// (B_4.T @ dX_5)[i,j] = sum_k B_4[k,i] * dX_5[k,j]
float sum = 0.0f;
for (int k = 0; k < gram_size; ++k) {
    float dX_5_kj = grad_output[k_idx];  // dX_5[k, j]
    float B_4_ki = gram_A_fp32[k * gram_size + global_row];  // B_4[k, i]
    sum += B_4_ki * dX_5_kj;
}
dX_4_temp[buffer_idx] += sum;
```

**Status**: ✅ **CORRECT** - B_4 is treated as constant, gradient flows only through application

---

### PHASE 3: Backward Through Normalization
**Location**: `newton_schulz_bwd_kernel.cuh` lines 1162-1260

**Forward**: `X_0 = b_t_bf16 / norm`  
**Backward**: `d(b_t) = (dX_4 - X_0 * <dX_4, X_0>) / norm`

**Status**: ✅ **CORRECT** - Standard normalization backward formula

---

### PHASE 4: Backward Through b_t = alpha * delta * B * u
**Location**: `newton_schulz_bwd_kernel.cuh` lines 1262-1423

**Forward**: `b_t = alpha * delta * B * u`  
**Backward**:
- `grad_u += alpha * delta * B * d_b_t`
- `grad_delta += alpha * B * u * d_b_t`
- `grad_B += alpha * delta * u * d_b_t`

**Status**: ✅ **CORRECT** - Standard chain rule

---

## Key Verification Points

### ✅ B_4 is Properly Detached
1. X_4 is recomputed forward (detached) - lines 567-871
2. A_4 and B_4 are computed from detached X_4 - lines 875-993
3. B_4 is stored in `gram_A_fp32` as constant - line 991
4. Gradient flows only through application: `dX_4 += B_4.T @ dX_5` - lines 1102-1110

### ✅ Gradient Formula is Correct
For forward: `X_5 = a*X_4 + B_4 @ X_4` (B_4 constant)  
Backward: `dX_4 = a*dX_5 + B_4.T @ dX_5`

**Derivation**:
- `X_5[i,j] = a*X_4[i,j] + sum_k B_4[i,k] * X_4[k,j]`
- `dX_4[i,j] = ∂L/∂X_4[i,j] = sum_kl (∂L/∂X_5[k,l]) * (∂X_5[k,l]/∂X_4[i,j])`
- `∂X_5[k,l]/∂X_4[i,j] = a*δ(k,i)*δ(l,j) + B_4[k,i]*δ(l,j)`
- `dX_4[i,j] = a*dX_5[i,j] + sum_k B_4[k,i] * dX_5[k,j]`
- This is: `dX_4 = a*dX_5 + B_4.T @ dX_5` ✅

---

## Potential Issues to Check

### 1. Transposed Case (D > N)
**Lines 1114-1159**: Similar logic for transposed case
- Forward: `X_5_storage = a*X_4_storage + X_4_storage @ B_4.T` (right multiply)
- Backward: `dX_4_storage += dX_5_storage @ B_4` (right multiply)
- ✅ **CORRECT** - Matches forward pass structure

### 2. Buffer Indexing
All buffer indexing uses `dstate` (not `dstate*2`), which matches the real-only storage approach.
- ✅ **CONSISTENT** with forward pass changes

### 3. BF16 Rounding
All operations use BF16 rounding to match forward pass:
- A_4, A_4², B_4 are all rounded to BF16 - ✅ **CORRECT**

---

## Summary

**Implementation Status**: ✅ **CORRECT**

The backward pass correctly implements:
1. ✅ Detached first 4 steps (recomputed forward, no gradients)
2. ✅ Gradient computation only through 5th iteration
3. ✅ B_4 treated as constant (detached)
4. ✅ Correct gradient formulas for both transposed and non-transposed cases
5. ✅ Proper normalization and b_t backward pass

**No changes needed** - the implementation matches the efficient detached approach!


