# Newton-Schulz Backward Pass Trace Verification

**Date:** Based on verified forward pass implementation
**Status:** ✅ **VERIFIED CORRECT - Both Transposed and Non-Transposed Cases**

## Forward Pass Reference (Verified Correct)

According to `NS_CUDA_VERIFICATION_COMPLETE.md`, the forward pass is **mathematically correct** and matches PyTorch to < 1% error.

### Forward Pass Formula (5th Iteration)

**Non-Transposed Case (D <= N):**
- X is [D, N], stored at: `buffer_idx = d * L * dstate + t * dstate + n`
- B_4 is [D, D]
- Forward: `X_5[i, j] = a*X_4[i, j] + sum_k B_4[i, k] * X_4[k, j]`
  - Where `i` ∈ [0, D-1] (dimension), `j` ∈ [0, dstate-1], `k` ∈ [0, D-1]

**Transposed Case (D > N):**
- Logical: X is [N, D]
- Storage: `X_storage[d, n] = X_logical[n, d]`
- Buffer: `buffer_idx = d * L * dstate + t * dstate + n`
- B_4 is [N, N]
- Forward: `X_5_storage[d, n] = a*X_4_storage[d, n] + sum_k B_4[n, k] * X_4_storage[d, k]`
  - Where `d` ∈ [0, D-1], `n` ∈ [0, dstate-1], `k` ∈ [0, dstate-1]

---

## Backward Pass Trace

The backward pass should reverse **only the 5th iteration** (first 4 are detached).

### Non-Transposed Case: Backward Trace

**Forward:** `X_5[i, j] = a*X_4[i, j] + sum_k B_4[i, k] * X_4[k, j]`

**Backward derivation:**
- `dX_4[i, j]` receives gradient from:
  1. **Direct term:** `∂X_5[i, j]/∂X_4[i, j] = a` → contributes `a*dX_5[i, j]` ✓
  2. **From X_5[k, j] terms:** For each `k`, `X_4[i, j]` appears in `X_5[k, j]` with coefficient `B_4[k, i]`
     - `∂X_5[k, j]/∂X_4[i, j] = B_4[k, i]` (since `X_5[k, j] = a*X_4[k, j] + sum_l B_4[k, l] * X_4[l, j]`)
     - Contributes: `sum_k B_4[k, i] * dX_5[k, j]` ✓

**Backward formula:** `dX_4[i, j] = a*dX_5[i, j] + sum_k B_4[k, i] * dX_5[k, j]`

This is: `dX_4 = a*dX_5 + B_4.T @ dX_5`

**CUDA Implementation Check:**
- ✅ **Step 1 (line 1015):** `dX_4[i, j] = a*dX_5[i, j]` ✓
- ✅ **Step 2 (lines 1084-1093):** `dX_4[i, j] += sum_k B_4[k, i] * dX_5[k, j]` ✓
  - Indexing: `B_4_ki = gram_A_fp32[k * gram_size + i]` ✓
  - Indexing: `dX_5_kj = grad_output[k_idx]` where `k_idx = k * L * dstate + t * dstate + j` ✓

**Result:** ✅ **CORRECT** - Matches forward pass reverse exactly.

---

### Transposed Case: Backward Trace

**Forward:** `X_5_storage[d, n] = a*X_4_storage[d, n] + sum_k B_4[n, k] * X_4_storage[d, k]`

**Backward derivation:**
- `dX_4_storage[d, n]` receives gradient from:
  1. **Direct term:** `∂X_5_storage[d, n]/∂X_4_storage[d, n] = a` → contributes `a*dX_5_storage[d, n]` ✓
  2. **From X_5_storage[d, k] terms:** For each `k`, `X_4_storage[d, n]` appears in `X_5_storage[d, k]` with coefficient `B_4[n, k]`
     - `∂X_5_storage[d, k]/∂X_4_storage[d, n] = B_4[n, k]`
     - Contributes: `sum_k B_4[n, k] * dX_5_storage[d, k]` ✓

**Backward formula:** `dX_4_storage[d, n] = a*dX_5_storage[d, n] + sum_k B_4[n, k] * dX_5_storage[d, k]`

**CUDA Implementation Check:**
- ✅ **Step 1 (line 1035):** `dX_4_storage[d, n] = a*dX_5_storage[d, n]` ✓
  - Buffer indexing: `global_col * L * dstate + time_idx * dstate + row` where `global_col=d`, `row=n` ✓
- ✅ **Step 2 (lines 1130-1139):** `dX_4_storage[d, n] += sum_k B_4[n, k] * dX_5_storage[d, k]` ✓
  - Indexing: `B_4_nk = gram_A_fp32[n * gram_size + k]` ✓
  - Indexing: `dX_5_dk = grad_output[k_idx]` where `k_idx = d * L * dstate + t * dstate + k` ✓

**Result:** ✅ **CORRECT** - Matches forward pass reverse exactly.

---

## Index Verification

### Buffer Layout Consistency

Both forward and backward use the same buffer layout:
- **Format:** `[batch, dim, seqlen, dstate]`
- **Index calculation:** `batch_idx * D * L * dstate + dim_idx * L * dstate + time_idx * dstate + state_idx`

This ensures:
- Forward writes to `X_5` at `buffer_idx`
- Backward reads `dX_5` from same `buffer_idx`
- **Consistency:** ✅ Verified in both cases

---

## Fixes Applied

### Fix 1: Step 1 - Initialize dX_4 (Both Cases)
**Issue:** Step 1 didn't distinguish between transposed/non-transposed indexing.
**Fix:** Split Step 1 into two cases:
- **Non-transposed (line 998-1017):** Uses `global_row * L * dstate + time_idx * dstate + col`
- **Transposed (line 1018-1038):** Uses `global_col * L * dstate + time_idx * dstate + row`

**Verification:** ✅ Correct indexing matches forward pass storage.

### Fix 2: Step 3 - Normalization Gradient (Both Cases)
**Issue:** Normalization gradient computation didn't handle transposed case.
**Fix:** Split Step 3 into two cases:
- **Non-transposed (line 1150-1194):** Uses `global_row` for `u`, `delta`, `B` indexing
- **Transposed (line 1195-1240):** Uses `global_col` for `u`, `delta`, and corrects `B` indexing to `global_col * B_d_stride + row * B_dstate_stride`

**Verification:** ✅ Correct indexing for both cases.

---

## Summary

### ✅ Non-Transposed Case: **VERIFIED CORRECT**
1. Forward: `X_5 = a*X_4 + B_4 @ X_4`
2. Backward: `dX_4 = a*dX_5 + B_4.T @ dX_5`
3. Implementation: ✅ Matches exactly

### ✅ Transposed Case: **VERIFIED CORRECT**
1. Forward: `X_5_storage = a*X_4_storage + sum_k B_4[n, k] * X_4_storage[d, k]`
2. Backward: `dX_4_storage = a*dX_5_storage + sum_k B_4[n, k] * dX_5_storage[d, k]`
3. Implementation: ✅ Matches exactly

### ✅ Both Cases: **PROPERLY FIXED**
- Step 1: Correct initialization for both cases ✓
- Step 2: Correct matrix multiplication gradient ✓
- Step 3: Correct normalization gradient for both cases ✓

---

## Conclusion

**The backward pass implementation is mathematically correct** and properly reverses the verified forward pass for both transposed and non-transposed cases.

The fixes applied ensure:
1. ✅ Correct indexing in Step 1 for both cases
2. ✅ Correct matrix multiplication gradient in Step 2 (already correct)
3. ✅ Correct normalization gradient in Step 3 for both cases

**Status:** ✅ **READY FOR TESTING**

---

**Trace completed by:** Comparison with verified forward pass implementation
**Reference:** `NS_CUDA_VERIFICATION_COMPLETE.md` (Forward pass verified to < 1% error with PyTorch)







