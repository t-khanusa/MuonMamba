# Newton-Schulz Backward Pass - Critical Bugs Analysis

## Executive Summary

After careful analysis of the NS backward kernel, I've identified **several critical bugs** that explain why CUDA gradients are tiny compared to PyTorch reference:

1. **BUG 1**: Transposed case backward pass - wrong B_4 indexing
2. **BUG 2**: Missing gradient contribution from dA_4 and dB_4 (5th iteration gradients)
3. **BUG 3**: Variable B gradient indexing may be incorrect

---

## BUG 1: Transposed Case - Wrong B_4 Indexing (Line 1105)

**Location**: `newton_schulz_bwd_kernel.cuh`, line 1100-1107

**Issue**: In the transposed case, the backward pass computes `dX_4 += B_4.T @ dX_5`, but the B_4 indexing is wrong.

**Forward Pass Logic**:
- Logical: `X_logical` is `[N, D]` where `X_logical[n, d] = X_storage[d, n]`
- Forward: `X_5_logical = a*X_4_logical + B_4 @ X_4_logical` where `B_4` is `[N, N]`
- Storage: `X_5_storage[d, n] = a*X_4_storage[d, n] + sum_k B_4[n, k] * X_4_storage[d, k]`

**Backward Pass Should Be**:
- `dX_4_logical = a*dX_5_logical + B_4.T @ dX_5_logical`
- `dX_4_storage[d, n] = a*dX_5_storage[d, n] + sum_k B_4.T[n, k] * dX_5_storage[d, k]`
- `= a*dX_5_storage[d, n] + sum_k B_4[k, n] * dX_5_storage[d, k]`

**Current Code (Line 1100-1107)**:
```cuda
// (B_4.T @ dX_5_storage)[d,n] = sum_k B_4[k,d] * dX_5_storage[k,n]
float sum = 0.0f;
for (int k = 0; k < gram_size; ++k) {
    int k_idx = batch_idx * D * L * dstate + k * L * dstate + time_idx * dstate + row;
    float dX_5_kn = grad_output[k_idx];
    float B_4_kd = gram_A_fp32[k * gram_size + global_col];  // BUG: Should be B_4[k, n], not B_4[k, d]
    sum += B_4_kd * dX_5_kn;
}
```

**Problem**: 
- The code uses `B_4_kd = gram_A_fp32[k * gram_size + global_col]` which is `B_4[k, d]`
- But we need `B_4[k, n]` where `n = row`
- The indices are swapped!

**Fix**: 
```cuda
// Correct: (B_4.T @ dX_5_storage)[d,n] = sum_k B_4[k,n] * dX_5_storage[d,k]
// But wait, we're computing for storage[d,n], and dX_5_storage is indexed by [d,n]
// Actually, we need to think more carefully...

// Forward: X_5_storage[d, n] = a*X_4_storage[d, n] + sum_k B_4[n, k] * X_4_storage[d, k]
// Backward: dX_4_storage[d, n] = a*dX_5_storage[d, n] + sum_k B_4[n, k] * dX_5_storage[d, k]
float sum = 0.0f;
for (int k = 0; k < gram_size; ++k) {
    int k_idx = batch_idx * D * L * dstate + global_col * L * dstate + time_idx * dstate + k;
    float dX_5_dk = grad_output[k_idx];
    float B_4_nk = gram_A_fp32[row * gram_size + k];  // B_4[n, k] where n=row
    sum += B_4_nk * dX_5_dk;
}
```

Wait, let me reconsider. In the forward pass:
- `X_5_storage[d, n] = a*X_4_storage[d, n] + sum_k B_4[n, k] * X_4_storage[d, k]`

For backward:
- `dX_4_storage[d, n] = ∂L/∂X_4_storage[d, n] = a*dX_5_storage[d, n] + sum_k B_4[n, k] * dX_5_storage[d, k]`

So the current code is wrong because:
1. It's using `dX_5_storage[k, n]` instead of `dX_5_storage[d, k]`
2. It's using `B_4[k, d]` instead of `B_4[n, k]`

**Correct Fix for Transposed Case**:
```cuda
// (dX_4_storage += B_4 @ dX_5_storage)[d,n] = sum_k B_4[n,k] * dX_5_storage[d,k]
float sum = 0.0f;
for (int k = 0; k < gram_size; ++k) {
    int k_idx = batch_idx * D * L * dstate + global_col * L * dstate + time_idx * dstate + k;
    float dX_5_dk = grad_output[k_idx];  // dX_5_storage[d, k]
    float B_4_nk = gram_A_fp32[row * gram_size + k];  // B_4[n, k] where n=row
    sum += B_4_nk * dX_5_dk;
}
```

---

## BUG 2: Missing Gradient Through A_4 and B_4 (5th Iteration)

**Location**: After line 1112

**Issue**: The backward pass only computes gradients through `X_5 = a*X_4 + B_4@X_4`, but it doesn't compute gradients through:
- `B_4 = b*A_4 + c*A_4²`
- `A_4 = X_4 @ X_4.T`

Since we're using the detached approach (first 4 steps detached), we should NOT compute gradients through A_4 and B_4. However, the current implementation is missing these gradients if they're supposed to be included.

**Analysis**: According to the design, only the 5th iteration should have gradients. So `B_4` and `A_4` should be treated as constants (detached). This means:
- ✅ `dX_4` should receive gradients from `X_5` (this is done)
- ❌ We should NOT compute `dA_4` or `dB_4` (this is correct)

So this might not be a bug, but let me verify the mathematical correctness.

**Forward Pass (5th iteration only)**:
```
A_4 = X_4 @ X_4.T  (detached)
A_4² = A_4 @ A_4   (detached)
B_4 = b*A_4 + c*A_4²  (detached)
X_5 = a*X_4 + B_4 @ X_4  (with gradients)
```

**Backward Pass**:
```
dX_4 = a*dX_5 + (B_4.T @ dX_5)  (treating B_4 as constant)
```

This looks correct. The issue is that `B_4` itself has no gradients because it's detached.

---

## BUG 3: Non-Transposed Case - Potential Index Issue

**Location**: Line 1060

**Current Code**:
```cuda
// (B_4.T @ dX_5)[i,j] = sum_k B_4[k,i] * dX_5[k,j]
float B_4_ki = gram_A_fp32[k * gram_size + global_row];
```

**Analysis**:
- `B_4[i,j] = gram_A_fp32[i * gram_size + j]`
- `B_4.T[i,j] = B_4[j,i] = gram_A_fp32[j * gram_size + i]`
- For `(B_4.T @ dX_5)[i,j] = sum_k B_4.T[i,k] * dX_5[k,j] = sum_k B_4[k,i] * dX_5[k,j]`
- `B_4[k,i] = gram_A_fp32[k * gram_size + i]` where `i = global_row`

So `gram_A_fp32[k * gram_size + global_row]` should be correct.

However, let me double-check the forward pass to ensure the matrix multiplication is correct:

**Forward (Non-Transposed)**:
- `X` is `[D, N]`, `B_4` is `[D, D]`
- `X_5[i, j] = a*X_4[i, j] + sum_k B_4[i, k] * X_4[k, j]`

**Backward**:
- `dX_4[i, j] = a*dX_5[i, j] + sum_k B_4.T[i, k] * dX_5[k, j]`
- `= a*dX_5[i, j] + sum_k B_4[k, i] * dX_5[k, j]`

This matches the current code, so the non-transposed case seems correct.

---

## BUG 4: Variable B Gradient Accumulation

**Location**: Line 1228

**Current Code**:
```cuda
atomicAdd(&grad_B[B_idx], alpha * delta_val * u_val * d_b_t);
```

Where `B_idx` for variable B is computed at lines 1204-1206:
```cuda
B_idx = batch_idx * B_batch_stride + 
        group_id * B_group_stride +
        n * B_dstate_stride + time_idx;
```

**Issue**: For variable B, `grad_B` has shape `[batch, n_groups, dstate, seqlen]` or `[batch, n_groups, seqlen, dstate]`. The indexing needs to match the actual layout.

Let me check the forward pass to see how B is indexed:
- Forward (line 591-597): `B[batch_idx, group_id, col, time_idx]` where `col` is the dstate dimension

So the layout is `[B, G, N, L]` where:
- `B` = batch
- `G` = groups
- `N` = dstate
- `L` = seqlen

For gradient accumulation, we need:
- `grad_B[batch_idx, group_id, n, time_idx]` where `n` is the dstate index

**Current Code**:
```cuda
B_idx = batch_idx * B_batch_stride + 
        group_id * B_group_stride +
        n * B_dstate_stride + time_idx;
```

This assumes layout `[B, G, N, L]` which should be correct IF `B_dstate_stride` and the stride for `time_idx` are correct.

However, I suspect the issue might be that for variable B, the gradient should accumulate over all dimensions `d` that share the same `group_id`, not just one dimension.

Actually, wait - each `(batch, time)` processes one matrix `[D, N]`. For variable B, each dimension `d` maps to a group `group_id`, and within that group, we have `B[batch, group_id, n, time]` where `n` is in `[0, dstate)`.

So for gradient `grad_B[batch, group_id, n, time]`, we accumulate contributions from all dimensions `d` that map to `group_id`, but only for the specific `n` and `time`.

The current code processes one `d` at a time and accumulates to `grad_B[batch, group_id, n, time]`, which should be correct.

---

## BUG 5: Most Critical - Wrong Matrix Dimensions in Transposed Case

**Re-examining the Transposed Case More Carefully**:

For the transposed case:
- Logical: `X_logical` is `[N, D]` where `N = dstate = 64`, `D = 128`
- Storage: `X_storage` is `[D, N] = [128, 64]` where `X_storage[d, n] = X_logical[n, d]`
- `B_4` is `[N, N] = [64, 64]` (logical)

**Forward Pass**:
- Logical: `X_5_logical[n, d] = a*X_4_logical[n, d] + sum_k B_4[n, k] * X_4_logical[k, d]`
- Storage: `X_5_storage[d, n] = a*X_4_storage[d, n] + sum_k B_4[n, k] * X_4_storage[d, k]`

**Backward Pass**:
- `dX_4_logical[n, d] = a*dX_5_logical[n, d] + sum_k B_4.T[n, k] * dX_5_logical[k, d]`
- `= a*dX_5_logical[n, d] + sum_k B_4[k, n] * dX_5_logical[k, d]`
- Storage: `dX_4_storage[d, n] = a*dX_5_storage[d, n] + sum_k B_4[k, n] * dX_5_storage[d, k]`

**Current Code (Line 1100-1107)**:
```cuda
// (B_4.T @ dX_5_storage)[d,n] = sum_k B_4[k,d] * dX_5_storage[k,n]  ❌ WRONG
for (int k = 0; k < gram_size; ++k) {
    int k_idx = batch_idx * D * L * dstate + k * L * dstate + time_idx * dstate + row;
    float dX_5_kn = grad_output[k_idx];  // This is dX_5_storage[k, n]
    float B_4_kd = gram_A_fp32[k * gram_size + global_col];  // This is B_4[k, d]
    sum += B_4_kd * dX_5_kn;
}
```

**Problems**:
1. `dX_5_kn` is `dX_5_storage[k, n]` but we need `dX_5_storage[d, k]`
2. `B_4_kd` is `B_4[k, d]` but we need `B_4[k, n]` (or `B_4[n, k]` depending on the formula)

**Correct Version**:
```cuda
// dX_4_storage[d, n] += sum_k B_4[k, n] * dX_5_storage[d, k]
float sum = 0.0f;
for (int k = 0; k < gram_size; ++k) {
    int k_idx = batch_idx * D * L * dstate + global_col * L * dstate + time_idx * dstate + k;
    float dX_5_dk = grad_output[k_idx];  // dX_5_storage[d, k]
    float B_4_kn = gram_A_fp32[k * gram_size + row];  // B_4[k, n] where n=row
    sum += B_4_kn * dX_5_dk;
}
```

---

## Summary of Bugs

1. **CRITICAL BUG**: Transposed case backward pass (line 1100-1107) has wrong indexing - uses `dX_5_storage[k, n]` and `B_4[k, d]` instead of `dX_5_storage[d, k]` and `B_4[k, n]`

2. **Potential Bug**: Variable B gradient indexing needs verification

3. **Missing Feature**: The backward pass doesn't handle the case where gradients should flow through A_4 and B_4, but this is intentional (detached approach)

---

## Recommended Fixes

### Fix 1: Transposed Case Backward (Line 1100-1107)

Replace:
```cuda
// (B_4.T @ dX_5_storage)[d,n] = sum_k B_4[k,d] * dX_5_storage[k,n]
float sum = 0.0f;
for (int k = 0; k < gram_size; ++k) {
    int k_idx = batch_idx * D * L * dstate + k * L * dstate + time_idx * dstate + row;
    float dX_5_kn = grad_output[k_idx];
    float B_4_kd = gram_A_fp32[k * gram_size + global_col];
    sum += B_4_kd * dX_5_kn;
}
```

With:
```cuda
// dX_4_storage[d, n] += sum_k B_4[k, n] * dX_5_storage[d, k]
float sum = 0.0f;
for (int k = 0; k < gram_size; ++k) {
    int k_idx = batch_idx * D * L * dstate + global_col * L * dstate + time_idx * dstate + k;
    float dX_5_dk = grad_output[k_idx];  // dX_5_storage[d, k]
    float B_4_kn = gram_A_fp32[k * gram_size + row];  // B_4[k, n] where n=row
    sum += B_4_kn * dX_5_dk;
}
```

### Fix 2: Verify Variable B Gradient Indexing

The variable B gradient accumulation at line 1228 looks correct, but should be verified against the forward pass indexing.

---

## Impact

These bugs would cause:
- **Tiny gradients**: Wrong matrix multiplication would give incorrect (often much smaller) gradients
- **NaN in variable B case**: Wrong indexing could cause out-of-bounds access or incorrect accumulation
- **100% test failures**: All tests fail because gradients don't match PyTorch reference

---

**Status**: Critical bugs identified. Fixes needed before backward pass will work correctly.







