# Correctness Analysis: Selective Scan with Newton-Schulz

## Equations to Verify

1. `b_t = alpha * delta * B * u_t` for t in range(sequence_length)
2. `b_ortho = Newton-Schulz5(b)` where NS applies to [D,N] matrices for each (batch, timestep) pair
3. `v_t = beta * v_{t-1} + b_t_ortho`
4. `h_t = exp(delta*A) * h_{t-1} + v_t`
5. `y_t = C_t * h_t + D_t * u_t`

---

## Critical Issues Found

### ❌ **ISSUE 1: Complex Case Indexing Bug in Scan Kernel**

**Location**: `selective_scan_fwd_kernel.cuh` lines 283-286

**Problem**: The complex case multiplies `params.dstate * 2` in the indexing, but `X_4_buffer` is stored as `[batch, dim, seqlen, dstate]` with float32 values. For complex weights, the storage should still be `[batch, dim, seqlen, dstate]` but each element represents a complex number.

**Current Code**:
```cuda
int global_idx = batch_id * params.dim * params.seqlen * params.dstate * 2 +
               d * params.seqlen * params.dstate * 2 +
               t * params.dstate * 2 +
               state_idx * 2;
delta_B_u_val = complex_t(velocity_ortho_buffer[global_idx], velocity_ortho_buffer[global_idx + 1]);
```

**Issue**: The buffer shape is `[batch, dim, seqlen, dstate]` in float32, not `[batch, dim, seqlen, dstate*2]`. The indexing should be:
```cuda
int global_idx = batch_id * params.dim * params.seqlen * params.dstate +
               d * params.seqlen * params.dstate +
               t * params.dstate +
               state_idx;
// For complex, we need to check how complex values are stored in X_4_buffer
// If stored as interleaved [real, imag], then we need * 2
// But if stored as separate real/imag arrays, this is wrong
```

**Fix Needed**: Verify how complex values are stored in `X_4_buffer`. If NS kernel outputs float32 (real part only), then complex case needs special handling.

---

### ❌ **ISSUE 2: Missing Alpha Multiplication in NS Path**

**Location**: `selective_scan_fwd_kernel.cuh` lines 255, 287

**Problem**: When loading from `X_4_buffer`, the value is already `b_t_ortho` (orthogonalized `b_t`). The NS kernel computes `b_t = alpha * delta * B * u` and then applies NS. However, the scan kernel should use `b_t_ortho` directly without additional scaling.

**Current Code**:
```cuda
delta_B_u = velocity_ortho_buffer[global_idx];  // Already b_t_ortho
```

**Analysis**: The NS kernel (line 1693) computes `b_t_val = alpha * delta_val * B_val * u_val`, so `alpha` is already applied. This appears CORRECT.

**However**: The scan kernel in non-NS mode (line 264) applies `params.alpha` again. But in NS mode, it doesn't. This is CORRECT because NS kernel already applied alpha.

✅ **This is actually correct** - no fix needed.

---

### ⚠️ **ISSUE 3: Efficiency - Synchronization After NS Kernel**

**Location**: `selective_scan_fwd_kernel.cuh` line 501

**Problem**: `cudaDeviceSynchronize()` is called after NS kernel, which blocks the entire GPU. This is inefficient.

**Current Code**:
```cuda
cudaDeviceSynchronize();
```

**Impact**: Blocks all GPU work until NS completes, preventing overlap with other operations.

**Fix**: Remove `cudaDeviceSynchronize()` and rely on stream synchronization. The scan kernel will naturally wait for NS kernel to complete if they're on the same stream.

---

### ❌ **ISSUE 4: Mathematical Correctness - NS Application Scope**

**Location**: `newton_schulz_fwd_kernel.cuh` lines 1604-2138

**Problem**: The user's equation states: "NS just apply for 2D matrix, we choose apply NS for full dim and full dstate [D,N] with B*L times"

**Current Implementation**: The NS kernel processes each (batch, timestep) pair independently, applying NS to a [D, N] matrix. This is `batch * seqlen` separate NS applications.

**Analysis**: 
- For each (batch, timestep), we have `b_t` with shape `[D, N]` (before NS)
- NS is applied to each `[D, N]` matrix independently
- This matches the user's description: "B*L times" (batch * seqlen times)

✅ **This is correct** - NS is applied per (batch, timestep) to [D,N] matrices.

---

### ⚠️ **ISSUE 5: Variable B Indexing Consistency**

**Location**: `newton_schulz_fwd_kernel.cuh` lines 1687-1689

**Problem**: Variable B shape is `[batch, n_groups, dstate, seqlen]` = `[B, G, N, L]`. The indexing uses:
```cuda
B_val = to_float(B[batch_idx * B_batch_stride + 
                   group_id * B_group_stride +
                   col * B_dstate_stride + time_idx]);
```

**Analysis**: This indexes as `B[b, g, n, t]` which matches `[B, G, N, L]` layout. ✅ **This is correct** (was fixed in a previous commit).

---

### ❌ **ISSUE 6: Hidden State Scan - Missing Exp Correction**

**Location**: `selective_scan_fwd_kernel.cuh` lines 334-337

**Problem**: The equation states `h_t = exp(delta*A) * h_{t-1} + v_t`, but the code uses `exp2f` with `LOG2E` scaling.

**Current Code**:
```cuda
thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]), velocity_data[i].y);
```

**Analysis**: 
- Line 174-178: `A_val[r]` is multiplied by `kLog2e` (M_LOG2E ≈ 1.4427)
- So `exp2f(delta * (A * LOG2E)) = exp(delta * A)` ✅ **This is correct**

---

### ⚠️ **ISSUE 7: Output Computation - C_t Handling**

**Location**: `selective_scan_fwd_kernel.cuh` lines 377-389

**Problem**: The equation is `y_t = C_t * h_t + D_t * u_t`, but the code has complex logic for when to use `B*C` vs just `C`.

**Current Code**:
```cuda
const weight_t C_val = !kIsVariableC
    ? BC_val[r]  // Either C (momentum) or B*C (original Mamba)
    : (!kIsVariableB ? 
        (params.use_newton_schulz || params.beta != 1.0f ? C_vals[i] : BC_val[r] * C_vals[i])  // B const, C var
        : C_vals[i]);  // B var, C var
```

**Analysis**: 
- For momentum mode (NS or beta != 1.0): B is already applied in `b_t`, so output should be `y = C * h` (not `B*C * h`)
- For original Mamba (beta == 1.0, no NS): B is applied at output, so `y = B*C * h`

✅ **This logic is correct** - matches the comment on lines 212-215.

---

### ✅ **ISSUE 8: Complex Case Storage in X_4_buffer - FIXED**

**Location**: `newton_schulz_fwd_kernel.cuh` and `selective_scan_fwd_kernel.cuh`

**Problem**: When `weight_t` is complex, the NS kernel only stored the **real part** of `b_t`, but the scan kernel tried to read it as a **complex** value.

**Fix Applied**:
1. ✅ Added `is_complex_type` trait to detect complex `weight_t`
2. ✅ Updated NS kernel to compute complex `b_t = alpha * delta * B * u` (handles both real and imag)
3. ✅ Updated NS kernel to store both real and imag parts in interleaved format: `[real, imag, real, imag, ...]`
4. ✅ Updated buffer allocation in `selective_scan.cpp` to `[batch, dim, seqlen, dstate*2]` for complex case
5. ✅ Updated scan kernel reading logic (already correct - uses `*2` for complex indexing)
6. ⚠️ **Partial**: NS iterations currently use real part only for complex matrices
   - TODO: Implement full complex NS using Hermitian transpose `A = X @ X.H`

**Current Status**: 
- Complex `b_t` computation: ✅ **FIXED**
- Complex storage: ✅ **FIXED** (interleaved real/imag)
- Buffer allocation: ✅ **FIXED**
- Scan kernel reading: ✅ **CORRECT** (already had correct indexing)
- NS iterations for complex: ⚠️ **SIMPLIFIED** (uses real part only; full complex NS needs implementation)

---

## Summary of Issues

| Issue | Severity | Status | Location |
|-------|----------|--------|----------|
| Complex case indexing | ❌ **CRITICAL** | Needs fix | `selective_scan_fwd_kernel.cuh:283-286` |
| Complex storage in X_4_buffer | ❌ **CRITICAL** | Needs fix | NS kernel + scan kernel |
| cudaDeviceSynchronize() | ⚠️ **EFFICIENCY** | Should remove | `selective_scan_fwd_kernel.cuh:501` |
| Alpha multiplication | ✅ **CORRECT** | No fix needed | - |
| NS application scope | ✅ **CORRECT** | No fix needed | - |
| Variable B indexing | ✅ **CORRECT** | Fixed previously | - |
| Hidden state exp | ✅ **CORRECT** | No fix needed | - |
| Output C handling | ✅ **CORRECT** | No fix needed | - |

---

## Recommendations

1. **Fix complex case**: Implement proper complex storage in `X_4_buffer` for complex weights
2. **Remove synchronization**: Remove `cudaDeviceSynchronize()` and use stream-based synchronization
3. **Add bounds checking**: The bounds check on line 250 is good, but should also verify `d < params.dim`
4. **Documentation**: Add comments explaining the complex storage format for `X_4_buffer`

