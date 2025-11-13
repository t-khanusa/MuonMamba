# Variable B Indexing Debug Report

## Issue
Variable B tests show ~2.4x error (58% relative error) between CUDA and reference.

## Root Cause Analysis

### Expected Tensor Layout
According to `selective_scan.cpp` line 278:
```cpp
CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
```
Shape: `[batch, n_groups, dstate, seqlen]` = `[B, G, N, L]`

### CUDA Kernel Indexing

#### Newton-Schulz Kernel (line 1683-1685)
```cpp
// Variable B: [B, G, L, N]  <- COMMENT SAYS THIS, BUT ACTUAL SHAPE IS [B, G, N, L]
int group_size = (D + n_groups - 1) / n_groups;
int group_id = min(global_row / group_size, n_groups - 1);
B_val = to_float(B[batch_idx * B_batch_stride + 
                   group_id * B_group_stride +
                   time_idx * dstate + col]);  // Uses time_idx * dstate + col
```
**This indexes as**: `B[b, g, t*N + n]` which assumes `[B, G, L, N]` layout.

#### Scan Kernel (line 188)
```cpp
load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals, ...);
```
Where `Bvar = B[batch_id, group_id, 0, 0]` and `B_dstate_stride = B.stride(2) = seqlen` for `[B, G, N, L]`.

**This indexes as**: `B[b, g, n, t]` which matches `[B, G, N, L]` layout.

### The Problem
**NS kernel and scan kernel use different indexing schemes!**

- NS kernel: Treats B as `[B, G, L, N]` → `B[b, g, t*N + n]`
- Scan kernel: Treats B as `[B, G, N, L]` → `B[b, g, n*L + t]`

This is a **bug in the CUDA NS kernel** - it uses the wrong indexing for variable B.

### Verification
Test script shows:
- NS kernel offset `42 = 5*8 + 2` accesses `B_flat[42] = 0.164`
- Correct access `B[0, 0, 2, 5]` = `B_flat[69] = 0.538`
- **They don't match!**

## Solutions Attempted

1. ✅ Matched group_id calculation (ceiling division)
2. ❌ Transposed B to match NS kernel layout - didn't fix error
3. ❌ Used flattened offset matching - still ~2x error

## Current Status

**Variable B indexing bug confirmed in CUDA NS kernel**

The NS kernel incorrectly assumes `[B, G, L, N]` layout when the actual shape is `[B, G, N, L]`.

## Recommendations

### Option 1: Fix CUDA Kernel (Recommended)
Update `newton_schulz_fwd_kernel.cuh` line 1683-1685 to use correct indexing:
```cpp
// Instead of: time_idx * dstate + col
// Use: col * B_dstate_stride + time_idx
// But B_dstate_stride needs to be passed or computed correctly
```

### Option 2: Fix Reference Implementation
Make reference match CUDA's (incorrect) behavior by:
- Transposing B before use (attempted, didn't fully fix)
- Or using the flattened offset matching NS kernel

### Option 3: Accept Current State
- Constant B cases work perfectly (7/11 tests passing)
- Variable B cases have known indexing bug
- Document the limitation

## Impact

- **Constant B**: ✅ 100% correct
- **Variable B**: ❌ ~2x error due to indexing bug
- **Overall**: 7/11 tests passing (64%)

## Next Steps

1. **Option A**: Fix CUDA NS kernel indexing for variable B
2. **Option B**: Update reference to match CUDA's buggy behavior exactly
3. **Option C**: Document limitation and use constant B in production

**Recommendation**: Option A - Fix the CUDA kernel to use correct `[B, G, N, L]` indexing.



