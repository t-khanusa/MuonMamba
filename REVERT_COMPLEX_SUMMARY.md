# Reverted Complex Storage - Keep X_4_buffer Real

## Changes Made

### 1. Buffer Allocation (`selective_scan.cpp`)
- ✅ Reverted `X_4_buffer` allocation to `[batch, dim, seqlen, dstate]` (real only)
- ✅ Reverted `grad_X_4_buffer` allocation to same shape
- ✅ Added comments noting complex support will be added after backward pass works

### 2. Newton-Schulz Kernel (`newton_schulz_fwd_kernel.cuh`)
- ✅ **Storage (b_t computation)**: Stores only real part of `b_t` (line 1736-1743)
- ✅ **Normalization**: Simplified to handle real values only (line 1797-1811)
- ✅ **Case 1 (Non-transposed, Gram computation)**: Fixed buffer indexing (lines 1849-1888)
- ✅ **Case 2 (Transposed, Gram computation)**: Fixed buffer indexing (lines 1909-1956)
- ✅ **Case 3 (X = a*X + B@X, non-transposed)**: Fixed buffer indexing (lines 2037-2083)
- ✅ **Case 3 (X = a*X + B@X, transposed)**: Fixed buffer indexing (lines 2104-2152)

### 3. Scan Kernel (`selective_scan_fwd_kernel.cuh`)
- ✅ Updated complex case reading to use real part only (line 286-294)
- ✅ Added bounds check for safety

## Current Behavior

For **complex weights**:
- `b_t` is computed as complex: `b_t = alpha * delta * B * u` (both real and imag)
- Only **real part** is stored in `X_4_buffer`
- Imaginary part is **discarded** (will be fixed after backward pass works)
- Scan kernel reads real part and sets imag to 0

## Next Steps (After Backward Pass Works)

1. Update buffer allocation to `[batch, dim, seqlen, dstate*2]` for complex
2. Store both real and imag parts in interleaved format
3. Update scan kernel to read both parts
4. Implement full complex NS with Hermitian transpose

## Files Modified

- `csrc/selective_scan/selective_scan.cpp` - Buffer allocation
- `csrc/selective_scan/newton_schulz_fwd_kernel.cuh` - All buffer indexing fixed
- `csrc/selective_scan/selective_scan_fwd_kernel.cuh` - Scan kernel reading

