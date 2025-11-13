# Complex Case Fix Summary

## Issue Fixed
Fixed critical bug where complex `b_t` values were only storing the real part, causing the scan kernel to read uninitialized memory for the imaginary part.

## Changes Made

### 1. Newton-Schulz Kernel (`newton_schulz_fwd_kernel.cuh`)
- ✅ Added `is_complex_type` trait to detect complex `weight_t`
- ✅ Updated `b_t` computation to handle complex: `b_t = alpha * delta * B * u` (both real and imag)
- ✅ Updated storage to interleaved format: `[real, imag, real, imag, ...]`
- ✅ Updated buffer indexing throughout to account for `dstate*2` for complex case
- ⚠️ NS iterations use real part only (full complex NS needs Hermitian transpose)

### 2. Buffer Allocation (`selective_scan.cpp`)
- ✅ Updated `X_4_buffer` allocation: `[batch, dim, seqlen, dstate*2]` for complex
- ✅ Updated `grad_X_4_buffer` allocation: same as above

### 3. Scan Kernel (`selective_scan_fwd_kernel.cuh`)
- ✅ Already correct - uses `*2` indexing for complex case (lines 283-287)

## Testing

To test the fix, run with complex weights:
```python
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# Test with complex A
batch, dim, seqlen, dstate = 2, 4, 8, 4
u = torch.randn(batch, dim, seqlen, dtype=torch.float32, device='cuda')
delta = torch.randn(batch, dim, seqlen, dtype=torch.float32, device='cuda')
A = torch.randn(dim, dstate, dtype=torch.complex64, device='cuda')
B = torch.randn(dim, dstate, dtype=torch.complex64, device='cuda')
C = torch.randn(dim, dstate, dtype=torch.complex64, device='cuda')

# Use momentum mode (beta != 0) to trigger NS
out, x, x_4 = selective_scan_fn(u, delta, A, B, C, beta=0.5, alpha=1.0)

# Check that x_4 buffer has correct shape (dstate*2 for complex)
assert x_4.shape == (batch, dim, seqlen, dstate * 2), f"Expected shape {(batch, dim, seqlen, dstate * 2)}, got {x_4.shape}"
```

## Known Limitations

1. **NS Iterations for Complex**: Currently uses real part only. Full complex NS would require:
   - Hermitian transpose: `A = X @ X.H` instead of `X @ X.T`
   - Complex matrix operations throughout NS iterations
   - This is a TODO for future enhancement

2. **Performance**: Interleaved storage doubles memory usage for complex case, but is necessary for correctness.

## Verification Checklist

- [x] Complex `b_t` computation includes both real and imag
- [x] Buffer stores interleaved real/imag values
- [x] Buffer allocation accounts for complex case
- [x] Scan kernel reads complex values correctly
- [ ] NS iterations fully support complex (currently simplified)
- [ ] Backward pass handles complex correctly (needs verification)

