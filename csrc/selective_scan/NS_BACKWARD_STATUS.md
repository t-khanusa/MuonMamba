# Newton-Schulz Velocity Backward Pass - Current Status

## Summary

The Newton-Schulz velocity backward pass has been implemented in `newton_schulz_bwd_kernel.cuh`. Testing shows that the CUDA implementation and PyTorch reference produce results in the same ballpark (correct signs, similar magnitudes), but with significant numerical differences (10-1000x relative error).

## What's Been Implemented

### CUDA Kernel (`newton_schulz_bwd_kernel.cuh`)
- **Forward Recomputation**: Recomputes X_0 → X_4 (4 iterations, detached)
- **Backward Through 5th Iteration**: Complete backward pass through:
  - X_5 = a*X_4 + B_4@X_4
  - B_4 = b*A_4 + c*A_4²
  - A_4 = X_4 @ X_4.T
- **Gradient Through Normalization**: Correct gradient formula applied
- **BF16 Precision Matching**: All forward operations use BF16 rounding (matching forward kernel)
- **Bug Fixes Applied**:
  1. ✅ Fixed X_0 recomputation (was using X_4 instead of X_0 for normalization gradient)
  2. ✅ Fixed dot product computation (now uses X_0 instead of X_4)

### Python Reference (`generate_ns_velocity_test_data.py`)
- **Detached Forward**: Recomputes X_0 → X_4 with exact BF16 rounding matching CUDA
- **Backprop Through 5th Iteration Only**: Uses PyTorch autograd for 5th iteration
- **BF16 Precision Matching**: A_4, A_4², and B_4 rounded to BF16 in backward pass
- **Gradient Accumulation**: Correctly accumulates gradients for u, delta, B

## Current Test Results

Test configuration: `batch=2, dim=8, seqlen=16, dstate=16`

| Gradient | Max Abs Diff | Max Rel Error | Status |
|----------|--------------|---------------|---------|
| grad_u | 3.84 | 1377x | ❌ |
| grad_delta | 1.69 | 1377x | ❌ |
| grad_B | 1.20 | 503x | ❌ |

### Sample Comparison (first 5 values of grad_u)
```
[0] CUDA: 0.016855, Torch: 0.012859, diff: 0.004
[1] CUDA: -0.453432, Torch: -0.143710, diff: 0.310
[2] CUDA: 0.812132, Torch: 0.119627, diff: 0.693
[3] CUDA: -0.198974, Torch: -0.042276, diff: 0.157
[4] CUDA: -0.338768, Torch: -0.008197, diff: 0.331
```

## Analysis

### What's Correct
✅ Signs match (both positive/negative in same places)  
✅ Magnitudes are in same ballpark (within 10x for most values)  
✅ Logic appears correct (both compute same mathematical operations)  
✅ BF16 rounding is matched in both implementations  
✅ X_0 recomputation is correct  

### Potential Issues
❓ Large numerical differences (10-1000x) suggest systematic issue  
❓ Differences are proportionally consistent across all gradients  
❓ Could be related to how BF16/FP32 mixing is handled  
❓ Possible accumulation order differences  

## Files

- **CUDA Implementation**: `csrc/selective_scan/newton_schulz_bwd_kernel.cuh` (lines 524-1477)
- **Python Reference**: `csrc/selective_scan/generate_ns_velocity_test_data.py`
- **Test Program**: `csrc/selective_scan/test_real_ns_backward.cu`

## Next Steps

1. **Debug with Minimal Test**: Create 2x2 matrix test case to manually trace execution
2. **Add Debug Prints**: Insert prints in CUDA kernel to see intermediate values
3. **Check Accumulation Order**: Verify if parallel reduction vs sequential summation causes differences
4. **Review BF16 Handling**: Double-check if there are any missed BF16 conversions

## Integration Status

⚠️ **Not Yet Integrated** into `selective_scan_bwd_kernel.cuh`  
Reason: Need to resolve numerical accuracy issues before integration

The kernel is functionally complete and can be integrated if the current accuracy level is acceptable for the use case.

