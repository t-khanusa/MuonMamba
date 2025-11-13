# Comprehensive NS Backward Test Suite - Summary

## What Was Created

A comprehensive test suite (`test_comprehensive_ns_backward.py`) with **20 test cases** covering:

### Test Categories

1. **Small Configurations (4 tests)**
   - Basic momentum with const/variable B and C
   - Quick verification tests

2. **Medium Configurations (4 tests)**
   - Medium-sized matrices (B=4, D=32, L=128, N=16)
   - Different beta/alpha values
   - Variable B testing

3. **Large Configurations (2 tests)**
   - Large matrices (B=8, D=64, L=256, N=32)
   - With and without skip connections

4. **Production Configurations (7 tests)**
   - **Full production size**: B=16, D=128, L=512, N=64
   - Different batch sizes (4, 8, 16)
   - Variable B/C combinations
   - Different beta/alpha values
   - Skip connections
   - Tall matrix handling (D > N)

5. **Edge Cases (3 tests)**
   - Very high beta (0.99)
   - Very low alpha (0.1)

## Current Status

**Issue Identified:** The test reveals significant differences between CUDA and PyTorch reference gradients. This suggests one of:

1. NS backward kernel may not be correctly integrated
2. NS backward kernel may not be computing gradients correctly
3. Reference implementation may not match CUDA's approach exactly

**Observed Behavior:**
- CUDA gradients are very small (order 1e-4 to 1e-2)
- Reference gradients are much larger (order 1e+0 to 1e+1)
- All gradients show 100% exceed tolerance ratio

## Next Steps for Debugging

### Step 1: Verify NS Backward is Being Called

Add debugging to check if `grad_X_4_buffer` is being populated:

```cpp
// In selective_scan_bwd_kernel.cuh, after accumulating to grad_X_4_buffer
if (params.use_newton_schulz && threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[NS BWD DEBUG] grad_X_4_buffer[0] = %.6f\n", grad_X_4_buffer[0]);
}
```

### Step 2: Verify NS Backward Kernel is Executing

Add debug prints in `newton_schulz_bwd_kernel.cuh`:

```cuda
if (tid == 0 && batch_idx == 0 && time_idx == 0) {
    printf("[NS BWD] Kernel launched: batch=%d, time=%d\n", batch_idx, time_idx);
    printf("[NS BWD] grad_output[0,0] = %.6f\n", grad_output[...]);
}
```

### Step 3: Check Gradient Accumulation

Verify that NS backward is adding gradients correctly:

```cuda
// Before adding gradients
float grad_u_before = grad_u[0];
// ... NS backward computation ...
float grad_u_after = grad_u[0];
if (tid == 0 && batch_idx == 0 && time_idx == 0) {
    printf("[NS BWD] grad_u changed: %.6f -> %.6f (diff=%.6f)\n", 
           grad_u_before, grad_u_after, grad_u_after - grad_u_before);
}
```

### Step 4: Simplify Reference Implementation

The reference implementation in `selective_scan_backward_ref` may have bugs. Consider:

1. **Testing without NS first**: Set `beta=0.0` and verify backward pass matches
2. **Testing NS backward in isolation**: Create a standalone NS backward test
3. **Matching CUDA's exact approach**: 
   - Accumulate gradients into `grad_X_4_buffer` 
   - Call NS backward which adds to original gradients

## Test File Structure

The test file includes:

1. **`pytorch_ns_backward_ref()`**: PyTorch reference for NS backward (detached first 4 steps)
2. **`selective_scan_backward_ref()`**: Full selective scan backward with NS integration
3. **`compare_gradients()`**: Gradient comparison with adaptive tolerance
4. **`test_backward_case()`**: Single test case runner
5. **`main()`**: Test suite runner with 20 production cases

## How to Run

```bash
# Run all tests
python test_comprehensive_ns_backward.py

# Run with verbose output
python test_comprehensive_ns_backward.py 2>&1 | tee ns_backward_test.log

# Run specific test (modify main() to filter)
```

## Expected Output Format

For each test:
```
================================================================================
Test: [Test Name]
================================================================================
  Config: B=X, D=Y, L=Z, N=W
  beta=X, alpha=Y
  variable_B=..., variable_C=...
  
  CUDA grad stats:
    du: mean=..., max=...
    ddelta: mean=..., max=...
    ...
  
✅/❌ du:
  Max abs diff: ...
  Max rel diff: ...
  ...
  
✅/❌ ddelta:
  ...
```

## Known Issues

1. **Large gradient differences**: CUDA and reference gradients differ significantly
   - May indicate NS backward not being called
   - May indicate NS backward implementation bug
   - May indicate reference implementation bug

2. **Reference implementation complexity**: The PyTorch reference tries to replicate CUDA's exact approach but may have bugs

## Recommendations

1. **First**: Verify NS backward kernel is actually being called (add debug prints)
2. **Second**: Test NS backward in isolation (without full selective scan)
3. **Third**: Fix reference implementation to match CUDA's exact gradient flow
4. **Fourth**: Re-run comprehensive test suite

## Files Modified

- `test_comprehensive_ns_backward.py`: Comprehensive test suite (created)

## Related Files

- `csrc/selective_scan/newton_schulz_bwd_kernel.cuh`: NS backward kernel implementation
- `csrc/selective_scan/selective_scan_bwd_kernel.cuh`: Selective scan backward with NS integration
- `csrc/selective_scan/selective_scan.cpp`: Backward entry point
- `csrc/selective_scan/test_ns_backward.py`: Standalone NS backward tests

---

**Status**: Test suite created and ready. Debugging required to resolve gradient differences.

**Date**: November 2025







