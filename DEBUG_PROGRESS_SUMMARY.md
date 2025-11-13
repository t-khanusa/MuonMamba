# Debug and Fix NS Backward Pass - Progress Summary

## Completed Tasks

### 1. Added Debug Output to CUDA Kernels ✅
- Added debug prints to `selective_scan_bwd_kernel.cuh`:
  - Lines 393-396: Print `dv` values per timestep before accumulation
  - Lines 402-404: Print `grad_X_4_buffer` values after accumulation
- Added debug prints to `newton_schulz_bwd_kernel.cuh`:
  - Lines 1017-1020: Print `grad_output` reading per timestep
  - Lines 1323-1335: Print `grad_u` before/after accumulation
  - Lines 1351-1363: Print `grad_delta` before/after accumulation

### 2. Created PyTorch Reference ✅
- Created `test_backward_debug_reference.py` with accurate PyTorch reference matching CUDA:
  - `pytorch_ns_backward_ref_debug`: NS backward with detached first 4 steps
  - `selective_scan_forward_ref_debug`: Forward pass matching CUDA
  - `selective_scan_backward_ref_debug`: Backward pass matching CUDA

### 3. Created Comparison Test Script ✅
- Created `test_backward_debug_comparison.py`:
  - Compares CUDA vs PyTorch gradients
  - Analyzes NS contribution per timestep
  - Identifies bug: timesteps 1-3 have zero NS contribution in CUDA

## Bug Confirmed

**Bug**: CUDA's NS backward only produces gradients for timestep 0. Timesteps 1-3 have NO NS backward contribution to `du`.

**Evidence**:
- Timestep 0: CUDA NS contrib = [0.208, 0.020] ✅
- Timestep 1: CUDA NS contrib = [0.000, 0.000] ❌ (PyTorch: [-0.003, 0.011])
- Timestep 2: CUDA NS contrib = [0.000, 0.000] ❌ (PyTorch: [0.100, -0.028])
- Timestep 3: CUDA NS contrib = [0.000, 0.000] ❌ (PyTorch: [-0.116, -0.121])

## Next Steps

### 1. Investigate Root Cause
The bug could be:
- `grad_X_4_buffer` not accumulated correctly for timesteps 1-3 in main backward kernel
- NS backward kernel not reading/writing correctly for timesteps 1-3
- Buffer indexing mismatch between main backward and NS backward

### 2. Fix CUDA Code
Once root cause is identified:
- Fix `grad_X_4_buffer` accumulation logic if needed
- Fix NS backward kernel reading/writing logic if needed
- Verify buffer indexing matches between kernels

### 3. Verify Fix
- Run comparison test again
- Verify all timesteps have non-zero NS contribution when expected
- Verify gradients match PyTorch reference within tolerance

## Files Modified

1. `csrc/selective_scan/selective_scan_bwd_kernel.cuh` - Added debug prints
2. `csrc/selective_scan/newton_schulz_bwd_kernel.cuh` - Added debug prints
3. `test_backward_debug_reference.py` - New PyTorch reference
4. `test_backward_debug_comparison.py` - New comparison test script




