# Comprehensive Forward Pass Test

## Overview

This test suite (`test_comprehensive_forward.py`) comprehensively validates the CUDA implementation of the selective scan forward pass against a PyTorch reference implementation.

## Test Coverage

The test suite covers:

1. **Momentum Mode (beta != 0)** - All combinations:
   - Constant B, constant C
   - Variable B, constant C
   - Constant B, variable C
   - Variable B, variable C

2. **Matrix Shapes**:
   - Tall matrices (dim > dstate)
   - Fat matrices (dim < dstate)
   - Square matrices (dim = dstate)
   - Large production-scale dimensions

3. **Parameters**:
   - Different beta values
   - Different alpha values
   - With/without skip connection (D)

## Mathematical Equations Verified

The test validates all 5 core equations:

1. **b_t = alpha × delta_t × B_t × u_t**
   - Computed per timestep
   - Handles constant and variable B

2. **b_t_ortho = NewtonSchulz5(b_t)**
   - Applied when momentum is enabled (beta != 0)
   - Uses BF16 arithmetic matching CUDA

3. **v_t = beta × v_{t-1} + b_t_ortho**
   - Velocity state update
   - Momentum accumulation

4. **h_t = exp(delta_t × A) × h_{t-1} + v_t**
   - Hidden state update
   - Exponential decay with velocity input

5. **y_t = C × h_t + D × u_t**
   - Output computation
   - **Critical fix**: For momentum mode with constant B, uses C (not B*C) since B already in b_t

## Key Fixes Validated

### Constant B Handling in Momentum Mode

**Before Fix:**
- Output used `B*C*h`, causing double application of B
- Mathematically incorrect

**After Fix:**
- Momentum mode (beta != 0): Output uses `C*h` (B already in b_t)
- Original mode (beta == 0): Output uses `B*C*h` (B deferred, original Mamba optimization)

This is validated in all test cases.

## Running the Test

```bash
cd /project/khanhnt/muontest/Momentum_correct
python test_comprehensive_forward.py
```

## Expected Output

The test will:
1. Run each test case
2. Compare CUDA output vs PyTorch reference
3. Report absolute and relative differences
4. Show worst mismatches if any
5. Provide a summary of passed/failed tests

## Tolerance Levels

- **Standard tests**: `tol_abs=1e-3`, `tol_rel=1e-2`
- **Production scale**: `tol_abs=5e-3`, `tol_rel=5e-2` (relaxed for BF16)

## Test Cases

1. Basic Momentum (const B, C)
2. Momentum (var B, const C)
3. Momentum (const B, var C)
4. Momentum (var B, var C)
5. Large Dimensions
6. Tall Matrix
7. Fat Matrix
8. With Skip Connection
9. Different Alpha
10. Different Beta
11. Production Scale

## Notes

- Complex A support is skipped for now (complex NS needs refinement)
- Original Mamba mode (beta == 0) tests removed (use original Mamba test suite)
- Focus is on validating the momentum mode fixes

## Reference Implementation

The PyTorch reference (`selective_scan_ref_fixed`) exactly matches the CUDA kernel logic:
- Same equation order
- Same B/C handling
- Same Newton-Schulz integration
- Same BF16 arithmetic for NS

This ensures mathematical correctness validation.



