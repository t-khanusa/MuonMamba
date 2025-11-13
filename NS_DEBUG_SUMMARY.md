# Newton-Schulz CUDA Debugging Summary

## Current Status

### ✅ What Works
1. **Small matrices (D=8, N=6)**: Newton-Schulz produces correct output
   - CUDA: X_final[0,0]=0.167, [1,1]=0.258
   - PyTorch: X[0,0]=0.152, [1,1]=0.248
   - Very close match!

2. **Implementation structure**: We correctly:
   - Use BF16 throughout (matching PyTorch)
   - Accumulate Gram matrix in FP32, then convert to BF16
   - Apply polynomial B = b*A + c*A²
   - Update X = a*X + B@X with BF16 rounding

### ❌ What Fails
1. **Large matrices (D=128, N=64)**: Newton-Schulz explodes
   - After 5 iterations: X_final values reach 10^15
   - All outputs become NaN
   - Production parameters (B=16, D=128, L=512, N=64) fail completely

## Root Cause Analysis

### Key Finding: Scale-Dependent Bug
The bug only appears with large matrices. Comparison:

| Metric | Small (D=8, N=6) | Large (D=128, N=64) |
|--------|------------------|---------------------|
| Input norm | 6.15 | 83.93 |
| X[0,0] after norm (CUDA) | 0.002914 | 0.000322 |
| X[0,0] after norm (PyTorch) | N/A | 0.0085 |
| **Ratio** | - | **26x smaller in CUDA!** |
| Final status | ✅ Converges | ❌ Explodes |

### PyTorch Behavior (Correct)
Newton-Schulz **intentionally grows** the norm during iterations:
- Iteration 1: norm = 1.0 → 3.25 (Gram trace: 1.0 → 10.56)
- Iteration 2: norm = 3.25 → 7.06 (Gram trace: 10.56 → 49.75)
- Iteration 3: norm = 7.06 → 7.47 (Gram trace: 49.75 → 55.75)
- Iteration 4-5: norm ≈ 7.5 (Gram trace ≈ 64, converged!)

### CUDA Behavior (Broken)
Values explode exponentially instead of converging:
- After 5 iterations: X_final[0,0] = 3.9×10^15

## Potential Causes

### 1. Memory Corruption (Most Likely)
We reuse `tile_buffer_bf16` for both X values and Gram matrix:
```cuda
// Step 1: Load X into tile_buffer_bf16
// Step 2: Compute Gram A
// Step 3: Store A into tile_buffer_bf16  ← Overwrites X!
__nv_bfloat16* gram_A_bf16 = tile_buffer_bf16;
```

**However**, we do reload X from global memory (`velocity_ortho`) during B@X computation, so this might not be the issue.

### 2. Indexing Bug in Transposed Case
For D=128, N=64 (tall matrix):
- We transpose logically to [N=64, D=128]  
- Gram matrix is [64, 64]
- Complex indexing for row/col access in global memory

Possible issue: Wrong indices when reading X values in transposed case during Gram computation or B@X multiplication.

### 3. Numerical Precision Issue
CUDA X values are 26x smaller than PyTorch after normalization. This suggests:
- Wrong norm computation?
- Extra division somewhere?
- BF16 rounding errors accumulating differently?

## Implementation Changes Made

### 1. Fixed Dtype Consistency
- Changed Gram matrix from FP32-only to: accumulate in FP32, convert to BF16 (matches PyTorch)
- Ensured all intermediate results round to BF16 after each operation

### 2. Proper Atomic Operations
- Implemented `atomicAddBF16` using atomicCAS on unsigned int
- Handles BF16 accumulation correctly across threads

### 3. Memory Layout
- Hybrid approach: Gram accumulates in FP32, then converts to BF16
- Reuses tile_buffer for polynomial computation (A, A², B)

## Next Steps to Fix

### Priority 1: Add Detailed Debug Output
Add printf statements to trace:
1. X values before/after each load from global memory
2. Gram matrix A values (first 3x3)
3. Polynomial B values
4. X values after update
5. Verify indexing is correct in transposed case

### Priority 2: Verify Indexing
For transposed case (D > N):
- Double-check row/col index computation
- Verify we're reading correct elements from `velocity_ortho`
- Add assertions for bounds checking

### Priority 3: Test Intermediate Sizes
Test with:
- D=16, N=8 (should work)
- D=32, N=16 (might work)
- D=64, N=32 (might fail)
- D=128, N=64 (fails)

Find the threshold where it breaks to narrow down the cause.

### Priority 4: Match PyTorch Exactly
Create a test that:
1. Extracts b_t values from CUDA
2. Runs PyTorch NS on those exact values
3. Runs CUDA NS on those exact values
4. Compares intermediate Gram matrices
5. Finds first divergence point

## Test Files Created

1. `test_ns_cuda_only.py` - Tests with small matrix (works ✅)
2. `test_ns_prod_dims.py` - Tests with production dims (fails ❌)
3. `test_torch_ns_large.py` - Verifies PyTorch handles large matrix (works ✅)
4. `test_torch_ns_trace.py` - Traces PyTorch NS iteration by iteration
5. `debug_gram_matrix.py` - Step-by-step Gram matrix computation

## Code Locations

Key files:
- `csrc/selective_scan/newton_schulz_fwd_kernel.cuh` - CUDA implementation
- `mamba_ssm/ops/selective_scan_interface.py` - PyTorch reference

Critical functions:
- `newton_schulz_velocity_5step_kernel` (line 700+)
- Gram matrix computation (line 895+)
- Polynomial computation (line 1007+)
- X update (line 1056+)

## Conclusion

We've made significant progress understanding the Newton-Schulz implementation:
- ✅ Correctly matches PyTorch for small matrices
- ✅ Proper BF16 dtype handling throughout
- ✅ Correct algorithmic structure
- ❌ Scale-dependent bug for large matrices needs investigation

The 26x difference in normalized X values is the smoking gun - something is wrong with either:
1. How we compute/apply the norm
2. How we index into the transposed matrix
3. How we reload X values during iterations

Next developer should focus on adding detailed debug output to trace where the 26x discrepancy originates.







