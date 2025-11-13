# Newton-Schulz CUDA Fix Strategy

## Critical Discovery

PyTorch Newton-Schulz **also** has Gram diagonal = 0.016 for large matrices (D=128, N=64).  
This is **NOT a bug** - the NS algorithm is designed for this!

### Evidence:
```python
# PyTorch with D=128, N=64:
Gram A diagonal: [0.0162, 0.0162, 0.0154]  # Small values
Gram A trace: 1.0  # Not 64!
Final output: norm=7.06, range=[-0.32, 0.30]  # ✅ Converges perfectly

# CUDA with same matrix:
Final output: values reach 10^15  # ❌ Explodes
```

## Root Cause

The bug is **NOT in the Gram matrix computation** (it matches PyTorch).  
The bug is in **how we iterate** or **index into the transposed matrix**.

### Working Cases:
- ✅ Small matrices (D=8, N=6): Perfect match with PyTorch
- ✅ PyTorch with large matrices: Converges correctly

### Failing Case:
- ❌ CUDA with large matrices (D=128, N=64): Explodes

## Most Likely Bug Locations

### 1. Transposed Matrix Indexing (HIGH PRIORITY)
When D > N, we transpose the matrix. The indexing becomes complex:
- Logical: X is [N=64, D=128]
- Storage: X is [D=128, N=64]
- Element X_logical[n, d] = X_storage[d, n]

**Potential bugs:**
- Wrong row/col calculation in Gram matrix computation
- Wrong indices when loading X for B@X multiplication
- Off-by-one errors in tile boundaries

### 2. Memory Overlap During Iteration
We reuse `tile_buffer_bf16` for:
1. Loading X values (in tiles)
2. Storing Gram matrix A
3. Storing polynomial B

**Check:** Are we accidentally reading stale/corrupted X values from the buffer instead of reloading from global memory?

### 3. Atomic Accumulation Race Conditions
For Gram matrix computation, multiple tiles accumulate using `atomicAdd`.  
With large matrices (gram_size=64), there are many atomic operations.

**Check:** Are atomic operations working correctly? Is there overflow in FP32 accumulation?

## Recommended Fix Approach

### Step 1: Verify Gram Matrix is Correct
Add simple kernel to compute and print ENTIRE Gram matrix (not just 3x3).  
Compare with PyTorch's A matrix element by element.

### Step 2: Verify B Matrix is Correct  
After computing B = b*A + c*A², print entire B matrix.  
Compare with PyTorch's B.

### Step 3: Verify First X Update
After first iteration X_new = a*X + B@X, print entire X_new.  
Compare with PyTorch's X after iteration 1.

### Step 4: Check Indexing in Transposed Case
Add bounds checking and print actual indices being accessed:
- When loading X from `velocity_ortho`
- When computing Gram elements
- When computing B@X

## Quick Test

Create a test that:
1. Extracts b_t from CUDA (before NS)
2. Runs PyTorch NS on CPU with those exact values
3. Manually calls CUDA Gram computation
4. Compares outputs element-by-element

## Alternative: Simpler Implementation

If debugging is too complex, consider:
1. Implement NS entirely in a single kernel (no tiling)
2. Limit to small gram_size (e.g., max 32) for guaranteed correctness
3. Fall back to identity for larger matrices (acceptable for momentum)

## Files to Focus On

- `csrc/selective_scan/newton_schulz_fwd_kernel.cuh`:
  - Lines 963-1005: Transposed Gram computation
  - Lines 1140-1180: Transposed B@X multiplication  
  - Check ALL indices: `batch_idx * D * L * dstate + d * L * dstate + time_idx * dstate + n`

##Status

Current blocker: Debug printf caching makes iteration slow.  
Recommendation: Use standalone kernel test or GDB/CUDA debugger for detailed inspection.







