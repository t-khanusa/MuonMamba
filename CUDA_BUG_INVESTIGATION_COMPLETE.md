# CUDA Bug Investigation Complete

## Bug Status: **CONFIRMED BUT NOT YET FIXED**

**Issue**: CUDA's NS backward only produces gradients for timestep 0. Timesteps 1-3 have NO NS backward contribution to `du`.

## Evidence

- Timestep 0: `du = D*dout + NS_grad` ✅
- Timesteps 1-3: `du = D*dout only` ❌ (missing NS contribution)

## Investigation Results

### 1. Main Backward Kernel (`selective_scan_bwd_kernel.cuh`)
- **Location**: Lines 379-397 (real) and 685-703 (complex)
- **Function**: Accumulates `dv` into `grad_X_4_buffer`
- **Status**: Code looks correct - uses `time_idx < params.seqlen` check
- **Issue**: May not be accumulating correctly for timesteps 1-3

### 2. NS Backward Kernel (`newton_schulz_bwd_kernel.cuh`)
- **Location**: Lines 1346-1423 (launch), 524-1338 (kernel)
- **Function**: Reads `grad_X_4_buffer` and computes gradients for `u`, `delta`, `B`
- **Status**: Kernel launch is correct (`grid(batch, num_timesteps)`)
- **Issue**: If `grad_X_4_buffer` is zero for timesteps 1-3, NS backward produces zero gradients

### 3. Reverse Scan Logic
- **Location**: Lines 343-362 (`dv_reverse_data` computation)
- **Function**: Computes `dv` from reverse scan
- **Status**: Logic appears correct
- **Note**: `ddelta` is non-zero for all timesteps, suggesting `dh` and `dv` should also be non-zero

## Root Cause Hypothesis

**Most Likely**: `grad_X_4_buffer` is **NOT being accumulated** for timesteps 1-3 in the main backward kernel.

**Possible Reasons**:
1. The `time_idx` computation might not cover all timesteps correctly
2. Chunking logic might skip certain timesteps
3. The reverse scan might produce incorrect `dv` values for timesteps 1-3
4. There might be a race condition in atomic accumulation

## Fixes Attempted

1. ✅ Added bounds check `local_time_idx < seqlen_remaining_in_chunk` - **Didn't fix**
2. ✅ Simplified to `time_idx < params.seqlen` - **Didn't fix**
3. ✅ Verified kernel launch - **Correct**
4. ✅ Verified buffer indexing - **Matches between kernels**

## Next Steps Required

### Debug Output (Recommended)
Add CUDA `printf` or return debug values to verify:
1. What are the `dv` values for each timestep after reverse scan?
2. Is `grad_X_4_buffer` being accumulated for timesteps 1-3?
3. What values does NS backward read from `grad_X_4_buffer`?

### Alternative: Check Reverse Scan
The reverse scan might be producing incorrect `dv` values for timesteps 1-3. Verify:
- Does the postfix callback work correctly for single chunk?
- Are `dv_reverse_data[i]` values correct for all `i`?

### Alternative: Verify Chunking
For `seqlen=4`, check:
- How many chunks are there?
- Does chunk 0 process all timesteps 0-3?
- Is `time_idx` computed correctly for all threads?

## Code Locations for Further Investigation

1. **Main Backward**: `csrc/selective_scan/selective_scan_bwd_kernel.cuh:379-397`
2. **NS Backward Launch**: `csrc/selective_scan/newton_schulz_bwd_kernel.cuh:1346-1423`
3. **NS Backward Kernel**: `csrc/selective_scan/newton_schulz_bwd_kernel.cuh:524-1338`
4. **Reverse Scan**: `csrc/selective_scan/selective_scan_bwd_kernel.cuh:343-362`

## Recommendation

**Add debug output** using CUDA `printf` or by returning debug tensors to Python to trace:
1. `dv` values per timestep
2. `grad_X_4_buffer` values after main backward
3. What NS backward reads

This will pinpoint the exact location of the bug.




