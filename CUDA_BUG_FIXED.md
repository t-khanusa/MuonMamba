# CUDA Bug: NS Backward Only Processes Timestep 0

## Bug Summary

**Confirmed Bug**: CUDA's NS backward pass only produces gradients for timestep 0. Timesteps 1-3 have NO NS backward contribution.

### Evidence:
- Timestep 0: `du = D*dout + NS_grad` ✅
- Timesteps 1-3: `du = D*dout only` (missing NS contribution) ❌

### Expected:
- ALL timesteps should have NS backward gradients
- `grad_X_4_buffer` should have non-zero values for ALL timesteps

## Root Cause Analysis

The kernel launch is correct:
- `grid(batch, num_timesteps) = (1, 4)` launches 4 blocks (one per timestep)
- `time_idx = t_start + blockIdx.y = 0 + (0,1,2,3) = (0,1,2,3)`

So all timesteps SHOULD be processed.

## Most Likely Causes

1. **Main backward kernel not accumulating `grad_X_4_buffer` for timesteps 1-3**
   - Location: `selective_scan_bwd_kernel.cuh` lines 380-392
   - Issue: `time_idx` computation or chunking logic might skip timesteps 1-3
   - Fix: Verify `time_idx` covers all timesteps in all chunks

2. **NS backward kernel reading zero from `grad_X_4_buffer` for timesteps 1-3**
   - Location: `newton_schulz_bwd_kernel.cuh` lines 1014, 1034
   - Issue: If `grad_X_4_buffer[timesteps 1-3]` is zero, backward produces zero
   - Fix: Ensure main backward accumulates correctly

3. **Index computation error**
   - Issue: Buffer indexing might be wrong for certain timesteps
   - Fix: Verify `buffer_idx` computation matches layout

## Fix Strategy

1. **Add debug output** to verify `grad_X_4_buffer` has values for all timesteps after main backward
2. **Verify indexing**: Ensure `time_idx` computation covers all timesteps
3. **Check chunking**: Verify chunking logic processes all timesteps correctly
4. **Fix the bug**: Based on findings, correct either accumulation or indexing

## Next Steps

Since timestep 0 works correctly, the structure is sound. The fix should focus on ensuring timesteps 1-3 are processed identically to timestep 0.




