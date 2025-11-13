# CUDA Bug Analysis and Fix Plan

## Critical Finding

CUDA's NS backward is **ONLY producing gradients for timestep 0**. Timesteps 1-3 have NO NS backward contribution:
- Timestep 0: du = D*dout + NS_grad = [1.4248, 0.0689] ✅
- Timesteps 1-3: du = D*dout only = [0.1464, -0.0324] (no NS contribution!) ❌

## Expected Behavior

All timesteps should have NS backward gradients because:
1. `grad_X_4_buffer` should have non-zero values for ALL timesteps
2. NS backward kernel is launched with `grid(batch, seqlen)` which processes all timesteps
3. Each timestep should produce gradients for `du`, `ddelta`, and `dB`

## Potential Bug Locations

### 1. Main Backward Kernel (`selective_scan_bwd_kernel.cuh`)
**Line 381-391**: Accumulates `dv` into `grad_X_4_buffer`
- Check: Is `time_idx` computed correctly for all timesteps?
- Check: Are all chunks processed correctly?
- Check: Is the condition `time_idx < params.seqlen` causing issues?

### 2. NS Backward Kernel (`newton_schulz_bwd_kernel.cuh`)
**Line 1010-1016**: Reads `grad_output` (which is `grad_X_4_buffer`)
- Check: Is `buffer_idx` computed correctly for all timesteps?
- Check: Is `grad_output` read correctly?
- Check: Are gradients written correctly for all timesteps?

**Line 1319-1335**: Writes gradients using `atomicAdd`
- Check: Is `u_idx` and `delta_idx` computed correctly for all timesteps?
- Check: Are gradients actually being written?

## Next Steps

1. **Add debug prints to CUDA** to verify:
   - What values are in `grad_X_4_buffer` after main backward?
   - What values does NS backward read from `grad_output`?
   - What gradients does NS backward write?

2. **Verify indexing**: Check if `time_idx` computation is correct for all timesteps

3. **Check kernel launch**: Verify all timesteps are actually launched

4. **Fix the bug**: Based on findings, fix either:
   - Main backward kernel accumulation
   - NS backward kernel reading/writing
   - Index computation

## Hypothesis

Most likely: The NS backward kernel has an indexing bug that causes it to only process/write timestep 0 correctly, or `grad_X_4_buffer` is not being accumulated correctly for timesteps 1-3.




