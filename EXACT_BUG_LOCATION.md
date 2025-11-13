# Exact Bug Location - Investigation Summary

## Bug Confirmed
- Timestep 0: NS contribution = [0.208, 0.020] ✓
- Timesteps 1-3: NS contribution = [0.000, 0.000] ✗

## Root Cause Analysis

### Mapping Verification
- For `seqlen=4`, only threads with `threadIdx.x < 4` have valid timesteps
- All valid timesteps use `i=0`:
  - Timestep 0: threadIdx.x=0, i=0 → `dv_reverse_data[0]`
  - Timestep 1: threadIdx.x=1, i=0 → `dv_reverse_data[0]`
  - Timestep 2: threadIdx.x=2, i=0 → `dv_reverse_data[0]`
  - Timestep 3: threadIdx.x=3, i=0 → `dv_reverse_data[0]`

### Potential Causes
1. **dout_vals[0] is zero for threads 1,2,3** - `BlockLoad` might not load correctly
2. **thread_reverse_data[0].y is zero after hidden state reverse scan** - reverse scan bug
3. **dv_reverse_data[0].y is zero after velocity reverse scan** - reverse scan bug
4. **gpuAtomicAdd doesn't accumulate correctly** - unlikely but possible
5. **NS backward doesn't read grad_X_4_buffer correctly** - but timestep 0 works, so unlikely

## Next Steps
1. Verify `dout_vals[0]` values for threads 1,2,3
2. Verify `thread_reverse_data[0].y` after hidden state reverse scan
3. Verify `dv_reverse_data[0].y` after velocity reverse scan
4. Verify `grad_X_4_buffer` values after accumulation
5. Fix the identified issue




