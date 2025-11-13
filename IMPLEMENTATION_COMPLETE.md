# Newton-Schulz Integration Implementation Complete

## Summary

Successfully implemented the three-phase Newton-Schulz orthogonalization approach following Option A from the memory flow analysis.

## What Was Implemented

### 1. Kernel Modifications (`selective_scan_fwd_kernel.cuh`)

**Added buffer read/write logic** in the velocity scan computation (lines ~235-261):

- **Read Mode**: If `b_t_ortho_buffer_ptr` is not null, load `b_t_ortho` values from the buffer
- **Write Mode**: Otherwise, compute `b_t` normally and optionally store to `b_t_buffer` when NS is enabled
- **Buffer indexing**: Correctly computed index for `[batch, dim, seqlen, dstate]` layout

### 2. C++ Integration (`selective_scan.cpp`)

**Implemented three-phase approach** (lines ~367-395):

- **Phase 1**: Launch kernel with `use_newton_schulz=true` → computes and stores `b_t` to buffer
- **Phase 2**: Launch NS kernel per chunk using `launch_newton_schulz_velocity_tiled()` → orthogonalizes `b_t` → writes to `b_t_ortho_buffer`
- **Phase 3**: Launch kernel again with `b_t_buffer_ptr=null` → reads from `b_t_ortho_buffer` → continues scan and computes final outputs

### 3. Infrastructure (Already Done)
- Parameters added to `SSMParamsBase`
- Buffer allocation in forward function
- Parameter passing in `set_ssm_params_fwd`

## How It Works

```
Forward Pass with NS Enabled:

1. Kernel Launch 1 (Phase 1)
   ├─ Computes: b_t = alpha * delta * B * u
   ├─ Stores: b_t → b_t_buffer[batch, dim, timestep, dstate]
   ├─ Also computes: v_t = beta * v_{t-1} + b_t (intermediate outputs)
   └─ But these outputs will be recomputed in Phase 3

2. Newton-Schulz Kernel (Phase 2)
   ├─ Grid: (batch, num_timesteps_per_chunk) blocks
   ├─ Reads: b_t_buffer[batch, :, timestep, :]
   ├─ Applies: NS orthogonalization to [dim, dstate] matrix
   └─ Writes: b_t_ortho_buffer[batch, :, timestep, :]

3. Kernel Launch 2 (Phase 3)
   ├─ Reads: b_t_ortho_buffer[batch, dim, timestep, dstate]
   ├─ Uses: v_t = beta * v_{t-1} + b_t_ortho
   ├─ Continues: Hidden state scan
   └─ Final outputs use orthogonalized values
```

## Memory Flow

Per chunk (2048 timesteps, dim=128, dstate=64):
- Write b_t: ~65 MB
- Read for NS: ~65 MB
- Write b_t_ortho: ~65 MB
- Read for scan: ~65 MB
- **Total**: ~260 MB per chunk

## How to Enable

1. Set `use_newton_schulz = true` in `selective_scan.cpp` line 327
2. Ensure complex=False (only real-valued supported currently)
3. The three-phase process will execute automatically

## Current Limitations

1. **Complex values**: Not yet supported (only real float32)
2. **Backward pass**: Not yet implemented
3. **Performance**: Launches kernel 2x (for phases 1 and 3), plus NS kernel launches
4. **Memory**: Extra ~260 MB per sequence for buffers

## Testing

To test:
1. Set `use_newton_schulz = true`
2. Run forward pass
3. Verify:
   - Buffers are allocated correctly
   - Kernel launches succeed
   - Outputs are different from NS=false (orthogonalization applied)
   - No crashes or errors

## Next Steps (Future Work)

1. Add backward pass integration
2. Optimize to reduce double kernel launch (combine phases 1 & 3)
3. Add support for complex values
4. Profile and optimize performance
5. Add unit tests

## Files Modified

✅ `csrc/selective_scan/selective_scan_fwd_kernel.cuh` - Added buffer read/write logic
✅ `csrc/selective_scan/selective_scan.cpp` - Implemented three-phase launch sequence
✅ `csrc/selective_scan/selective_scan.h` - Added NS parameters (already done)

















