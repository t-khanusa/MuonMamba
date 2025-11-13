# Newton-Schulz Orthogonalization Implementation Status

## Summary

Infrastructure for Newton-Schulz orthogonalization has been added to the MomentumMamba codebase. The implementation is **partially complete** and currently disabled by default.

## What Has Been Implemented

### 1. Parameter Structures (`csrc/selective_scan/selective_scan.h`)
- Added `use_newton_schulz` boolean flag to `SSMParamsBase`
- Added `b_t_buffer_ptr` for storing raw b_t values
- Added `b_t_ortho_buffer_ptr` for storing orthogonalized values

### 2. Buffer Allocation (`csrc/selective_scan/selective_scan.cpp`)
- Added buffer allocation in `selective_scan_fwd()` function
- Buffers have shape `[batch, dim, seqlen, dstate]`
- Using float32 for numerical stability in NS operations
- Currently disabled by default with `use_newton_schulz = false`

### 3. Parameter Passing
- Updated `set_ssm_params_fwd()` signature to accept NS parameters
- Updated `selective_scan_fwd()` to pass buffer pointers
- Updated backward pass to pass null pointers (NS not yet implemented in backward)

## What Still Needs to Be Done

### Phase 1: Kernel Integration (NOT YET IMPLEMENTED)
The challenging part is integrating NS into the scan kernel due to architectural constraints:

**Current Architecture:**
- Grid: `(batch, dim)` blocks
- Each block processes 2048 timesteps (ChunkSize)
- Velocity scan computes `v_t = beta * v_{t-1} + alpha * delta * B * u` inline

**Challenge:**
- NS needs to process [dim, dstate] matrices per timestep
- Current blocks only have ONE dim per block
- NS requires accessing ALL dims to compute Gram matrix

**Proposed Solutions:**

**Option A: Separate Kernel Phases (Recommended)**
```cuda
// Phase 1: Compute b_t for all dims
Kernel with grid (batch, dim) blocks
→ Store b_t[batch, dim, timestep, dstate] to global buffer

// Phase 2: Apply NS per timestep  
Kernel with grid (batch, chunk) blocks
→ Load b_t[:, timestep, :] for each timestep in chunk
→ Apply NS in shared memory
→ Store b_t_ortho back

// Phase 3: Continue with velocity scan
Kernel with grid (batch, dim) blocks
→ Load b_t_ortho[batch, dim, timestep, dstate]
→ Continue scan with orthogonalized values
```

**Option B: In-Kernel with Global Memory (Simpler)**
- Store b_t values to global buffer as computed
- Sync across all blocks
- Launch NS kernel with different grid structure
- Read back b_t_ortho and continue

### Phase 2: Kernel Modifications Needed

1. **Modify `selective_scan_fwd_kernel.cuh`:**
   - When `use_newton_schulz` is true, store `alpha * delta * B * u` to `b_t_buffer` instead of using directly
   - After NS kernel completes, load from `b_t_ortho_buffer` for velocity scan

2. **Launch NS Kernel:**
   - After computing b_t for a chunk, launch `launch_newton_schulz_velocity_tiled()`
   - Process each chunk independently
   - Use existing kernel from `newton_schulz_fwd_kernel.cuh`

### Phase 3: Backward Pass (NOT YET IMPLEMENTED)
- Currently using straight-through estimator (passthrough)
- Full backward would require computing gradients through:
  - Orthogonalization: G' = a*G + G@B
  - Polynomial: B = b*A + c*A^2
  - Gram matrix: A = G^T @ G
  - Normalization: G = V / norm

## Files Modified

1. `csrc/selective_scan/selective_scan.h` - Added NS parameters
2. `csrc/selective_scan/selective_scan.cpp` - Added buffer allocation and parameter passing

## Testing

To enable NS (once kernel integration is complete):
1. Set `use_newton_schulz = true` in `selective_scan.cpp` line 327
2. Ensure complex=False (only real valued supported for now)
3. The infrastructure will allocate buffers and pass pointers

## Performance Considerations

**Expected overhead when enabled:**
- Memory: Additional 256KB - 2MB per forward pass (depending on batch size)
- Compute: ~5-10% increase in kernel time for NS operations
- Kernel launches: Additional 1 kernel launch per chunk (if using separate phases)

**Shared memory usage:**
- NS kernel needs ~50KB shared memory per block
- May require `cudaFuncSetAttribute` to increase beyond 48KB default

## Next Steps

1. **Decide on architecture approach** (Option A vs Option B)
2. **Implement kernel modifications** to store/load b_t values
3. **Add NS kernel launch** between b_t computation and velocity scan
4. **Test with small examples** to verify correctness
5. **Profile performance** impact
6. **Implement full backward pass** if needed (or keep straight-through estimator)

## Notes

- Existing `newton_schulz_velocity_tiled_kernel` is already implemented and ready to use
- The kernel expects buffer shape `[batch, dim, seqlen, dstate]`
- Grid structure for NS: `(batch, num_timesteps)` - different from scan's `(batch, dim)`
- This architectural mismatch is why integration is non-trivial

















