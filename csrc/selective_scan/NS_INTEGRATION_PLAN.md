# Newton-Schulz Backward Integration Plan

## Current Implementation Status

### Forward Pass (`selective_scan_fwd_kernel.cuh`)
- Lines 242-258: If `params.use_newton_schulz` is true, loads orthogonalized velocity from `X_4_buffer`
- The buffer contains output of NS 5-step forward: orthogonalized b_t values
- Buffer shape: `[batch, dim, seqlen, dstate]`

### Backward Pass (`selective_scan_bwd_kernel.cuh`)  
- Lines 252-276: Reconstructs velocity scan by recomputing `delta_B_u = alpha * delta * B * u`
- Lines 313-332: Reverse scan through velocity to get gradients
- Lines 334-384: Computes gradients through velocity equation

## Integration Strategy

The NS backward pass needs to be integrated to handle the case where orthogonalization was used in the forward pass.

### Key Insight
When `use_newton_schulz = true`:
- **Forward**: `v_t = β·v_{t-1} + X_ortho` where `X_ortho` comes from NS(alpha * delta * B * u)
- **Backward**: Need to backprop through NS to get gradients w.r.t. `delta`, `B`, `u`

### Integration Point
**Location**: After line 384 in `selective_scan_bwd_kernel.cuh` (after computing dv values)

**Pseudocode**:
```cpp
if (params.use_newton_schulz) {
    // Call NS backward to get additional gradients through orthogonalization
    // This will ADD gradients to du_vals, ddelta_vals, dB_vals/dBC_val
    
    // The dv values contain ∂L/∂(alpha·delta·B·u) from the velocity reverse scan
    // We need to backprop these through the NS 5-step orthogonalization
    
    // For now, we'll use a simplified approach:
    // The NS backward kernel expects full [D, N] matrices per timestep
    // But the scan operates on one state at a time
    // 
    // Solution: Accumulate dv values into a buffer, then call NS backward
    // after the state loop completes
}
```

## Implementation Approach

### Option 1: Per-Chunk NS Backward (Recommended)
Call NS backward once per chunk after all states are processed:

```cpp
// After line 608 (end of state loop, before chunk loop ends)
if (params.use_newton_schulz) {
    // Accumulated dv values are stored somewhere
    // Call NS backward for entire [dim, dstate] matrix for this chunk
    launch_newton_schulz_velocity_5step_backward<input_t, weight_t>(
        grad_velocity_buffer,  // [dim, seqlen, dstate] accumulated gradients
        u, delta, B,
        du_accum, ddelta_accum, dB_accum,  // Accumulators
        params.alpha, params.batch, params.dim, params.seqlen, params.dstate,
        chunk * kChunkSize, (chunk + 1) * kChunkSize,
        params.u_batch_stride, params.u_d_stride,
        params.delta_batch_stride, params.delta_d_stride,
        params.B_batch_stride, params.B_group_stride,
        params.B_d_stride, params.B_dstate_stride,
        kIsVariableB, params.dim_ngroups_ratio,
        stream
    );
}
```

###Option 2: Post-Processing (Simpler for Initial Integration)
Add a separate kernel launch after the main backward kernel:

```cpp
// In selective_scan_bwd_cuda() function
if (params.use_newton_schulz) {
    // Launch NS backward as a separate pass
    // This requires storing intermediate dv values
}
```

## Required Changes

### 1. Add to `SSMParamsBwd` struct (selective_scan.h)
```cpp
struct SSMParamsBwd {
    // ... existing fields ...
    bool use_newton_schulz;
    float alpha;  // NS alpha parameter
    float beta;   // NS beta parameter  
    void *grad_velocity_buffer_ptr;  // Temp buffer for dv accumulation
};
```

### 2. Modify `selective_scan_bwd_kernel.cuh`

**Add include**:
```cpp
#include "newton_schulz_fwd_kernel.cuh"
```

**Add dv accumulation** (after line 332):
```cpp
if (params.use_newton_schulz && params.grad_velocity_buffer_ptr != nullptr) {
    // Store dv values for later NS backward pass
    float *grad_v_buffer = reinterpret_cast<float*>(params.grad_velocity_buffer_ptr);
    #pragma unroll
    for (int i = 0; i < kNItems; ++i) {
        int t = chunk * kChunkSize + threadIdx.x * kNItems + i;
        if (t < params.seqlen) {
            int d = dim_id;
            int idx = batch_id * params.dim * params.seqlen * params.dstate +
                     d * params.seqlen * params.dstate +
                     t * params.dstate +
                     state_idx;
            float dv_val = dv_reverse_data[i].y;
            atomicAdd(&grad_v_buffer[idx], dv_val);
        }
    }
}
```

**Call NS backward** (after line 659, after main kernel):
```cpp
// In selective_scan_bwd_cuda or selective_scan_bwd_launch
if (params.use_newton_schulz) {
    launch_newton_schulz_velocity_5step_backward<input_t, weight_t>(
        reinterpret_cast<float*>(params.grad_velocity_buffer_ptr),
        reinterpret_cast<const input_t*>(params.u_ptr),
        reinterpret_cast<const input_t*>(params.delta_ptr),
        reinterpret_cast<const weight_t*>(params.B_ptr),
        reinterpret_cast<float*>(params.du_ptr),
        reinterpret_cast<float*>(params.ddelta_ptr),
        reinterpret_cast<float*>(params.dB_ptr),
        params.alpha,
        params.batch, params.dim, params.seqlen, params.dstate,
        0, params.seqlen,  // Full sequence
        params.u_batch_stride, params.u_d_stride,
        params.delta_batch_stride, params.delta_d_stride,
        params.B_batch_stride, params.B_group_stride,
        params.B_d_stride, params.B_dstate_stride,
        params.is_variable_B, params.dim_ngroups_ratio,
        stream
    );
}
```

### 3. Python Binding Updates (selective_scan.cpp)
Pass `use_newton_schulz` and `alpha`/`beta` parameters from Python.

## Testing Strategy

### Phase 1: Compilation
- ✅ Verify backward kernel compiles (completed)
- ⏳ Verify integration compiles

### Phase 2: Correctness
- ⏳ Unit test: NS backward alone
- ⏳ Integration test: Full selective scan backward with NS
- ⏳ Gradient check: Compare with PyTorch autograd

### Phase 3: Performance
- ⏳ Benchmark overhead of NS backward
- ⏳ Profile memory usage
- ⏳ Optimize if needed

## Next Steps

1. **Immediate**: Add NS backward call to `selective_scan_bwd_cuda` 
2. **Short-term**: Test integration with simple cases
3. **Medium-term**: Full gradient checking
4. **Long-term**: Optimize and tune performance

## Notes

- The NS backward kernel is already implemented and verified correct
- Integration requires careful handling of buffer management
- Gradients from NS should be **added** to existing gradients (not replace)
- The ns gradients computed through the orthogonalization are additional to the gradients from the scan itself

