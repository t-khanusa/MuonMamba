# Step-by-Step Implementation Plan for Newton-Schulz Integration

## Current Status
✅ Infrastructure added (buffers, parameters)
⏳ Kernel integration (in progress)

## Implementation Approach

### Option A: Full Three-Phase Implementation (Complex)
Split scan into three separate kernels:
1. Phase 1 kernel: Compute b_t only
2. Launch NS kernel 
3. Phase 2 kernel: Continue scan with b_t_ortho

**Pros**: Clean separation
**Cons**: Major refactoring, need to duplicate scan logic

### Option B: Conditional In-Kernel Phase Split (Recommended for Start)
Modify existing kernel to optionally work in phases within the chunk loop:
1. When `use_newton_schulz=true`: Compute entire chunk's b_t first, store
2. Sync and launch NS for this chunk
3. Load b_t_ortho and continue

**Pros**: Minimal changes to existing code
**Cons**: Conditional logic makes kernel more complex

### Option C: Simple Pass-Through First (Getting Started)
For now, just add the machinery but don't actually apply NS:
1. Compute b_t as before
2. Store to buffer (even if not using NS yet)
3. Copy buffer to b_t_ortho (passthrough)
4. Use as before

**Pros**: Get the infrastructure working first
**Cons**: Not actually orthogonalizing yet

## Recommended Implementation Order

### Phase 1: Get Infrastructure Working (Option C)
Goal: Prove buffers work, kernel launches succeed

1. Modify kernel to store b_t values to buffer after computing
2. Add logic to copy b_t_buffer → b_t_ortho_buffer (passthrough)
3. Modify kernel to optionally load from b_t_ortho_buffer instead of computing inline
4. Test: Should produce identical results with NS disabled

### Phase 2: Add NS Kernel Launch (Option B)
Goal: Actually orthogonalize the data

1. After computing b_t for a chunk, launch NS kernel
2. Kernel modifies b_t_ortho_buffer in-place
3. Continue scan using b_t_ortho_buffer
4. Test: Verify NS is being applied

### Phase 3: Optimize (Option A if needed)
Goal: Better performance

1. If performance issues, consider full kernel split
2. Profile and optimize

## Code Changes Needed

### 1. Kernel: Store b_t values

In `selective_scan_fwd_kernel.cuh`, after line 234:

```cuda
// Store b_t if Newton-Schulz enabled
if (params.use_newton_schulz && !kIsComplex) {
    float* b_t_buffer_ptr = reinterpret_cast<float*>(params.b_t_buffer_ptr);
    int timestep = chunk * kChunkSize + threadIdx.x * kNItems + i;
    int buffer_idx = batch_id * params.dim * params.seqlen * params.dstate 
                     + dim_id * params.seqlen * params.dstate
                     + timestep * params.dstate
                     + state_idx * params.dstate_stride;  // Need to handle stride
    
    // Store b_t value for this (batch, dim, timestep, dstate)
    b_t_buffer_ptr[buffer_idx] = params.alpha * delta_B_u;
}
```

### 2. cpp: Launch NS between chunks

In `selective_scan.cpp`, after computing each chunk:

```cpp
// After kernel launch for computing b_t
if (use_newton_schulz && !is_complex) {
    // Launch NS kernel for this chunk
    int t_start = chunk * 2048;
    int t_end = min((chunk + 1) * 2048, seqlen);
    
    launch_newton_schulz_velocity_tiled(
        reinterpret_cast<float*>(b_t_buffer.data_ptr()),
        reinterpret_cast<float*>(b_t_ortho_buffer.data_ptr()),
        batch, dim, seqlen, dstate,
        t_start, t_end,
        stream
    );
}
```

### 3. Kernel: Load b_t_ortho

Modify velocity_data computation to load from buffer:

```cuda
if (params.use_newton_schulz && !kIsComplex) {
    float* b_t_ortho_ptr = reinterpret_cast<float*>(params.b_t_ortho_buffer_ptr);
    // Calculate buffer index
    // Load b_t_ortho value
    float b_t_ortho = b_t_ortho_ptr[buffer_idx];
    velocity_data[i] = make_float2(params.beta, b_t_ortho);
} else {
    // Original computation
    velocity_data[i] = make_float2(params.beta, params.alpha * delta_B_u);
}
```

## Testing Steps

1. **Test 1**: Run with `use_newton_schulz=false`, verify no changes
2. **Test 2**: Run with `use_newton_schulz=true` and passthrough, verify identical results  
3. **Test 3**: Run with actual NS, verify different results and numerical stability
4. **Test 4**: Profile performance impact

## Current Decision

Start with **Option C (Passthrough)** to get infrastructure working, then move to **Option B** for actual NS integration.

Would you like me to:
1. Start with Option C (passthrough - simplest)
2. Jump to Option B (full integration with kernel launch)
3. Something else?

















