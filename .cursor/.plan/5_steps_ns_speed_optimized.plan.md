# 5-Step Newton-Schulz Speed-Optimized Implementation

## Overview

Implement 5-step Newton-Schulz with:

- On-the-fly b_t computation (eliminates Phase 1)
- Store X_4 only for fast backward pass
- Shared memory optimization: 33 KB (fits 48 KB default)
- A^2 computed on-the-fly to save 16 KB shared memory

## Architecture Changes

### Two-Phase Design (down from 3 phases)

**Phase 1 (NEW)**: Newton-Schulz with integrated b_t computation

- Grid: `(batch, num_timesteps)` blocks
- Compute b_t = alpha × delta × B × u on-the-fly in registers
- Apply 5-step NS with shared memory reuse
- Store X_4 before final iteration
- Write orthogonalized result

**Phase 2**: Velocity/Hidden state scan (unchanged)

- Grid: `(batch, dim)` blocks
- Read orthogonalized b_t
- Continue with momentum scan

### Backward Pass

- Recompute b_t on-the-fly (~0.01ms, negligible)
- Load X_4 from storage (fast read)
- Recompute A_4 from X_4
- Backward through step 5 only
- Propagate gradients to u, delta, B

## Key Files to Modify

### 1. Forward Kernel (`newton_schulz_fwd_kernel.cuh`)

- Create new `newton_schulz_velocity_5step_kernel` with:
  - Parameters: u, delta, B (raw inputs, not pre-computed b_t)
  - Shared memory: 33 KB with buffer reuse
  - On-the-fly b_t computation
  - 5 NS iterations with A^2 recomputed each iteration
  - X_4 storage before final step

### 2. Backward Kernel (`newton_schulz_bwd_kernel.cuh`)

- Create new `newton_schulz_velocity_5step_backward_kernel` with:
  - Load X_4 from storage
  - Recompute b_t (cheap)
  - Recompute A_4 from X_4
  - Backward through step 5 only
  - Chain rule to u, delta, B gradients

### 3. Main Integration (`selective_scan_fwd_kernel.cuh`)

- Remove Phase 1 (b_t computation)
- Update Phase 2 launch to pass u, delta, B directly
- Update Phase 3 to read orthogonalized results

### 4. Parameter Structure (`selective_scan.h`)

- Remove `b_t_buffer_ptr` (no longer needed)
- Keep `X_4_buffer_ptr` for backward pass
- Add strides for u, delta, B access in NS kernel

### 5. Launch Wrapper

- Create `launch_newton_schulz_velocity_5step`
- Remove Phase 1 kernel launch
- Update Phase 2 with new parameters

## Detailed Implementation Steps

### Step 1: Update Parameter Structure

**File**: `csrc/selective_scan/selective_scan.h`

Remove b_t_buffer, keep X_4_buffer:

```cpp
struct SSMParamsBase {
    // ... existing fields ...
    
    // Newton-Schulz buffers
    void *__restrict__ X_4_buffer_ptr;  // [batch, dim, seqlen, dstate] - only X_4
    // b_t_buffer_ptr removed - computed on-the-fly
};

struct SSMParamsBwd: public SSMParamsBase {
    // ... existing fields ...
    void *__restrict__ grad_X_4_buffer_ptr;  // For backward
};
```

### Step 2: Implement 5-Step Forward Kernel with SM Optimization

**File**: `csrc/selective_scan/newton_schulz_fwd_kernel.cuh`

Key features:

- Shared memory layout: tile_buffer (16KB) + gram_A_then_B (16KB) + partial_sums (1KB) = 33 KB
- Compute b_t from u, delta, B in registers
- 5 NS iterations with A^2 computed on-the-fly
- Store X_4 before 5th iteration

Structure:

```cuda
template<int kBlockSize = 256, int kTileSize = 64>
__global__ void newton_schulz_velocity_5step_kernel(
    const float* __restrict__ u,            // [B, D, L]
    const float* __restrict__ delta,        // [B, D, L]
    const float* __restrict__ B,            // [D, N] or [B, G, L, N]
    float* __restrict__ velocity_ortho,     // [B, D, L, N]
    float* __restrict__ X_4_buffer,         // [B, D, L, N]
    float alpha, int B, int D, int L, int dstate, int t_start,
    int u_batch_stride, int u_d_stride,
    int delta_batch_stride, int delta_d_stride,
    int B_batch_stride, int B_group_stride, 
    int B_d_stride, int B_dstate_stride,
    bool is_variable_B, int n_groups
)
```

Implementation flow:

1. Compute b_t on-the-fly per element
2. Write to velocity_ortho buffer (used as working buffer)
3. Compute Frobenius norm
4. For each of 5 iterations:

   - Compute A = X @ X.T (accumulate from tiles)
   - Compute B = b*A + c*A^2 (A^2 on-the-fly, reuse buffer)
   - If step==4: save current X to X_4_buffer
   - Apply X = a*X + B@X

5. Final X_5 written to velocity_ortho

### Step 3: Implement Helper Functions

**File**: `csrc/selective_scan/newton_schulz_fwd_kernel.cuh`

Helper functions needed:

- `compute_b_t_element()` - Load u, delta, B and compute product
- `compute_frobenius_norm_tiled()` - Norm computation across tiles
- `compute_gram_matrix_tiled()` - A = X @ X.T with tiling
- `compute_polynomial_inplace()` - B = b*A + c*A^2 with A^2 on-the-fly
- `apply_orthogonalization_tiled()` - X = a*X + B@X
- `save_X_to_buffer()` - Store X_4 to global memory

### Step 4: Implement Backward Kernel

**File**: `csrc/selective_scan/newton_schulz_bwd_kernel.cuh`

Key features:

- Load X_4 from storage (fast!)
- Recompute b_t on-the-fly (cheap!)
- Recompute A_4 = X_4 @ X_4.T
- Backward through step 5 only:
  - grad_X_4 from direct term: a * grad_X_5
  - grad_X_4 from B@X term: B_4.T @ grad_X_5
  - grad_B_4 = grad_X_5 @ X_4.T
  - grad_A_4 from polynomial chain rule
  - grad_X_4 from Gram matrix: 2 * grad_A_4 @ X_4
- Backward through normalization
- Chain rule to u, delta, B

### Step 5: Create Launch Wrapper

**File**: `csrc/selective_scan/newton_schulz_fwd_kernel.cuh`

```cuda
inline void launch_newton_schulz_velocity_5step(
    const float* u, const float* delta, const float* B,
    float* velocity_ortho, float* X_4_buffer,
    float alpha, int batch, int dim, int seqlen, int dstate,
    int t_start, int t_end,
    // strides...
    cudaStream_t stream
)
```

Grid configuration: `(batch, t_end - t_start)`

Shared memory: 33 KB (fits default 48 KB)

Handle dstate > 64 case with extended SM config

### Step 6: Update Main Scan Kernel Integration

**File**: `csrc/selective_scan/selective_scan_fwd_kernel.cuh`

Changes to `selective_scan_fwd_cuda()`:

1. Remove Phase 1 (lines 474-500)
2. Update Phase 2 to call new 5-step kernel with u, delta, B
3. Phase 3 unchanged (reads orthogonalized results)

Remove old three-phase logic:

```cuda
// OLD: Phase 1, 2, 3 with ns_phase parameter
// NEW: Only Phase 1 (NS) and Phase 2 (scan)
```

### Step 7: Update Python Interface

**File**: Python binding file (likely `mamba_ssm/ops/selective_scan_interface.py`)

Buffer allocation changes:

- Remove b_t_buffer allocation
- Allocate X_4_buffer: `batch * dim * seqlen * dstate * sizeof(float)`
- Pass X_4_buffer to forward/backward

### Step 8: Shared Memory Configuration

**File**: `csrc/selective_scan/newton_schulz_fwd_kernel.cuh`

For dstate > 64, configure extended shared memory:

```cuda
const int smem_size = (kTileSize * dstate + 
                       dstate * dstate + 
                       kBlockSize) * sizeof(float);

if (smem_size > 48 * 1024) {
    cudaFuncSetAttribute(
        newton_schulz_velocity_5step_kernel<kBlockSize, kTileSize>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
}
```

### Step 9: Testing and Validation

- Unit test: Forward pass produces orthogonal matrices
- Unit test: Backward pass gradients correct (compare with numerical)
- Integration test: Full MomentumMamba forward/backward
- Performance test: Measure Phase 1 time (target ~48ms)
- Performance test: Measure backward time (target ~20ms)
- Memory test: Verify 128 MB buffer size (batch=8)

### Step 10: Documentation Updates

- Update MEMORY_FLOW_ANALYSIS.md with new 2-phase design
- Document shared memory optimization strategy
- Document performance characteristics
- Add comments explaining A^2 on-the-fly computation

## Performance Targets

| Metric | Target | Notes |

|--------|--------|-------|

| Forward time | 48ms | Phase 1 NS (eliminates old Phase 1) |

| Backward time | 20ms | Fast X_4 load, cheap b_t recompute |

| Shared memory | 33 KB | Fits 48 KB default for dstate=64 |

| Extra memory | 128 MB | X_4 buffer only (batch=8) |

| Orthogonality quality | ~1e-6 | 5-step NS residual norm |

## Edge Cases to Handle

1. **Variable B handling**: Support both constant and time-varying B
2. **dstate > 64**: Configure extended shared memory (97 KB for dstate=128)
3. **Small sequences**: Handle L < 512 gracefully
4. **Transpose case**: Handle M > N (tall matrices) with transpose flag
5. **Numerical stability**: Epsilon in norm computation, safe division

## Files Modified Summary

1. `csrc/selective_scan/selective_scan.h` - Parameter structure
2. `csrc/selective_scan/newton_schulz_fwd_kernel.cuh` - New 5-step kernel
3. `csrc/selective_scan/newton_schulz_bwd_kernel.cuh` - New backward kernel  
4. `csrc/selective_scan/selective_scan_fwd_kernel.cuh` - Remove Phase 1, update integration
5. Python binding file - Buffer allocation updates
6. `MEMORY_FLOW_ANALYSIS.md` - Documentation update

## Success Criteria

- ✅ Forward pass completes in ~48ms (batch=8, L=512, D=128, N=64)
- ✅ Backward pass completes in ~20ms
- ✅ Shared memory usage ≤ 48 KB for dstate=64
- ✅ Orthogonality quality: ||X^T X - I|| < 1e-5
- ✅ Gradients match numerical gradients (relative error < 1e-3)
- ✅ Memory usage: +128 MB (X_4 buffer only)
- ✅ All existing tests pass