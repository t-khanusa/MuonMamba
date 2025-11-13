# 5-Step Newton-Schulz Implementation Status

## ✅ Completed Tasks

### 1. Parameter Structure Updates (`selective_scan.h`)
- ✅ Removed `b_t_buffer_ptr` and `b_t_ortho_buffer_ptr`
- ✅ Removed `ns_phase` (no longer needed)
- ✅ Added `X_4_buffer_ptr` for storing intermediate state
- ✅ Added `ns_steps` parameter (default 5)
- ✅ Updated backward parameter structure with `grad_X_4_buffer_ptr`

### 2. Forward Kernel Implementation (`newton_schulz_fwd_kernel.cuh`)
- ✅ Implemented `newton_schulz_velocity_5step_kernel` with:
  - On-the-fly b_t computation from u, delta, B
  - 5 Newton-Schulz iterations
  - Shared memory optimization (33 KB):
    - tile_buffer: 16 KB
    - gram_A_then_B: 16 KB (reused for A and B)
    - partial_sums: 1 KB
  - A² computed on-the-fly to save memory
  - X_4 storage before 5th iteration
  - Support for both constant and variable B
- ✅ Implemented `launch_newton_schulz_velocity_5step` wrapper with:
  - Grid configuration (batch, timesteps)
  - Shared memory allocation
  - Extended SM configuration for dstate > 64

### 3. Main Integration (`selective_scan_fwd_kernel.cuh`)
- ✅ Removed old Phase 1 (b_t computation in scan kernel)
- ✅ Removed ns_phase logic
- ✅ Updated `selective_scan_fwd_cuda` to:
  - Call 5-step NS kernel with u, delta, B parameters
  - Launch scan kernel that reads from X_4_buffer
- ✅ Updated scan kernel to:
  - Read orthogonalized b_t from X_4_buffer
  - Removed old Phase 1 computation code
  - Simplified velocity scan logic

### 4. Buffer Allocation (`selective_scan.cpp`)
- ✅ Added X_4_buffer allocation when beta != 0.0
- ✅ Size: [batch, dim, seqlen, dstate] in float32
- ✅ Set X_4_buffer_ptr in params
- ✅ Set use_newton_schulz and ns_steps flags

### 5. Documentation (`MEMORY_FLOW_ANALYSIS_5STEP.md`)
- ✅ Comprehensive design documentation
- ✅ Shared memory layout analysis
- ✅ Memory traffic analysis
- ✅ Performance characteristics
- ✅ Equation flow and architecture overview

---

## 🔄 Remaining Tasks

### 6. Backward Kernel Implementation (Not Yet Started)
**File**: `csrc/selective_scan/newton_schulz_bwd_kernel.cuh`

Need to implement:
- `newton_schulz_velocity_5step_backward_kernel` with:
  - Load X_4 from storage
  - Recompute b_t on-the-fly (cheap)
  - Recompute A_4 = X_4 @ X_4^T
  - Backward through step 5 only:
    - grad_X_4 from direct term: a × grad_X_5
    - grad_X_4 from B@X term: B_4^T @ grad_X_5
    - grad_B_4 = grad_X_5 @ X_4^T
    - grad_A_4 from polynomial chain rule
    - grad_X_4 from Gram matrix: 2 × grad_A_4 @ X_4
  - Backward through normalization
  - Chain rule to u, delta, B gradients

- `launch_newton_schulz_velocity_5step_backward` wrapper

### 7. Backward Integration (Not Yet Started)
**File**: `csrc/selective_scan/selective_scan_bwd_kernel.cuh`

Need to:
- Update backward pass to call 5-step NS backward kernel
- Ensure gradients flow correctly from scan to NS to inputs

### 8. Backward Buffer Allocation (Not Yet Started)
**File**: `csrc/selective_scan/selective_scan.cpp`

Need to:
- Update `selective_scan_bwd` function
- Allocate grad_X_4_buffer if needed
- Pass gradients correctly

### 9. Testing (Not Yet Started)
Need to create:
- Unit test: Forward pass orthogonality (||X^T X - I|| < 1e-5)
- Unit test: Backward pass gradient correctness
- Integration test: Full MomentumMamba forward/backward
- Performance benchmark: Forward ~48ms, backward ~20ms
- Memory test: Verify 128 MB buffer size

### 10. Bug Fixes and Optimization
- Fix any compilation errors
- Test with various batch sizes, sequence lengths, dstate values
- Profile and optimize if needed
- Handle edge cases (very small/large sequences)

---

## Current Architecture Summary

### Two-Phase Forward Pass

**Phase 1: 5-Step Newton-Schulz**
```
Grid: (batch, timesteps)
- Compute b_t = alpha × delta × B × u on-the-fly
- Apply 5 NS iterations with SM reuse
- Store X_4 before 5th iteration
- Output: velocity_ortho [batch, dim, seqlen, dstate]
```

**Phase 2: Velocity/Hidden State Scan**
```
Grid: (batch, dim)
- Read b_t_ortho from X_4_buffer
- Velocity scan: v_t = beta × v_{t-1} + b_t_ortho
- Hidden scan: h_t = exp(delta×A) × h_{t-1} + v_t
- Output: y_t = C × h_t + D × u
```

### Memory Usage
- X_4_buffer: batch × dim × seqlen × dstate × 4 bytes
- For batch=8, dim=128, seqlen=512, dstate=64: **128 MB**
- Shared memory per block: **33 KB** (fits 48 KB default)

### Performance Targets
- Forward: ~48ms (Phase 1) + ~15ms (Phase 2) = **63ms**
- Backward: **~20ms** (fast X_4 load, cheap b_t recompute)
- Orthogonality quality: **||X^T X - I|| < 1e-5**

---

## Next Steps

1. **Immediate**: Implement backward kernel (`newton_schulz_velocity_5step_backward_kernel`)
2. **Then**: Integrate backward kernel into main backward pass
3. **Then**: Allocate grad buffers in backward pass
4. **Then**: Compile and test for errors
5. **Then**: Create unit tests for orthogonality and gradients
6. **Finally**: Performance benchmarking and optimization

---

## Key Design Decisions Made

1. ✅ **5-step NS**: Better quality (1e-5 vs 1e-3 orthogonality)
2. ✅ **On-the-fly b_t**: Eliminates Phase 1 kernel launch, saves bandwidth
3. ✅ **Store X_4 only**: Fast backward (20ms vs 35ms if recompute)
4. ✅ **SM reuse**: A and B share 16 KB buffer (33 KB total vs 49 KB)
5. ✅ **A² on-the-fly**: Saves 16 KB SM, minimal compute overhead

---

## Compilation Status

**No linter errors** in modified files as of last check.

Files ready for compilation:
- ✅ `selective_scan.h`
- ✅ `newton_schulz_fwd_kernel.cuh`
- ✅ `selective_scan_fwd_kernel.cuh`
- ✅ `selective_scan.cpp`

---

## Testing Plan

Once backward pass is implemented:

1. **Orthogonality Test**
   ```python
   # Test that ||X^T X - I||_F < 1e-5
   b_t_ortho = forward_pass(...)
   gram = b_t_ortho.T @ b_t_ortho
   error = torch.norm(gram - torch.eye(N))
   assert error < 1e-5
   ```

2. **Gradient Test**
   ```python
   # Test gradients match numerical
   grad_analytical = backward_pass(...)
   grad_numerical = compute_numerical_gradient(...)
   rel_error = torch.norm(grad_analytical - grad_numerical) / torch.norm(grad_numerical)
   assert rel_error < 1e-3
   ```

3. **Performance Test**
   ```python
   # Benchmark forward/backward
   forward_time = benchmark_forward(batch=8, L=512, D=128, N=64)
   backward_time = benchmark_backward(...)
   assert forward_time < 65  # ms
   assert backward_time < 25  # ms
   ```

---

## References

- Design document: `MEMORY_FLOW_ANALYSIS_5STEP.md`
- Implementation plan: `5-step-ns-speed-optimized.plan.md`
- Original analysis: `MEMORY_FLOW_ANALYSIS.md`





