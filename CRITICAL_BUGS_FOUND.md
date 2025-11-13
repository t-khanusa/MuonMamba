# Critical Bugs Found in MuonMamba Backward Pass

## Bug Summary

The MuonMamba backward pass with Newton-Schulz has several critical issues that prevent correct gradient computation.

## Bug 1: Main Backward Kernel Does NOT Compute du/ddelta/dB When NS is Enabled ✅ CORRECT

**Status**: This is actually the CORRECT design!

When NS is enabled:
- Main backward kernel only computes `ddelta` from exp path (line 396)
- Main backward kernel accumulates `dv` into `grad_X_4_buffer` (line 391)
- NS backward kernel computes ALL gradients for `u`, `delta`, and `B` from velocity path

## Bug 2: Zero Initialization ✅ FIXED

**Fixed in**: `csrc/selective_scan/selective_scan.cpp` line 498-499

```cpp
at::Tensor du = use_newton_schulz ? torch::zeros_like(u) : torch::empty_like(u);
at::Tensor ddelta = use_newton_schulz ? torch::zeros_like(delta) : torch::empty_like(delta);
```

When NS is enabled, `du` and `ddelta` start at zero, NS backward adds to them.

## Bug 3: Temporary Float32 Buffers ✅ IMPLEMENTED

**Implemented in**: Lines 509-532 of `selective_scan.cpp`

NS backward writes float32 gradients, but `du/ddelta/dB` might be float16/bfloat16.
Solution: Use temporary float32 buffers, then convert and add.

## Bug 4: NS Backward Not Writing Gradients? ❌ NEEDS INVESTIGATION

**Symptoms**:
- `dB` is all zeros after backward
- `du` is 1.0 (suspicious - should be zeros + NS contributions)

**Possible causes**:
1. `grad_X_4_buffer` is all zeros (no gradients flowing from main kernel)
2. NS backward kernel has a bug
3. Gradients not being accumulated correctly

## Next Steps

1. Add debug output to verify `grad_X_4_buffer` has non-zero values
2. Verify NS backward kernel is actually being launched
3. Check if there's an indexing bug in NS backward kernel
4. Verify the gradient accumulation logic

## Key Design Points

### Forward Pass:
1. Compute `b_t = alpha * delta * B * u`
2. Apply NS5: `b_t_ortho = NS5(b_t)` on `[dim, dstate]` matrices
3. Velocity: `v_t = beta * v_{t-1} + b_t_ortho`
4. Hidden: `h_t = exp(delta*A) * h_{t-1} + v_t`

### Backward Pass:
1. **Main kernel**: 
   - Reverse scan hidden states
   - Reverse scan velocity to get `dv` (gradient w.r.t. `b_t_ortho`)
   - Accumulate `dv` into `grad_X_4_buffer`
   - Compute `ddelta` from exp path only
   
2. **NS backward kernel**:
   - Takes `grad_X_4_buffer` as input (gradient w.r.t. `b_t_ortho`)
   - Recomputes X_0 → X_4 (detached)
   - Backprops through 5th iteration only
   - Computes and accumulates gradients for `u`, `delta`, `B`

## Code Flow

```
selective_scan_bwd() in C++:
  ├─ Allocate du=zeros, ddelta=zeros (for NS mode)
  ├─ Allocate du_ns_temp, ddelta_ns_temp, dB_ns_temp (float32)
  ├─ Set params.du_ns_temp_ptr, etc.
  ├─ Launch main backward kernel:
  │    ├─ Accumulate dv → grad_X_4_buffer
  │    └─ Compute ddelta (exp path only)
  ├─ Launch NS backward kernel (in selective_scan_bwd_cuda):
  │    ├─ Read grad_X_4_buffer
  │    ├─ Recompute NS forward (X_0→X_4)
  │    ├─ Backprop through 5th iteration
  │    └─ Write to du_ns_temp, ddelta_ns_temp, dB_ns_temp
  └─ Add NS gradients to final gradients:
       ├─ du += du_ns_temp
       ├─ ddelta += ddelta_ns_temp
       └─ dB += dB_ns_temp
```





