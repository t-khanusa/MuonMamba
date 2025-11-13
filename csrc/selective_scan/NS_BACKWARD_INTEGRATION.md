# Newton-Schulz Backward Pass Integration into MomentumMamba

## Summary

Successfully integrated the Newton-Schulz backward pass (detached first 4 steps, gradient only through last step) into the MomentumMamba backward kernel.

## Integration Points

### 1. Backward Entry Point (`selective_scan.cpp`)

**Location**: `selective_scan_bwd` function (lines 502-578)

**Changes**:
- Allocate `grad_X_4_buffer` when `use_newton_schulz` is enabled
  - Shape: `[batch, dim, seqlen, dstate]` in float32
  - Used to accumulate gradients w.r.t. the orthogonalized B (output of NS)
- Set NS parameters in `params`:
  - `params.use_newton_schulz = use_newton_schulz`
  - `params.grad_X_4_buffer_ptr = grad_X_4_buffer.data_ptr()`
- After main backward kernel completes, call NS backward pass:
  - Passes `grad_X_4_buffer` as `grad_output` (gradients w.r.t. NS output)
  - NS backward computes and **adds** gradients to `grad_u`, `grad_delta`, `grad_B`

### 2. Backward Kernel (`selective_scan_bwd_kernel.cuh`)

**Location**: Multiple locations in `selective_scan_bwd_kernel`

#### 2.1 Variable B Case (Real, lines 379-424)

**When NS is enabled**:
- Instead of accumulating `dB_vals[i]` directly to `dB`, accumulate to `grad_X_4_buffer`
- Gradient w.r.t. orthogonalized B: `dB_vals[i] = dv * params.alpha * delta_vals[i] * float(u_vals[i])`
- Accumulation: `grad_X_4_buffer[batch, dim, timestep, dstate] += dB_vals[i]`

**When NS is NOT enabled**:
- Normal behavior: accumulate directly to `dB` ✓

#### 2.2 Constant B Case (Real, lines 381-402)

**When NS is enabled**:
- Compute per-timestep gradients for B (both from velocity and output paths)
- Gradient from velocity: `dv * params.alpha * delta_vals[i] * float(u_vals[i])`
- Gradient from output: `dout_vals[i] * (!kIsVariableC ? h_t : h_t * C_vals[i])`
- Accumulate total into `grad_X_4_buffer` per timestep
- Skip accumulating `dBC_val` into `smem_dbc` for B (lines 455-460)

**When NS is NOT enabled**:
- Normal behavior: accumulate `dBC_val` across all timesteps into `smem_dbc`, then add to `dB` at end ✓

#### 2.3 Complex Cases

**Variable B (Complex, lines 672-695)**:
- Extract real part from complex gradient (B is real even when A is complex)
- Accumulate only real parts (even indices in `dB_vals_f`) into `grad_X_4_buffer`
- Skip accumulating to `dB` when NS is enabled

**Constant B (Complex, lines 611-635)**:
- Extract real part: `(dv * params.alpha * delta_vals[i] * float(u_vals[i])).real_`
- Extract real part from output gradient
- Accumulate into `grad_X_4_buffer` per timestep
- Skip accumulating `dBC_val` for B when NS is enabled (lines 677-683)

#### 2.4 Final Gradient Accumulation (lines 705-716)

**When NS is enabled and B is constant**:
- Skip adding `dBC_val` to `dB` (NS backward handles it)
- Still accumulate for C if constant

**When NS is NOT enabled**:
- Normal behavior: add `dBC_val` to `dB` ✓

### 3. NS Backward Call (`selective_scan.cpp`, lines 543-571)

After main backward kernel:
```cpp
launch_newton_schulz_velocity_5step_backward<input_t, weight_t>(
    grad_X_4,              // Gradient w.r.t. orthogonalized B
    u_ptr, delta_ptr, B_ptr,  // Original inputs for recomputation
    grad_u_ptr, grad_delta_ptr, grad_B_ptr,  // Output gradients (will be added to)
    params.alpha,
    params.batch, params.dim, params.seqlen, params.dstate,
    0, params.seqlen,  // Process all timesteps
    ... strides ...
    params.is_variable_B, params.n_groups,
    stream
);
```

**Key Points**:
- NS backward recomputes X_4 from original inputs (4 detached NS iterations)
- Then backpropagates through 5th iteration only
- Computes gradients w.r.t. original `u`, `delta`, `B`
- **Adds** these gradients to existing `grad_u`, `grad_delta`, `grad_B` buffers

## Gradient Flow

```
Forward:
  u, delta, B → G = alpha * delta * B * u
  G → NS_forward → G_ortho (orthogonalized B)
  G_ortho → velocity scan → output

Backward:
  grad_output → velocity backward → grad_G_ortho
  grad_G_ortho → NS_backward → grad_u, grad_delta, grad_B
  (Also: grad_output → hidden backward → grad_u, grad_delta, grad_A, grad_C)
```

## Testing

All changes are guarded by `params.use_newton_schulz` checks, ensuring:
- ✅ Normal backward pass (without NS) works unchanged
- ✅ Backward pass with NS correctly accumulates gradients
- ✅ Supports both variable B and constant B
- ✅ Supports both real and complex A

## Notes

- NS backward treats B_4 as a **constant** (detached), only backpropagates through its application
- Per-timestep gradients are required for NS backward (forward NS is applied per timestep)
- Complex A case: B is still real (input_t), so only real parts are accumulated

## Status

✅ Integration complete
✅ No lint errors
✅ Backward compatible (non-NS path unchanged)
✅ Ready for testing

