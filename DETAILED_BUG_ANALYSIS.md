# Detailed Bug Analysis

## Current Status

NS backward function is **verified correct** - matches working reference exactly.
Large gradient differences remain in full backward pass:
- ddelta: max diff ~27.8 (CUDA: 1.33, Ref: 42.56)
- dC: max diff ~12.1 (CUDA: 12.65, Ref: 35.53)

## Key Observations

1. **NS backward produces large gradients** for some timesteps (e.g., grad_delta = [0.1389, 27.7676] at timestep 2)
2. **db_t_ortho values are reasonable** (mean ~0.0007-0.002)
3. **NS backward is called per timestep** in PyTorch reference, but CUDA calls it once at end (should be equivalent)

## Potential Root Causes

### 1. Gradient Accumulation Order
- NS backward gradients are added during backward loop
- Exp path gradients are added after NS backward
- Need to verify this matches CUDA's order

### 2. dh Computation
- Reverse scan for hidden states: `dh[t] = dout[t]*C + exp(delta[t+1]*A) * dh[t+1]`
- Current implementation may not exactly match CUDA's inclusive reverse scan
- Need to verify: does `dh` correctly accumulate all future contributions?

### 3. dv Computation  
- Velocity reverse scan: `dv[t] = dh[t] + beta * dv[t+1]`
- Current: `dv_t = dh + beta * dv`
- Need to verify: does `dv` correctly accumulate all future contributions?

### 4. h_t_minus_v_t
- Computed from forward pass states: `h_t - v_t`
- These should match CUDA exactly
- Need to verify states are identical

## Next Steps

1. Add detailed comparison of `dh` and `dv` values at each timestep between CUDA and reference
2. Verify `h_t_minus_v_t` matches CUDA exactly
3. Check if gradient accumulation happens in correct order
4. Verify exp path gradient computation matches CUDA exactly





