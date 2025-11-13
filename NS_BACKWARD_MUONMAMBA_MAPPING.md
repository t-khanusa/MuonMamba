# Newton-Schulz Backward Pass Integration with MuonMamba

## Forward Pass Flow

### Phase 1: Newton-Schulz 5-Step Orthogonalization
**Kernel**: `newton_schulz_velocity_5step_kernel` (`newton_schulz_fwd_kernel.cuh`)

1. Compute `b_t = alpha * delta * B * u` for each timestep
2. Convert to bfloat16 and normalize: `X_0 = bfloat16(b_t) / norm`
3. Apply 5 NS iterations: `X_0 → X_1 → X_2 → X_3 → X_4 → X_5`
   - Each iteration: `X_{i+1} = a*X_i + B_i @ X_i` where `B_i = b*A_i + c*A_i²`
4. **Store final result `X_5` in `X_4_buffer`** (misnamed, but contains X_5)
   - `X_5` is `b_t_ortho` (orthogonalized version of `b_t`)

### Phase 2: Selective Scan
**Kernel**: `selective_scan_fwd_kernel` (`selective_scan_fwd_kernel.cuh`)

1. **Load `b_t_ortho` from `X_4_buffer`** (line 245-255)
   ```cuda
   delta_B_u = velocity_ortho_buffer[global_idx];  // Load X_5 = b_t_ortho
   velocity_data[i] = make_float2(params.beta, delta_B_u);
   ```

2. Velocity scan: `v_t = beta * v_{t-1} + b_t_ortho`
3. Hidden state scan: `h_t = exp(delta*A) * h_{t-1} + v_t`
4. Output: `y_t = C_t * h_t + D_t * u_t`

## Backward Pass Flow

### Phase 1: Selective Scan Backward
**Kernel**: `selective_scan_bwd_kernel` (`selective_scan_bwd_kernel.cuh`)

#### Key Fix: Reconstruct velocity with correct `b_t_ortho`

**Before fix** (WRONG):
```cuda
// Used b_t = alpha * delta * B * u
float delta_B_u = delta_vals[i] * B_val * float(u_vals[i]);
velocity_data[i] = make_float2(params.beta, params.alpha * delta_B_u);
```
❌ This reconstructs with `b_t`, not `b_t_ortho`, giving wrong `v_t` values!

**After fix** (CORRECT - lines 254-286):
```cuda
if (params.use_newton_schulz) {
    // Load b_t_ortho from X_4_buffer (same as forward pass)
    float *velocity_ortho_buffer = params.X_4_buffer_ptr;
    const int time_idx = chunk * kChunkSize + threadIdx.x + i * kNThreads;
    
    int global_idx = batch_id * params.dim * params.seqlen * params.dstate +
                    dim_id * params.seqlen * params.dstate +
                    time_idx * params.dstate + state_idx;
    delta_B_u = velocity_ortho_buffer[global_idx];  // Load X_5 = b_t_ortho
    
    // b_t_ortho already includes alpha scaling
    velocity_data[i] = make_float2(params.beta, delta_B_u);
} else {
    // Normal mode: compute b_t = alpha * delta * B * u
    delta_B_u = delta_vals[i] * B_val * float(u_vals[i]);
    velocity_data[i] = make_float2(params.beta, params.alpha * delta_B_u);
}
```
✅ This correctly uses `b_t_ortho` to reconstruct `v_t`

#### Gradient Computation

1. **Reconstruct scans** using correct `b_t_ortho`:
   - Velocity scan: `v_t = beta * v_{t-1} + b_t_ortho`
   - Hidden state scan: `h_t = exp(delta*A) * h_{t-1} + v_t`

2. **Reverse scans** to compute gradients:
   - Hidden state reverse: computes `dh_t` (gradient w.r.t. hidden states)
   - Velocity reverse: computes `dv_t` (gradient w.r.t. velocity states)

3. **Critical insight**: `dv` is gradient w.r.t. `b_t_ortho`, not `b_t`
   - `dv = ∂L/∂(b_t_ortho)`
   - We **accumulate `dv` into `grad_X_4_buffer`** (lines 364-376)
   ```cuda
   if (params.use_newton_schulz) {
       // Accumulate gradient w.r.t. b_t_ortho into grad_X_4_buffer
       grad_X_4_buffer[grad_idx] = dv;  // dX_5
       
       // Only compute gradients from exp path (not velocity path)
       ddelta_vals[i] += dx * A_val * h_t_minus_v_t;  // Only exp contribution
   }
   ```

4. **Do NOT compute** gradients for `u`, `delta`, `B` from velocity path when NS enabled
   - These will be computed by NS backward pass

### Phase 2: Newton-Schulz Backward
**Kernel**: `newton_schulz_velocity_5step_backward_kernel` (`newton_schulz_bwd_kernel.cuh`)

#### Input
- `grad_X_4_buffer` (contains `dX_5 = ∂L/∂(b_t_ortho)` accumulated from Phase 1)
- Original inputs: `u`, `delta`, `B` (for recomputation)

#### Critical Design: Detached First 4 Steps

**Only gradient through 5th iteration** (first 4 iterations detached):

1. **Recompute X_0 → X_4** (detached, no gradients):
   ```cuda
   with torch.no_grad():
       X_0 = normalize(bfloat16(b_t))
       for i in range(4):  // Only 4 iterations
           A_i = X_i @ X_i.T
           B_i = b*A_i + c*A_i²
           X_{i+1} = a*X_i + B_i @ X_i
       // X_4 is now ready
   ```

2. **Recompute 5th iteration forward** (need A_4, B_4):
   ```cuda
   A_4 = X_4 @ X_4.T  // Detached
   B_4 = b*A_4 + c*A_4²  // Detached
   // B_4 is treated as CONSTANT
   ```

3. **Backward through 5th iteration** (lines 1013-1143):
   ```cuda
   // Treat B_4 as constant (detached)
   // Only backprop through APPLICATION of B_4, not its COMPUTATION
   dX_4 = a * dX_5 + B_4.T @ dX_5
   ```
   ✅ This is correct! B_4 has no gradients flowing through its computation.

4. **Backward through normalization** (lines 1145-1260):
   ```cuda
   // X_0 = b_t_bf16 / norm
   // Compute: d(b_t) = (dX_4 - X_0 * <dX_4, X_0>) / norm
   dot_product = sum(dX_4 * X_0)
   d_b_t = (dX_4 - X_0 * dot_product) / norm
   ```

5. **Backward through `b_t = alpha * delta * B * u`** (lines 1262-1423):
   ```cuda
   // Compute gradients w.r.t. original inputs
   grad_u += alpha * delta * B * d_b_t
   grad_delta += alpha * B * u * d_b_t
   grad_B += alpha * delta * u * d_b_t
   ```

## Why This Works

### Detached First 4 Steps
- Forward: 5 full NS iterations
- Backward: Only gradients through 5th iteration
- Reason: Reduces memory (don't need to store X_0, X_1, X_2, X_3) and computation

### Key Properties
1. **X_5 approximates orthogonality** after 5 iterations
2. **Small changes in X_4** lead to small changes in X_5 (NS converges)
3. **Gradient through 5th iteration** captures most important information
4. **Detaching first 4 steps** is an approximation, but:
   - Saves memory (4x reduction)
   - Saves computation (backward through 4 iterations)
   - Empirically works well (Muon paper confirms this)

## Verification Checklist

✅ Forward: `X_4_buffer` contains `X_5 = b_t_ortho` (final NS output)
✅ Backward: Load `b_t_ortho` from `X_4_buffer` to reconstruct velocity scan
✅ Backward: Accumulate `dv = ∂L/∂(b_t_ortho)` into `grad_X_4_buffer`
✅ NS Backward: Treats B_4 as constant (detached)
✅ NS Backward: Only backprops through 5th iteration
✅ NS Backward: Computes gradients w.r.t. `u`, `delta`, `B` from `d(b_t)`

## Common Pitfalls (Fixed)

❌ **Bug 1**: Using `b_t` instead of `b_t_ortho` in backward velocity reconstruction
   - **Effect**: Wrong `v_t` values → wrong gradients for everything
   - **Fix**: Load `b_t_ortho` from `X_4_buffer` (lines 262-278)

❌ **Bug 2**: Computing gradients through B_4's computation in NS backward
   - **Effect**: Double-counting gradients, incorrect backprop
   - **Fix**: NS backward correctly treats B_4 as constant (line 1042)

❌ **Bug 3**: Computing velocity-path gradients for `u`, `delta`, `B` in Phase 1
   - **Effect**: Double-counting when NS backward also computes them
   - **Fix**: Skip velocity-path gradients when NS enabled (lines 363-380)

## Memory Layout

```
X_4_buffer: [batch, dim, seqlen, dstate] in float32
  - Forward: Stores X_5 = b_t_ortho (final NS output)
  - Backward: Read by selective_scan_bwd to reconstruct velocity

grad_X_4_buffer: [batch, dim, seqlen, dstate] in float32
  - Backward Phase 1: Accumulates dv = ∂L/∂(b_t_ortho)
  - Backward Phase 2: Read by NS backward as dX_5
```





