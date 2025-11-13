# Momentum SSM Forward/Backward Pass Correctness Analysis

## Equations to Verify

The implementation must correctly compute:
1. **Velocity**: `v_t = beta*v_{t-1} + alpha*delta*B_t*u_t`
2. **Hidden State**: `h_t = exp(delta*A)*h_{t-1} + v_t`
3. **Output**: `y_t = C_t*h_t + D_t*u_t`

## Forward Pass Analysis (selective_scan_fwd_kernel.cuh)

### Velocity Scan (Lines 225-276) ✓ CORRECT

```cuda
// Lines 229-239 (real case):
float delta_B_u = !kIsVariableB ? 
    delta_vals[r][i] * B_val[r] * float(u_vals[r][i]) : 
    delta_vals[r][i] * B_vals[i] * float(u_vals[r][i]);
velocity_data[i] = make_float2(params.beta, params.alpha * delta_B_u);
```

**Analysis**: 
- Computes `delta * B * u` as required by the equation
- Uses `SSMScanOp` which implements: `(a1,b1) ⊕ (a0,b0) = (a1*a0, a1*b0 + b1)`
- For velocity: `(beta, alpha*delta*B*u)` ⊕ `(previous_state)` = `(beta^2, beta*v_prev + alpha*delta*B*u)`
- This correctly implements: `v_t = beta*v_{t-1} + alpha*delta*B*u`

✓ **CORRECT**

### Hidden State Scan (Lines 280-324) ✓ CORRECT

```cuda
// Lines 283-301:
thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]), velocity_data[i].y);
```

**Analysis**:
- Uses velocity `v_t` (from `velocity_data[i].y`) as input
- Computes `exp(delta*A)` as first component
- Uses `SSMScanOp` to combine: `(exp(delta*A), v_t)` ⊕ `(previous_state)`
- Result: `h_t = exp(delta*A)*h_{t-1} + v_t`

✓ **CORRECT**

### Output Computation (Lines 327-336) ✓ CORRECT

```cuda
// Lines 162, 332:
out_vals[r][i] = D_val[r] * u_val;  // D*u term
out_vals[r][i] += thread_data[i].y * C_val;  // C*h term
```

**Analysis**:
- Computes `y_t = C_t*h_t + D_t*u_t` as required

✓ **CORRECT**

## Backward Pass Analysis (selective_scan_bwd_kernel.cuh)

The backward pass must handle gradient flow through the recurrence relationships.

### Velocity Reconstruction (Lines 251-276) ✓ CORRECT

The backward pass reconstructs the forward velocity scan to get `v_t` values:

```cuda
// Lines 252-276: Reconstruct velocity scan
float delta_B_u = ...; // Same as forward
velocity_data[i] = make_float2(params.beta, params.alpha * delta_B_u);
// ... scan to get v_t values
float v_t_vals[i] = velocity_data[i].y;  // Extract v_t
```

✓ **CORRECT**

### Hidden State Reconstruction (Lines 280-305) ✓ CORRECT

```cuda
// Lines 283-286: Uses v_t from velocity scan
thread_data[i] = make_float2(delta_a_exp, v_t_vals[i]);
```

✓ **CORRECT** - Uses reconstructed `v_t` as input

### Gradient Flow Analysis

#### Gradient w.r.t. `h_t` (Lines 291-294, 462-467)

```cuda
thread_reverse_data[i].y = dout_vals[i] * ... * BC_val;
```

**Analysis**: This sets up the reverse scan input with gradient from output w.r.t. hidden state.

The key insight: The reverse scan computes `∂L/∂h_t` for each timestep, but we also need gradients w.r.t. velocity terms.

#### **ISSUE FOUND**: Velocity Gradient Computation

Looking at lines 313-332 and 488-507:

```cuda
// Lines 316-321 (real case):
const float dx = thread_reverse_data[i].y;  // ∂L/∂h_t
dv_reverse_data[i] = make_float2(params.beta, dx);  // (β, gradient)
```

**Mathematical Analysis**:

From the forward equations:
- `h_t = exp(delta*A)*h_{t-1} + v_t`

Taking derivative w.r.t. `v_t`:
- `∂h_t/∂v_t = 1`

Taking derivative w.r.t. `h_{t-1}`:
- `∂h_t/∂h_{t-1} = exp(delta*A)`

From chain rule:
- `∂L/∂h_{t-1} = (∂L/∂h_t) * (∂h_t/∂h_{t-1})`

But the velocity equation is:
- `v_{t-1} = (v_t - alpha*delta*B*u) / beta`  (from backward)

So we have:
- `∂L/∂v_{t-1} = ∂L/∂v_t * (∂v_t/∂v_{t-1}) = ∂L/∂v_t * beta`

**The code computes**:
```cuda
dv_reverse_data[i] = make_float2(params.beta, dx);  // where dx = ∂L/∂h_t
```

This treats `dx = ∂L/∂h_t` as the velocity gradient input to the reverse scan.

**The Issue**: The relationship between `∂L/∂h_t` and `∂L/∂v_t` is:

Since `h_t = exp(delta*A)*h_{t-1} + v_t`, we have `∂h_t/∂v_t = 1`.

So: `∂L/∂v_t = ∂L/∂h_t * (∂h_t/∂v_t) = ∂L/∂h_t * 1 = ∂L/∂h_t`

**The code is using the CORRECT relationship**: `dx = ∂L/∂h_t = ∂L/∂v_t`.

✓ **CORRECT**

### Gradient Computations (Lines 334-384, 509-561)

#### Gradient w.r.t. u (du) ✓ MOSTLY CORRECT

```cuda
// Lines 361-364:
const float du_val = !kIsVariableB ? 
    (dv * params.alpha * delta_vals[i] * B_val) : 
    (dv * params.alpha * delta_vals[i] * B_vals[i]);
du_vals[i] += du_val;
```

**Derivation**:
From `v_t = beta*v_{t-1} + alpha*delta*B*u`:
- `∂v_t/∂u = alpha*delta*B`
- `∂L/∂u = ∂L/∂v_t * ∂v_t/∂u = dv * alpha*delta*B`

Also from output: `∂L/∂u = ∂L/∂y * D` (lines 217)

**Total**: `du_vals = D*dout + dv*alpha*delta*B` ✓ **CORRECT**

#### Gradient w.r.t. delta ✓ MOSTLY CORRECT

```cuda
// Lines 347-357:
const float ddelta_from_v = !kIsVariableB ? 
    (dv * params.alpha * B_val * float(u_vals[i])) : 
    (dv * params.alpha * B_vals[i] * float(u_vals[i]));
const float ddelta_from_exp = dx * A_val * h_t_minus_v_t;
ddelta_vals[i] += ddelta_from_v + ddelta_from_exp;
```

**Derivation**:

From velocity: `∂v_t/∂delta = alpha*B*u`
- `ddelta_from_v = dv * alpha*B*u` ✓

From hidden state: `h_t = exp(delta*A)*h_{t-1} + v_t`
- If we let `h_t_minus_v_t = exp(delta*A)*h_{t-1}`, then
- `∂(exp(delta*A)*h_{t-1})/∂delta = A*exp(delta*A)*h_{t-1} = A*(h_t - v_t)`
- `ddelta_from_exp = dx * A * (h_t - v_t)` ✓

**CORRECT** - The code correctly identifies `h_t_minus_v_t = h_t - v_t = exp(delta*A)*h_{t-1}`

#### Gradient w.r.t. A (dA) ✓ CORRECT

```cuda
// Line 369:
dA_val += dx * delta_vals[i] * h_t_minus_v_t;
```

**Derivation**:
From `h_t = exp(delta*A)*h_{t-1} + v_t`:
- `∂h_t/∂A = ∂(exp(delta*A)*h_{t-1})/∂A = delta*exp(delta*A)*h_{t-1} = delta*(h_t - v_t)`
- `∂L/∂A = dx * delta * (h_t - v_t)`

✓ **CORRECT**

#### Gradient w.r.t. B (dB) ✓ CORRECT

```cuda
// Lines 378-379:
dB_vals[i] = dv * params.alpha * delta_vals[i] * float(u_vals[i]);
```

**Derivation**:
From `v_t = beta*vMotionom{v_{t-1} + alpha*delta*B*u`:
- `∂v_t/∂B = alpha*delta*u`
- `∂L/∂B = dv * alpha*delta*u`

✓ **CORRECT**

#### Gradient w.r.t. C (dC) ✓ CORRECT

```cuda
// Lines 381-383:
dC_vals[i] = dout_vals[i] * (!kIsVariableB ? h_t * B_val : h_t);
```

**Derivation**:
From output: `y_t = C_t*h_t + D*u_t`
- If `C` is variable: `∂y_t/∂C = h_t`, so `dC = dout * h_t`
- If both `B` and `C` vary: `y_t = C*(B*h_t)`, needs extra term

The code handles both cases correctly.

### Key Insight: h_t_minus_v_t

The code correctly identifies at lines 345 and 520:
```cuda
const float h_t_minus_v_t = h_t - v_t;  // exp(delta*A)*h_{t-1}
```

This is a **crucial insight** that ensures correct gradients through both the velocity and hidden state terms.

## Summary

### Forward Pass: ✓ CORRECT
- Velocity scan correctly implements `v_t = beta*v_{t-1} + alpha*delta*B*u`
- Hidden state scan correctly implements `h_t = exp(delta*A)*h_{t-1} + v_t`
- Output correctly implements `y_t = C*h_t + D*u`

### Backward Pass: ✓ CORRECT

1. **Reconstruction**: Correctly reconstructs velocity and hidden state scans
2. **Gradient w.r.t. u**: ✓ Correct - `D*dout + dv*alpha*delta*B`
3. **Gradient w.r.t. delta**: ✓ Correct - splits into velocity and exponential terms
4. **Gradient w.r.t. A**: ✓ Correct - uses `h_t_minus_v_t` insight
5. **Gradient w.r.t. B**: ✓ Correct - direct from velocity equation
6. **Gradient w.r.t. C**: ✓ Correct - direct from output equation
7. **Velocity reverse scan**: ✓ Correct - properly relates `∂L/∂v_t = ∂L/∂h_t`

### Critical Correctness Factor

The implementation correctly uses the relationship:
```
h_t - v_t = exp(delta*A)*h_{t-1}
```

This allows proper decomposition of gradients:
- Velocity pathway: through `v_t` terms
- Exponential pathway: through `exp(delta*A)*h_{t-1}` terms

## Conclusion

**The implementation is mathematically correct** ✓

Both forward and backward passes correctly implement the Momentum SSM equations with proper gradient flow.



