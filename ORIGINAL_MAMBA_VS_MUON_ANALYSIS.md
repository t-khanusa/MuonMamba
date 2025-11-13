# Original Mamba vs Muon with Momentum: Complete Analysis

## ✅ CONFIRMATION: The B*C Pattern is Correct in Original Mamba!

After reviewing the **original Mamba implementation** (without momentum), I can confirm that the `B*C` storage pattern is **intentional and correct**.

---

## Original Mamba SSM Equations

### Standard SSM (Continuous):
```
dh/dt = A*h + B*u
y = C*h + D*u
```

### Discretized SSM (Original Mamba):
```
h_t = exp(delta*A) * h_{t-1} + (delta*B*u)
y_t = C * h_t + D*u
```

**Key insight:** The B parameter is **multiplied into the hidden state update**, not just a separate term!

---

## Original Mamba Implementation Analysis

### Line 162: Precompute delta*u
```cpp
delta_u_vals[r][i] = delta_vals[r][i] * u_val;
```

### Line 222: Scan input preparation (constant B case)
```cpp
thread_data[i] = make_float2(
    exp2f(delta_vals[r][i] * A_val[r]),  // Multiplicative: exp(delta*A)
    delta_u_vals[r][i]                    // Additive: delta*u (B=1 implicit!)
);
```

**When B is constant, it's implicitly set to 1 in the scan!**

### Line 210: Store B*C when both constant
```cpp
BC_val[r] = B[...] * C[...];  // Precompute B*C
```

### Line 266: Output computation
```cpp
out_vals[r][i] += thread_data[i].y * C_val;
// thread_data[i].y = h_t (contains accumulated delta*u, not delta*B*u)
// C_val = B*C (when both constant)
// Result: h_t * (B*C) = (delta*u accumulated) * (B*C)
```

---

## Why B*C is Correct in Original Mamba

### Case 1: Both B and C Constant

**Scan input (line 222):**
```cpp
thread_data[i].y = delta_u_vals[r][i];  // = delta*u (no B yet!)
```

**After scan:**
```cpp
h_t = exp(delta*A) * h_{t-1} + delta*u  // Accumulated, but NO B factor yet
```

**Output (line 266):**
```cpp
out += h_t * BC_val[r];  // = h_t * (B*C)
```

**Mathematical result:**
```
y_t = D*u + (B*C) * h_t
    = D*u + (B*C) * (sum of delta*u terms)
    = D*u + B*C * (delta_1*u_1 + exp*delta_2*u_2 + ...)
```

**This effectively applies B to the accumulated hidden state, then C for output!**

---

### Case 2: B Variable

**Scan input (line 222):**
```cpp
thread_data[i].y = B_vals[i] * delta_u_vals[r][i];  // = B*delta*u
```

**After scan:**
```cpp
h_t = exp(delta*A) * h_{t-1} + B*delta*u  // B already included
```

**Output (line 266):**
```cpp
C_val = BC_val[r];  // = C (just C, since B varies)
out += h_t * C_val;  // = h_t * C
```

**Mathematical result:**
```
y_t = D*u + C * h_t
    = D*u + C * (sum of B*delta*u terms)
```

**B is already in h_t, so only multiply by C!**

---

## The Key Insight: B Application Timing

| Case | B Application | H_t Contains | Output Multiplies | Final Output |
|------|---------------|--------------|-------------------|--------------|
| B const, C const | At output | delta*u (no B) | B*C | (B*C)*h = correct |
| B const, C var | At output | delta*u (no B) | B*C_t | (B*C_t)*h = correct |
| B var, C const | In scan | B*delta*u | C | C*h = correct |
| B var, C var | In scan | B*delta*u | C_t | C_t*h = correct |

**All cases are mathematically equivalent and correct!**

---

## Muon with Momentum: Comparison

### Your Modified Implementation (Momentum + NS)

**Your equations:**
```
b_t = alpha * delta * B * u              [Equation 1]
b_t_ortho = NS_5step(b_t)                [Equation 2]
v_t = beta * v_{t-1} + b_t_ortho         [Equation 3]
h_t = exp(delta*A) * h_{t-1} + v_t       [Equation 4]
y_t = C * h_t + D*u                      [Equation 5]
```

**Key difference:** Your implementation:
1. Computes `b_t = alpha * delta * B * u` (B applied early)
2. Stores in NS kernel output buffer
3. Loads from buffer in scan (B already applied)
4. Output should multiply by C only, not B*C

---

## 🚨 THE ACTUAL BUG IN YOUR CODE

Looking at your implementation again:

### Your Code (Lines 251-257):
```cpp
} else {
    // Normal mode: compute b_t on-the-fly
    delta_B_u = delta_vals[r][i] * B_val[r] * float(u_vals[r][i]);
    delta_B_u = params.alpha * delta_B_u;  // = alpha * delta * B * u
}
```

**You apply B here!** ✅ Correct for momentum formulation.

### But then (Lines 371-375):
```cpp
const weight_t C_val = !kIsVariableC
    ? BC_val[r]  // When B constant, this is B*C
    : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
out_vals[r][i] += thread_data[i].y * C_val;
```

**You multiply by B*C again!** ❌ **DOUBLE B APPLICATION!**

---

## The Problem

**Original Mamba:** Delays B application to output (stores B*C)
- Scan input: `delta*u` (no B)
- Output: `h * (B*C)` ✅ Correct

**Your Muon (constant B case):** Applies B twice!
- Scan input: `alpha * delta * B * u` (B applied)
- Output: `h * (B*C)` (B applied again) ❌ Wrong!

**Result:** `y = D*u + (B*C) * h` where h already contains B factor
- Effective: `y = D*u + B²*C * (accumulated terms)` ❌ **Extra B factor!**

---

## ✅ THE FIX

You need to change how BC_val is used when B is constant:

### Option 1: Store C instead of B*C (Recommended)
```cpp
if constexpr (!kIsVariableB && !kIsVariableC) {
    for (int r = 0; r < kNRows; ++r) {
        B_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];  // Just C!
    }
}
if constexpr (!kIsVariableB && kIsVariableC) {
    for (int r = 0; r < kNRows; ++r) {
        B_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
        // BC_val not needed for this case
    }
}
```

### Option 2: Check if momentum mode and adjust
```cpp
// In output computation:
const weight_t C_val = !kIsVariableC
    ? (params.use_newton_schulz || params.beta != 1.0 ? 
       C[state_idx * params.C_dstate_stride + r * params.C_d_stride] :  // Just C for momentum
       BC_val[r])  // B*C for original Mamba
    : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
```

**Option 1 is cleaner since you always apply B early in momentum mode.**

---

## Summary Table

| Mode | B Applied When | H_t Contains | Output Should Use | Current Bug |
|------|----------------|--------------|-------------------|-------------|
| Original Mamba (B const) | At output | delta*u | B*C | ✅ Correct |
| Original Mamba (B var) | In scan | B*delta*u | C | ✅ Correct |
| **Your Muon (B const, NS)** | **In NS kernel** | **alpha*delta*B*u** | **C only** | **❌ Uses B*C** |
| **Your Muon (B const, normal)** | **Line 256** | **alpha*delta*B*u** | **C only** | **❌ Uses B*C** |
| Your Muon (B var, NS) | In NS kernel | alpha*delta*B*u | C | ✅ Correct |
| Your Muon (B var, normal) | Line 255 | alpha*delta*B*u | C | ✅ Correct |

---

## ✅ Final Confirmation

**Original Mamba behavior with BC_val:** ✅ **CORRECT** 
- Intentional optimization to delay B application
- Comment explicitly states this (lines 181-183)

**Your Muon behavior with BC_val:** ❌ **BUG WHEN B CONSTANT**
- You apply B early (in equation 1)
- But still use B*C at output (inherited from Mamba)
- Results in double B application

**Fix:** When B is constant and momentum is enabled, store C (not B*C) in BC_val.

---

## Recommended Action

Add this fix to your `selective_scan_fwd_kernel.cuh`:

```cpp
// Around line 208-212
if constexpr (!kIsVariableB && !kIsVariableC) {
    #pragma unroll
    for (int r = 0; r < kNRows; ++r) {
        if (params.use_newton_schulz || params.beta != 1.0) {
            // Momentum mode: B already applied in b_t computation
            B_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
            BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];  // Just C
        } else {
            // Original Mamba mode: B applied at output
            B_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
            BC_val[r] = B_val[r] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];  // B*C
        }
    }
}
```

This maintains backward compatibility with original Mamba while fixing the momentum mode.

---

**Date:** 2025-11-01  
**Status:** ✅ Original Mamba verified correct, ❌ Muon momentum has bug with constant B  
**Severity:** HIGH (causes incorrect outputs when B is constant)  
**Fix:** Simple - store C instead of B*C when momentum enabled






