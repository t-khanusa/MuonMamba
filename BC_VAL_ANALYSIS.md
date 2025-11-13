# BC_val Analysis - Is there a bug on line 212?

## The Question

Line 212 in `selective_scan_fwd_kernel.cuh`:
```cpp
BC_val[r] = B_val[r] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
```

When both B and C are constant, this stores `B * C` in `BC_val`.

Later, line 375 uses:
```cpp
out_vals[r][i] += thread_data[i].y * C_val;  // where C_val = BC_val[r] in this case
```

This results in: `output += h_t * (B * C)`

**Is this correct, or should it be `output += h_t * C`?**

---

## Complete Case Analysis

### Case 1: Both B and C Constant (!kIsVariableB && !kIsVariableC)

**Line 211-212:**
```cpp
B_val[r] = B[...];
BC_val[r] = B_val[r] * C[...];  // BC_val = B * C
```

**Line 254 (b_t computation):**
```cpp
delta_B_u = delta_vals[r][i] * B_val[r] * float(u_vals[r][i]);  // Uses B_val
```

**Line 372-373 (C_val computation):**
```cpp
const weight_t C_val = BC_val[r];  // C_val = B * C
```

**Line 375 (output):**
```cpp
out_vals[r][i] += thread_data[i].y * C_val;  // output += h_t * (B*C)
```

**Mathematical result:**
```
b_t = alpha * delta * B * u
h_t = exp(delta*A) * h_{t-1} + beta * v_{t-1} + b_t
output = D*u + (B*C) * h_t
```

**Expected (standard SSM):**
```
output = D*u + C * h_t
```

**❌ LOOKS WRONG - Extra B factor!**

---

### Case 2: B Constant, C Variable (!kIsVariableB && kIsVariableC)

**Line 204:**
```cpp
BC_val[r] = B[...];  // BC_val = B (just B, not B*C)
```

**Line 218:**
```cpp
B_val[r] = B[...];  // B_val = B
```

**Line 254 (b_t computation):**
```cpp
delta_B_u = delta_vals[r][i] * B_val[r] * float(u_vals[r][i]);  // Uses B_val
```

**Line 373 (C_val computation):**
```cpp
const weight_t C_val = BC_val[r] * C_vals[i];  // C_val = B * C_t
```

**Line 375 (output):**
```cpp
out_vals[r][i] += thread_data[i].y * C_val;  // output += h_t * (B * C_t)
```

**Mathematical result:**
```
b_t = alpha * delta * B * u
output = D*u + (B * C_t) * h_t
```

**❌ ALSO WRONG - Extra B factor!**

---

### Case 3: B Variable, C Constant (kIsVariableB && !kIsVariableC)

**Line 193:**
```cpp
BC_val[r] = C[...];  // BC_val = C (just C, not B*C)
```

**Line 254-255 (b_t computation):**
```cpp
delta_B_u = delta_vals[r][i] * B_vals[i] * float(u_vals[r][i]);  // Uses B_vals[i]
```

**Line 372 (C_val computation):**
```cpp
const weight_t C_val = BC_val[r];  // C_val = C
```

**Line 375 (output):**
```cpp
out_vals[r][i] += thread_data[i].y * C_val;  // output += h_t * C
```

**Mathematical result:**
```
b_t = alpha * delta * B_t * u
output = D*u + C * h_t
```

**✅ CORRECT!**

---

### Case 4: Both B and C Variable (kIsVariableB && kIsVariableC)

**BC_val not used in this case.**

**Line 254-255 (b_t computation):**
```cpp
delta_B_u = delta_vals[r][i] * B_vals[i] * float(u_vals[r][i]);  // Uses B_vals[i]
```

**Line 373 (C_val computation):**
```cpp
const weight_t C_val = C_vals[i];  // C_val = C_t
```

**Line 375 (output):**
```cpp
out_vals[r][i] += thread_data[i].y * C_val;  // output += h_t * C_t
```

**Mathematical result:**
```
b_t = alpha * delta * B_t * u
output = D*u + C_t * h_t
```

**✅ CORRECT!**

---

## Summary Table

| Case | B Type | C Type | b_t uses | C_val = | Output | Correct? |
|------|--------|--------|----------|---------|--------|----------|
| 1 | Const | Const | B_val | B*C | h*(B*C) | ❌ |
| 2 | Const | Var | B_val | B*C_t | h*(B*C_t) | ❌ |
| 3 | Var | Const | B_vals[i] | C | h*C | ✅ |
| 4 | Var | Var | B_vals[i] | C_t | h*C_t | ✅ |

---

## 🚨 CRITICAL BUG CONFIRMED

**Cases 1 and 2 are INCORRECT!**

When B is constant, the code multiplies the output by B, resulting in:
- Case 1: `output = D*u + (B*C)*h` instead of `D*u + C*h`
- Case 2: `output = D*u + (B*C_t)*h` instead of `D*u + C_t*h`

---

## Why Wasn't This Caught in Testing?

**Possible reasons:**

1. **Tests use variable B:** If all tests use `kIsVariableB=true`, Cases 3 and 4 would be tested (which are correct), but not Cases 1 and 2.

2. **B ≈ 1:** If B values are close to 1, the bug's effect would be minimal and might not be noticed.

3. **No explicit validation:** If tests don't compare against known-correct reference implementations with constant B.

4. **Bug exists in reference too:** If the PyTorch reference implementation has the same bug, tests comparing CUDA vs PyTorch would pass despite both being wrong.

---

## How to Fix

### Option 1: Store C instead of B*C (Recommended)
```cpp
if constexpr (!kIsVariableB && !kIsVariableC) {
    for (int r = 0; r < kNRows; ++r) {
        B_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];  // Just C
    }
}
if constexpr (!kIsVariableB && kIsVariableC) {
    for (int r = 0; r < kNRows; ++r) {
        B_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
        // BC_val not needed, will use C_vals[i] directly
    }
}
```

### Option 2: Adjust the output computation
```cpp
const weight_t C_val = !kIsVariableC
    ? BC_val[r] / B_val[r]  // Divide out the B factor
    : (!kIsVariableB ? C_vals[i] : C_vals[i]);
```

**Option 1 is cleaner and avoids division.**

---

## Wait - Let Me Check the Original Mamba Code

Before declaring this a bug, I should verify against the original Mamba implementation to see if this is intentional or if I'm misunderstanding the formulation...

Actually, looking at the comment on lines 181-183:
```cpp
// This variable holds B * C if both B and C are constant across seqlen. If only B varies
// across seqlen, this holds C. If only C varies across seqlen, this holds B.
// If both B and C vary, this is unused.
```

This comment explicitly states that `BC_val` holds `B * C` when both are constant. So this is **intentional**, not an accident.

**This means either:**
1. The formulation is non-standard (uses B*C in output for optimization?)
2. There's a fundamental misunderstanding of the SSM equations
3. This is a known bug that hasn't been fixed

Let me check if there's any rescaling happening elsewhere...







