# Selective Scan Forward Kernel: Line-by-Line Verification

## Mathematical Equations to Verify

```
1. b_t = alpha * delta_t * B_t * u_t
2. Newton-Schulz: b_t_ortho = NS_5step(b_t)  [VERIFIED ✅]
3. v_t = beta * v_{t-1} + b_t_ortho
4. h_t = exp(delta_t × A) × h_{t-1} + v_t
5. y_t = C * h_t + D * u_t
```

---

## Part 1: Input Loading and Preprocessing (Lines 138-165)

### Lines 138-151: Load Input Data
```cpp
// Load u and delta for this chunk
for (int r = 0; r < kNRows; ++r) {
    load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, ...);
    load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, ...);
}
u += kChunkSize;
delta += kChunkSize;
```
**✅ Correct:** Loads inputs for current chunk, advances pointers.

### Lines 153-165: Process Delta and Initialize Output
```cpp
for (int r = 0; r < kNRows; ++r) {
    for (int i = 0; i < kNItems; ++i) {
        float u_val = float(u_vals[r][i]);
        
        // Apply delta bias
        delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
        
        // Apply softplus if enabled
        if (params.delta_softplus) {
            delta_vals[r][i] = delta_vals[r][i] <= 20.f ? 
                log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
        }
        
        // Initialize output with D*u term (equation 5)
        out_vals[r][i] = D_val[r] * u_val;  // ✅ y_t = D*u (partial)
    }
}
```
**✅ Correct:** 
- Applies delta bias and softplus transformation
- Initializes output with `D*u` term from equation 5

---

## Part 2: Load A, B, C Parameters (Lines 168-220)

### Lines 169-180: Load A and Prepare for exp(delta×A)
```cpp
for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
    weight_t A_val[kNRows];
    for (int r = 0; r < kNRows; ++r) {
        A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
        
        // Multiply by LOG2E to use exp2f instead of expf
        constexpr float kLog2e = M_LOG2E;
        if constexpr (!kIsComplex) {
            A_val[r] *= kLog2e;  // ✅ Prepare for exp(delta×A)
        } else {
            A_val[r].real_ *= kLog2e;
        }
    }
```
**✅ Correct:** 
- Loads A parameter for each state dimension
- Pre-multiplies by LOG2E for efficient exp2f computation
- This prepares for `exp(delta×A)` in equation 4

### Lines 187-220: Load B and C (Variable or Constant)
```cpp
// Handle different combinations of variable/constant B and C
if constexpr (kIsVariableB) {
    load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals, ...);
    if constexpr (!kIsVariableC) {
        BC_val[r] = C[...];  // Store C if B varies
    }
}
if constexpr (kIsVariableC) {
    load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals, ...);
    if constexpr (!kIsVariableB) {
        BC_val[r] = B[...];  // Store B if C varies
    }
}
if constexpr (!kIsVariableB && !kIsVariableC) {
    B_val[r] = B[...];
    BC_val[r] = B_val[r] * C[...];  // Precompute B*C
}
```
**✅ Correct:** Efficiently handles all combinations of variable/constant B and C.

---

## Part 3: EQUATION 1 - Compute b_t (Lines 229-257)

### Lines 233-250: Newton-Schulz Mode (Load Precomputed b_t_ortho)
```cpp
if (params.use_newton_schulz) {
    // Load from X_4_buffer (orthogonalized b_t from NS kernel)
    float *velocity_ortho_buffer = reinterpret_cast<float *>(params.X_4_buffer_ptr);
    int d = dim_id * kNRows + r;  // Global dim index
    int t = chunk * kChunkSize + threadIdx.x * kNItems + i;  // Global time index
    
    // Bounds check
    if (t < params.seqlen && d < params.dim) {
        int global_idx = batch_id * params.dim * params.seqlen * params.dstate +
                       d * params.seqlen * params.dstate +
                       t * params.dstate +
                       state_idx;
        delta_B_u = velocity_ortho_buffer[global_idx];  // ✅ Load b_t_ortho
    } else {
        delta_B_u = 0.0f;
    }
}
```
**✅ CORRECT:**
- Loads precomputed `b_t_ortho` from NS kernel output
- Indexing matches NS kernel output layout exactly
- Bounds checking prevents out-of-range access
- **Note:** This is `b_t_ortho`, NOT raw `b_t` (already orthogonalized)

### Lines 251-257: Normal Mode (Compute b_t On-the-Fly)
```cpp
} else {
    // Normal mode: compute b_t on-the-fly
    delta_B_u = !kIsVariableB ? 
        delta_vals[r][i] * B_val[r] * float(u_vals[r][i]) : 
        delta_vals[r][i] * B_vals[i] * float(u_vals[r][i]);
    delta_B_u = params.alpha * delta_B_u;  // ✅ b_t = alpha * delta * B * u
}
```
**✅ CORRECT:**
- Computes `b_t = alpha * delta_t * B_t * u_t` (equation 1)
- Handles both constant and variable B
- Applies alpha scaling

**⚠️ IMPORTANT DISTINCTION:**
- **NS mode:** `delta_B_u = b_t_ortho` (alpha already applied in NS kernel)
- **Normal mode:** `delta_B_u = alpha * delta * B * u` (alpha applied here)

This is correct because NS kernel applies alpha internally (line 804 in newton_schulz_fwd_kernel.cuh).

---

## Part 4: EQUATION 3 - Velocity Scan: v_t = beta * v_{t-1} + b_t_ortho (Lines 259-319)

### Lines 259-264: Prepare Velocity Scan Data
```cpp
// Non-complex case
velocity_data[i] = make_float2(params.beta, delta_B_u);
```

**Data Structure:**
```
velocity_data[i] = (a, b) where:
  a = beta      (multiplicative factor)
  b = delta_B_u (additive term, either b_t_ortho or raw b_t)
```

### Lines 299-313: Velocity Scan Operation
```cpp
// Load velocity running prefix from previous chunk
scan_t velocity_running_prefix;
if constexpr (!kIsComplex) {
    velocity_running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? 
        x[(r * params.n_chunks + chunk - 1) * params.dstate * 2 + state_idx * 2] :
        make_float2(1.f, 0.f);  // Initial: (1, 0) means v_0 = 0
}

SSMScanPrefixCallbackOp<weight_t> velocity_prefix_op(velocity_running_prefix);
typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
    velocity_data, velocity_data, SSMScanOp<weight_t>(), velocity_prefix_op
);
```

**SSMScanOp Operation (from selective_scan_common.h line 149):**
```cpp
float2 operator()(const float2 &ab0, const float2 &ab1) const {
    return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
}
```

**Mathematical Meaning:**
```
Given: (a0, b0) and (a1, b1)
Output: (a1*a0, a1*b0 + b1)

For scan with (beta, b_t):
  Result: (beta_accumulated, v_t)
  where v_t = beta * v_{t-1} + b_t
```

**✅ CORRECT:** This implements `v_t = beta * v_{t-1} + b_t_ortho` (equation 3)

### Lines 316-319: Store Velocity State for Next Chunk
```cpp
if (threadIdx.x == 0) {
    x[(r * params.n_chunks + chunk) * params.dstate * 2 + state_idx * 2] = 
        velocity_prefix_op.running_prefix;
}
```
**✅ Correct:** Stores final velocity state in **even index** (`state_idx * 2`) for next chunk.

---

## Part 5: EQUATION 4 - Hidden State Scan: h_t = exp(delta×A) × h_{t-1} + v_t (Lines 323-367)

### Lines 326-345: Prepare Hidden State Scan Data
```cpp
for (int i = 0; i < kNItems; ++i) {
    if constexpr (!kIsComplex) {
        // Use velocity from stage 1 as the input
        thread_data[i] = make_float2(
            exp2f(delta_vals[r][i] * A_val[r]),  // ✅ exp(delta×A)
            velocity_data[i].y                    // ✅ v_t from scan 1
        );
    }
}
```

**Data Structure:**
```
thread_data[i] = (a, b) where:
  a = exp(delta_t × A)  (multiplicative factor)
  b = v_t               (additive term from velocity scan)
```

**✅ CORRECT:**
- `exp2f(delta * A)` computes `exp(delta×A)` efficiently (A already scaled by LOG2E)
- `velocity_data[i].y` is `v_t` from the velocity scan

### Lines 347-361: Hidden State Scan Operation
```cpp
// Load hidden state running prefix from previous chunk
scan_t running_prefix;
if constexpr (!kIsComplex) {
    running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? 
        x[(r * params.n_chunks + chunk - 1) * params.dstate * 2 + state_idx * 2 + 1] :
        make_float2(1.f, 0.f);  // Initial: (1, 0) means h_0 = 0
}

SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
);
```

**SSMScanOp Operation (same as velocity scan):**
```
Given: (exp(delta_{t-1}×A), h_{t-1}) and (exp(delta_t×A), v_t)
Output: (exp_accumulated, h_t)
where h_t = exp(delta_t×A) × h_{t-1} + v_t
```

**✅ CORRECT:** This implements `h_t = exp(delta_t×A) × h_{t-1} + v_t` (equation 4)

### Lines 364-367: Store Hidden State for Next Chunk
```cpp
if (threadIdx.x == 0) {
    x[(r * params.n_chunks + chunk) * params.dstate * 2 + state_idx * 2 + 1] = 
        prefix_op.running_prefix;
}
```
**✅ Correct:** Stores final hidden state in **odd index** (`state_idx * 2 + 1`) for next chunk.

---

## Part 6: EQUATION 5 - Output: y_t = C*h_t + D*u (Lines 369-379)

### Lines 369-379: Compute Final Output
```cpp
for (int i = 0; i < kNItems; ++i) {
    const weight_t C_val = !kIsVariableC
        ? BC_val[r]  // Precomputed B*C or just C (if B varies)
        : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
    
    if constexpr (!kIsComplex) {
        out_vals[r][i] += thread_data[i].y * C_val;  // ✅ y_t = D*u + C*h_t
    } else {
        out_vals[r][i] += (complex_t(thread_data[i].z, thread_data[i].w) * C_val).real_ * 2;
    }
}
```

**Breakdown:**
- `thread_data[i].y` contains `h_t` from the hidden state scan
- `out_vals[r][i]` was initialized with `D*u` (line 163)
- This line adds `C*h_t` to complete the equation

**✅ CORRECT:** Implements `y_t = C*h_t + D*u` (equation 5)

---

## State Storage Layout

The kernel stores two types of running states in the `x` buffer:

```
For each (row, chunk, state_idx):
  x[... + state_idx * 2 + 0] = velocity running state (v_t)
  x[... + state_idx * 2 + 1] = hidden state running state (h_t)
```

**✅ CORRECT:** Interleaved storage with even indices for velocity, odd for hidden state.

---

## Data Flow Summary

```
Input: u, delta, A, B, C, D

Step 1: Preprocess
  delta_vals = delta + bias + softplus(delta)
  out_vals = D * u  (initialize output)

Step 2: Compute/Load b_t
  IF use_newton_schulz:
    b_t = load from NS kernel output (b_t_ortho)
  ELSE:
    b_t = alpha * delta * B * u

Step 3: Velocity Scan (v_t = beta * v_{t-1} + b_t)
  velocity_data = (beta, b_t)
  InclusiveScan(velocity_data) → v_t

Step 4: Hidden State Scan (h_t = exp(delta×A) * h_{t-1} + v_t)
  thread_data = (exp(delta×A), v_t)
  InclusiveScan(thread_data) → h_t

Step 5: Output
  out_vals += C * h_t
  Final: y_t = D*u + C*h_t

Output: y_t stored to out_ptr
```

---

## Critical Verification Points

### ✅ 1. Equation 1: b_t = alpha * delta_t * B_t * u_t
**Lines 251-257 (Normal mode):**
```cpp
delta_B_u = delta_vals[r][i] * B_val[r] * float(u_vals[r][i]);
delta_B_u = params.alpha * delta_B_u;
```
**✅ CORRECT**

**Lines 233-250 (NS mode):**
```cpp
delta_B_u = velocity_ortho_buffer[global_idx];  // b_t_ortho (alpha already applied)
```
**✅ CORRECT** (alpha applied in NS kernel)

### ✅ 2. Equation 2: Newton-Schulz 5-step
**Handled by `newton_schulz_velocity_5step_kernel` (already verified)**
**✅ CORRECT**

### ✅ 3. Equation 3: v_t = beta * v_{t-1} + b_t_ortho
**Lines 259 + 311-313:**
```cpp
velocity_data[i] = make_float2(params.beta, delta_B_u);
typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
    velocity_data, velocity_data, SSMScanOp<weight_t>(), velocity_prefix_op
);
```
**SSMScanOp:** `(a1*a0, a1*b0 + b1)` → `(beta_acc, beta*v_{t-1} + b_t)`
**✅ CORRECT**

### ✅ 4. Equation 4: h_t = exp(delta_t×A) × h_{t-1} + v_t
**Lines 329 + 359-361:**
```cpp
thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]), velocity_data[i].y);
typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
);
```
**SSMScanOp:** `(a1*a0, a1*b0 + b1)` → `(exp_acc, exp(delta×A)*h_{t-1} + v_t)`
**✅ CORRECT**

### ✅ 5. Equation 5: y_t = C*h_t + D*u
**Lines 163 + 375:**
```cpp
out_vals[r][i] = D_val[r] * u_val;           // Initialize with D*u
out_vals[r][i] += thread_data[i].y * C_val;  // Add C*h_t
```
**✅ CORRECT**

---

## Potential Issues Analysis

### Issue 1: Double Alpha Application? ❌ (NOT A PROBLEM)
**Concern:** Is alpha applied twice (once in NS kernel, once in scan)?

**Analysis:**
- **NS mode:** Alpha applied in NS kernel (line 804), NOT in scan
- **Normal mode:** Alpha applied in scan (line 256), NO NS kernel

**Verification:**
```cpp
// NS mode (line 247)
delta_B_u = velocity_ortho_buffer[global_idx];  // No alpha multiplication

// Normal mode (line 256)
delta_B_u = params.alpha * delta_B_u;  // Alpha applied here
```

**✅ CORRECT:** Alpha is applied exactly once in both modes.

### Issue 2: State Storage Overlap? ❌ (NOT A PROBLEM)
**Concern:** Do velocity and hidden state overwrite each other?

**Analysis:**
- Velocity state: `x[... + state_idx * 2 + 0]` (even index)
- Hidden state: `x[... + state_idx * 2 + 1]` (odd index)

**✅ CORRECT:** Interleaved storage prevents overlap.

### Issue 3: Scan Order? ❌ (NOT A PROBLEM)
**Concern:** Are the two scans in the correct order?

**Analysis:**
1. First scan: Velocity (needs b_t)
2. Second scan: Hidden state (needs v_t from first scan)
3. Sync between scans (line 321): `__syncthreads()`

**✅ CORRECT:** Velocity scan completes before hidden state scan starts.

### Issue 4: C Value Computation? ⚠️ (NEEDS CLARIFICATION)

**Analysis:**
```cpp
const weight_t C_val = !kIsVariableC
    ? BC_val[r]  // If C is constant, BC_val may be B*C or just C
    : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
```

**Cases:**
1. **Both constant:** `BC_val[r] = B * C` (precomputed, line 212)
   - Used: `BC_val[r]` → **WRONG! Should multiply by h_t, not B*C*h_t**
   
2. **B constant, C varies:** `BC_val[r] = B` (line 204)
   - Used: `BC_val[r] * C_vals[i]` → B * C → **CORRECT**
   
3. **B varies, C constant:** `BC_val[r] = C` (line 193)
   - Used: `BC_val[r]` → C → **CORRECT**
   
4. **Both vary:** 
   - Used: `C_vals[i]` → C → **CORRECT**

**⚠️ WAIT - RE-ANALYSIS NEEDED:**

Looking at the comment on lines 181-183:
```cpp
// This variable holds B * C if both B and C are constant across seqlen. If only B varies
// across seqlen, this holds C. If only C varies across seqlen, this holds B.
// If both B and C vary, this is unused.
```

And looking at how it's used (line 375):
```cpp
out_vals[r][i] += thread_data[i].y * C_val;
```

Where `thread_data[i].y = h_t`.

**RE-EVALUATION:**
- Case 1 (both constant): `C_val = B*C`, output = `h_t * (B*C)` 
  - **This seems wrong! Should be `C * h_t`, not `B*C*h_t`**
  - Unless... the variable B is used elsewhere?

Let me check if B is used in the computation of b_t when both are constant...

Looking back at lines 251-257 for normal mode:
```cpp
delta_B_u = !kIsVariableB ? 
    delta_vals[r][i] * B_val[r] * float(u_vals[r][i]) :  // Uses B_val
    delta_vals[r][i] * B_vals[i] * float(u_vals[r][i]);
```

**AH! I see it now:**
- When B is constant: `B_val[r]` is used in computing b_t (line 254)
- When C is constant: precomputing `B*C` saves computation in output
- The `B` in the output equation `y = C*h + D*u` is NOT the same B as in `b_t = delta*B*u`

**WAIT - I need to check the original Mamba equations...**

Actually, looking at standard SSM equations:
```
b_t = delta * B * u
h_t = exp(delta*A) * h_{t-1} + b_t
y_t = C * h_t + D * u
```

There's no B in the output equation! So when both B and C are constant, precomputing `B*C` would be wrong.

**🔍 LET ME CHECK THE ORIGINAL MAMBA PAPER/CODE:**

Actually, I realize the variable naming might be confusing. Let me re-read the original equations more carefully...

In standard Mamba, B and C are different parameters:
- B: used in computing the hidden state update (b_t = B*u term)
- C: used in computing the output (y = C*h term)

When the comment says "B * C", it might mean the product is used somewhere specific...

Let me trace through Case 1 more carefully:

**Case 1: Both B and C constant, normal mode (no NS)**
```cpp
// Line 211-212: Precompute B*C
B_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
BC_val[r] = B_val[r] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];

// Line 254: Compute b_t using B_val
delta_B_u = delta_vals[r][i] * B_val[r] * float(u_vals[r][i]);

// Line 371-375: Compute output
const weight_t C_val = BC_val[r];  // = B * C
out_vals[r][i] += thread_data[i].y * C_val;  // = h_t * (B*C)
```

**This IS wrong! It should be `h_t * C`, not `h_t * (B*C)`!**

**❌ CRITICAL BUG FOUND!**

---

## 🚨 CRITICAL BUG IDENTIFIED

**Location:** Lines 371-375

**Issue:** When both B and C are constant, the code uses `BC_val[r] = B * C` as the C value, resulting in output `y = (B*C)*h + D*u` instead of `y = C*h + D*u`.

**Root Cause:** The variable `BC_val` is named to suggest it stores `B*C`, but it's used directly as `C_val` without accounting for the fact that B was already used in computing `b_t`.

**Impact:** Incorrect output scaling by factor of B when both parameters are constant.

**Fix Required:** 
```cpp
// Option 1: Don't precompute B*C, just store C
if constexpr (!kIsVariableB && !kIsVariableC) {
    for (int r = 0; r < kNRows; ++r) {
        B_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];  // Just C, not B*C
    }
}

// Option 2: Use B_val for b_t computation, C separately for output
// (requires additional storage)
```

---

## WAIT - Let me double-check by looking at the original Mamba implementation...

Actually, I should verify this against test results. If there's a bug, tests would fail. Let me reconsider...

**Alternative interpretation:** Maybe in some SSM formulations, the output uses `B*C*h`? Let me think about the dimensions...

- h_t is [dstate] dimensional
- C should map [dstate] → [1] (scalar output per state dim)
- B maps [1] → [dstate] (scalar input to state)

Actually, in continuous-time SSMs:
```
dh/dt = A*h + B*u
y = C*h + D*u
```

When discretized:
```
h_t = (discretized_A) * h_{t-1} + (discretized_B * u)
y_t = C * h_t + D * u
```

So B and C are definitely separate, and output should be `C*h`, not `B*C*h`.

**Unless... wait, let me check if this kernel is actually correct by looking at test results.**







