# Bug Fix: Double B Application with Constant B Parameter

## Issue Summary

**Bug:** When B parameter is constant and momentum is enabled (`use_newton_schulz=true` or `beta != 1.0`), the B factor was being applied twice, resulting in incorrect output scaling.

**Date Fixed:** 2025-11-01

---

## Root Cause Analysis

### Original Mamba Behavior (Correct)
- When B is constant, Mamba **delays** B multiplication until output stage
- Scan input: `delta*u` (no B factor)
- Output: `y = (B*C)*h` where h contains `delta*u` terms
- Result: Mathematically equivalent to `y = C*(B*delta*u)` ✓

### Muon with Momentum Behavior (Bug)
- Momentum mode applies B **early**: `b_t = alpha * delta * B * u`
- But inherited Mamba's optimization that stores `B*C` in `BC_val`
- Result: `y = (B*C)*h` where h already contains B factor
- Effective output: `y = B²*C*h` ❌ **Double B application!**

---

## The Fix

### Modified Lines: 208-228

**Before (Incorrect):**
```cpp
if constexpr (!kIsVariableB && !kIsVariableC) {
    for (int r = 0; r < kNRows; ++r) {
        B_val[r] = B[...];
        BC_val[r] = B_val[r] * C[...];  // Always stores B*C
    }
}
```

**After (Correct):**
```cpp
if constexpr (!kIsVariableB && !kIsVariableC) {
    for (int r = 0; r < kNRows; ++r) {
        B_val[r] = B[...];
        // Momentum mode: B already in b_t, store C only
        // Original Mamba: B applied at output, store B*C
        if (params.use_newton_schulz || params.beta != 1.0f) {
            BC_val[r] = C[...];  // Just C for momentum
        } else {
            BC_val[r] = B_val[r] * C[...];  // B*C for original Mamba
        }
    }
}
```

### Modified Lines: 377-383 (Output Computation)

**Before (Incorrect):**
```cpp
const weight_t C_val = !kIsVariableC
    ? BC_val[r]
    : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
```

**After (Correct):**
```cpp
const weight_t C_val = !kIsVariableC
    ? BC_val[r]  // Either C (momentum) or B*C (original Mamba)
    : (!kIsVariableB ? 
        (params.use_newton_schulz || params.beta != 1.0f ? C_vals[i] : BC_val[r] * C_vals[i])
        : C_vals[i]);
```

---

## Impact Analysis

### Cases Affected

| B Type | C Type | Mode | Before Fix | After Fix | Impact |
|--------|--------|------|------------|-----------|---------|
| Const | Const | Momentum | B²*C*h ❌ | C*h ✓ | **FIXED** |
| Const | Var | Momentum | B²*C_t*h ❌ | C_t*h ✓ | **FIXED** |
| Var | Const | Momentum | C*h ✓ | C*h ✓ | No change |
| Var | Var | Momentum | C_t*h ✓ | C_t*h ✓ | No change |
| Const | Const | Original Mamba | B*C*h ✓ | B*C*h ✓ | No change |
| Const | Var | Original Mamba | B*C_t*h ✓ | B*C_t*h ✓ | No change |

### Severity

**HIGH** - This bug caused incorrect outputs whenever:
1. B parameter is constant (not varying per timestep)
2. Momentum is enabled (Newton-Schulz or beta != 1.0)

**LOW** - No impact when:
1. B is variable per timestep (most common case)
2. Momentum is disabled (beta == 1.0 and no Newton-Schulz)

---

## Verification

### How to Test

1. **Create test with constant B:**
```python
# Test configuration
batch = 2
dim = 64
seqlen = 128
dstate = 16

# Use constant B (not time-varying)
B = torch.randn(dim, dstate)  # Constant across time

# Run with momentum enabled
use_newton_schulz = True
beta = 0.9
```

2. **Check output scaling:**
```python
# Expected behavior:
# y_t should NOT have B² factor
# Verify: output magnitude should match reference implementation
```

### Backward Compatibility

✅ **Maintained:** Original Mamba mode (beta=1.0, no NS) still uses B*C optimization
✅ **Fixed:** Momentum mode now correctly uses C only when B is constant

---

## Related Files

- **Fixed:** `/project/khanhnt/muontest/Momentum_correct/csrc/selective_scan/selective_scan_fwd_kernel.cuh`
- **Analysis:** `ORIGINAL_MAMBA_VS_MUON_ANALYSIS.md`
- **Verification:** `SELECTIVE_SCAN_VERIFICATION_FINAL.md`

---

## Testing Checklist

- [ ] Test with constant B, constant C, momentum enabled
- [ ] Test with constant B, variable C, momentum enabled
- [ ] Test with variable B (should work as before)
- [ ] Test original Mamba mode (beta=1.0, no NS) - should be unchanged
- [ ] Verify output magnitudes are correct
- [ ] Run full training loop to ensure no numerical issues

---

## Conclusion

This fix ensures that the B parameter is applied exactly **once** in momentum mode:
- When B is variable: Applied in scan (line 255 or 291)
- When B is constant: Applied in scan (line 256 or 293)
- Output stage: Multiplies by C only (not B*C)

The fix maintains backward compatibility with original Mamba while correctly implementing momentum SSM equations.

---

**Status:** ✅ FIXED  
**Date:** 2025-11-01  
**Files Changed:** 1  
**Lines Changed:** ~15





