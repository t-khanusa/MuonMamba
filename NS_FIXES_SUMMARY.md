# Newton-Schulz 5-Step Implementation - Fixes Applied

## Date: 2025-10-30

## Critical Bugs Fixed

### 1. ✅ Gram Matrix Indexing Bug
**Problem**: Used `D` instead of `gram_size` when indexing `gram_A_then_B`
- When `transposed==true`, `gram_size=dstate` but code used `D` width
- Caused out-of-bounds reads/writes, corrupting Gram matrix

**Fix**: Changed all `gram_A_then_B[i * D + j]` to `gram_A_then_B[i * gram_size + j]`

### 2. ✅ Variable B Group ID Calculation
**Problem**: Integer truncation and potential division by zero
- `group_id = global_row / (D / n_groups)` truncates if `D % n_groups != 0`
- Divides by zero if `n_groups > D`

**Fix**: 
```cuda
int group_size = (D + n_groups - 1) / n_groups;  // Ceiling division
int group_id = min(global_row / group_size, n_groups - 1);  // Cap at n_groups-1
```

### 3. ✅ Constant B Indexing
**Problem**: Ambiguous stride usage
**Fix**: Added comments clarifying `B[global_row * B_d_stride + col * B_dstate_stride]` assumes row-major layout

### 4. ✅ Tile Buffer Overflow for A²
**Problem**: `tile_buffer` size may be smaller than `gram_size*gram_size`
**Fix**: Added size check before using tile_buffer to store A²:
```cuda
const int tile_buffer_size = kTileSize * (transposed ? D : dstate);
if (gram_size * gram_size <= tile_buffer_size) {
    // Use tile_buffer
} else {
    // Use fallback
}
```

### 5. ✅ BFloat16 Conversion Optimization
**Problem**: Excessive round-trips inside inner loops caused numerical error and performance overhead
**Fix**: Convert once on load, accumulate in FP32:
```cuda
// Old (bad):
for (k) {
    __nv_bfloat16 a = __float2bfloat16(x);
    float prod = __bfloat162float(a) * ...;
    sum += __bfloat162float(__float2bfloat16(prod));  // Extra round
}

// New (good):
for (k) {
    float a = __bfloat162float(__float2bfloat16(x));  // Convert once
    sum += a * b;  // Accumulate in FP32
}
```

## Current Status

### ❌ Still Failing
Test with realistic dimensions (batch=2, dim=32, seqlen=8, dstate=16) shows:
- **CUDA output**: Contains `-inf` and very large values (40960.0)
- **Reference output**: Bounded values (-1.14 to 0.81)
- This indicates numerical instability somewhere

### Possible Remaining Issues

1. **NS Output Verification Needed**
   - Need to check if NS kernel itself produces inf/nan
   - Or if the scan diverges due to NS output

2. **Shared Memory Layout**
   - Verify no conflicts between tile_buffer and gram_A_then_B
   - Check if gram_size calculation is correct for both cases

3. **BFloat16 Precision**
   - While optimized, may still cause accumulation errors
   - Consider if certain operations need FP32 throughout

4. **Scan Integration**
   - Buffer indexing between NS write and scan read
   - Verify scan isn't reading stale/wrong data

## Testing Configuration

Comprehensive test uses:
- `batch=2, dim=32, seqlen=8, dstate=16`
- `delta_softplus=True` (realistic Mamba setting)
- `beta=0.9, alpha=1.0` (momentum enabled)

## Next Steps

1. Add targeted debug to isolate where inf values appear
2. Verify NS output independently before scan
3. Check if issue is in fat matrix (D<N) or tall matrix (D>N) case
4. Consider if normalization step needs adjustment
















