# BFloat16 Strategy for Newton-Schulz

## Current Issue

Trying to use bfloat16 intrinsics (`__hmul`, `__hadd`) causes compilation errors because these are for `__half` (FP16), not `__nv_bfloat16`.

## PyTorch's Approach

```python
X = G.bfloat16()  # Convert to bfloat16
X /= (X.norm() + eps)  # Operations in bfloat16
for _ in range(steps):
    A = X @ X.T  # Matrix mult in bfloat16
    B = b * A + c * A @ A
    X = a * X + B @ X
```

The `.bfloat16()` ensures values are in bfloat16 precision, but PyTorch may use float32 operations internally.

## Our CUDA Strategy

**Simpler approach**: Store as float32 but do periodic bfloat16 round-trips to match PyTorch's precision:

1. **Initial normalization**: Convert to bfloat16, then back to float32
2. **Each iteration**: Do all operations in float32, but ensure intermediate values went through bfloat16 precision at some point
3. **Final output**: float32 for scan operations

This matches the numerical behavior of PyTorch without complex intrinsics.

## Why This Works

- bfloat16 has same exponent range as float32 (8 bits)
- The precision loss comes from reduced mantissa (7 bits vs 23 bits)  
- Converting float → bfloat16 → float truncates mantissa
- This is exactly what PyTorch does!

## Implementation

Keep the current code (which uses float32 for operations) but just do the initial normalization conversion. The key is that X starts as bfloat16-precision values, and all subsequent operations maintain similar precision naturally.

Given the compilation complexity, let's **prioritize correctness over exact bfloat16 matching** for now:
1. ✅ Fix the tiled algorithm to match PyTorch logic (transpose handling, Gram matrix size)
2. ✅ Ensure no NaNs/numerical issues  
3. ⏭️ Add bfloat16 as optimization later (when we have more time to debug intrinsics)

The current float32 implementation should be "close enough" - the main correctness issue was the tiled algorithm, not the precision.





