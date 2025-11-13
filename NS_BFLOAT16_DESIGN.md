# Newton-Schulz bfloat16 Design

## Key Insight from Official Paper

> "Newton-Schulz iterations CAN BE STABLY run in bfloat16"

This is a critical design choice for numerical stability in orthogonalization.

## Mixed Precision Strategy

### Newton-Schulz Operations (bfloat16)

**Why bfloat16?**
- Wider dynamic range than fp16 (8-bit exponent like fp32)
- NS involves matrix multiplications that benefit from bf16's range
- Official implementation explicitly uses `.bfloat16()`
- Iterative refinement is stable in bf16

**What runs in bf16:**
```cuda
// 1. Normalization
X = X / norm  // bf16 division

// 2-6. Five NS iterations
for (int step = 0; step < 5; ++step) {
    A = X @ X.T      // bf16 matmul
    A2 = A @ A       // bf16 matmul
    B = b*A + c*A2   // bf16 arithmetic
    X = a*X + B@X    // bf16 matmul + addition
}
```

### Scan Operations (float32)

**Why float32?**
- Long sequence accumulation (L=512 to 8192)
- Momentum beta near 1.0 compounds errors exponentially
- State persistence across entire sequence

**What runs in fp32:**
```cuda
// Velocity scan
v_t = beta * v_{t-1} + b_t_ortho  // fp32 accumulation

// Hidden state scan  
h_t = exp(delta*A) * h_{t-1} + v_t  // fp32 accumulation

// Output
y_t = C * h_t + D * u  // fp32 computation
```

## Implementation Flow

```
Input: u (fp16), delta (fp16), B (fp32), A (fp32), C (fp32)
       ↓
1. Compute b_t = alpha × delta × B × u
   → fp32 (promote from fp16 for accuracy)
       ↓
2. Newton-Schulz Orthogonalization
   a) Convert b_t to bfloat16
   b) Normalize: X = b_t / ||b_t||  (bf16)
   c) 5 iterations in bfloat16
   d) Convert result back to float32
   → Orthogonalized b_t in fp32
       ↓
3. Scan Operations
   v_t = beta × v_{t-1} + b_t_ortho  (fp32)
   h_t = A_t × h_{t-1} + v_t         (fp32)
   y_t = C_t × h_t + D_t × u_t       (fp32)
   → Output in original input dtype (fp16)
```

## CUDA Implementation

### Shared Memory Layout

```cuda
// All buffers store bfloat16 during NS iterations
__shared__ __nv_bfloat16 tile_buffer_bf16[kTileSize * dstate];
__shared__ float gram_A_then_B[dstate * dstate];  // Can stay fp32 for accumulation
__shared__ float partial_sums[kBlockSize];
```

### Conversion Points

```cuda
// STEP 0: Compute b_t in fp32, convert to bf16 for NS
float b_t_val = alpha * delta_val * B_val * u_val;  // fp32
__nv_bfloat16 b_t_bf16 = __float2bfloat16(b_t_val); // Convert to bf16

// STEPS 1-5: NS iterations in bf16
// ... (all operations use bf16)

// FINAL: Convert back to fp32 for scan
float x_ortho = __bfloat162float(x_bf16);
velocity_ortho[idx] = x_ortho;  // Store as fp32 for scan
```

## Performance Considerations

### Benefits of bfloat16
- **2× memory bandwidth**: Reading/writing bf16 vs fp32
- **Faster Tensor Core operations**: NVIDIA GPUs optimize for bf16 matmul
- **Numerical stability**: Wider range than fp16 prevents overflow/underflow

### Cost of Conversions
- `__float2bfloat16`: ~1 cycle (very cheap)
- `__bfloat162float`: ~1 cycle (very cheap)
- Total conversion overhead: negligible (<0.1ms)

### Memory Traffic Comparison

**With fp32 NS**:
- Read/write 4 bytes per element
- 5 iterations × 16 MB = 80 MB read + 80 MB write = 160 MB

**With bf16 NS**:
- Read/write 2 bytes per element  
- 5 iterations × 8 MB = 40 MB read + 40 MB write = 80 MB
- **50% bandwidth reduction!** ✅

## Testing Strategy

### Verify Numerical Stability

```python
# Test that bf16 NS produces same orthogonality as fp32
G_fp32 = torch.randn(128, 64, dtype=torch.float32)
G_bf16 = G_fp32.bfloat16()

ortho_fp32 = newtonschulz5_ref(G_fp32)
ortho_bf16 = newtonschulz5_ref(G_bf16)

# Check orthogonality error is similar
error_fp32 = torch.norm(ortho_fp32.T @ ortho_fp32 - I)
error_bf16 = torch.norm(ortho_bf16.float().T @ ortho_bf16.float() - I)

print(f"FP32 error: {error_fp32:.2e}")
print(f"BF16 error: {error_bf16:.2e}")
# Both should be < 1e-3
```

### Verify Scan Accuracy

```python
# Test that fp32 scan accumulation is accurate over long sequences
# With bf16, errors would compound exponentially
```

## Recommendation: Mixed Precision

**Implement**: 
- ✅ **bfloat16 for Newton-Schulz** (as per paper)
- ✅ **float32 for state accumulation** (v_t, h_t)
- ✅ **Convert at boundaries** (cheap conversions)

**Benefits**:
- Faster NS (2× bandwidth, Tensor Core acceleration)
- Accurate scans (no accumulation errors)
- Matches official implementation philosophy

**Trade-offs**:
- Slightly more complex code (conversion points)
- But: Conversions are nearly free on modern GPUs

## Alternative: Full float32

If mixed precision proves problematic:
- Fallback to fp32 for everything
- Simpler but slower (~1.5× NS time)
- Still correct, just less efficient

## Conclusion

Use **mixed precision**:
- bfloat16 for Newton-Schulz (stable + fast)
- float32 for scans (accurate accumulation)

This matches the official paper's design and provides optimal performance/accuracy trade-off.





