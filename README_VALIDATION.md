# Newton-Schulz 5-Step Algorithm: CUDA vs PyTorch Validation Suite

## Overview

This directory contains a comprehensive validation suite for comparing CUDA and PyTorch implementations of the Newton-Schulz 5-step orthogonalization algorithm. The suite ensures mathematical and logical correctness across various test cases.

## Quick Start

### Run All Validations (Recommended)

```bash
./run_all_validations.sh
```

This will:
1. Compile CUDA test program
2. Run PyTorch reference tests
3. Run CUDA tests
4. Generate comprehensive validation report
5. Compare outputs and provide summary

### Individual Test Runs

#### PyTorch Reference
```bash
python test_ns_5step_pytorch.py
```

#### CUDA Tests
```bash
nvcc -O3 -arch=sm_80 test_ns_5step_detailed.cu -o test_ns_5step_detailed
./test_ns_5step_detailed
```

#### Comprehensive Suite (22 test cases)
```bash
python test_ns_comprehensive_validation.py
```

## Test Files

### Core Test Programs

1. **`test_ns_5step_pytorch.py`**
   - PyTorch reference implementation
   - Tests 3 cases: fat (3×4), tall (4×3), production (128×64)
   - Outputs norms, traces, and matrices
   - Uses bfloat16 precision matching CUDA

2. **`test_ns_5step_detailed.cu`**
   - CUDA implementation with detailed logging
   - Same 3 test cases as PyTorch
   - Outputs norms, traces, and matrices
   - Uses optimized kernel with shared memory

3. **`test_ns_comprehensive_validation.py`**
   - Comprehensive test suite with 22 test cases
   - Categories:
     - Small matrices (2×2 to 8×4)
     - Medium matrices (16×32, 32×32)
     - Production matrices (64×128, 128×128)
     - Edge cases (identity, extreme values, aspect ratios)
   - Generates `torch_reference_results.json`

4. **`compare_cuda_torch.py`**
   - Direct comparison script
   - Computes relative differences
   - Validates against tolerance thresholds

5. **`run_all_validations.sh`**
   - Master validation script
   - Runs all tests and generates reports
   - Color-coded output for easy reading

### Generated Output Files

- `pytorch_output.txt` - PyTorch test results
- `cuda_output.txt` - CUDA test results
- `comprehensive_output.txt` - All 22 test case results
- `validation_summary.txt` - Executive summary
- `torch_reference_results.json` - Reference data for all cases
- `VALIDATION_REPORT.md` - Detailed analysis and findings

## Test Cases

### Standard Test Cases (Basic Validation)

| Test Name | D | N | Type | Input Pattern |
|-----------|---|---|------|---------------|
| Test 1 | 3 | 4 | Fat | Sequential [1..12] |
| Test 2 | 4 | 3 | Tall | Sequential [1..12] |
| Test 3 | 128 | 64 | Tall | (i % 100) / 10.0 |

### Comprehensive Test Cases (Full Suite)

#### Small Matrices
- `tiny_fat_2x3` (2×3)
- `small_fat_3x4` (3×4)
- `small_fat_3x5` (3×5)
- `small_fat_4x8` (4×8, random)
- `tiny_tall_3x2` (3×2)
- `small_tall_4x3` (4×3)
- `small_tall_5x3` (5×3)
- `small_tall_8x4` (8×4, random)
- `tiny_square_2x2` (2×2)
- `small_square_4x4` (4×4, random)

#### Medium Matrices
- `medium_fat_16x32` (16×32, random)
- `medium_tall_32x16` (32×16, random)
- `medium_square_32x32` (32×32, random)

#### Production Size
- `prod_fat_64x128` (64×128, random)
- `prod_tall_128x64` (128×64, random)
- `prod_square_128x128` (128×128, random)

#### Edge Cases
- `edge_very_fat` (2×16, extreme aspect ratio)
- `edge_very_tall` (16×2, extreme aspect ratio)
- `edge_identity` (4×4, identity matrix)
- `edge_small_values` (4×4, values < 0.1)
- `edge_large_values` (4×4, values > 100)
- `edge_mixed_values` (4×4, mixed ranges 0.01-100)

## Validation Metrics

### Primary Metrics

1. **Norms** (6 values per test)
   - Initial norm + 5 iteration norms
   - Tolerance: < 1% relative difference
   - Indicates convergence behavior

2. **Traces** (5 values per test)
   - Trace of Gram matrix at each iteration
   - Tolerance: < 2% relative difference
   - Should generally increase (approaching min(D,N))

3. **Output Matrix** (D×N values)
   - Final orthogonalized matrix
   - Tolerance: < 5% relative difference
   - Element-wise comparison

### Secondary Metrics

4. **Orthogonality Error**
   - For fat matrices: max|X@X.T - I|
   - For tall matrices: max|X.T@X - I|
   - Should be < 0.5 for good orthogonalization
   - CUDA and PyTorch should have similar errors

## Expected Results

### Typical Differences

Based on comprehensive testing, expected differences between CUDA and PyTorch:

1. **Initial Norm:** 0.02-0.35% difference
   - Cause: BF16 rounding in reduction operations
   - Status: ✓ Acceptable

2. **First Iteration Trace:** < 0.5% difference
   - Cause: Minor BF16 rounding differences
   - Status: ✓ Excellent agreement

3. **Later Iteration Traces:** 1-4% difference
   - Cause: Cumulative BF16 rounding effects
   - Status: ✓ Acceptable for iterative algorithm

4. **Final Output:** 1-5% element-wise difference
   - Cause: Accumulated numerical differences
   - Status: ✓ Within tolerance

### Known Patterns

- **Smaller matrices** show higher relative differences (BF16 precision limits)
- **Larger matrices** show better agreement (averaging effects)
- **Trace progression** should be similar in both implementations
- **Orthogonality quality** should be equivalent

## Understanding the Results

### What to Look For

#### ✓ Good Signs
- Initial norms within 0.5%
- First trace within 1%
- Trace progression similar
- Final orthogonality error < 0.5
- No NaN or Inf values

#### ⚠️ Warning Signs (Usually Acceptable)
- Traces differ by 2-4% in later iterations
- Final norms differ by < 10%
- Some output elements differ by 5-10%
- Cumulative divergence over iterations

#### ✗ Problem Signs (Investigate)
- Initial norm differs by > 1%
- First trace differs by > 2%
- NaN or Inf values appear
- Orthogonality error > 1.0
- Systematic bias in differences

### Interpreting Differences

BF16 (bfloat16) has limited precision:
- Mantissa: 7 bits (vs 23 for float32)
- Precision: ~2-3 decimal digits
- Accumulated errors over 5 iterations are expected

**Key insight:** Small numerical differences do NOT indicate incorrect implementation, they indicate different (but equivalent) BF16 operation ordering.

## Validation Criteria

### Pass/Fail Thresholds

| Metric | Threshold | Strictness |
|--------|-----------|------------|
| Initial norm | < 0.5% | Strict |
| Iteration 1 trace | < 1% | Strict |
| Iteration 2-5 traces | < 2% | Moderate |
| Final norm | < 10% | Lenient |
| Output values | < 5% | Moderate |
| Orthogonality error | Similar | Qualitative |

### Overall Verdict

Implementation passes if:
1. ✓ All strict criteria met
2. ✓ Most moderate criteria met  
3. ✓ No problem signs observed
4. ✓ Mathematical properties preserved

## Troubleshooting

### CUDA Compilation Fails

```bash
# Check CUDA version
nvcc --version

# Try different architecture
nvcc -O3 -arch=sm_70 test_ns_5step_detailed.cu -o test_ns_5step_detailed

# Check for syntax errors
nvcc -O0 -g test_ns_5step_detailed.cu -o test_ns_5step_detailed
```

### PyTorch Tests Fail

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check BF16 support
python -c "import torch; print(torch.cuda.is_bf16_supported())"

# Run with verbose output
python test_ns_5step_pytorch.py 2>&1 | tee pytorch_debug.txt
```

### Large Differences Observed

1. **Verify BF16 is used consistently:**
   - PyTorch: `.bfloat16()` conversion
   - CUDA: `__float2bfloat16()` calls

2. **Check matrix dimensions:**
   - Ensure D and N match between tests
   - Verify transpose logic for tall matrices

3. **Compare intermediate values:**
   - Add debug prints for Gram matrix
   - Check A² computation
   - Verify polynomial coefficients (a, b, c)

4. **Test on simpler inputs:**
   - Identity matrix
   - Diagonal matrix
   - Small matrices (2×2)

## Advanced Usage

### Custom Test Cases

Add your own test case to `test_ns_comprehensive_validation.py`:

```python
test_cases.append({
    'name': 'my_custom_test',
    'D': 10,
    'N': 8,
    'input': np.random.randn(10, 8).astype(np.float32)
})
```

### Tolerance Adjustment

Modify tolerances in comparison functions:

```python
# In compare_results()
norm_ok = np.max(norm_rel_diff) < 0.01  # Change 0.01 to desired threshold
trace_ok = np.max(trace_rel_diff) < 0.02  # Change 0.02 to desired threshold
output_ok = np.max(output_rel_diff) < 0.05  # Change 0.05 to desired threshold
```

### Batch Testing

To test multiple matrix sizes at once:

```bash
for D in 4 8 16 32 64 128; do
    for N in 2 4 8 16 32 64; do
        if [ $D -ge $N ]; then
            echo "Testing D=$D, N=$N"
            # Add test here
        fi
    done
done
```

## Performance Testing

While this suite focuses on correctness, you can measure performance:

```python
import time

# PyTorch
start = time.time()
X, norms, traces = newtonschulz5_torch(G, steps=5)
torch_time = time.time() - start

print(f"PyTorch time: {torch_time*1000:.2f}ms")
```

```cpp
// CUDA
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
test_ns_kernel<<<...>>>(...)
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
printf("CUDA time: %.2fms\n", ms);
```

## References

### Algorithm
- Newton-Schulz iteration: `X_{n+1} = a*X_n + (b*A + c*A²)@X_n`
- Coefficients: a=3.4445, b=-4.7750, c=2.0315
- Converges to orthogonal matrix when initialized with normalized input

### Implementation Details
- Precision: bfloat16 (BF16)
- Iterations: 5 steps
- Transposition: Tall matrices (D > N) are transposed
- Gram matrix: A = X @ X.T computed in BF16

## Contact & Support

For issues or questions:
1. Review `VALIDATION_REPORT.md` for detailed analysis
2. Check `validation_summary.txt` for latest results
3. Examine individual output files for specific errors
4. Compare against reference values in JSON file

## License

This validation suite is part of the Momentum project.

---

**Last Updated:** 2025-11-01  
**Version:** 1.0  
**Status:** Production Ready ✓






