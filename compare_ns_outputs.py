#!/usr/bin/env python3
"""
Compare CUDA and PyTorch NS outputs to identify differences
"""

import sys

# CUDA outputs from test_ns_5step_detailed
cuda_results = {
    "test1": {
        "D": 3, "N": 4,
        "norms": [25.495098, 0.741700, 1.336678, 1.281155, 1.295264, 1.348625],
        "traces": [1.003418, 0.549805, 1.787109, 1.644531, 1.677734],
        "output_00": -0.703125
    },
    "test2": {
        "D": 4, "N": 3,
        "norms": [25.495098, 0.717038, 1.258388, 1.404842, 1.439126, 1.018379],
        "traces": [1.001953, 0.514648, 1.585938, 1.973633, 2.074219],
        "output_00": -0.427734
    },
    "test3": {
        "D": 128, "N": 64,
        "norms": [518.188171, 1.708395, 3.130939, 4.160120, 3.663187, 3.649228],
        "traces": [0.997925, 2.918213, 9.800781, 17.302734, 13.415039],
        "output_00": -0.066406
    }
}

# PyTorch outputs
pytorch_results = {
    "test1": {
        "D": 3, "N": 4,
        "norms": [25.500000, 0.738281, 1.343750, 1.289062, 1.265625, 1.375000],
        "traces": [1.003418, 0.544434, 1.802734, 1.656250, 1.609375],
        "output_00": -0.7227
    },
    "test2": {
        "D": 4, "N": 3,
        "norms": [25.500000, 0.722656, 1.250000, 1.406250, 1.460938, 1.203125],
        "traces": [1.001953, 0.525391, 1.554688, 1.975586, 2.130859],
        "output_00": -0.3828
    },
    "test3": {
        "D": 128, "N": 64,
        "norms": [520.000000, 1.710938, 3.125000, 4.156250, 3.671875, 3.656250],
        "traces": [0.994324, 2.920898, 9.797852, 17.323242, 13.532227],
        "output_00": -0.0684
    }
}

def compare_test(test_name, cuda, pytorch):
    print(f"\n{'='*80}")
    print(f"COMPARISON: {test_name} (D={cuda['D']}, N={cuda['N']})")
    print(f"{'='*80}")
    
    # Compare initial norm
    norm_diff = abs(cuda['norms'][0] - pytorch['norms'][0])
    norm_pct = (norm_diff / pytorch['norms'][0]) * 100
    print(f"\nInitial Norm:")
    print(f"  CUDA:    {cuda['norms'][0]:.6f}")
    print(f"  PyTorch: {pytorch['norms'][0]:.6f}")
    print(f"  Diff:    {norm_diff:.6f} ({norm_pct:.4f}%)")
    
    if norm_pct > 0.1:
        print(f"  ⚠️  SIGNIFICANT DIFFERENCE in initial norm!")
    
    # Compare traces
    print(f"\nTraces (should monotonically increase):")
    print(f"  Iter  CUDA       PyTorch    Diff       Monotonic?")
    print(f"  ----  ---------  ---------  ---------  ----------")
    prev_cuda = 0
    prev_pytorch = 0
    for i in range(5):
        cuda_trace = cuda['traces'][i]
        pytorch_trace = pytorch['traces'][i]
        diff = abs(cuda_trace - pytorch_trace)
        
        cuda_mono = "✓" if cuda_trace > prev_cuda else "✗"
        pytorch_mono = "✓" if pytorch_trace > prev_pytorch else "✗"
        
        print(f"  {i+1:4d}  {cuda_trace:9.6f}  {pytorch_trace:9.6f}  {diff:9.6f}  CUDA:{cuda_mono} PT:{pytorch_mono}")
        
        prev_cuda = cuda_trace
        prev_pytorch = pytorch_trace
    
    # Compare final output
    output_diff = abs(cuda['output_00'] - pytorch['output_00'])
    output_pct = (output_diff / abs(pytorch['output_00'])) * 100 if pytorch['output_00'] != 0 else 0
    print(f"\nFinal Output [0,0]:")
    print(f"  CUDA:    {cuda['output_00']:.6f}")
    print(f"  PyTorch: {pytorch['output_00']:.6f}")
    print(f"  Diff:    {output_diff:.6f} ({output_pct:.4f}%)")
    
    # Analysis
    print(f"\nAnalysis:")
    
    # Check if trace is monotonic
    cuda_monotonic = all(cuda['traces'][i] < cuda['traces'][i+1] for i in range(4))
    pytorch_monotonic = all(pytorch['traces'][i] < pytorch['traces'][i+1] for i in range(4))
    
    if not cuda_monotonic:
        print(f"  ❌ CUDA trace is NOT monotonic - this is a bug!")
    else:
        print(f"  ✓ CUDA trace is monotonic")
    
    if not pytorch_monotonic:
        print(f"  ❌ PyTorch trace is NOT monotonic - this is a bug!")
    else:
        print(f"  ✓ PyTorch trace is monotonic")
    
    # Check if converging to gram_size
    gram_size = min(cuda['D'], cuda['N'])
    final_trace_cuda = cuda['traces'][-1]
    final_trace_pytorch = pytorch['traces'][-1]
    
    print(f"  Expected final trace: ~{gram_size}")
    print(f"  CUDA final trace: {final_trace_cuda:.4f} (gap: {gram_size - final_trace_cuda:.4f})")
    print(f"  PyTorch final trace: {final_trace_pytorch:.4f} (gap: {gram_size - final_trace_pytorch:.4f})")
    
    if gram_size - final_trace_cuda > 1:
        print(f"  ⚠️  CUDA is far from convergence!")
    if gram_size - final_trace_pytorch > 1:
        print(f"  ⚠️  PyTorch is far from convergence!")

if __name__ == "__main__":
    print("="*80)
    print("CUDA vs PyTorch Newton-Schulz Comparison")
    print("="*80)
    
    compare_test("Test 1 (Fat Matrix)", cuda_results["test1"], pytorch_results["test1"])
    compare_test("Test 2 (Tall Matrix)", cuda_results["test2"], pytorch_results["test2"])
    compare_test("Test 3 (Production)", cuda_results["test3"], pytorch_results["test3"])
    
    print("\n" + "="*80)
    print("SUMMARY OF ISSUES FOUND")
    print("="*80)
    
    issues = []
    
    # Check for non-monotonic traces
    for test_name, cuda in cuda_results.items():
        if not all(cuda['traces'][i] < cuda['traces'][i+1] for i in range(4)):
            issues.append(f"❌ {test_name}: CUDA trace not monotonic")
    
    for test_name, pytorch in pytorch_results.items():
        if not all(pytorch['traces'][i] < pytorch['traces'][i+1] for i in range(4)):
            issues.append(f"❌ {test_name}: PyTorch trace not monotonic")
    
    # Check for norm differences
    for test_name in cuda_results.keys():
        norm_diff_pct = abs(cuda_results[test_name]['norms'][0] - pytorch_results[test_name]['norms'][0]) / pytorch_results[test_name]['norms'][0] * 100
        if norm_diff_pct > 0.1:
            issues.append(f"⚠️  {test_name}: Initial norm differs by {norm_diff_pct:.4f}%")
    
    # Check convergence
    for test_name in cuda_results.items():
        test_data = cuda_results[test_name]
        gram_size = min(test_data['D'], test_data['N'])
        if gram_size - test_data['traces'][-1] > 1:
            issues.append(f"⚠️  {test_name}: CUDA far from convergence (trace={test_data['traces'][-1]:.2f}, expected~{gram_size})")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No major issues found!")
    
    print("\nNEXT STEPS:")
    print("1. Check if the official PyTorch reference normalizes X after each iteration")
    print("2. Verify the Newton-Schulz coefficients (a, b, c)")
    print("3. Check if we need X = X / X.norm() after each iteration for stability")

