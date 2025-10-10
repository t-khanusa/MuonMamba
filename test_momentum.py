#!/usr/bin/env python3
"""
Test script for Mamba momentum implementation.

This script tests:
1. Reference implementation (Python) works correctly
2. CUDA implementation compiles and runs
3. Momentum parameters affect output as expected
"""

import torch
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
from calflops import calculate_flops


def test_reference_implementation():
    """Test the Python reference implementation of momentum."""
    print("Testing reference implementation...")
    
    batch, dim, seqlen, dstate = 2, 4, 8, 3
    
    # Create test inputs
    u = torch.randn(batch, dim, seqlen, dtype=torch.float32)
    delta = torch.randn(batch, dim, seqlen, dtype=torch.float32) * 0.1
    A = -torch.exp(torch.randn(dim, dstate, dtype=torch.float32))
    B = torch.randn(batch, dstate, seqlen, dtype=torch.float32)
    C = torch.randn(batch, dstate, seqlen, dtype=torch.float32)
    D = torch.randn(dim, dtype=torch.float32)
    
    # Test 1: beta=0 should give same result as original (no momentum)
    out_no_momentum = selective_scan_ref(
        u, delta, A, B, C, D=D, 
        delta_softplus=False, 
        beta=0.0, alpha=1.0
    )
    
    # Test 2: With momentum
    out_with_momentum = selective_scan_ref(
        u, delta, A, B, C, D=D,
        delta_softplus=False,
        beta=0.9, alpha=1.0
    )
    
    # Test 3: Return last state
    out, last_h, last_v = selective_scan_ref(
        u, delta, A, B, C, D=D,
        delta_softplus=False,
        beta=0.9, alpha=1.0,
        return_last_state=True
    )
    
    print(f"  Output shape: {out.shape}")
    print(f"  Last hidden state shape: {last_h.shape}")
    print(f"  Last velocity state shape: {last_v.shape}")
    print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"  Momentum changes output: {not torch.allclose(out_no_momentum, out_with_momentum)}")
    
    # Verify shapes are correct
    assert out.shape == (batch, dim, seqlen)
    assert last_h.shape == (batch, dim, dstate)
    assert last_v.shape == (batch, dim, dstate)
    
    print("  ✓ Reference implementation passed\n")
    return True


def test_momentum_effect():
    """Test that momentum parameters have the expected effect."""
    print("Testing momentum parameter effects...")
    
    batch, dim, seqlen, dstate = 1, 2, 16, 4
    
    u = torch.randn(batch, dim, seqlen, dtype=torch.float32)
    delta = torch.ones(batch, dim, seqlen, dtype=torch.float32) * 0.1
    A = -torch.ones(dim, dstate, dtype=torch.float32)
    B = torch.randn(batch, dstate, seqlen, dtype=torch.float32)
    C = torch.randn(batch, dstate, seqlen, dtype=torch.float32)
    
    # Different beta values
    out_beta0 = selective_scan_ref(u, delta, A, B, C, beta=0.0, alpha=1.0)
    out_beta05 = selective_scan_ref(u, delta, A, B, C, beta=0.5, alpha=1.0)
    out_beta09 = selective_scan_ref(u, delta, A, B, C, beta=0.9, alpha=1.0)
    
    # Different alpha values
    out_alpha0 = selective_scan_ref(u, delta, A, B, C, beta=0.9, alpha=0.0)
    out_alpha1 = selective_scan_ref(u, delta, A, B, C, beta=0.9, alpha=1.0)
    out_alpha2 = selective_scan_ref(u, delta, A, B, C, beta=0.9, alpha=2.0)
    
    print(f"  Beta=0.0 mean abs: {out_beta0.abs().mean():.4f}")
    print(f"  Beta=0.5 mean abs: {out_beta05.abs().mean():.4f}")
    print(f"  Beta=0.9 mean abs: {out_beta09.abs().mean():.4f}")
    print(f"  Alpha=0.0 mean abs: {out_alpha0.abs().mean():.4f}")
    print(f"  Alpha=1.0 mean abs: {out_alpha1.abs().mean():.4f}")
    print(f"  Alpha=2.0 mean abs: {out_alpha2.abs().mean():.4f}")
    
    # Verify outputs are different
    assert not torch.allclose(out_beta0, out_beta09, rtol=0.01)
    assert not torch.allclose(out_alpha0, out_alpha2, rtol=0.01)
    
    print("  ✓ Momentum effects verified\n")
    return True


def test_cuda_implementation():
    """Test CUDA implementation if available."""
    if not torch.cuda.is_available():
        print("Skipping CUDA tests (CUDA not available)\n")
        return True
        
    print("Testing CUDA implementation...")
    
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        
        batch, dim, seqlen, dstate = 2, 8, 32, 16
        
        # Create CUDA tensors
        u = torch.randn(batch, dim, seqlen, device='cuda', dtype=torch.float32)
        delta = torch.randn(batch, dim, seqlen, device='cuda', dtype=torch.float32) * 0.1
        A = -torch.exp(torch.randn(dim, dstate, device='cuda', dtype=torch.float32))
        B = torch.randn(batch, dstate, seqlen, device='cuda', dtype=torch.float32)
        C = torch.randn(batch, dstate, seqlen, device='cuda', dtype=torch.float32)
        D = torch.randn(dim, device='cuda', dtype=torch.float32)
        
        # Test CUDA forward pass
        out_cuda = selective_scan_fn(
            u, delta, A, B, C, D=D,
            delta_softplus=True,
            beta=0.9, alpha=1.0
        )
        
        print(f"  CUDA output shape: {out_cuda.shape}")
        print(f"  CUDA output range: [{out_cuda.min():.3f}, {out_cuda.max():.3f}]")
        print(f"  CUDA output is finite: {torch.isfinite(out_cuda).all()}")
        
        # Test with return_last_state
        out_cuda, last_h, last_v = selective_scan_fn(
            u, delta, A, B, C, D=D,
            delta_softplus=True,
            beta=0.9, alpha=1.0,
            return_last_state=True
        )
        
        print(f"  Last hidden state shape: {last_h.shape}")
        print(f"  Last velocity state shape: {last_v.shape}")
        
        assert out_cuda.shape == (batch, dim, seqlen)
        assert last_h.shape == (batch, dim, dstate)
        assert last_v.shape == (batch, dim, dstate)
        assert torch.isfinite(out_cuda).all()
        
        print("  ✓ CUDA implementation passed\n")
        return True
        
    except Exception as e:
        print(f"  ✗ CUDA test failed: {e}\n")
        return False


def test_cuda_vs_cpu_correctness():
    """Test that CUDA implementation matches CPU reference."""
    if not torch.cuda.is_available():
        print("Skipping CUDA vs CPU test (CUDA not available)\n")
        return True
        
    print("Testing CUDA vs CPU correctness...")
    
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        
        batch, dim, seqlen, dstate = 2, 4, 16, 8
        
        # Create CPU tensors
        u_cpu = torch.randn(batch, dim, seqlen, dtype=torch.float32)
        delta_cpu = torch.randn(batch, dim, seqlen, dtype=torch.float32) * 0.1
        A_cpu = -torch.exp(torch.randn(dim, dstate, dtype=torch.float32))
        B_cpu = torch.randn(batch, dstate, seqlen, dtype=torch.float32)
        C_cpu = torch.randn(batch, dstate, seqlen, dtype=torch.float32)
        D_cpu = torch.randn(dim, dtype=torch.float32)
        
        # Copy to CUDA
        u_cuda = u_cpu.cuda()
        delta_cuda = delta_cpu.cuda()
        A_cuda = A_cpu.cuda()
        B_cuda = B_cpu.cuda()
        C_cuda = C_cpu.cuda()
        D_cuda = D_cpu.cuda()
        
        # Test with different momentum parameters
        beta_values = [0.0, 0.5, 0.9]
        alpha_values = [0.5, 1.0, 2.0]
        
        max_diff = 0.0
        for beta in beta_values:
            for alpha in alpha_values:
                # CPU reference
                out_cpu = selective_scan_ref(
                    u_cpu, delta_cpu, A_cpu, B_cpu, C_cpu, D=D_cpu,
                    delta_softplus=True,
                    beta=beta, alpha=alpha
                )
                
                # CUDA implementation
                out_cuda = selective_scan_fn(
                    u_cuda, delta_cuda, A_cuda, B_cuda, C_cuda, D=D_cuda,
                    delta_softplus=True,
                    beta=beta, alpha=alpha
                )
                
                # Compare
                diff = (out_cuda.cpu() - out_cpu).abs().max().item()
                max_diff = max(max_diff, diff)
                
                print(f"  β={beta}, α={alpha}: max diff = {diff:.6f}")
        
        # Test with return_last_state
        print("\n  Testing with return_last_state...")
        out_cpu, last_h_cpu, last_v_cpu = selective_scan_ref(
            u_cpu, delta_cpu, A_cpu, B_cpu, C_cpu, D=D_cpu,
            delta_softplus=True,
            beta=0.9, alpha=1.0,
            return_last_state=True
        )
        
        out_cuda, last_h_cuda, last_v_cuda = selective_scan_fn(
            u_cuda, delta_cuda, A_cuda, B_cuda, C_cuda, D=D_cuda,
            delta_softplus=True,
            beta=0.9, alpha=1.0,
            return_last_state=True
        )
        
        out_diff = (out_cuda.cpu() - out_cpu).abs().max().item()
        h_diff = (last_h_cuda.cpu() - last_h_cpu).abs().max().item()
        v_diff = (last_v_cuda.cpu() - last_v_cpu).abs().max().item()
        
        print(f"  Output diff: {out_diff:.6f}")
        print(f"  Hidden state diff: {h_diff:.6f}")
        print(f"  Velocity state diff: {v_diff:.6f}")
        
        # Check tolerance (allow some numerical difference)
        tolerance = 1e-3
        if max_diff < tolerance and out_diff < tolerance and h_diff < tolerance and v_diff < tolerance:
            print(f"\n  ✓ CUDA matches CPU (max diff: {max_diff:.6f} < {tolerance})\n")
            return True
        else:
            print(f"\n  ✗ CUDA differs from CPU (max diff: {max_diff:.6f} >= {tolerance})\n")
            return False
            
    except Exception as e:
        print(f"  ✗ CUDA vs CPU test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_mamba_module():
    """Test the Mamba module with momentum."""
    print("Testing Mamba module...")
    
    try:
        from mamba_ssm import Mamba
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("  ⚠️  Skipping Mamba module test (CUDA not available)\n")
            return True
        
        # Create module on CUDA
        mamba = Mamba(
            d_model=128,
            d_state=64,
            beta=0.9,
            alpha=1.0,
        ).cuda()
        
        print(f"  Beta parameter: {mamba.beta.item()}")
        print(f"  Alpha parameter: {mamba.alpha.item()}")
        
        # Test forward pass with CUDA tensors
        x = torch.randn(1, 512, 128, device='cuda')
        y = mamba(x)

        flops = calculate_flops(mamba, input_shape=(1, 512, 128))
        print(f"  FLOPS: {flops}")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        assert y.shape == x.shape
        
        # Test state allocation
        conv_state, ssm_state, velocity_state = mamba.allocate_inference_cache(
            batch_size=1, max_seqlen=512
        )
        
        print(f"  Conv state shape: {conv_state.shape}")
        print(f"  SSM state shape: {ssm_state.shape}")
        print(f"  Velocity state shape: {velocity_state.shape}")
        
        assert velocity_state.shape == ssm_state.shape
        
        print("  ✓ Mamba module passed\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Mamba module test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Mamba Momentum Implementation Tests")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Reference Implementation", test_reference_implementation()))
    results.append(("Momentum Effects", test_momentum_effect()))
    results.append(("CUDA Implementation", test_cuda_implementation()))
    results.append(("CUDA vs CPU Correctness", test_cuda_vs_cpu_correctness()))
    results.append(("Mamba Module", test_mamba_module()))
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<50} {status}")
    print("="*60)
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. See above for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

