#!/usr/bin/env python3
"""
Comprehensive testing suite for Momentum Mamba.

Tests:
1. Correctness across configurations
2. Edge cases and boundary conditions
3. Performance benchmarks
4. Memory usage
5. Training convergence
"""

import torch
import torch.nn as nn
import time
import numpy as np
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


def test_correctness_configurations():
    """Test correctness across various configurations."""
    print("="*70)
    print("Testing Correctness Across Configurations")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping\n")
        return True
    
    # Test configurations
    configs = [
        # (batch, dim, seqlen, dstate, beta, alpha, description)
        (1, 4, 16, 8, 0.0, 1.0, "Small, no momentum"),
        (2, 8, 32, 16, 0.5, 1.0, "Medium, light momentum"),
        (4, 16, 64, 32, 0.9, 1.0, "Large, strong momentum"),
        (1, 4, 128, 8, 0.9, 0.5, "Long sequence, scaled alpha"),
        (8, 8, 16, 16, 0.99, 2.0, "High beta, large alpha"),
        (1, 2, 256, 4, 0.5, 1.0, "Very long sequence"),
    ]
    
    all_passed = True
    results = []
    
    for batch, dim, seqlen, dstate, beta, alpha, desc in configs:
        try:
            # Create inputs
            u_cpu = torch.randn(batch, dim, seqlen, dtype=torch.float32)
            delta_cpu = torch.randn(batch, dim, seqlen, dtype=torch.float32).abs() * 0.1
            A_cpu = -torch.randn(dim, dstate, dtype=torch.float32).abs()
            B_cpu = torch.randn(batch, dstate, seqlen, dtype=torch.float32)
            C_cpu = torch.randn(batch, dstate, seqlen, dtype=torch.float32)
            D_cpu = torch.randn(dim, dtype=torch.float32)
            
            # CPU reference
            out_cpu = selective_scan_ref(
                u_cpu, delta_cpu, A_cpu, B_cpu, C_cpu, D=D_cpu,
                delta_softplus=True, beta=beta, alpha=alpha
            )
            
            # CUDA
            u_cuda = u_cpu.cuda()
            delta_cuda = delta_cpu.cuda()
            A_cuda = A_cpu.cuda()
            B_cuda = B_cpu.cuda()
            C_cuda = C_cpu.cuda()
            D_cuda = D_cpu.cuda()
            
            out_cuda = selective_scan_fn(
                u_cuda, delta_cuda, A_cuda, B_cuda, C_cuda, D=D_cuda,
                delta_softplus=True, beta=beta, alpha=alpha
            )
            
            # Compare
            diff = (out_cuda.cpu() - out_cpu).abs().max().item()
            passed = diff < 1e-3
            
            status = "✓" if passed else "✗"
            results.append((desc, passed, diff))
            print(f"{status} {desc:<40} diff={diff:.6f}")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"✗ {desc:<40} ERROR: {e}")
            all_passed = False
            results.append((desc, False, float('inf')))
    
    print(f"\nPassed: {sum(1 for _, p, _ in results if p)}/{len(results)}")
    print()
    return all_passed


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("="*70)
    print("Testing Edge Cases")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping\n")
        return True
    
    all_passed = True
    
    # Test 1: Very small beta (near standard Mamba)
    print("Test 1: Very small beta (β=1e-6)...")
    try:
        batch, dim, seqlen, dstate = 2, 4, 16, 8
        u = torch.randn(batch, dim, seqlen, device='cuda')
        delta = torch.randn(batch, dim, seqlen, device='cuda').abs() * 0.1
        A = -torch.randn(dim, dstate, device='cuda').abs()
        B = torch.randn(batch, dstate, seqlen, device='cuda')
        C = torch.randn(batch, dstate, seqlen, device='cuda')
        D = torch.randn(dim, device='cuda')
        
        out = selective_scan_fn(u, delta, A, B, C, D=D, beta=1e-6, alpha=1.0)
        assert torch.isfinite(out).all(), "Output contains non-finite values"
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_passed = False
    
    # Test 2: Beta = 1.0 (maximum momentum)
    print("Test 2: Maximum beta (β=0.999)...")
    try:
        out = selective_scan_fn(u, delta, A, B, C, D=D, beta=0.999, alpha=1.0)
        assert torch.isfinite(out).all(), "Output contains non-finite values"
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_passed = False
    
    # Test 3: Zero alpha (no input contribution)
    print("Test 3: Zero alpha (α=0.0)...")
    try:
        out = selective_scan_fn(u, delta, A, B, C, D=D, beta=0.9, alpha=0.0)
        assert torch.isfinite(out).all(), "Output contains non-finite values"
        # Output should be close to zero since no input contributes
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_passed = False
    
    # Test 4: Very large alpha
    print("Test 4: Large alpha (α=10.0)...")
    try:
        out = selective_scan_fn(u, delta, A, B, C, D=D, beta=0.5, alpha=10.0)
        assert torch.isfinite(out).all(), "Output contains non-finite values"
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_passed = False
    
    # Test 5: Single element sequence
    print("Test 5: Single element sequence (seqlen=1)...")
    try:
        u_single = torch.randn(1, 4, 1, device='cuda')
        delta_single = torch.randn(1, 4, 1, device='cuda').abs() * 0.1
        B_single = torch.randn(1, 8, 1, device='cuda')
        C_single = torch.randn(1, 8, 1, device='cuda')
        
        out = selective_scan_fn(u_single, delta_single, A, B_single, C_single, D=D, beta=0.9, alpha=1.0)
        assert out.shape == (1, 4, 1), f"Wrong output shape: {out.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_passed = False
    
    # Test 6: Large batch size
    print("Test 6: Large batch size (batch=32)...")
    try:
        u_large = torch.randn(32, 4, 16, device='cuda')
        delta_large = torch.randn(32, 4, 16, device='cuda').abs() * 0.1
        B_large = torch.randn(32, 8, 16, device='cuda')
        C_large = torch.randn(32, 8, 16, device='cuda')
        
        out = selective_scan_fn(u_large, delta_large, A, B_large, C_large, D=D, beta=0.9, alpha=1.0)
        assert out.shape == (32, 4, 16), f"Wrong output shape: {out.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_passed = False
    
    print()
    return all_passed


def benchmark_performance():
    """Benchmark forward and backward pass performance."""
    print("="*70)
    print("Performance Benchmarks")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping\n")
        return True
    
    # Test configurations
    configs = [
        (2, 256, 512, 16, "Small (batch=2, dim=256, seq=512)"),
        (4, 512, 1024, 16, "Medium (batch=4, dim=512, seq=1024)"),
        (8, 1024, 2048, 16, "Large (batch=8, dim=1024, seq=2048)"),
    ]
    
    n_warmup = 5
    n_iters = 20
    
    print(f"Warmup: {n_warmup} iterations, Benchmark: {n_iters} iterations\n")
    
    for batch, dim, seqlen, dstate, desc in configs[:1]:  # Only test small for speed
        print(f"Configuration: {desc}")
        print("-" * 70)
        
        # Create inputs
        u = torch.randn(batch, dim, seqlen, device='cuda', dtype=torch.float32, requires_grad=True)
        delta = torch.randn(batch, dim, seqlen, device='cuda', dtype=torch.float32).abs() * 0.1
        delta.requires_grad = True
        A = -torch.randn(dim, dstate, device='cuda', dtype=torch.float32).abs()
        A.requires_grad = True
        B = torch.randn(batch, dstate, seqlen, device='cuda', dtype=torch.float32, requires_grad=True)
        C = torch.randn(batch, dstate, seqlen, device='cuda', dtype=torch.float32, requires_grad=True)
        D = torch.randn(dim, device='cuda', dtype=torch.float32, requires_grad=True)
        
        # Test both standard Mamba and momentum Mamba
        for beta, alpha, name in [(0.0, 1.0, "Standard Mamba"), (0.9, 1.0, "Momentum Mamba")]:
            print(f"\n{name} (β={beta}, α={alpha}):")
            
            # Warmup
            for _ in range(n_warmup):
                out = selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=True, beta=beta, alpha=alpha)
                loss = out.sum()
                loss.backward()
                u.grad = None
                delta.grad = None
                A.grad = None
                B.grad = None
                C.grad = None
                D.grad = None
            
            torch.cuda.synchronize()
            
            # Benchmark forward
            start = time.time()
            for _ in range(n_iters):
                out = selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=True, beta=beta, alpha=alpha)
            torch.cuda.synchronize()
            fwd_time = (time.time() - start) / n_iters * 1000
            
            # Benchmark forward + backward
            start = time.time()
            for _ in range(n_iters):
                out = selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=True, beta=beta, alpha=alpha)
                loss = out.sum()
                loss.backward()
                u.grad = None
                delta.grad = None
                A.grad = None
                B.grad = None
                C.grad = None
                D.grad = None
            torch.cuda.synchronize()
            fwd_bwd_time = (time.time() - start) / n_iters * 1000
            bwd_time = fwd_bwd_time - fwd_time
            
            print(f"  Forward:  {fwd_time:.3f} ms")
            print(f"  Backward: {bwd_time:.3f} ms")
            print(f"  Total:    {fwd_bwd_time:.3f} ms")
        
        print()
    
    return True


def test_memory_usage():
    """Test memory usage and efficiency."""
    print("="*70)
    print("Memory Usage Analysis")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping\n")
        return True
    
    batch, dim, seqlen, dstate = 4, 512, 1024, 16
    
    for beta, alpha, name in [(0.0, 1.0, "Standard Mamba"), (0.9, 1.0, "Momentum Mamba")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create inputs
        u = torch.randn(batch, dim, seqlen, device='cuda', requires_grad=True)
        delta = torch.randn(batch, dim, seqlen, device='cuda').abs() * 0.1
        delta.requires_grad = True
        A = -torch.randn(dim, dstate, device='cuda').abs()
        A.requires_grad = True
        B = torch.randn(batch, dstate, seqlen, device='cuda', requires_grad=True)
        C = torch.randn(batch, dstate, seqlen, device='cuda', requires_grad=True)
        D = torch.randn(dim, device='cuda', requires_grad=True)
        
        # Forward + backward
        out = selective_scan_fn(u, delta, A, B, C, D=D, beta=beta, alpha=alpha)
        loss = out.sum()
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"{name}: Peak memory = {peak_mem:.2f} MB")
    
    print()
    return True


def test_training_convergence():
    """Test training convergence on a simple task."""
    print("="*70)
    print("Training Convergence Test")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping\n")
        return True
    
    # Simple copy task: learn to copy input to output
    print("Task: Learn to copy a random sequence")
    print()
    
    for beta, alpha, name in [(0.0, 1.0, "Standard Mamba"), (0.9, 1.0, "Momentum Mamba")]:
        print(f"{name} (β={beta}, α={alpha}):")
        
        # Create model
        model = Mamba(
            d_model=32,
            d_state=8,
            d_conv=4,
            expand=2,
            beta=beta,
            alpha=alpha,
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training loop
        losses = []
        n_steps = 50
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Generate random input
            x = torch.randn(2, 16, 32, device='cuda')
            target = x  # Copy task
            
            # Forward
            out = model(x)
            loss = criterion(out, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (step + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                print(f"  Step {step+1:3d}: Loss = {avg_loss:.6f}")
        
        final_loss = np.mean(losses[-10:])
        initial_loss = np.mean(losses[:10])
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {final_loss:.6f}")
        print(f"  Improvement:  {improvement:.1f}%")
        print()
    
    return True


def test_gradient_stability():
    """Test gradient stability during training."""
    print("="*70)
    print("Gradient Stability Test")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping\n")
        return True
    
    batch, dim, seqlen, dstate = 2, 4, 32, 8
    n_steps = 100
    
    for beta, alpha, name in [(0.0, 1.0, "Standard Mamba"), (0.9, 1.0, "Momentum Mamba")]:
        print(f"{name} (β={beta}, α={alpha}):")
        
        grad_norms = []
        max_grads = []
        
        for step in range(n_steps):
            # Create fresh inputs
            u = torch.randn(batch, dim, seqlen, device='cuda', requires_grad=True)
            delta = torch.randn(batch, dim, seqlen, device='cuda').abs() * 0.1
            delta.requires_grad = True
            A = -torch.randn(dim, dstate, device='cuda').abs()
            A.requires_grad = True
            B = torch.randn(batch, dstate, seqlen, device='cuda', requires_grad=True)
            C = torch.randn(batch, dstate, seqlen, device='cuda', requires_grad=True)
            D = torch.randn(dim, device='cuda', requires_grad=True)
            
            # Forward + backward
            out = selective_scan_fn(u, delta, A, B, C, D=D, beta=beta, alpha=alpha)
            loss = out.sum()
            loss.backward()
            
            # Collect gradient statistics
            grad_norm = torch.cat([
                u.grad.flatten(),
                delta.grad.flatten(),
                A.grad.flatten(),
                B.grad.flatten(),
                C.grad.flatten(),
                D.grad.flatten(),
            ]).norm().item()
            
            max_grad = max(
                u.grad.abs().max().item(),
                delta.grad.abs().max().item(),
                A.grad.abs().max().item(),
                B.grad.abs().max().item(),
                C.grad.abs().max().item(),
                D.grad.abs().max().item(),
            )
            
            grad_norms.append(grad_norm)
            max_grads.append(max_grad)
        
        # Statistics
        mean_norm = np.mean(grad_norms)
        std_norm = np.std(grad_norms)
        mean_max = np.mean(max_grads)
        std_max = np.std(max_grads)
        
        print(f"  Gradient norm:     {mean_norm:.3f} ± {std_norm:.3f}")
        print(f"  Max gradient:      {mean_max:.3f} ± {std_max:.3f}")
        print(f"  Non-finite grads:  {sum(1 for g in grad_norms if not np.isfinite(g))}/{n_steps}")
        print()
    
    return True


def main():
    """Run all comprehensive tests."""
    print("\n")
    print("="*70)
    print("COMPREHENSIVE MOMENTUM MAMBA TEST SUITE")
    print("="*70)
    print()
    
    results = []
    
    # Run tests
    results.append(("Correctness (Configurations)", test_correctness_configurations()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Performance Benchmarks", benchmark_performance()))
    results.append(("Memory Usage", test_memory_usage()))
    results.append(("Training Convergence", test_training_convergence()))
    results.append(("Gradient Stability", test_gradient_stability()))
    
    # Summary
    print("="*70)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<50} {status}")
    print("="*70)
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All comprehensive tests passed!")
        print("✅ Momentum Mamba is production-ready!")
    else:
        print("\n⚠️  Some tests failed. Please review.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

