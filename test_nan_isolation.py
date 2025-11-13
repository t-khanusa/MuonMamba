#!/usr/bin/env python3
"""
Test to isolate NaN bug: Compare selective_scan_fn vs MuonMamba
This helps identify if NaN comes from selective_scan_fn itself or from MuonMamba's usage
"""

import torch
import torch.nn.functional as F
from mamba_ssm.modules.muon_mamba import MuonMamba, MuonMambaConfig
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange

# Set random seed
torch.manual_seed(42)

def test_selective_scan_fn_direct():
    """Test selective_scan_fn directly (low-level)"""
    print("=" * 80)
    print("TEST 1: selective_scan_fn Direct (Low-Level)")
    print("=" * 80)
    
    batch_size = 2
    dim = 128
    seq_len = 512
    dstate = 64
    beta = 0.95
    alpha = 1.0
    
    # Create inputs with extreme values (more likely to trigger NaN)
    u = torch.randn(batch_size, dim, seq_len, dtype=torch.float32, device='cuda', requires_grad=True) * 10.0
    delta = torch.randn(batch_size, dim, seq_len, dtype=torch.float32, device='cuda', requires_grad=True) * 0.1
    delta = F.softplus(delta)
    
    A = -torch.rand(dim, dstate, dtype=torch.float32, device='cuda', requires_grad=True) - 1.0
    B = torch.randn(dim, dstate, dtype=torch.float32, device='cuda', requires_grad=True) * 0.1
    C = torch.randn(dim, dstate, dtype=torch.float32, device='cuda', requires_grad=True) * 0.1
    D = torch.randn(dim, dtype=torch.float32, device='cuda', requires_grad=True) * 0.1
    
    # Create z for gating
    z = torch.randn(batch_size, dim, seq_len, dtype=torch.float32, device='cuda') * 0.1
    
    print(f"Input shapes: u={u.shape}, delta={delta.shape}, A={A.shape}, B={B.shape}, C={C.shape}, D={D.shape}")
    print(f"Input stats: u mean={u.mean().item():.6f}, std={u.std().item():.6f}")
    print(f"            delta mean={delta.mean().item():.6f}, std={delta.std().item():.6f}")
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        result = selective_scan_fn(
            u, delta, A, B, C, D=D, z=z,
            delta_softplus=False,  # Already applied softplus
            beta=beta, alpha=alpha
        )
        
        if isinstance(result, tuple):
            out, out_z = result
        else:
            out = result
            out_z = None
        
        print(f"Output stats: mean={out.mean().item():.6f}, std={out.std().item():.6f}, "
              f"max={out.max().item():.6f}, min={out.min().item():.6f}")
        print(f"Has NaN: {torch.isnan(out).any().item()}")
        print(f"Has Inf: {torch.isinf(out).any().item()}")
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("  ❌ FAILED: Forward pass contains NaN/Inf")
            return False
        
    except Exception as e:
        print(f"  ❌ FAILED: Forward pass exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Backward pass
    print("\nRunning backward pass...")
    dout = torch.randn_like(out) * 0.01
    
    try:
        loss = (out * dout).sum()
        loss.backward()
        
        # Check gradients
        print("\nChecking gradients...")
        all_ok = True
        
        for name, param in [("u", u), ("delta", delta), ("A", A), ("B", B), ("C", C), ("D", D)]:
            if param.grad is not None:
                grad_nan = torch.isnan(param.grad).any().item()
                grad_inf = torch.isinf(param.grad).any().item()
                grad_norm = param.grad.norm().item()
                
                if grad_nan or grad_inf:
                    print(f"  ❌ {name}: Has NaN={grad_nan}, Has Inf={grad_inf}, norm={grad_norm:.6e}")
                    all_ok = False
                elif grad_norm > 1e6:
                    print(f"  ⚠️  {name}: Very large gradient norm={grad_norm:.6e}")
                elif grad_norm < 1e-10:
                    print(f"  ⚠️  {name}: Very small gradient norm={grad_norm:.6e}")
                else:
                    print(f"  ✅ {name}: OK (norm={grad_norm:.6e})")
            else:
                print(f"  ⚠️  {name}: Gradient is None")
        
        if all_ok:
            print("\n  ✅ PASSED: All gradients are finite")
        else:
            print("\n  ❌ FAILED: Some gradients contain NaN/Inf")
        
        return all_ok
        
    except Exception as e:
        print(f"  ❌ FAILED: Backward pass exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_muon_mamba_model():
    """Test MuonMamba model (high-level)"""
    print("\n" + "=" * 80)
    print("TEST 2: MuonMamba Model (High-Level)")
    print("=" * 80)
    
    # Create config with high beta (more likely to cause overflow)
    config = MuonMambaConfig(
        d_model=128,
        n_layers=1,
        d_state=64,
        momentum_beta=0.95,  # High beta - more likely to cause overflow
        momentum_alpha=1.0,
    )
    
    # Create model
    model = MuonMamba(config).cuda()
    model.train()
    
    # Create input with extreme values
    batch_size = 2
    seq_len = 512
    x = torch.randn(batch_size, seq_len, config.d_model, device='cuda') * 10.0  # Large values
    
    print(f"Input shape: {x.shape}")
    print(f"Input stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}, "
          f"max={x.max().item():.6f}, min={x.min().item():.6f}")
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        out = model(x)
        print(f"Output stats: mean={out.mean().item():.6f}, std={out.std().item():.6f}, "
              f"max={out.max().item():.6f}, min={out.min().item():.6f}")
        print(f"Has NaN: {torch.isnan(out).any().item()}")
        print(f"Has Inf: {torch.isinf(out).any().item()}")
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("  ❌ FAILED: Forward pass contains NaN/Inf")
            return False
        
    except Exception as e:
        print(f"  ❌ FAILED: Forward pass exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Backward pass
    print("\nRunning backward pass...")
    loss = out.sum()
    
    try:
        loss.backward()
        
        # Check gradients
        print("\nChecking gradients...")
        all_ok = True
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_nan = torch.isnan(param.grad).any().item()
                grad_inf = torch.isinf(param.grad).any().item()
                grad_norm = param.grad.norm().item()
                
                if grad_nan or grad_inf:
                    print(f"  ❌ {name}: Has NaN={grad_nan}, Has Inf={grad_inf}, norm={grad_norm:.6e}")
                    all_ok = False
                elif grad_norm > 1e6:
                    print(f"  ⚠️  {name}: Very large gradient norm={grad_norm:.6e}")
                elif grad_norm < 1e-10:
                    print(f"  ⚠️  {name}: Very small gradient norm={grad_norm:.6e}")
                else:
                    print(f"  ✅ {name}: OK (norm={grad_norm:.6e})")
            else:
                print(f"  ⚠️  {name}: Gradient is None")
        
        if all_ok:
            print("\n  ✅ PASSED: All gradients are finite")
        else:
            print("\n  ❌ FAILED: Some gradients contain NaN/Inf")
        
        return all_ok
        
    except Exception as e:
        print(f"  ❌ FAILED: Backward pass exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """Compare both approaches with same inputs"""
    print("\n" + "=" * 80)
    print("TEST 3: Direct Comparison (Same Inputs)")
    print("=" * 80)
    
    batch_size = 2
    dim = 128
    seq_len = 512
    dstate = 64
    beta = 0.95
    alpha = 1.0
    
    # Same inputs for both tests
    torch.manual_seed(42)
    u = torch.randn(batch_size, dim, seq_len, dtype=torch.float32, device='cuda') * 10.0
    delta = torch.randn(batch_size, dim, seq_len, dtype=torch.float32, device='cuda') * 0.1
    delta = F.softplus(delta)
    
    A = -torch.rand(dim, dstate, dtype=torch.float32, device='cuda') - 1.0
    B = torch.randn(dim, dstate, dtype=torch.float32, device='cuda') * 0.1
    C = torch.randn(dim, dstate, dtype=torch.float32, device='cuda') * 0.1
    D = torch.randn(dim, dtype=torch.float32, device='cuda') * 0.1
    
    print("Comparing outputs from selective_scan_fn vs MuonMamba...")
    print("(This helps identify if the issue is in selective_scan_fn or in MuonMamba's wrapper)")
    
    # Test selective_scan_fn
    z_ss = torch.randn(batch_size, dim, seq_len, dtype=torch.float32, device='cuda') * 0.1
    result_ss = selective_scan_fn(
        u.clone(), delta.clone(), A.clone(), B.clone(), C.clone(), D=D.clone(), z=z_ss,
        delta_softplus=False, beta=beta, alpha=alpha
    )
    out_ss = result_ss[0] if isinstance(result_ss, tuple) else result_ss
    
    # Test MuonMamba (simulate the same computation)
    # Note: MuonMamba applies additional operations (conv1d, projections), so this is approximate
    config = MuonMambaConfig(d_model=dim, n_layers=1, d_state=dstate, momentum_beta=beta, momentum_alpha=alpha)
    model = MuonMamba(config).cuda()
    model.eval()
    
    # Manually set weights to match (approximately)
    with torch.no_grad():
        # This is just for comparison - won't match exactly due to MuonMamba's architecture
        pass
    
    x_mm = rearrange(u, "b d l -> b l d")
    out_mm = model(x_mm)
    out_mm = rearrange(out_mm, "b l d -> b d l")
    
    print(f"\nselective_scan_fn output: mean={out_ss.mean().item():.6f}, std={out_ss.std().item():.6f}")
    print(f"MuonMamba output: mean={out_mm.mean().item():.6f}, std={out_mm.std().item():.6f}")
    print(f"\nNote: Outputs won't match exactly due to MuonMamba's additional layers")
    print(f"      (conv1d, projections, etc.), but both should be finite.")


def main():
    print("\n" + "=" * 80)
    print("NaN ISOLATION TEST: selective_scan_fn vs MuonMamba")
    print("=" * 80)
    print("\nThis test helps identify where NaN originates:")
    print("1. If selective_scan_fn produces NaN → bug in CUDA kernel")
    print("2. If MuonMamba produces NaN but selective_scan_fn doesn't → bug in MuonMamba wrapper")
    print("3. If both produce NaN → bug in selective_scan_fn (affects both)")
    print("=" * 80)
    
    results = {}
    
    # Test 1: selective_scan_fn directly
    results['selective_scan_fn'] = test_selective_scan_fn_direct()
    
    # Test 2: MuonMamba model
    results['muon_mamba'] = test_muon_mamba_model()
    
    # Test 3: Comparison
    test_comparison()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED ✅")
        print("  → No NaN detected in either path")
    else:
        print("SOME TESTS FAILED ❌")
        if not results['selective_scan_fn']:
            print("  → NaN originates from selective_scan_fn (CUDA kernel bug)")
        if not results['muon_mamba']:
            print("  → NaN originates from MuonMamba (wrapper bug)")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


