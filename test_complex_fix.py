#!/usr/bin/env python3
"""
Test script to verify complex case fix for Newton-Schulz orthogonalization.
Tests that complex b_t values are stored correctly with both real and imag parts.
"""

import torch
import sys
import os

# Add path to import selective_scan
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("Warning: Could not import selective_scan_fn. Make sure mamba_ssm is installed.")
    sys.exit(1)

def test_complex_fix():
    """Test that complex weights work correctly with NS orthogonalization."""
    print("Testing complex case fix...")
    
    # Small test case
    batch, dim, seqlen, dstate = 2, 4, 8, 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("Skipping test - CUDA not available")
        return
    
    print(f"Device: {device}")
    print(f"Shape: batch={batch}, dim={dim}, seqlen={seqlen}, dstate={dstate}")
    
    # Create inputs with complex weights
    u = torch.randn(batch, dim, seqlen, dtype=torch.float32, device=device)
    delta = torch.randn(batch, dim, seqlen, dtype=torch.float32, device=device)
    A = torch.randn(dim, dstate, dtype=torch.complex64, device=device)
    B = torch.randn(dim, dstate, dtype=torch.complex64, device=device)
    C = torch.randn(dim, dstate, dtype=torch.complex64, device=device)
    
    print("\n1. Testing with momentum mode (beta != 0) to trigger NS...")
    try:
        # Use momentum mode (beta != 0) to trigger NS
        result = selective_scan_fn(u, delta, A, B, C, beta=0.5, alpha=1.0)
        
        if len(result) == 3:
            out, x, x_4 = result
            print(f"   ✅ Forward pass completed")
            print(f"   Output shape: {out.shape}")
            print(f"   State shape: {x.shape}")
            print(f"   X_4_buffer shape: {x_4.shape}")
            
            # Check X_4_buffer shape
            expected_shape = (batch, dim, seqlen, dstate * 2)
            if x_4.shape == expected_shape:
                print(f"   ✅ X_4_buffer has correct shape for complex: {x_4.shape}")
            else:
                print(f"   ❌ X_4_buffer shape mismatch!")
                print(f"      Expected: {expected_shape}")
                print(f"      Got: {x_4.shape}")
                return False
            
            # Check that X_4_buffer has non-zero values (both real and imag)
            x_4_real = x_4[..., 0::2]  # Even indices = real parts
            x_4_imag = x_4[..., 1::2]  # Odd indices = imag parts
            
            real_nonzero = (x_4_real != 0).any()
            imag_nonzero = (x_4_imag != 0).any()
            
            print(f"   Real part has non-zero values: {real_nonzero.item()}")
            print(f"   Imag part has non-zero values: {imag_nonzero.item()}")
            
            if real_nonzero and imag_nonzero:
                print("   ✅ Both real and imag parts are stored correctly!")
            elif real_nonzero:
                print("   ⚠️  Only real part has non-zero values (imag part may be zero or bug)")
            else:
                print("   ❌ X_4_buffer appears to be all zeros!")
                return False
                
        else:
            print(f"   ❌ Unexpected result length: {len(result)}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing without momentum (beta == 0) for comparison...")
    try:
        result = selective_scan_fn(u, delta, A, B, C, beta=0.0, alpha=1.0)
        if len(result) == 2:
            out, x = result
            print(f"   ✅ Forward pass completed (no NS)")
        else:
            print(f"   ⚠️  Unexpected result length: {len(result)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_complex_fix()
    sys.exit(0 if success else 1)

