#!/usr/bin/env python3
"""
Test the Python backward function against known values
"""

import torch
import sys
sys.path.insert(0, '/project/khanhnt/muontest/Momentum_correct/csrc/selective_scan')
from generate_ns_velocity_test_data import newtonschulz5_velocity_detached_backward

# Same inputs as debug_tiny_case.py
torch.manual_seed(42)
G = torch.randn(2, 2, dtype=torch.float32)
grad_output = torch.randn(2, 2, dtype=torch.float32)

print("Testing Python backward function...")
print(f"Input G:\n{G}")
print(f"Grad output:\n{grad_output}")

# Call the function
grad_G_func = newtonschulz5_velocity_detached_backward(G.detach(), grad_output)

print(f"\ngrad_G from function:\n{grad_G_func}")

# Expected from manual computation
print(f"\nExpected grad_G:\n" + """tensor([[ 0.4498, -3.7940],
        [ 0.3690,  1.0977]])""")

print(f"\nDo they match? {torch.allclose(grad_G_func, torch.tensor([[ 0.4498, -3.7940], [ 0.3690,  1.0977]]), rtol=1e-3, atol=1e-4)}")

