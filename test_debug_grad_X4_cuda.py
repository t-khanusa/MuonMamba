#!/usr/bin/env python3
"""
Test to verify grad_X_4_buffer is actually populated in CUDA
We'll need to add debug output to CUDA to check this
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import selective_scan_cuda

# Minimal test
batch, dim, seqlen, dstate = 1, 2, 4, 2
beta, alpha = 0.9, 1.0
device = 'cuda'
dtype = torch.float32

torch.manual_seed(42)
u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * 0.1
delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) * 0.1
A = -torch.rand(dim, dstate, dtype=dtype, device=device) * 0.1
B = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
C = torch.randn(dim, dstate, dtype=dtype, device=device) * 0.1
D = torch.randn(dim, dtype=dtype, device=device) * 0.1
dout = torch.ones(batch, dim, seqlen, dtype=dtype, device=device)

print("="*80)
print("DEBUG: Checking if we can inspect grad_X_4_buffer")
print("="*80)
print("Unfortunately, grad_X_4_buffer is an internal CUDA buffer")
print("We cannot directly access it from Python.")
print("\nWe need to add debug output in CUDA code to verify:")
print("1. Is grad_X_4_buffer being accumulated for all timesteps?")
print("2. What are the dv values for each timestep?")
print("3. What does NS backward kernel read from grad_X_4_buffer?")




