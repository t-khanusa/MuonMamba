#!/usr/bin/env python3
"""
Test how PyTorch handles gradients through BF16 rounding
"""

import torch

x = torch.tensor([1.5, 2.5, 3.5], requires_grad=True)

# Method 1: With BF16 rounding
y1 = x.bfloat16().float()
y1.backward(torch.ones_like(y1))
grad1 = x.grad.clone()
x.grad.zero_()

print("Gradient through BF16 rounding:")
print(f"  Input: {[1.5, 2.5, 3.5]}")
print(f"  BF16 rounded: {y1}")
print(f"  Gradient: {grad1}")

# Method 2: Without BF16 rounding
y2 = x
y2.backward(torch.ones_like(y2))
grad2 = x.grad.clone()

print("\nGradient without BF16 rounding:")
print(f"  Gradient: {grad2}")

print("\nConclusion:")
if torch.allclose(grad1, grad2):
    print("  Gradients are the same - PyTorch uses straight-through estimator")
else:
    print("  Gradients are different!")
    print(f"  Difference: {grad1 - grad2}")

