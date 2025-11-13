#!/usr/bin/env python3
"""
Test to understand how dv values should be distributed across timesteps
"""

import torch

# Simple test case
seqlen = 4
beta = 0.9

# Assume dh values (gradients from hidden state scan)
# For simplicity, assume dh[t] = 1.0 for all t
dh = torch.ones(seqlen)

# Reverse scan for velocity: dv[t] = dh[t] + beta * dv[t+1]
# Process in reverse order
dv = torch.zeros(seqlen)
for t in range(seqlen - 1, -1, -1):
    if t < seqlen - 1:
        dv[t] = dh[t] + beta * dv[t + 1]
    else:
        dv[t] = dh[t]
    print(f"Timestep {t}: dh={dh[t]:.3f}, dv={dv[t]:.6f}")

print("\nExpected: dv[0] > dv[1] > dv[2] > dv[3] (accumulated backward)")
print(f"Actual: dv[0]={dv[0]:.6f}, dv[1]={dv[1]:.6f}, dv[2]={dv[2]:.6f}, dv[3]={dv[3]:.6f}")




