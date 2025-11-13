#!/usr/bin/env python3
"""
Test to understand the exact mapping between i, threadIdx.x, and time_idx
"""

# For seqlen=4, kChunkSize=128 (if kNThreads=32, kNItems=4)
seqlen = 4
kChunkSize = 128  # Assume kNThreads=32, kNItems=4
chunk = 0

print("Timestep mapping for chunk 0:")
print("time_idx = chunk * kChunkSize + threadIdx.x + i * kNThreads")
print("time_idx = 0 * 128 + threadIdx.x + i * 32")
print()

for threadIdx_x in range(8):  # Check first 8 threads
    for i in range(4):  # kNItems=4
        time_idx = chunk * kChunkSize + threadIdx_x + i * 32
        print(f"threadIdx.x={threadIdx_x:2d}, i={i}, time_idx={time_idx:3d}", end="")
        if time_idx < seqlen:
            print(f" ✓ VALID")
        else:
            print(f" ✗ INVALID (>={seqlen})")

print("\nOnly threadIdx.x < 4 with i=0 gives valid time_idx!")
print("So for timesteps 1, 2, 3:")
print("  - Timestep 1: threadIdx.x=1, i=0")
print("  - Timestep 2: threadIdx.x=2, i=0")
print("  - Timestep 3: threadIdx.x=3, i=0")
print()
print("So dv_reverse_data[0] for threads 1,2,3 should correspond to timesteps 1,2,3")




