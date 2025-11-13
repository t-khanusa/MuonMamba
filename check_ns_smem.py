#!/usr/bin/env python3
"""
Calculate shared memory usage for NS kernel
"""

def calc_smem(dim, dstate):
    kBlockSize = 256
    kTileSize = 64
    
    tile_buffer = kTileSize * dstate * 4  # float
    gram_A_then_B = dstate * dstate * 4  # float
    partial_sums = kBlockSize * 4  # float
    
    total = tile_buffer + gram_A_then_B + partial_sums
    
    print(f"\nDimension: dim={dim}, dstate={dstate}")
    print(f"  tile_buffer ({kTileSize} × {dstate}): {tile_buffer:,} bytes ({tile_buffer/1024:.1f} KB)")
    print(f"  gram_A_then_B ({dstate} × {dstate}): {gram_A_then_B:,} bytes ({gram_A_then_B/1024:.1f} KB)")
    print(f"  partial_sums ({kBlockSize}): {partial_sums:,} bytes ({partial_sums/1024:.1f} KB)")
    print(f"  TOTAL: {total:,} bytes ({total/1024:.1f} KB)")
    
    if total > 48 * 1024:
        print(f"  ❌ EXCEEDS 48 KB limit!")
    else:
        print(f"  ✅ Within 48 KB limit")
    
    return total

print("="*80)
print("Newton-Schulz Kernel Shared Memory Analysis")
print("="*80)

# Test cases
calc_smem(4, 3)
calc_smem(4, 16)
calc_smem(64, 16)
calc_smem(64, 32)
calc_smem(128, 64)
calc_smem(256, 64)

print("\n" + "="*80)
print("Analysis:")
print("="*80)
print("The NS kernel uses:")
print("  - tile_buffer: kTileSize × dstate floats (working buffer for D rows)")
print("  - gram_A_then_B: dstate × dstate floats (Gram matrix, reused for B)")
print("  - partial_sums: kBlockSize floats (for norm reduction)")
print()
print("Critical: gram_A_then_B grows as O(dstate²)!")
print("  dstate=16: 16² × 4 = 1 KB")
print("  dstate=32: 32² × 4 = 4 KB")
print("  dstate=64: 64² × 4 = 16 KB")





