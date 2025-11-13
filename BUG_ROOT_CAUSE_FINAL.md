# Root Cause Analysis - CUDA NS Backward Bug

## Problem
Timesteps 1-3 have zero NS backward contribution, while timestep 0 works correctly.

## Root Cause
The reverse scan in `selective_scan_bwd_kernel.cuh` accumulates gradients backward through time. For threads 1-3 (timesteps 1-3), only `i=0` corresponds to valid timesteps, while `i=1,2,3` are invalid (out of bounds).

`ThreadReverseReduce` processes items from `[kNItems-1]` down to `[0]`, accumulating with `SSMScanOp`. Even with identity elements `(1, 0)` for invalid items, the scan still processes them, which can cause incorrect accumulation if `thread_reverse_data[0].y` is zero for threads 1-3 after the hidden state reverse scan.

## Attempted Fixes
1. Zero out invalid items in `thread_reverse_data[i].y` before hidden state reverse scan
2. Use identity element `(1, 0)` for invalid items in `dv_reverse_data` before velocity reverse scan

Both fixes didn't resolve the bug, suggesting the issue is deeper:
- Either `thread_reverse_data[0].y` is zero for threads 1-3 after hidden state reverse scan
- OR the block scan doesn't correctly propagate gradients from thread 0 to threads 1-3

## Next Steps
1. Add Python script to verify `grad_X_4_buffer` values after main backward kernel
2. Check if NS backward correctly reads `grad_X_4_buffer` for all timesteps
3. Verify block scan postfix propagation for threads 1-3




