# Root Cause Analysis - Reverse Scan Bug

## Problem
Timesteps 1-3 have zero NS backward contribution, while timestep 0 works correctly.

## Root Cause Hypothesis
The reverse scan accumulates gradients backward through time. For threads 1-3, `dv_reverse_data[0]` corresponds to valid timesteps 1-3, but `dv_reverse_data[1,2,3]` are invalid (zero or out of bounds).

When `ThreadReverseScanInclusive` processes items from `i=kNItems-1` down to `i=0`:
- For thread 1 (timestep 1): only `input[0]` is valid
- `ThreadReverseReduce` reduces: `scan_op(scan_op(scan_op(input[3], input[2]), input[1]), input[0])`
  - Since input[1,2,3] have zeros, this multiplies beta unnecessarily
- Then `ThreadReverseScanInclusive` scans with this `thread_postfix`, causing incorrect accumulation

## The Fix
We need to ensure that invalid items don't interfere with the scan. The identity element `(1, 0)` doesn't help because it still multiplies beta.

Actually, wait - with `SSMScanOp((1, 0), (beta, g)) = (beta*1, beta*0 + g) = (beta, g)`, the identity should preserve the value. But the issue is that the scan accumulates in reverse order, so invalid items still get processed.

The REAL fix: We should only process valid items in the scan. But the scan processes all items in the array.

Actually, I think the issue is simpler: Maybe `thread_reverse_data[0].y` is zero AFTER the hidden state reverse scan for threads 1-3. This would happen if the hidden state reverse scan doesn't accumulate correctly.

Let me check if the hidden state reverse scan has the same issue.
