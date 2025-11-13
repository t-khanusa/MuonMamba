#!/usr/bin/env python3
"""
Understanding the reverse scan accumulation order issue
"""

# For thread 1 (timestep 1), with kNItems=4:
# input[0] = (beta, dx1) - valid
# input[1,2,3] = (beta, 0) - invalid (zeros)

# ThreadReverseScanInclusive processes from i=3 down to i=0:
# inclusive starts as postfix (from block scan)
# i=3: inclusive = scan_op(postfix, input[3]) = scan_op(postfix, (beta, 0))
# i=2: inclusive = scan_op(inclusive, input[2]) = scan_op(..., (beta, 0))
# i=1: inclusive = scan_op(inclusive, input[1]) = scan_op(..., (beta, 0))
# i=0: inclusive = scan_op(inclusive, input[0]) = scan_op(..., (beta, dx1))

# With SSMScanOp((a0, b0), (a1, b1)) = (a1*a0, a1*b0 + b1):
# If postfix = (beta, dx_from_thread_0), then:
# i=3: (beta, dx_from_thread_0) -> scan_op -> (beta^2, beta*dx_from_thread_0)
# i=2: (beta^2, beta*dx_from_thread_0) -> scan_op -> (beta^3, beta^2*dx_from_thread_0)
# i=1: (beta^3, beta^2*dx_from_thread_0) -> scan_op -> (beta^4, beta^3*dx_from_thread_0)
# i=0: (beta^4, beta^3*dx_from_thread_0) -> scan_op((beta, dx1)) -> (beta^5, beta^4*dx_from_thread_0 + beta*dx1)

# So output[0] = (beta^5, beta^4*dx_from_thread_0 + beta*dx1)

# This is WRONG! output[0] should be (beta^2, beta*dx_from_thread_0 + dx1) 
# (accumulation from timestep 1 backward: timestep 1 itself + timestep 0)

# The issue is that invalid items (i=1,2,3) are being accumulated, multiplying beta unnecessarily.

# FIX: For threads where only i=0 is valid, we should extract the value BEFORE the scan processes invalid items,
# OR we need to ensure invalid items don't multiply beta.

# Actually wait - the scan processes items in reverse order (from high index to low index).
# So for thread 1:
# - i=3: processes first (invalid, zero)
# - i=0: processes last (valid)

# The postfix from block scan accumulates from thread 1 backward through thread 0.
# So thread 1's postfix = accumulation from threads 1, 0.

# Then ThreadReverseScanInclusive accumulates items within thread 1's array from i=3 down to i=0.

# So output[0] = accumulation from thread 1's item 0, plus contributions from thread 0 (from postfix).

# This should be correct! But maybe the issue is that the block scan doesn't correctly accumulate from thread 0?

# OR maybe the issue is that thread_reverse_data[0].y is zero for threads 1-3 AFTER the hidden state reverse scan?

print("The reverse scan should work correctly IF:")
print("1. thread_reverse_data[0].y is non-zero for threads 1-3 (after hidden state reverse scan)")
print("2. The block scan correctly accumulates from thread 0")

print("\nIf thread_reverse_data[0].y is ZERO for threads 1-3, then:")
print("- dv_reverse_data[0] = (beta, 0) before reverse scan")
print("- After reverse scan, output[0] will still have contributions from thread 0 (via postfix)")
print("- But if thread_reverse_data[0].y is zero, there's no local contribution from this timestep")
print("- So output[0] only has contributions from future timesteps (thread 0), not from this timestep")

print("\nThis means the bug is likely that thread_reverse_data[0].y is ZERO for threads 1-3!")
print("This would happen if the hidden state reverse scan doesn't accumulate correctly for those threads.")




