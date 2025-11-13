template<typename input_t, typename weight_t, int kBlockSize = 256, int kTileSize = 64>
__global__ void newton_schulz_velocity_5step_backward_kernel(
    const input_t* __restrict__ u,        // [B, D, L]
    const input_t* __restrict__ delta,    // [B, D, L]
    const weight_t* __restrict__ Bmat,    // [D, N] or [B,G,L,N]
    const float* __restrict__ velocity_out, // forward X_final stored as BF16-as-float [B,D,L,N]
    const float* __restrict__ grad_out,     // upstream dL/dX5 [B,D,L,N] (FP32)
    float* __restrict__ grad_u_global,      // output grad_u accumulator (FP32)
    float* __restrict__ grad_delta_global,  // output grad_delta accumulator (FP32)
    float* __restrict__ grad_B_global,      // output grad_B accumulator (FP32)
    float alpha,
    // strides and shape params:
    int B_dim, int D, int L, int dstate,
    int u_batch_stride, int u_d_stride,
    int delta_batch_stride, int delta_d_stride,
    int B_batch_stride, int B_group_stride,
    int B_d_stride, int B_dstate_stride,
    bool is_variable_B, int n_groups,
    float eps
) {
    const int batch_idx = blockIdx.x;
    const int t_local = blockIdx.y;
    const int time_idx = /* t_start */ + t_local; // set t_start appropriately

    if (batch_idx >= B_dim || time_idx >= L) return;

    const int tid = threadIdx.x;

    // compute logical sizes
    const bool transposed = (D > dstate);
    const int gram_size = transposed ? dstate : D; // m = min(M,N) used by forward
    const int M = D;     // number of rows in storage for a block; forward used rows D
    const int N = dstate;

    // ---------- Shared memory layout ----------
    extern __shared__ float smem[]; 
    // We'll reinterpret memory into typed regions:
    // region 0: tile_buffer_bf16 (as __nv_bfloat16) - size = actual_tile_buffer_size (in BF16 elements)
    // region 1: gram_A_fp32  (float) size = gram_size * gram_size
    // region 2: partial_sums (float) size = kBlockSize (for reductions)
    // region 3: dA_fp32 buffer (float) size = gram_size * gram_size (may reuse gram_A_fp32 if careful)
    // region 4: small per-row partial accumulators (float) size = kTileSize (for grad_u, grad_delta)
    // We'll layout: tile_bf16 | gram_A_fp32 | partial_sums | grad_row_partial
    __nv_bfloat16* tile_bf16 = (__nv_bfloat16*)smem; // note size measured in BF16 elements
    float* gram_A_fp32 = (float*)(tile_bf16 + actual_tile_buffer_size); // actual_tile_buffer_size must be computed host-side
    float* partial_sums = gram_A_fp32 + gram_size * gram_size;
    float* grad_row_partial = partial_sums + kBlockSize; // length at least kTileSize

    // For convenience, small helper lambdas (pseudocode)
    auto get_velocity_idx = [&](int row, int col) {
        // identical indexing as forward:
        return batch_idx * D * L * dstate + row * L * dstate + time_idx * dstate + col;
    };
    auto get_u_idx = [&](int row) {
        return batch_idx * u_batch_stride + row * u_d_stride + time_idx;
    };
    auto get_delta_idx = [&](int row) {
        return batch_idx * delta_batch_stride + row * delta_d_stride + time_idx;
    };

    // ---------- 1) RECOMPUTE X0..X4 (forward-style, 4 iterations) ----------
    // We'll compute b_t (FP32), convert to BF16 then recompute normalization / X0 and run 4 NS iterations
    // To reproduce forward exactly, do BF16 conversion at same points and compute norm from BF16-rounded values.

    // initialize local accumulators for norm (two: one from BF16-rounded values and optionally FP32)
    float local_norm_sq_bf16 = 0.0f;
    float local_norm_sq_fp32 = 0.0f;

    // We will iteratively fill the tile buffers with BF16 values for recompute.
    // Loop over D in tiles of kTileSize (same pattern as forward)
    for (int d_start = 0; d_start < D; d_start += kTileSize) {
        int d_end = min(d_start + kTileSize, D);
        int tile_rows = d_end - d_start;

        // Each thread loads some entries (tile_rows * N) into local registers,
        // computes b_t_val = alpha * delta * B * u per element, then converts to BF16 and stores into velocity_out-like temp
        // Since forward stored BF16-rounded b_t into velocity_ortho before NS, we must reproduce same rounding.
        for (int idx = tid; idx < tile_rows * N; idx += kBlockSize) {
            int local_row = idx / N;
            int col = idx % N;
            int global_row = d_start + local_row;

            // load u and delta
            float u_val = to_float(u[get_u_idx(global_row)]);
            float delta_val = to_float(delta[get_delta_idx(global_row)]);
            // load B value (handle constant vs variable)
            float B_val;
            if (!is_variable_B) {
                B_val = to_float(Bmat[global_row * B_d_stride + col * B_dstate_stride]);
            } else {
                int group_size = (D + n_groups - 1) / n_groups;
                int group_id = min(global_row / group_size, n_groups - 1);
                // layout should match forward's variable B indexing
                B_val = to_float(Bmat[batch_idx * B_batch_stride + group_id * B_group_stride + time_idx * dstate + col]);
            }

            float b_t_fp32 = alpha * delta_val * B_val * u_val;

            // record FP32 norm (for optional debug)
            local_norm_sq_fp32 += b_t_fp32 * b_t_fp32;

            // Convert to BF16 (rounding) and back to FP32 to recreate BF16-rounded value
            __nv_bfloat16 b_t_bf16 = FP32_TO_BF16(b_t_fp32);
            float b_t_bf16_as_fp32 = BF16_TO_FLOAT(b_t_bf16);

            // accumulate BF16-norm
            local_norm_sq_bf16 += b_t_bf16_as_fp32 * b_t_bf16_as_fp32;

            // store BF16-rounded value temporarily into velocity_out-like place (we can use a temp global scratch or a per-block buffer)
            // To keep memory local, we write directly into the same velocity_out buffer but we should NOT overwrite forward outputs
            // For recomputation we can write to a local shared buffer location or reuse velocity_out if safe.
            int store_idx = get_velocity_idx(global_row, col);
            // store BF16-rounded values as float to shared/global (match forward)
            // NOTE: in your forward you wrote these to velocity_ortho before NS — replicate same store to preserve bit-exactness
            // We'll write into velocity_out (assuming it's safe to overwrite or you have a separate buffer)
            // For safety, write to shared memory tile_bf16 as BF16 representation:
            tile_bf16[local_row * N + col] = float_to_bf16_reinterpret(b_t_bf16_as_fp32);
        } // idx load loop
        __syncthreads();

        // Reduce local_norm_sq_bf16 across block -> partial_sums[tid], then block reduction to get norm for entire matrix
        // We'll do reductions after all tiles processed, for simplicity accumulate tile contributions into partial_sums.
        // Here we store partial per-thread
        // We'll not perform final sqrt yet; after all tiles processed we'll do global block reduction.
    } // d_start tile loop

    // Now perform block-level reduction to compute global norm_bf16 (over full matrix elements handled by block)
    // Each thread must accumulate the partial sums it computed across tiles into partial_sums[tid]
    partial_sums[tid] = local_norm_sq_bf16;   // store local into shared
    __syncthreads();
    // tree-reduce
    for (int stride = kBlockSize>>1; stride > 0; stride >>= 1) {
        if (tid < stride) partial_sums[tid] += partial_sums[tid + stride];
        __syncthreads();
    }
    float norm_bf16 = sqrtf(partial_sums[0] + 1e-8f);

    // (Optionally compute norm_fp32 from local_norm_sq_fp32 and compare / debug)
    // We now have the BF16-based norm used in forward.

    // ---------- Normalize to get X0 (BF16-rounded values) ----------
    // We'll overwrite the tile_bf16 locations with normalized BF16 values (matching forward: normalized then BF16 rounded)
    for (int d_start = 0; d_start < D; d_start += kTileSize) {
        int d_end = min(d_start + kTileSize, D);
        int tile_rows = d_end - d_start;

        for (int idx = tid; idx < tile_rows * N; idx += kBlockSize) {
            int local_row = idx / N;
            int col = idx % N;
            int global_row = d_start + local_row;

            // read BF16 value from tile_bf16 (we stored it earlier)
            __nv_bfloat16 bf16_val = tile_bf16[local_row * N + col];
            float bf16_fp32 = BF16_TO_FLOAT(bf16_val);

            // normalize in FP32
            float normalized = bf16_fp32 / norm_bf16;

            // round normalized to BF16 (forward does this)
            __nv_bfloat16 normalized_bf16 = FP32_TO_BF16(normalized);
            float normalized_bf16_as_fp32 = BF16_TO_FLOAT(normalized_bf16);

            // write back to tile_bf16 (as BF16)
            tile_bf16[local_row * N + col] = float_to_bf16_reinterpret(normalized_bf16_as_fp32);
        }
        __syncthreads();
    }

    // At this point tile_bf16 contains X0 tiles (BF16-coded) for the last processed tile only.
    // For the full recomputation of X1..X4 we must run 4 NS iterations over the entire matrix,
    // using the same tiled strategy as forward. We'll use tile_bf16 to hold local tiles during each iteration,
    // and gram_A_fp32 to accumulate Gram matrices as forward did.

    // ---------- Run 4 NS iterations (detached) to produce X4 ----------
    for (int step = 0; step < 4; ++step) {
        // zero gram_A_fp32
        for (int idx = tid; idx < gram_size * gram_size; idx += kBlockSize) {
            gram_A_fp32[idx] = 0.0f;
        }
        __syncthreads();

        if (!transposed) {
            // Fat/square case: A is D x D
            // Tile over D rows, same as forward: for each tile load BF16 values (normalized or current X)
            for (int d_start = 0; d_start < D; d_start += kTileSize) {
                int d_end = min(d_start + kTileSize, D);
                int tile_rows = d_end - d_start;

                // Load X tile into tile_bf16 (BF16 element stored in shared)
                for (int idx = tid; idx < tile_rows * N; idx += kBlockSize) {
                    int lr = idx / N;
                    int col = idx % N;
                    int global_row = d_start + lr;

                    // get stored BF16-normalized value: either we kept it in velocity_out or tile_bf16 global scratch
                    // For recomputation we reconstruct from velocity_out if necessary:
                    int global_idx = get_velocity_idx(global_row, col);
                    // velocity_out contains BF16-rounded floats (forward stored into it)
                    float stored_val = velocity_out[global_idx];          // BF16-as-FP32
                    tile_bf16[lr * N + col] = float_to_bf16_reinterpret(stored_val);
                }
                __syncthreads();

                // compute partial A contributions for rows in this tile
                // accumulate A[i,j] += sum_k X[i,k] * X[j,k] (accumulate in FP32)
                for (int i_local = tid; i_local < tile_rows; i_local += kBlockSize) {
                    int i_global = d_start + i_local;
                    for (int j = 0; j < gram_size; ++j) {
                        float sum = 0.0f;
                        // iterate k over cols N
                        for (int k = 0; k < N; ++k) {
                            float xi = BF16_TO_FLOAT(tile_bf16[i_local * N + k]);
                            // X[j,k] may not be in tile; load from global storage (BF16-as-FP32)
                            int idx_jk = get_velocity_idx(j, k); // careful mapping if j outside tile
                            float xjk = velocity_out[idx_jk];
                            sum += xi * xjk;
                        }
                        // atomicAdd into gram_A_fp32[i_global * gram_size + j] (we accumulate across tiles)
                        atomicAdd(&gram_A_fp32[i_global * gram_size + j], sum);
                    }
                }
                __syncthreads();
            } // d_start tiles
        } else {
            // Tall (transposed) case: similar but tile layout differs; follow forward's logic
            // ... (omitted for brevity; implement same tiling approach as forward)
        }

        // convert A (FP32) to BF16 in-place if forward uses BF16 A; also compute A^2 in BF16 then compute B = b*A + c*A^2
        // For backward recomputation we must produce B4 and possibly store both A4_fp32 & A4_bf16 for use in gradient computation.
        // Convert A_fp32 -> gram_A_bf16 in shared memory
        // compute A2 (A @ A) with BF16 inputs but FP32 accumulation (store as FP32 maybe)
        // compute B4_bf16 = FP32_TO_BF16(b * A_ij + c * A2_ij) exactly as forward
        // Then update X = a*X + B@X (as forward) and store results back to velocity_out (BF16-as-FP32)
        // Repeat for next iteration
        // IMPORTANT: must do BF16 rounding at same points as forward

        // -- convert to BF16, compute A^2, B, update X (tile loops, identical to forward) --
        // (Keep implementation identical to forward; for brevity not repeated fully here.)
    } // 4 iterations => now have X4 stored in the same storage as forward's X after 4th iteration

    __syncthreads();

    // ---------- Now we have X4 (BF16-as-FP32) in velocity_out or in shared memory; begin BACKPROP of 5th step ----------

    // 1) Load G5 (upstream gradient) for this block (elements for rows 0..D-1, cols 0..N-1)
    // Promote grad_out entries to FP32 and write into shared memory buffer if needed
    // Compute dX_direct = a * G5 + B.T @ G5  (we'll compute B.T @ G5 via GEMM-style tiling)
    // We'll also compute grad_B (dB) = G5 @ X4.T
    // and then grad_A etc.

    // For simplicity, do tiled GEMM for dB and dX_direct using same tile strategy:
    // zero local accumulators for dX (per element), grad_B_partial for elements, and per-row grad_u, grad_delta partials.
    // We'll accumulate in FP32.

    // ZERO local accumulators in shared memory (dX_fp32 per element maybe large; we instead compute dX per element on the fly)
    // We'll compute grad_B per element and atomicAdd to global, compute grad_u and grad_delta per row via reduction.

    // ==== Compute grad_B = G5 @ X4.T ====
    // Tile loops over rows i (0..gram_size-1) & cols j (0..gram_size-1) similar to forward GEMM, but computing G5 @ X4.T
    for (int i_start = 0; i_start < gram_size; i_start += kTileSize) {
        int i_end = min(i_start + kTileSize, gram_size);
        for (int j_start = 0; j_start < gram_size; j_start += kTileSize) {
            int j_end = min(j_start + kTileSize, gram_size);

            // Each thread computes partial product for block of size [tile_i x tile_j]
            // load required subtiles of G5 and X4 into shared memory BF16/FP32 as in forward
            // accumulate partials in FP32
            // finally atomicAdd per-element (i,j) into grad_B_global (or accumulate in per-block buffer then atomicAdd)
            for (int ii = i_start; ii < i_end; ++ii) {
                for (int jj = j_start; jj < j_end; ++jj) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; ++k) {
                        // G5 element (ii,k) and X4 element (jj,k)
                        int idx_g5 = batch_idx * D * L * dstate + ii * L * dstate + time_idx * dstate + k;
                        float g5 = grad_out[idx_g5]; // FP32
                        int idx_x4 = get_velocity_idx(jj, k); // X4 stored as BF16-as-float
                        float x4_val = velocity_out[idx_x4]; // BF16-as-FP32
                        sum += g5 * x4_val;
                    }
                    // Now sum is dB_partial for element (ii,jj)
                    // Multiply by any scalar if B had coefficients? No, it's grad_B = G5 @ X4.T directly
                    // We may need to scale with device constants b,c later for grad_A; but grad_B is dB_4
                    // Accumulate into grad_B_global: do atomicAdd
                    int gB_idx = batch_idx * D * gram_size + ii * gram_size + jj; // adjust layout to match grad_B_global storage
                    atomicAdd(&grad_B_global[gB_idx], sum);
                }
            }
            __syncthreads();
        }
    }

    // ==== Compute grad_X_direct = a * G5 + B.T @ G5 ====
    // Compute B.T @ G5 using tile GEMM: for each row r and col c compute sum_k B^T[r,k] * G5[k,c]
    // Then add a * G5
    // Store result in local per-element result dX_fp32 (write back to global dX buffer or use on the fly)
    // (Omitted detailed loop to avoid redundancy; use tiled GEMM same as forward.)

    // ==== Compute grad_A (using grad_B) ====
    // We have grad_B available (in grad_B_global) but we likely want it in shared memory for current block:
    // Load relevant submatrix grad_B_block into shared memory and compute:
    // grad_A = b * grad_B + c * (A.T @ grad_B + grad_B @ A.T)
    // Implement with tiled GEMMs; accum in FP32 into a shared grad_A_fp32 buffer.

    // ==== Compute grad_X_from_A = (grad_A + grad_A.T) @ X4 ====
    // Another GEMM: compute (grad_A + grad_A.T) multiplied by X4
    // Accumulate with grad_X_direct to obtain G4_fp32 per element

    // ==== Now normalization backward: compute grad_b per element (FP32) ====
    // Need u_bf16_as_fp32 values — these are the BF16-rounded b_t entries you recomputed earlier.
    // We can read them from velocity_out pre-NS storage if you preserved them, or recompute them again (cheap).
    // We'll compute dot = sum(u * G4) and s = sqrt(sum(u*u)) via reductions.

    // First compute dot and s partials per-thread across assigned subset of elements
    float local_dot = 0.0f;
    float local_u_sq = 0.0f;
    // Each thread processes a subset of matrix entries; accumulate u*G4 and u*u
    for (int row = tid_tile_start; row < tid_tile_end; row += kBlockSize) {
        for (int col = 0; col < N; ++col) {
            int idx = get_velocity_idx(row, col);
            float u_elem = velocity_out[idx];    // BF16-as-FP32, this is u
            float g4_elem = /* read G4_fp32 at (row,col) computed earlier */;
            local_dot += u_elem * g4_elem;
            local_u_sq += u_elem * u_elem;
        }
    }
    // block reduce local_dot, local_u_sq into partial_sums[tid], then root thread computes dot and s
    partial_sums[tid] = local_dot;
    __syncthreads();
    for (int stride = kBlockSize>>1; stride > 0; stride >>= 1) {
        if (tid < stride) partial_sums[tid] += partial_sums[tid + stride];
        __syncthreads();
    }
    float dot = partial_sums[0];

    partial_sums[tid] = local_u_sq;
    __syncthreads();
    for (int stride = kBlockSize>>1; stride > 0; stride >>= 1) {
        if (tid < stride) partial_sums[tid] += partial_sums[tid + stride];
        __syncthreads();
    }
    float s_val = sqrtf(partial_sums[0] + 1e-12f);
    float norm = s_val + eps;

    // Now compute grad_b per element:
    // If s_val small, grad_b = G4 / norm; else grad_b = G4 / norm - u * (dot / (s * norm * norm))
    for (int row = tid_row_start; row < tid_row_end; row += kBlockSize) {
        for (int col = 0; col < N; ++col) {
            int idx = get_velocity_idx(row, col);
            float u_elem = velocity_out[idx];        // BF16-as-FP32
            float g4_elem = /* G4 at (row,col) */;
            float grad_b_elem;
            if (s_val < 1e-12f) {
                grad_b_elem = g4_elem / norm;
            } else {
                grad_b_elem = g4_elem / norm - u_elem * (dot / (s_val * norm * norm));
            }
            // Save grad_b_elem into a shared buffer or process chain to grad_u, grad_delta, grad_B
        }
    }

    // ==== Finally chain grad_b to grad_u, grad_delta, and grad_B ====
    // For each row (global_row), accumulate:
    //   grad_u[row]     += alpha * delta[row] * sum_col( B[row,col] * grad_b[row,col] )
    //   grad_delta[row] += alpha * sum_col( B[row,col] * u[row] * grad_b[row,col] ) = alpha * u[row] * sum_col( B[row,col] * grad_b[row,col] )
    //   grad_B[row,col] += alpha * delta[row] * u[row] * grad_b[row,col]   (atomicAdd per element)
    //
    // Implementation:
    // - Each thread computes partial sums for a set of rows/cols and writes to per-row shared partial accumulators
    // - After per-row reduction, thread 0 atomically adds to global grad_u and grad_delta.
    // - For grad_B, each computed grad_B_elem uses atomicAdd into grad_B_global index.

    // Example compute loop:
    for (int row = 0; row < D; ++row) {
        float partial_sum_grad_u = 0.0f;
        float partial_sum_grad_delta = 0.0f;
        float u_row = to_float(u[get_u_idx(row)]);
        float delta_row = to_float(delta[get_delta_idx(row)]);
        for (int col = tid; col < N; col += kBlockSize) {
            // B_val depends on variable B or constant
            float B_val;
            if (!is_variable_B) {
                B_val = to_float(Bmat[row * B_d_stride + col * B_dstate_stride]);
            } else {
                int group_size = (D + n_groups - 1) / n_groups;
                int group_id = min(row / group_size, n_groups - 1);
                B_val = to_float(Bmat[batch_idx * B_batch_stride + group_id * B_group_stride + time_idx * dstate + col]);
            }
            float grad_b_elem = /* obtained earlier */;
            partial_sum_grad_u += B_val * grad_b_elem;
            partial_sum_grad_delta += B_val * u_row * grad_b_elem;

            // grad_B_elem = alpha * delta_row * u_row * grad_b_elem
            float grad_B_elem = alpha * delta_row * u_row * grad_b_elem;
            int gB_global_idx = /* layout mapping */;
            atomicAdd(&grad_B_global[gB_global_idx], grad_B_elem);
        }
        // now each thread has partial sums for this row; do block reduce across threads into grad_row_partial[threadIdx]
        grad_row_partial[tid] = partial_sum_grad_u; // store for reduction
        __syncthreads();
        // tree reduce into grad_row_partial[0]
        for (int stride = kBlockSize>>1; stride > 0; stride >>= 1) {
            if (tid < stride) grad_row_partial[tid] += grad_row_partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            float total_grad_u = alpha * delta_row * grad_row_partial[0];
            atomicAdd(&grad_u_global[batch_idx * D + row], total_grad_u);
        }

        // Similarly reduce partial_sum_grad_delta into grad_delta_global
        grad_row_partial[tid] = partial_sum_grad_delta;
        __syncthreads();
        for (int stride = kBlockSize>>1; stride > 0; stride >>= 1) {
            if (tid < stride) grad_row_partial[tid] += grad_row_partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            float total_grad_delta = alpha * u_row * grad_row_partial[0];
            atomicAdd(&grad_delta_global[batch_idx * D + row], total_grad_delta);
        }
        __syncthreads();
    }

    // End of kernel
}
