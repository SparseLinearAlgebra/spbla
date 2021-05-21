#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256
#define NNZ_ESTIMATION 32

#endif

uint upper_bound_unique(__global const uint *data, uint data_size, uint val) {
    uint left = 0;
    uint right = data_size;
    if (left == right) return 0;
    uint m;

    while (left + 1 != right) {
        m = left + ((right - left) / 2);
        if (data[m] < val) {
            left = m;
            continue;
        }
        // we can stop if we find
        if (data[m] == val) {
            return m + 1;
        }

        right = m;
    }

    // if we less than first value
    if (data[left] > val) {
        return left;
    }

    return right;
}


__kernel void count_nnz_per_row(
        __global uint *c_rpt,
        __global uint *c_rows,

        __global const uint *a_rpt,
        __global const uint *b_rpt,

        __global const uint *a_rows,
        __global const uint *b_rows,

        uint c_nzr,
        uint b_nzr,
        uint b_nrows
        ) {
    uint global_id = get_global_id(0);

    if (global_id == 0) {
        // to get correct total_sum in prefix sum routine
        c_rpt[c_nzr] = 0;
    }

    if (global_id >= c_nzr) return;

    uint a_ridx = global_id / b_nzr;
    uint b_ridx = global_id % b_nzr;

    c_rpt[global_id] = (a_rpt[a_ridx + 1] - a_rpt[a_ridx]) * (b_rpt[b_ridx + 1] - b_rpt[b_ridx]);
    c_rows[global_id] = a_rows[a_ridx] * b_nrows + b_rows[b_ridx];
}


__kernel void calculate_kronecker_product(
        __global const uint *c_rpt,
        __global uint *c_cols,

        __global const uint *a_rpt,
        __global const uint *b_rpt,

        __global const uint *a_cols,
        __global const uint *b_cols,

        uint b_nzr,
        uint c_nnz,
        uint c_nzr,
        uint b_ncols
        ) {

    uint global_id = get_global_id(0);
    if (global_id >= c_nnz) return;
    uint row_idx = upper_bound_unique(c_rpt, c_nzr + 1, global_id) - 1;
    uint col_idx = global_id - c_rpt[row_idx];
    uint a_ridx = row_idx / b_nzr;
    uint b_ridx = row_idx % b_nzr;

    uint b_rlen = b_rpt[b_ridx + 1] - b_rpt[b_ridx];

    uint a_cidx = col_idx / b_rlen;
    uint b_cidx = col_idx % b_rlen;

    c_cols[global_id] = a_cols[a_rpt[a_ridx] + a_cidx] * b_ncols + b_cols[b_rpt[b_ridx] + b_cidx];
}