#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

uint search_global(__global const unsigned int* array, uint value, uint size) {
    uint l = 0;
    uint r = size;
    uint m =  l + ((r - l) / 2);
    while (l < r) {
        if (array[m] == value) {
            return m;
        }

        if (array[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }

        m =  l + ((r - l) / 2);
    }

    return size;
}


__kernel void count_workload(__global uint* nnz_est,
                             __global const uint* a_rpt,
                             __global const uint* a_cols,
                             __global const uint* b_rows,
                             __global const uint* b_rpt,
                             uint a_nzr,
                             uint b_nzr

) {
    uint global_id = get_global_id(0);
    if (global_id >= a_nzr) return;
    // important zeroe value!!!!
    if (global_id == 0) nnz_est[a_nzr] = 0;

    nnz_est[global_id] = 0;
    uint start = a_rpt[global_id];
    uint end = a_rpt[global_id + 1];
    for (uint col_idx = start; col_idx < end; col_idx ++) {
        uint col_ptr = a_cols[col_idx];
        uint col_ptr_pos = search_global(b_rows, col_ptr, b_nzr);
        nnz_est[global_id] += col_ptr_pos == b_nzr ? 0 :
                              b_rpt[col_ptr_pos + 1] - b_rpt[col_ptr_pos];
    }
}


__kernel void count_workload_csr(__global uint* nnz_est,
                                __global const uint* a_rpt,
                                __global const uint* a_cols,
                                __global const uint* b_rpt,
                                uint a_nzr,
                                uint b_nzr

) {
    uint global_id = get_global_id(0);
    if (global_id >= a_nzr) return;
    // important zeroe value!!!!
    if (global_id == 0) nnz_est[a_nzr] = 0;

    nnz_est[global_id] = 0;
    uint start = a_rpt[global_id];
    uint end = a_rpt[global_id + 1];
    for (uint col_idx = start; col_idx < end; col_idx ++) {
        uint col_ptr = a_cols[col_idx];
        nnz_est[global_id] += b_rpt[col_ptr + 1] - b_rpt[col_ptr];
    }
}