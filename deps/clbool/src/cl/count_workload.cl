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


__kernel void count_workload(__global unsigned int* nnz_estimation,
                             __global const unsigned int* a_rows_pointers,
                             __global const unsigned int* a_cols,
                             __global const unsigned int* b_rows_compressed,
                             __global const unsigned int* b_rows_pointers,
                             unsigned int a_nzr,
                             unsigned int b_nzr

) {
    uint global_id = get_global_id(0);
    if (global_id >= a_nzr) return;
    // important zeroe value!!!!
    if (global_id == 0) nnz_estimation[a_nzr] = 0;
    nnz_estimation[global_id] = 0;
    uint start = a_rows_pointers[global_id];
    uint end = a_rows_pointers[global_id + 1];

    for (uint col_idx = start; col_idx < end; col_idx ++) {
        uint col_ptr = a_cols[col_idx];
        uint col_ptr_position = search_global(b_rows_compressed, col_ptr, b_nzr);
        nnz_estimation[global_id] += col_ptr_position == b_nzr ? 0 :
                b_rows_pointers[col_ptr_position + 1] - b_rows_pointers[col_ptr_position];
    }
}
