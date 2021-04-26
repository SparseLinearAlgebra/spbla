#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

uint search_global(__global const unsigned int *array,
                   uint value, uint size) {
    uint l = 0;
    uint r = size;
    uint m = l + ((r - l) / 2);
    while (l < r) {
        if (array[m] == value) {
            return m;
        }

        if (array[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }

        m = l + ((r - l) / 2);
    }

    return size;
}

__kernel void copy_one_value(__global const unsigned int *indices,
                             unsigned int group_start, // indices_pointers[workload_group_id], workload_group_id = 1
                             unsigned int group_length,

                             __global const unsigned int *pre_matrix_rows_pointers,
                             __global unsigned int *pre_matrix_cols_indices,

                             __global const unsigned int *a_rows_pointers,
                             __global const unsigned int *a_cols,

                             __global const unsigned int *b_rows_pointers,
                             __global const unsigned int *b_rows_compressed,
                             __global const unsigned int *b_cols,

                             unsigned int b_nzr

) {
    uint global_id = get_global_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;

    if (row_pos >= group_end) return;
    uint a_row_index = indices[row_pos];
    uint start = a_rows_pointers[a_row_index];
    uint end = a_rows_pointers[a_row_index + 1];

    for (uint col_idx = start; col_idx < end; col_idx++) {
        uint col_ptr = a_cols[col_idx];
        uint col_ptr_position = search_global(b_rows_compressed, col_ptr, b_nzr);
        if (col_ptr_position != b_nzr) {
            uint value_pointer = b_rows_pointers[col_ptr_position];
            pre_matrix_cols_indices[pre_matrix_rows_pointers[a_row_index]] = b_cols[value_pointer];
            return;
        }
    }
}
