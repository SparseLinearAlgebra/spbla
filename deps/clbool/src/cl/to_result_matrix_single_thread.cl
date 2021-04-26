#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

__kernel void to_result(__global const unsigned int *indices,
                        unsigned int group_start,
                        unsigned int group_length,

                        __global const unsigned int *c_rows_pointers,
                        __global unsigned int *c_cols_indices,

                        __global const unsigned int *pre_matrix_rows_pointers,
                        __global const unsigned int *pre_matrix_cols_indices

) {
    uint global_id = get_global_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;

    if (row_pos >= group_end) return;

    uint a_row_index = indices[row_pos];

    uint prev_pos = pre_matrix_rows_pointers[a_row_index];
    uint new_pos = c_rows_pointers[a_row_index];

    c_cols_indices[new_pos] = pre_matrix_cols_indices[prev_pos];
}
