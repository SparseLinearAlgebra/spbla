#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

__kernel void to_result(__global const unsigned int *indices,
                        unsigned int group_start, // indices_pointers[workload_group_id], workload_group_id = 1

                        __global const unsigned int *c_rows_pointers,
                        __global unsigned int *c_cols_indices,

                        __global const unsigned int *pre_matrix_rows_pointers,
                        __global const unsigned int *pre_matrix_cols_indices

) {
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);

    uint row_pos = group_start + group_id;

    uint row_index = indices[row_pos];
    uint prev_row_start = pre_matrix_rows_pointers[row_index];
    uint new_row_start = c_rows_pointers[row_index];
    uint row_length = c_rows_pointers[row_index + 1] - c_rows_pointers[row_index];


    uint steps = (row_length + GROUP_SIZE - 1) / GROUP_SIZE;

    for (uint i = 0; i < steps; ++i) {
        uint pos_in_row = GROUP_SIZE * i + local_id;
        uint prev_pos = prev_row_start + pos_in_row;
        uint new_pos = new_row_start + pos_in_row;
        if (pos_in_row < row_length) {
            c_cols_indices[new_pos] = pre_matrix_cols_indices[prev_pos];
        }
    }
}
