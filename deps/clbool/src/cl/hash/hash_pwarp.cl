#ifndef RUN

#include "../clion_defines.cl"

#define GROUP_SIZE 32
#define NNZ_ESTIMATION 32

#endif
#define TABLE_SIZE 32
// 4 threads for 4 roes
#define PWARP 4
// how many rows_gpu (tables) can wo process by one threadblock
#define ROWS_PER_TB (GROUP_SIZE / PWARP)
#define HASH_SCAL 107

uint search_global(__global const uint *array, uint value, uint size) {
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

__kernel void hash_symbolic_pwarp(__global const uint *indices,
                                  uint group_start,
                                  uint group_length,

                                  __global uint *nnz_estimation,

                                  __global const uint *a_rows_pointers,
                                  __global const uint *a_cols,

                                  __global const uint *b_rows_pointers,
                                  __global const uint *b_rows_compressed,
                                  __global const uint *b_cols,
                                  const uint b_nzr
) {

    uint hash, old, row_index, a_start, a_end, col_index, b_col, b_rpt;

    uint row_id_bin = get_global_id(0) / PWARP;
    uint local_row_id = row_id_bin & (ROWS_PER_TB - 1); // row_id_bin & (ROWS_PER_TB - 1) == row_id_bin % ROWS_PER_TB
    uint id_in_pwarp = get_global_id(0) & (PWARP - 1); // get_global_id(0) & (PWARP - 1) == get_global_id(0) % PWARP
    uint row_pos = group_start + row_id_bin; // row for pwarp

    __local uint hash_table[ROWS_PER_TB * TABLE_SIZE];
    __local uint nz_count[ROWS_PER_TB * PWARP];
    __local uint *thread_nz = nz_count + (PWARP * local_row_id + id_in_pwarp);
    thread_nz[0] = 0;
    __local uint *local_table = hash_table + (TABLE_SIZE * local_row_id);
    // init hash_table

    for (uint i = id_in_pwarp; i < TABLE_SIZE; i += PWARP) {
        local_table[i] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // if (pwarp_id >= group_start + group_length) return; -- cannot return because of barrier later
    if (row_pos < group_start + group_length) {
        row_index = indices[row_pos];
        a_start = a_rows_pointers[row_index];
        a_end = a_rows_pointers[row_index + 1];

        for (uint a_prt = a_start + id_in_pwarp; a_prt < a_end; a_prt += PWARP) {
            col_index = a_cols[a_prt]; // позицию этого будем искать в матрице B
            b_rpt = search_global(b_rows_compressed, col_index, b_nzr);
            if (b_rpt == b_nzr) {
                continue;
            }

            for (uint k = b_rows_pointers[b_rpt]; k < b_rows_pointers[b_rpt + 1]; ++k) {
                b_col = b_cols[k];
                // Now go to hashtable and search for b_col
                hash = (b_col * HASH_SCAL) & (TABLE_SIZE - 1);
                while (true) {
                    if (local_table[hash] == b_col) {
                        break;
                    } else if (local_table[hash] == -1) {
                        old = atom_cmpxchg(local_table + hash, -1, b_col);
                        if (old == -1) {
                            thread_nz[0]++;
                            break;
                        }
                    } else {
                        hash = (hash + 1) & (TABLE_SIZE - 1);
                    }
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (row_pos >= group_start + group_length) return;


    if (id_in_pwarp == 0) {
        nnz_estimation[row_index] = thread_nz[0] + thread_nz[1] + thread_nz[2] + thread_nz[3];
    }
}


void print_array(__local uint *data, uint size) {
    printf("\n");
    for (uint i = 0; i < size; ++i) {
        printf("%d: %d, ", i, data[i]);
    }
    printf("\n");
}

__kernel void hash_numeric_pwarp(__global const uint *indices,
                                 uint group_start,
                                 uint group_length,

                                 __global
                                 const uint *pre_matrix_rows_pointers,
                                 __global uint *c_cols,

                                 __global const uint *a_rows_pointers,
                                 __global const uint *a_cols,

                                 __global const uint *b_rows_pointers,
                                 __global const uint *b_rows_compressed,
                                 __global const uint *b_cols,
                                 const uint b_nzr
) {

    uint hash, old, row_index, a_start, a_end, col_index, b_col, b_rpt, index;

    uint row_id_bin = get_global_id(0) / PWARP;
    uint local_row_id = row_id_bin & (ROWS_PER_TB - 1); // row_id_bin & (ROWS_PER_TB - 1) == row_id_bin % ROWS_PER_TB
    uint id_in_pwarp = get_global_id(0) & (PWARP - 1); // get_global_id(0) & (PWARP - 1) == get_global_id(0) % PWARP
    uint row_pos = group_start + row_id_bin; // row for pwarp

    __local uint hash_table[ROWS_PER_TB * TABLE_SIZE];
    __local uint c_cols_local[ROWS_PER_TB * TABLE_SIZE];
    __local uint nz_count[ROWS_PER_TB];
    __local uint *local_table = hash_table + (TABLE_SIZE * local_row_id);
    __local uint *c_cols_cur_local = c_cols_local + (TABLE_SIZE * local_row_id);

    if (id_in_pwarp == 0) {
        nz_count[local_row_id] = 0;
    }

    for (uint i = id_in_pwarp; i < TABLE_SIZE; i += PWARP) {
        local_table[i] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // if (pwarp_id >= group_start + group_length) return; -- cannot return because of barrier later
    if (row_pos < group_start + group_length) {
        row_index = indices[row_pos];
        a_start = a_rows_pointers[row_index];
        a_end = a_rows_pointers[row_index + 1];

        for (uint a_prt = a_start + id_in_pwarp; a_prt < a_end; a_prt += PWARP) {
            col_index = a_cols[a_prt]; // позицию этого будем искать в матрице B
            b_rpt = search_global(b_rows_compressed, col_index, b_nzr);
            if (b_rpt == b_nzr) {
                continue;
            }

            for (uint k = b_rows_pointers[b_rpt]; k < b_rows_pointers[b_rpt + 1]; ++k) {
                b_col = b_cols[k];
                // Now go to hashtable and search for b_col
                hash = (b_col * HASH_SCAL) & (TABLE_SIZE - 1);
                while (true) {
                    if (local_table[hash] == b_col) {
                        break;
                    } else if (local_table[hash] == -1) {
                        old = atom_cmpxchg(local_table + hash, -1, b_col);
                        if (old == -1) {
                            break;
                        }
                    } else {
                        hash = (hash + 1) & (TABLE_SIZE - 1);
                    }
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
//    if (get_global_id(0) == 0) {
//        print_array(local_table, TABLE_SIZE);
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);

    if (row_pos < group_start + group_length) {

        // only 4 threads per row, not enough for goof prefix sum

        for (uint i = id_in_pwarp; i < TABLE_SIZE; i += PWARP) {
            if (local_table[i] != -1) {
                index = atomic_add(nz_count + local_row_id, 1);
                c_cols_cur_local[index] = local_table[i];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
//    if (get_global_id(0) == 0) {
//        print_array(c_cols_cur_local, nz_count[local_row_id]);
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);

    if (row_pos >= group_start + group_length) return;

    __global uint *c_cols_cur_global = c_cols + pre_matrix_rows_pointers[row_index];
    // Sorting
    // TODO добавить if ?
    uint nz = nz_count[local_row_id];
    uint count, target;
    for (int i = id_in_pwarp; i < nz; i += PWARP) {
        target = c_cols_cur_local[i];
        count = 0;
        for (uint k = 0; k < nz; ++k) {
            count += (uint) (c_cols_cur_local[k] - target) >> 31;
        }
        c_cols_cur_global[count] = target;
    }

}