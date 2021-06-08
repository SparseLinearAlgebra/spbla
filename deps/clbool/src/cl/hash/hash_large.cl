#ifndef RUN

#include "../clion_defines.cl"

#define GROUP_SIZE 32
#define NNZ_ESTIMATION 32

#endif
#define TABLE_SIZE 32
// 4 threads for 4 roes
#define WARP 32 // TODO add define for amd to 64
// how many rows_gpu (tables) can wo process by one threadblock
#define ROWS_PER_TB (GROUP_SIZE / PWARP)
#define HASH_SCAL 107

uint search_global(__global const unsigned int *array, uint value, uint size) {
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

// need to monitor table_size

__kernel void hash_symbolic_tb(__global const unsigned int *indices, // indices -- aka premutation
                                  unsigned int group_start,
                                  unsigned int group_length,

                                  __global
                                  const unsigned int *pre_matrix_rows_pointers, // указатели, куда записывать, или преф сумма по nnz_estimation
                                  __global unsigned int *pre_matrix_cols_indices, // указатели сюда, записываем сюда

                                  __global unsigned int *nnz_estimation, // это нужно обновлять

                                  __global const unsigned int *a_rows_pointers,
                                  __global const unsigned int *a_cols,

                                  __global const unsigned int *b_rows_pointers,
                                  __global const unsigned int *b_rows_compressed,
                                  __global const unsigned int *b_cols,
                                  const unsigned int b_nzr,

                                  // to monitor fails

                                  __global uint *fail_count,
                                  __global uint *failed_rows

) {
    uint hash, old, row_index, a_start, a_end, col_index, b_col, b_rpt, count, border;
    uint local_id = get_local_id(0); // 0 - 255
    uint row_id_bin = get_group_id(0); // 0, 1, ...
    uint warps_per_group = GROUP_SIZE / WARP; // 256 / 32 -- how many vals of A row you can process in ones
    // от 0 до 31
    uint id_in_warp = get_local_id(0) & (WARP - 1); // 0 - 31 , get_global_id(0) & (WARP - 1) == get_global_id(0) % WARP
    uint warp_id = local_id / WARP; // (0 - 255) / 32 -- warp id of thread
    uint row_pos = group_start + row_id_bin; //

    __local uint hash_table[TABLE_SIZE];
    // to monitor table size
    __local volatile uint snz[1];

    if (local_id == 0) {
        snz[0] = 0;
    }

    for (uint i = local_id; i < TABLE_SIZE; i += GROUP_SIZE) {
        hash_table[i] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    row_index = indices[row_pos];
    a_start = a_rows_pointers[row_index];
    a_end = a_rows_pointers[row_index + 1];
    count = 0;
    border = TABLE_SIZE >> 1;
    for (uint a_prt = a_start + warp_id; a_prt < a_end; a_prt += warps_per_group) {
        col_index = a_cols[a_prt]; // позицию этого будем искать в матрице B
        b_rpt = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_rpt == b_nzr) {
            continue;
        }

        for (uint k = b_rows_pointers[b_rpt] + id_in_warp; k < b_rows_pointers[b_rpt + 1]; k += WARP) {
            b_col = b_cols[k];
            // Now go to hashtable and search for b_col
            hash = (b_col * HASH_SCAL) & (TABLE_SIZE - 1);
            while (count < border && snz[0] < border) {
                if (hash_table[hash] == b_col) {
                    break;
                } else if (hash_table[hash] == -1) {
                    old = atom_cmpxchg(hash_table + hash, -1, b_col);
                    if (old == -1) {
                        atomic_add(snz, 1);
                        break;
                    }
                } else {
                    hash = (hash + 1) & (TABLE_SIZE - 1);
                    count ++;
                }
            }
            if (count >= border || snz[0] >= border) {
                break;
            }
        }
        if (count >= border || snz[0] >= border) {
            break;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (count >= border || snz[0] >= border) {
        if (local_id == 0) {
            uint d = atomic_add(fail_count, 1);
            failed_rows[d] = row_id_bin;
        }
    } else {
        if (id_in_warp == 0) {
            nnz_estimation[0] = snz[0];
        }
    }
}