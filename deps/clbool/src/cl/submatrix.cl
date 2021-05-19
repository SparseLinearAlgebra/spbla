#ifndef RUN

#include "clion_defines.cl"

#endif

// submatrix(i, j, nrows, ncols)


uint lower_bound_unique(__global const uint *data, uint data_size, uint val) {
    uint left = 0;
    uint right = data_size;
    if (left == right) return 0;
    uint m;

    while (left + 1 != right) {
        m = left + ((right - left) / 2);
        if (data[m] >= val) {
            right = m;
            continue;
        }
        // we can stop if we find, all values are unique
        if (data[m] == val) {
            return m;
        }
        left = m;
    }
    if (data[left] >= val) {
        return left;
    }
    return right;
}

// в первую очередь нужно понять, где располагатся ряды, может нет смысла выделять nrows памяти
// запишем в массив из 2 элементов, range be like (]
__kernel void rows_range(__global uint *rows_begin_end,
                         __global const uint *m_rows,
                         const uint m_nzr,
                         uint i,
                         uint nrows
) {
    uint local_id = get_local_id(0);
    // need 0 and 1 thread for bsearch
    if (local_id >= 2) return;
    uint to_find = local_id == 0 ? i : i + nrows;
    rows_begin_end[local_id] = lower_bound_unique(m_rows, m_nzr, to_find);
}


// рабочая группа не будет меньше чем 128, поэтому каждый первый
// поток в ворпе можно отправить на поиски границ.
// опять же заифать для AMD
// В эту функцию менее двух ворпов тогда отправлять нельзя
__kernel void submatrix_count_nnz(__global uint *out_rows_nnz, // size = nzr_tmp

                                  __global const uint *m_rpt,
                                  __global const uint *m_rows,
                                  __global const uint *m_cols,
                                  const uint m_nzr,

                                  // positions of submatrix rows in m_rpt
                                  const uint rows_begin,
                                  const uint rows_end,

                                  const uint j, // start for each subrow
                                  const uint ncols // length of each subrow
) {
    uint local_id = get_local_id(0);
    uint global_id = get_global_id(0);
    // temporary nzr for result matrix, final nzr might be less because of empty rows
    uint nzr_tmp = rows_end - rows_begin;
    if (local_id == 0) {
        out_rows_nnz[nzr_tmp] = 0;
    }

    if (global_id >= nzr_tmp) return;

    uint row_pos = rows_begin + global_id; // pos in m_rpt and m_rows
    uint row_start = m_rpt[row_pos];
    uint row_length = m_rpt[row_pos + 1] - row_start;

    uint subrow_start = lower_bound_unique(m_cols + row_start, row_length, j);
    uint subrow_end = lower_bound_unique(m_cols + row_start, row_length, j + ncols);
    out_rows_nnz[global_id] = subrow_end - subrow_start;
}


__kernel void submatrix_fill_nnz(__global const uint *out_rows_ptr,
                                 __global uint *m_cols_out,

                                 __global const uint *m_rpt,
                                 __global const uint *m_cols,

                                 const uint rows_begin,
                                 const uint j
) {
    uint group_id = get_group_id(0);
    uint local_id = get_local_id(0);
    uint wg_size = get_local_size(0);

    uint m_row_pos = rows_begin + group_id; // pos in m_rpt and m_rows
    uint m_row_start = m_rpt[m_row_pos];
    uint m_row_length = m_rpt[m_row_pos + 1] - m_row_start;
    uint subrow_start = lower_bound_unique(m_cols + m_row_start, m_row_length, j);
    uint subrow_length = out_rows_ptr[group_id + 1] - out_rows_ptr[group_id];
    uint shift = m_row_start + subrow_start;
    __global uint *row_out = m_cols_out + out_rows_ptr[group_id];
    for (uint idx = local_id; idx < subrow_length; idx += wg_size) {
        row_out[idx] = m_cols[shift + idx] - j;
    }
}
