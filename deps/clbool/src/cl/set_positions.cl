#ifndef RUN

#include "clion_defines.cl"

#define GROUP_SIZE 256

#endif


__kernel void set_positions(__global uint *newRows,
                            __global uint *newCols,
                            __global const uint *rows,
                            __global const uint *cols,
                            __global const uint *positions,
                            uint size
) {
    uint global_id = get_global_id(0);

    if (global_id == size - 1 && positions[global_id] != size) {
        newRows[positions[global_id]] = rows[global_id];
        newCols[positions[global_id]] = cols[global_id];
        return;
    }

    if (global_id >= size) return;

    if (positions[global_id] != positions[global_id + 1]) {
        newRows[positions[global_id]] = rows[global_id];
        newCols[positions[global_id]] = cols[global_id];
    }
}

__kernel void set_positions1d(__global uint *output,
                              __global const uint *input,
                              __global const uint *positions,
                              uint size
) {
    uint global_id = get_global_id(0);

    if (global_id == size - 1 && positions[global_id] != size) {
        output[positions[global_id]] = input[global_id];
        return;
    }

    if (global_id >= size) return;

    if (positions[global_id] != positions[global_id + 1]) {
        output[positions[global_id]] = input[global_id];
    }
}


__kernel void set_positions_pointers_and_rows(__global uint *newRowsPosition,
                                              __global uint *newRows,
                                              __global const uint *rowsPositions,
                                              __global const uint *rows,
                                              __global const uint *positions,
                                              uint nnz, // old nzr
                                              uint old_nzr,
                                              uint new_nzr
) {
    uint global_id = get_global_id(0);

    if (global_id >= old_nzr) return;

    if (global_id == old_nzr - 1) {
        if (positions[global_id] != old_nzr) {
            newRowsPosition[positions[global_id]] = rowsPositions[global_id];
            newRows[positions[global_id]] = rows[global_id];
        }
        newRowsPosition[new_nzr] = nnz;
        return;
    }

    if (positions[global_id] != positions[global_id + 1]) {
        newRowsPosition[positions[global_id]] = rowsPositions[global_id];
        newRows[positions[global_id]] = rows[global_id];
    }
}

__kernel void set_positions_rows(__global uint *rows_pointers,
                                 __global uint *rows_compressed,
                                 __global const uint *rows,
                                 __global const uint *positions,
                                 uint size,
                                 uint nzr
) {
    uint global_id = get_global_id(0);

    if (global_id == size - 1) {
        if (positions[global_id] != size) {
            rows_pointers[positions[global_id]] = global_id;
            rows_compressed[positions[global_id]] = rows[global_id];
        }
        rows_pointers[nzr] = size;
        return;
    }

    if (global_id >= size) return;

    if (positions[global_id] != positions[global_id + 1]) {
        rows_pointers[positions[global_id]] = global_id;
        rows_compressed[positions[global_id]] = rows[global_id];
    }
}

void __kernel set_positions_with_offset(__global uint *rpt_out,
                                        __global uint *rows_out,
                                        __global const uint *rows_in,
                                        __global const uint *rows_tmp_positions, // преф сумма по новым размерам рядов
                                        __global const uint *positions, // куда писать
                                        uint nzr_tmp,
                                        uint nnz,
                                        uint nzr,
                                        uint rows_begin, // с какого ряда копируем
                                        uint i
) {
    uint global_id = get_global_id(0);
    __global const uint *rows_in_shifted = rows_in + rows_begin;

    if (global_id == nzr_tmp - 1) {
        rpt_out[nzr] = nnz;
    }

    if (global_id >= nzr_tmp) return;

    if (positions[global_id] != positions[global_id + 1]) {
        rpt_out[positions[global_id]] = rows_tmp_positions[global_id];
        rows_out[positions[global_id]] = rows_in_shifted[global_id] - i;
    }
}