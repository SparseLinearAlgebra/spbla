#ifndef RUN

#include "clion_defines.cl"

#define GROUP_SIZE 256
#define NNZ_ESTIMATION 32

#endif

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


void bitonic_sort(__local uint *cols, uint fill_pointer) {
    uint local_id = get_local_id(0);

    uint outer = pow(2, ceil(log2((float) fill_pointer)));
    uint segment_length = 2;
    while (outer != 1) {
        uint half_segment_length = segment_length / 2;
        // id внутри сегмента
        uint local_line_id = local_id % half_segment_length;
        // тот, с кем будем сравниваться и меняться в сегменте
        uint local_twin_id = segment_length - local_line_id - 1;
        // id сегмента
        uint group_line_id = local_id / half_segment_length;
        // индексы элементов в массиве
        uint line_id = segment_length * group_line_id + local_line_id;
        uint twin_id = segment_length * group_line_id + local_twin_id;

        if (line_id < fill_pointer && twin_id < fill_pointer && cols[line_id] > cols[twin_id]) {
            uint tmp = cols[line_id];
            cols[line_id] = cols[twin_id];
            cols[twin_id] = tmp;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint j = half_segment_length; j > 1; j >>= 1) {
            uint half_j = j / 2;
            local_line_id = local_id % half_j;
            local_twin_id = local_line_id + half_j;
            group_line_id = local_id / half_j;
            line_id = j * group_line_id + local_line_id;
            twin_id = j * group_line_id + local_twin_id;

            if (line_id < fill_pointer && twin_id < fill_pointer && cols[line_id] > cols[twin_id]) {
                uint tmp = cols[line_id];
                cols[line_id] = cols[twin_id];
                cols[twin_id] = tmp;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        outer >>= 1;
        segment_length <<= 1;
    }


}

void scan(__local uint *positions, __local const uint *cols, uint fill_pointer) {
    uint group_size = get_local_size(0);
    uint local_id = get_local_id(0);
    uint dp = 1;

    positions[local_id] = local_id == 0 ? 1 :
                          (local_id >= fill_pointer) || (cols[local_id] == cols[local_id - 1])
                          ? 0 : 1;

    for (uint s = group_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;
            positions[j] += positions[i];
        }

        dp <<= 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == NNZ_ESTIMATION - 1) {
        positions[NNZ_ESTIMATION] = positions[local_id];
        positions[local_id] = 0;
    }

    for (uint s = 1; s < group_size; s <<= 1) {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;

            unsigned int t = positions[j];
            positions[j] += positions[i];
            positions[i] = t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

void scan_half_sized(__local uint *positions, __local const uint *cols, uint fill_pointer) {
    uint local_id = get_local_id(0);
    uint local_id_second_half = local_id + GROUP_SIZE;
    uint block_size = get_local_size(0);
    uint doubled_block_size = get_local_size(0) * 2;

    positions[local_id] = local_id == 0 ? 1 :
                          (local_id >= fill_pointer) || (cols[local_id] == cols[local_id - 1])
                          ? 0 : 1;

    positions[local_id_second_half] =
            (local_id_second_half >= fill_pointer) || (cols[local_id_second_half] == cols[local_id_second_half - 1])
            ? 0 : 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    uint dp = 1;
    for (uint s = doubled_block_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;
            positions[j] += positions[i];

        }

        dp <<= 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == block_size - 1) {
        positions[NNZ_ESTIMATION] = positions[local_id_second_half];
        positions[local_id_second_half] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = 1; s < doubled_block_size; s <<= 1) {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;
            unsigned int t = positions[j];
            positions[j] += positions[i];
            positions[i] = t;
        }

    }
}

void set_positions(__local const uint *positions, __local const uint *cols, uint fill_pointer,
                   __global uint *result) {
    uint local_id = get_local_id(0);

    if (local_id >= fill_pointer) return;
    if (local_id == fill_pointer - 1 && positions[local_id] != positions[NNZ_ESTIMATION]) {
        result[positions[local_id]] = cols[local_id];
    }
    if (positions[local_id] != positions[local_id + 1]) {
        result[positions[local_id]] = cols[local_id];
    }
}

void set_positions_half_sized(__local const uint *positions, __local const uint *cols, uint fill_pointer,
                              __global uint *result) {
    uint local_id = get_local_id(0);
    uint local_id_second_half = local_id + GROUP_SIZE;

    if (local_id >= fill_pointer) return;

    if (local_id == fill_pointer - 1 && positions[local_id] != positions[NNZ_ESTIMATION]) {
        result[positions[local_id]] = cols[local_id];
    }

    if (local_id_second_half == fill_pointer - 1 && positions[local_id_second_half] != positions[NNZ_ESTIMATION]) {
        result[positions[local_id_second_half]] = cols[local_id_second_half];
    }

    if (positions[local_id] != positions[local_id + 1]) {
        result[positions[local_id]] = cols[local_id];
    }

    if (local_id_second_half >= fill_pointer) return;

    if (positions[local_id_second_half] != positions[local_id_second_half + 1]) {
        result[positions[local_id_second_half]] = cols[local_id_second_half];
    }
}


// Выделить половину от группы на ряд, но не менее 32, ибо менее 32 не имеет смысла
__kernel void bitonic_esc(__global const unsigned int *indices,
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
                          const unsigned int b_nzr
) {

    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0); // главный за индекс ряда
    uint row_pos = group_start + group_id; // какой ряд обрабатывать будем
    uint group_size = get_local_size(0);

    // так как по группе на ряд, можем выйти
    if (row_pos >= group_start + group_length) return;

    __local uint cols[NNZ_ESTIMATION]; //

    // ------------------ fill cols  -------------------

    uint row_index = indices[row_pos];
    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[row_index];

    uint a_start = a_rows_pointers[row_index];
    uint a_end = a_rows_pointers[row_index + 1];
    uint fill_pointer = 0; // сколько удалось заполнить за шаг

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {
        uint col_index = a_cols[a_pointer]; // позицию этого будем искать в матрице B
        uint b_row_pointer = search_global(b_rows_compressed, col_index, b_nzr);

        if (b_row_pointer == b_nzr) {
            continue;
        }

        uint b_start = b_rows_pointers[b_row_pointer];
        uint b_end = b_rows_pointers[b_row_pointer + 1];
        uint b_row_length = b_end - b_start;

        uint steps = (b_row_length + group_size - 1) / group_size;

        for (uint group_step = 0; group_step < steps; ++group_step) {
            // b_start --  где насинается рад
            // group_step -- какой шаг относительно разбиения задачи по размеру группы
            // group_size -- stride размером с группу
            // local_id -- индивидуальный сдвиг потока
            uint elem_id_local = group_step * group_size + local_id;
            uint elem_id = b_start + elem_id_local;
            if (elem_id < b_end) {
                cols[fill_pointer + elem_id_local] = b_cols[elem_id];
            }

        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);
        fill_pointer += b_row_length;
    }

    bitonic_sort(cols, fill_pointer);
    barrier(CLK_LOCAL_MEM_FENCE);

    __local uint positions[NNZ_ESTIMATION + 1];

    if (NNZ_ESTIMATION > group_size) {

        scan_half_sized(positions, cols, fill_pointer);
        barrier(CLK_LOCAL_MEM_FENCE);
        set_positions_half_sized(positions, cols, fill_pointer, result);
    } else {
        scan(positions, cols, fill_pointer);
        barrier(CLK_LOCAL_MEM_FENCE);
        set_positions(positions, cols, fill_pointer, result);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        nnz_estimation[row_index] = positions[NNZ_ESTIMATION];
    }

}
