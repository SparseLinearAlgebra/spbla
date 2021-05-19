#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif


// TODO: maybe split task to call less threads

__kernel void kronecker(__global uint* rowsRes,
                        __global uint* colsRes,
                        __global const uint* rowsA,
                        __global const uint* colsA,
                        __global const uint* rowsB,
                        __global const uint* colsB,

                        uint rezSize,
                        uint nnzB,
                        uint nRowsB,
                        uint nColsB
                        ) {
    uint global_id = get_global_id(0);

    if (global_id >= rezSize) return;

    uint block_id = global_id / nnzB;
    uint elem_id = global_id % nnzB;

//    __global uint *rowA = rowsAblock_id;
//    __global uint *colA = colsA + block_id;
//
//    __global uint *rowB = rowsB + elem_id;
//    __global uint *colB = colsB + elem_id;

    rowsRes[global_id] = nRowsB * rowsA[block_id] + rowsB[elem_id];
    colsRes[global_id] = nColsB * colsA[block_id] + colsB[elem_id];
}

