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

                        uint nnzB,
                        uint nRowsB,
                        uint nColsB
                        ) {
    uint global_id = get_global_id(0);

    uint block_id = global_id / nnzB;
    uint elem_id = global_id % nnzB;

    uint rowA = rowsA[block_id];
    uint colA = colsA[block_id];

    uint rowB = rowsB[elem_id];
    uint colB = colsB[elem_id];

    rowsRes[global_id] = nRowsB * rowA + rowB;
    colsRes[global_id] = nColsB * colA + colB;
}



