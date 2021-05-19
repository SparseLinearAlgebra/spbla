#pragma once
#include "coo.hpp"

#include "controls.hpp"
#include "matrix_coo.hpp"
#include "../common/cl_includes.hpp"
#include "../common/utils.hpp"

namespace clbool::coo {

    void matrix_addition(
            Controls &controls,
            matrix_coo &matrix_out,
            const matrix_coo &a,
            const matrix_coo &b
    );
    //
    //void check_merge_correctness(
    //        Controls &controls,
    //        cl::Buffer &rows_gpu,
    //        cl::Buffer &cols,
    //        uint32_t merged_size
    //);

    void merge(
            Controls &controls,
            cl::Buffer &merged_rows,
            cl::Buffer &merged_cols,
            const matrix_coo &a,
            const matrix_coo &
    );

    void prepare_positions(
            Controls &controls,
            cl::Buffer &positions,
            cl::Buffer &merged_rows,
            cl::Buffer &merged_cols,
            uint32_t merged_size
    );



    void set_positions(
            Controls &controls,
            cl::Buffer &compressed_rows,
            cl::Buffer &new_cols,
            cl::Buffer &merged_rows,
            cl::Buffer &merged_cols,
            cl::Buffer &positions,
            uint32_t merged_size
    );

    void reduce_duplicates(
            Controls &controls,
            cl::Buffer &merged_rows,
            cl::Buffer &merged_cols,
            uint32_t &new_size,
            uint32_t merged_size
    );
}