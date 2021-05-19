
#include "controls.hpp"
#include "kernel.hpp"
#include "../common/utils.hpp"
#include "../common/cl_operations.hpp"
#include "coo_utils.hpp"
#include "coo_matrix_addition.hpp"


namespace clbool::coo {

    void matrix_addition(Controls &controls,
                         matrix_coo &matrix_out,
                         const matrix_coo &a,
                         const matrix_coo &b) {

        cl::Buffer merged_rows;
        cl::Buffer merged_cols;
        uint32_t new_size;

        merge(controls, merged_rows, merged_cols, a, b);

        reduce_duplicates(controls, merged_rows, merged_cols, new_size, a.nnz() + b.nnz());

        matrix_out = matrix_coo(a.nrows(), a.ncols(), new_size, merged_rows, merged_cols);
    }


    void merge(Controls &controls,
               cl::Buffer &merged_rows_out,
               cl::Buffer &merged_cols_out,
               const matrix_coo &a,
               const matrix_coo &b) {

        uint32_t merged_size = a.nnz() + b.nnz();

        auto coo_merge = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        uint32_t, uint32_t>
                        ("merge_path", "merge");
        coo_merge.set_needed_work_size(merged_size);

        cl::Buffer merged_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);
        cl::Buffer merged_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

        coo_merge.run(controls,
                      merged_rows, merged_cols,
                      a.rows_gpu(), a.cols_gpu(),
                      b.rows_gpu(), b.cols_gpu(),
                      a.nnz(), b.nnz());

    //        check_merge_correctness(controls, merged_rows, merged_cols, merged_size);

        merged_rows_out = std::move(merged_rows);
        merged_cols_out = std::move(merged_cols);
    }


    void reduce_duplicates(Controls &controls,
                           cl::Buffer &merged_rows,
                           cl::Buffer &merged_cols,
                           uint32_t &new_size,
                           uint32_t merged_size
    ) {
        // ------------------------------------ prepare array to count positions ----------------------

        cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

        prepare_positions(controls, positions, merged_rows, merged_cols, merged_size);

        // ------------------------------------ calculate positions, get new_size -----------------------------------

        prefix_sum(controls, positions, new_size, merged_size);

        cl::Buffer new_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_size);
        cl::Buffer new_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_size);

        set_positions(controls, new_rows, new_cols, merged_rows, merged_cols, positions, merged_size);

        merged_rows = std::move(new_rows);
        merged_cols = std::move(new_cols);
    }


    void prepare_positions(Controls &controls,
                           cl::Buffer &positions,
                           cl::Buffer &merged_rows,
                           cl::Buffer &merged_cols,
                           uint32_t merged_size
    ) {
        auto prepare_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("prepare_positions", "prepare_array_for_positions");
        prepare_positions.set_needed_work_size(merged_size);

        prepare_positions.run(controls, positions, merged_rows, merged_cols, merged_size);
    }


    void set_positions(Controls &controls,
                       cl::Buffer &new_rows,
                       cl::Buffer &new_cols,
                       cl::Buffer &merged_rows,
                       cl::Buffer &merged_cols,
                       cl::Buffer &positions,
                       uint32_t merged_size) {

        auto set_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>
                ("set_positions", "set_positions");
        set_positions.set_needed_work_size(merged_size);

        set_positions.run(controls, new_rows, new_cols, merged_rows, merged_cols, positions, merged_size).wait();
    }


    //void check_pref_correctness(const std::vector<uint32_t> &result,
    //                            const std::vector<uint32_t> &before) {
    //    uint32_t n = before.size();
    //    uint32_t acc = 0;
    //
    //    for (uint32_t i = 0; i < n; ++i) {
    //        acc = i == 0 ? 0 : before[i - 1] + acc;
    //
    //        if (acc != result[i]) {
    //            throw std::runtime_error("incorrect result");
    //        }
    //    }
    //    std::cout << "correct pref sum, the last value is " << result[n - 1] << std::endl;
    //}


    //// check weak correctness
    //void check_merge_correctness(Controls &controls, cl::Buffer &rows_gpu, cl::Buffer &cols, uint32_t merged_size) {
    //    std::vector<uint32_t> rowsC(merged_size);
    //    std::vector<uint32_t> colsC(merged_size);
    //
    //    controls.queue.enqueueReadBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * merged_size, rowsC.data());
    //    controls.queue.enqueueReadBuffer(cols, CL_TRUE, 0, sizeof(uint32_t) * merged_size, colsC.data());
    //
    //    coo_utils::check_correctness(rowsC, colsC);
    //}
}