#include "dcsr.hpp"

#include "../cl/headers/submatrix.h"
#include "../cl/headers/prepare_positions.h"
#include "../cl/headers/set_positions.h"

#define FILL_WG_SIZE 128 // work group size for each row in fill_rows

namespace clbool {

    namespace sbm_delails {

        // count [begin, end) of target rows_gpu
        void find_rows_range(Controls &controls, uint32_t &rows_begin, uint32_t &rows_end,
                             const matrix_dcsr &matrix_in, uint32_t i, uint32_t nrows) {

            cl::Buffer rows_begin_end_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * 2);

            auto find_range_program = program<cl::Buffer, cl::Buffer,
                    uint32_t, uint32_t, uint32_t>(submatrix_kernel, submatrix_kernel_length);
            find_range_program.set_kernel_name("rows_range")
                    .set_needed_work_size(2)
                    .set_block_size(WARP_SIZE);

            find_range_program.run(controls, rows_begin_end_gpu, matrix_in.rows_gpu(), matrix_in.nzr(), i, nrows).wait();

            cpu_buffer rows_begin_end_cpu(2);
            controls.queue.enqueueReadBuffer(rows_begin_end_gpu, CL_TRUE, 0, sizeof(uint32_t) * 2,
                                             rows_begin_end_cpu.data());

            rows_begin = rows_begin_end_cpu[0];
            rows_end = rows_begin_end_cpu[1];

            LOG << "GPU rows_begin = " << rows_begin << ", rows_end = " << rows_end;
        }

        void count_subrows_nnz(Controls &controls, cl::Buffer &subrows_nnz, const matrix_dcsr &matrix_in,
                               uint32_t j, uint32_t ncols,
                               uint32_t rows_begin, uint32_t rows_end) {

            uint32_t nzr_tmp = rows_end - rows_begin;
            subrows_nnz = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr_tmp + 1));
            auto count_program = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
                    (submatrix_kernel, submatrix_kernel_length);

            count_program.set_block_size(controls.block_size)
                    .set_kernel_name("submatrix_count_nnz")
                    .set_needed_work_size(nzr_tmp);

            count_program.run(controls, subrows_nnz,
                              matrix_in.rpt_gpu(), matrix_in.rows_gpu(), matrix_in.cols_gpu(), matrix_in.nzr(),
                              rows_begin, rows_end, j, ncols).wait();
        }

        void
        fill_rows(Controls &controls, cl::Buffer &cols_out, const matrix_dcsr &matrix_in, const cl::Buffer &subrows_nnz,
                  uint32_t nnz_out, uint32_t nzr_tmp,
                  uint32_t rows_begin, uint32_t j
        ) {
            cols_out = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nnz_out);

            auto fill_rows_nnz = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    uint32_t, uint32_t>(submatrix_kernel, submatrix_kernel_length);

            fill_rows_nnz.set_needed_work_size(nzr_tmp * FILL_WG_SIZE)
                    .set_block_size(FILL_WG_SIZE)
                    .set_kernel_name("submatrix_fill_nnz");
            fill_rows_nnz.run(controls, subrows_nnz, cols_out, matrix_in.rpt_gpu(), matrix_in.cols_gpu(),
                              rows_begin, j).wait();
        }

        void rpt_and_rows(Controls &controls, cl::Buffer &rpt_out, cl::Buffer &rows_out, uint32_t &nzr_out,
                          const matrix_dcsr &matrix_in, const cl::Buffer &subrows_nnz,
                          uint32_t nzr_tmp, uint32_t nnz_out, uint32_t rows_begin, uint32_t i
        ) {
            // prepare positions

            cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr_tmp + 1));

            auto prepare_pos_program = program<cl::Buffer, cl::Buffer, uint32_t>
                    (prepare_positions_kernel, prepare_positions_kernel_length);
            prepare_pos_program.set_kernel_name("prepare_for_shift_empty_rows")
                    .set_block_size(controls.block_size)
                    .set_needed_work_size(nzr_tmp);

            prepare_pos_program.run(controls, positions, subrows_nnz, nzr_tmp).wait();

            prefix_sum(controls, positions, nzr_out, nzr_tmp + 1);

            auto set_pos_program = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
                    (set_positions_kernel, set_positions_kernel_length);

            rpt_out = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr_out + 1));
            rows_out = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nzr_out);

            set_pos_program.set_kernel_name("set_positions_with_offset")
                    .set_block_size(controls.block_size)
                    .set_needed_work_size(nzr_tmp);
            set_pos_program.run(controls, rpt_out, rows_out,
                                matrix_in.rows_gpu(), subrows_nnz, positions,
                                nzr_tmp, nnz_out, nzr_out, rows_begin, i).wait();
        }
    }


    void submatrix(Controls &controls, matrix_dcsr &matrix_out, const matrix_dcsr &matrix_in,
                   uint32_t i, uint32_t j, uint32_t nrows, uint32_t ncols) {
        SET_TIMER


        if (matrix_in.nnz() == 0) {
            matrix_out = matrix_dcsr();
            return;
        }


        uint32_t rows_begin;
        uint32_t rows_end;
        sbm_delails::find_rows_range(controls, rows_begin, rows_end, matrix_in, i, nrows);

        if (rows_begin == rows_end) {
            matrix_out = matrix_dcsr();
            return;
        }

        cl::Buffer subrows_nnz;
        {
            START_TIMING
            sbm_delails::count_subrows_nnz(controls, subrows_nnz, matrix_in, j, ncols, rows_begin, rows_end);
            END_TIMING("count_subrows_nnz: ")
        }


        uint32_t nzr_tmp = rows_end - rows_begin;
        uint32_t nnz_out;
        prefix_sum(controls, subrows_nnz, nnz_out, nzr_tmp + 1);

        if (nnz_out == 0) {
            matrix_out = matrix_dcsr();
            return;
        }

        cl::Buffer cols_out;
        {
            START_TIMING
            sbm_delails::fill_rows(controls, cols_out, matrix_in, subrows_nnz, nnz_out, nzr_tmp, rows_begin, j);
            END_TIMING("fill_rows: ")
        }

        cl::Buffer rpt_out;
        cl::Buffer rows_out;
        uint32_t nzr_out;

        {
            START_TIMING
            sbm_delails::rpt_and_rows(controls, rpt_out, rows_out, nzr_out,
                     matrix_in, subrows_nnz, nzr_tmp, nnz_out, rows_begin, i);
            END_TIMING("rpt_and_rows: ")
        }

        matrix_out = matrix_dcsr(rpt_out, rows_out, cols_out, nrows, ncols, nnz_out, nzr_out);
    }
}