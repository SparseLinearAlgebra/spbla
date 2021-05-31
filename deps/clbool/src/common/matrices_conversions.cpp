#include "matrices_conversions.hpp"

namespace clbool {
    namespace {
    #define CONV_GROUP_SIZE 64

        void create_rows_pointers(Controls &controls,
                                  cl::Buffer &rows_pointers_out,
                                  cl::Buffer &rows_compressed_out,
                                  const cl::Buffer &rows,
                                  uint32_t size,
                                  uint32_t &nzr // non zero rows_gpu
        ) {

            cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

            auto prepare_positions = kernel<cl::Buffer, cl::Buffer, uint32_t>
                    ("prepare_positions", "prepare_array_for_rows_positions");
            prepare_positions.set_work_size(size);

            prepare_positions.run(controls, positions, rows, size);

            prefix_sum(controls, positions, nzr, size);

            cl::Buffer rows_pointers(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr + 1));
            cl::Buffer rows_compressed(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nzr);

            auto set_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>
                    ("set_positions", "set_positions_rows");
            set_positions.set_kernel_name("set_positions_rows")
                    .set_work_size(size);

            set_positions.run(controls, rows_pointers, rows_compressed, rows, positions, size, nzr);

            rows_pointers_out = std::move(rows_pointers);
            rows_compressed_out = std::move(rows_compressed);
        }
    }

    /*
     * .cols array in output and input is the same
     */
    matrix_coo dcsr_to_coo_shallow(Controls &controls, matrix_dcsr &a) {
        if (a.empty()) {
            return matrix_coo(a.nrows(), a.ncols());
        }

        cl::Buffer c_rows(controls.context, CL_MEM_READ_WRITE, sizeof(matrix_dcsr::index_type) * a.nnz());

        auto dscr_to_coo = kernel<cl::Buffer, cl::Buffer, cl::Buffer>
                ("dscr_to_coo", "dscr_to_coo");
        dscr_to_coo.set_block_size(CONV_GROUP_SIZE);
        dscr_to_coo.set_work_size(a.nzr() * CONV_GROUP_SIZE);

        dscr_to_coo.run(controls, a.rpt_gpu(), a.rows_gpu(), c_rows);
        return matrix_coo(a.nrows(), a.ncols(), a.nnz(), c_rows, a.cols_gpu());
    }

    matrix_coo dcsr_to_coo_deep(Controls &controls, const matrix_dcsr &a) {
        if (a.empty()) {
            return matrix_coo(a.nrows(), a.ncols());
        }

        cl::Buffer c_rows(controls.context, CL_MEM_READ_WRITE, sizeof(matrix_dcsr::index_type) * a.nnz());
        cl::Buffer c_cols(controls.context, CL_MEM_READ_WRITE, sizeof(matrix_dcsr::index_type) * a.nnz());

        controls.queue.enqueueCopyBuffer(a.cols_gpu(), c_cols, 0, 0, sizeof(matrix_dcsr::index_type) * a.nnz());

        auto dscr_to_coo = kernel<cl::Buffer, cl::Buffer, cl::Buffer>
                ("dscr_to_coo", "dscr_to_coo");
        dscr_to_coo.set_block_size(CONV_GROUP_SIZE)
                .set_work_size(a.nzr() * CONV_GROUP_SIZE);

        dscr_to_coo.run(controls, a.rpt_gpu(), a.rows_gpu(), c_rows);
        return matrix_coo(a.nrows(), a.ncols(), a.nnz(), c_rows, c_cols);
    }

    matrix_dcsr coo_to_dcsr_shallow(Controls &controls, const matrix_coo &a) {
        if (a.empty()) {
            return matrix_dcsr(a.nrows(), a.ncols());
        }
        cl::Buffer rpt;
        cl::Buffer rows;
        uint32_t nzr;
        if (a.nnz() != 0)
            create_rows_pointers(controls, rpt, rows, a.rows_gpu(), a.nnz(), nzr);

        return matrix_dcsr(rpt, rows, a.cols_gpu(),
                           a.nrows(), a.ncols(), a.nnz(), nzr
        );
    }

    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, const matrix_dcsr_cpu &m, uint32_t size) {
        if (m.cols().empty()) {
            return matrix_dcsr(size, size);
        }

        cl::Buffer rows_pointers(controls.context,
                                 (const_cast<matrix_dcsr_cpu&>(m)).rpt().begin(), (const_cast<matrix_dcsr_cpu&>(m)).rpt().end(), false);
        cl::Buffer rows_compressed(controls.context,
                                   (const_cast<matrix_dcsr_cpu&>(m)).rows().begin(), (const_cast<matrix_dcsr_cpu&>(m)).rows().end(), false);
        cl::Buffer cols_indices(controls.context,
                                (const_cast<matrix_dcsr_cpu&>(m)).cols().begin(), (const_cast<matrix_dcsr_cpu&>(m)).cols().end(), false);

        return matrix_dcsr(rows_pointers, rows_compressed, cols_indices,
                           size, size, m.cols().size(), m.rows().size());

    }

    matrix_coo matrix_coo_from_cpu(Controls &controls, matrix_coo_cpu &m, uint32_t size) {
        if (m.cols().empty()) {
            return matrix_coo(size, size);
        }

        cl::Buffer rows_indices(controls.context, m.rows().begin(), m.rows().end(), false);
        cl::Buffer cols_indices(controls.context, m.cols().begin(), m.cols().end(), false);

        return matrix_coo(size, size, m.rows().size(), rows_indices, cols_indices);
    }

    matrix_dcsr_cpu matrix_dcsr_from_gpu(Controls &controls, matrix_dcsr &m) {
        if (m.empty()) {
            return matrix_dcsr_cpu();
        }

        cpu_buffer rows_pointers(m.nzr() + 1);
        cpu_buffer rows_compressed(m.nzr());
        cpu_buffer cols_indices(m.nnz());

        controls.queue.enqueueReadBuffer(m.rpt_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * rows_pointers.size(), rows_pointers.data());
        controls.queue.enqueueReadBuffer(m.rows_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * rows_compressed.size(), rows_compressed.data());
        controls.queue.enqueueReadBuffer(m.cols_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * cols_indices.size(), cols_indices.data());

        return matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);

    }


    matrix_coo_cpu matrix_coo_from_gpu(Controls &controls, matrix_coo &m) {
        if (m.empty()) {
            return matrix_coo_cpu();
        }

        cpu_buffer rows_indices(m.nnz());
        cpu_buffer cols_indices(m.nnz());

        controls.queue.enqueueReadBuffer(m.rows_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * rows_indices.size(), rows_indices.data());
        controls.queue.enqueueReadBuffer(m.cols_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * cols_indices.size(), cols_indices.data());

        return matrix_coo_cpu(rows_indices, cols_indices);
    }
}