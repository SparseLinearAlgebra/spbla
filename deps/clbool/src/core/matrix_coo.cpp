#include "../common/cl_operations.hpp"

namespace clbool {
    matrix_coo::matrix_coo(index_type nrows,
                           index_type ncols)
            : matrix_base(nrows, ncols, 0) {}

    matrix_coo::matrix_coo(cl::Buffer &rows,
                           cl::Buffer &cols,

                           index_type nrows,
                           index_type ncols,
                           index_type nnz)
            : matrix_base(nrows, ncols, nnz), _rows(rows), _cols(cols) {}


    matrix_coo::matrix_coo(Controls &controls,
                           const index_type *rows_indices,
                           const index_type *cols_indices,

                           index_type nrows,
                           index_type ncols,
                           index_type nnz,
                           bool sorted,
                           bool noDuplicates
    )
            : matrix_base(nrows, ncols, nnz) {

        if (_nnz == 0) return;

        CLB_CREATE_BUF(_rows = utils::create_buffer(controls, nnz));
        CLB_CREATE_BUF(_cols = utils::create_buffer(controls, nnz));
        cl::Event e1, e2;
        CLB_WRITE_BUF(controls.queue.enqueueWriteBuffer(_rows, CL_FALSE, 0, sizeof(index_type) * nnz, rows_indices,
                                                        nullptr, &e1));
        CLB_WRITE_BUF(controls.queue.enqueueWriteBuffer(_cols, CL_FALSE, 0, sizeof(index_type) * nnz, cols_indices,
                                                        nullptr, &e2));
        CLB_WAIT({e1.wait(); e2.wait();});

        if (!sorted) {
            coo::sort_arrays(controls, _rows, _cols, _nnz);
        }

        if (!noDuplicates) {
            reduce_duplicates2(controls);
        }
    }


    matrix_coo::matrix_coo(Controls &controls,
                           index_type nrows,
                           index_type ncols,
                           index_type nnz,
                           cl::Buffer &rows,
                           cl::Buffer &cols,
                           bool sorted,
                           bool noDuplicates
    )
            : matrix_base(nrows, ncols, nnz), _rows(rows), _cols(cols) {

        if (!sorted) {
            coo::sort_arrays(controls, _rows, _cols, _nnz);
        }

        if (!noDuplicates) {
            reduce_duplicates2(controls);
        }
    }

    matrix_coo &matrix_coo::operator=(const matrix_coo &other) {
        _ncols = other._ncols;
        _nrows = other._nrows;
        _nnz = other._nnz;
        _rows = other._rows;
        _cols = other._cols;
        return *this;
    }

    void matrix_coo::reduce_duplicates(Controls &controls) {

        // ------------------------------------ prepare array to count positions -----------------------------------

        cl::Buffer positions;
        CLB_CREATE_BUF(positions = utils::create_buffer(controls, _nnz + 1));

        auto prepare_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("prepare_positions", "prepare_array_for_positions");
        prepare_positions.set_work_size(_nnz);

        CLB_RUN(TIME_RUN("prepare_positions run in: ",
        prepare_positions.run(controls, positions, _rows, _cols, _nnz)));

        // ------------------------------------ calculate positions, get new_size -----------------------------------

        uint32_t new_nnz;
        {
            START_TIMING
            prefix_sum(controls, positions, new_nnz, _nnz + 1);
            END_TIMING("prefix_sum run in: ")
        }

        cl::Buffer new_rows;
        CLB_CREATE_BUF(new_rows = utils::create_buffer(controls, new_nnz));
        cl::Buffer new_cols;
        CLB_CREATE_BUF(new_cols =  utils::create_buffer(controls, new_nnz));

        auto set_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>
                ("set_positions", "set_positions");
        set_positions.set_work_size(_nnz);
        CLB_RUN(TIME_RUN("set_positions run in: ",
        set_positions.run(controls, new_rows, new_cols, _rows, _cols, positions, _nnz)));
        _rows = std::move(new_rows);
        _cols = std::move(new_cols);
        _nnz = new_nnz;
    }

    void matrix_coo::reduce_duplicates2(Controls &controls) {
        // ------------------------------------ prepare array to count positions ----------------------

        uint32_t groups_num = (_nnz + controls.max_wg_size - 1) / controls.max_wg_size;
        cl::Buffer duplicates_per_tb;
        CLB_CREATE_BUF(duplicates_per_tb = utils::create_buffer(controls, groups_num + 1));

        // compare equal values on the edge of group
        auto init_duplicates = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>
                ("coo_reduce_duplicates", "init_duplicates");
        init_duplicates.set_block_size(controls.max_wg_size);
        init_duplicates.set_work_size(groups_num);

        CLB_RUN(TIME_RUN("init_duplicates run in: ",
        init_duplicates.run(controls, _rows, _cols, duplicates_per_tb, _nnz, groups_num)));

        auto reduce_tb = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("coo_reduce_duplicates", "reduce_duplicates_tb");
        reduce_tb.set_block_size(controls.max_wg_size);
        reduce_tb.set_work_size(_nnz);

        CLB_RUN(TIME_RUN("reduce_tb run in: ",
        reduce_tb.run(controls, _rows, _cols, duplicates_per_tb, _nnz)));

        uint32_t total_duplicates;
        uint32_t new_nnz;
        {
            START_TIMING
            prefix_sum(controls, duplicates_per_tb, total_duplicates, groups_num + 1);
            END_TIMING("prefix_sum run in: ")
        }

        if (total_duplicates == 0) return;

        new_nnz = _nnz - total_duplicates;

        cl::Buffer new_rows;
        CLB_CREATE_BUF(new_rows = utils::create_buffer(controls, new_nnz));

        cl::Buffer new_cols;
        CLB_CREATE_BUF(new_cols = utils::create_buffer(controls, new_nnz));

        auto shift_tb = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("coo_reduce_duplicates", "shift_tb");
        shift_tb.set_block_size(controls.max_wg_size);
        shift_tb.set_work_size(_nnz);

        CLB_RUN(TIME_RUN("shift_tb run in ",
        shift_tb.run(controls, _rows, _cols, new_rows, new_cols, duplicates_per_tb, _nnz)));

        _nnz = new_nnz;
        _rows = std::move(new_rows);
        _cols = std::move(new_cols);
    }
}
