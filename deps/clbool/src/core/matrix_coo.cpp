#include <cl_operations.hpp>
#include "matrix_coo.hpp"
#include "kernel.hpp"

namespace clbool {
    matrix_coo::matrix_coo(Controls &controls,
                           index_type nrows,
                           index_type ncols,
                           index_type nnz)
        : matrix_base(nrows, ncols, nnz) {
        if (_nnz == 0) return;
        _rows = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * _nnz);
        _cols = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * _nnz);
        }

    matrix_coo::matrix_coo(index_type nrows,
                           index_type ncols,
                           index_type nnz,
                           cl::Buffer &rows,
                           cl::Buffer &cols)
        : matrix_base(nrows, ncols, nnz)
        , _rows(rows)
        , _cols(cols)
        {}


    matrix_coo::matrix_coo(Controls &controls,
                           index_type nrows,
                           index_type ncols,
                           index_type nnz,
                           const index_type* rows_indices,
                           const index_type* cols_indices,
                           bool sorted,
                           bool noDuplicates
                           )
        : matrix_base(nrows, ncols, nnz) {

        if (_nnz == 0) return;
        _rows = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * nnz);
        _cols = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * nnz);

        try {
            controls.queue.enqueueWriteBuffer(_rows, CL_TRUE, 0, sizeof(index_type) * nnz, rows_indices);
            controls.queue.enqueueWriteBuffer(_cols, CL_TRUE, 0, sizeof(index_type) * nnz, cols_indices);

            if (!sorted) {
                coo::sort_arrays(controls, _rows, _cols, _nnz);
            }

            if (!noDuplicates) {
                reduce_duplicates(controls);
            }

        } catch (const cl::Error &e) {
            std::stringstream exception;
            exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
            throw std::runtime_error(exception.str());
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
        : matrix_base(nrows, ncols, nnz)
        , _rows(rows)
        , _cols(cols)
     {
        try {
            if (!sorted) {
                coo::sort_arrays(controls, _rows, _cols, _nnz);
            }

            if (!noDuplicates) {
                reduce_duplicates(controls);
            }

        } catch (const cl::Error &e) {
            std::stringstream exception;
            exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
            throw std::runtime_error(exception.str());
        }
    }

    matrix_coo &matrix_coo::operator=(matrix_coo other) {
        _ncols = other._ncols;
        _nrows = other._nrows;
        _nnz = other._nnz;
        _rows = other._rows;
        _cols = other._cols;
        return *this;
    }

    void matrix_coo::reduce_duplicates(Controls& controls) {
        // ------------------------------------ prepare array to count positions ----------------------

        cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (_nnz + 1));

        auto prepare_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("prepare_positions", "prepare_array_for_positions");
        prepare_positions.set_needed_work_size(_nnz);
        prepare_positions.run(controls, positions, _rows, _cols, _nnz).wait();

        // ------------------------------------ calculate positions, get new_size -----------------------------------

        uint32_t new_nnz;
        prefix_sum(controls, positions, new_nnz, _nnz + 1);

        cl::Buffer new_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_nnz);
        cl::Buffer new_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_nnz);
        auto set_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>
                ("set_positions", "set_positions");
        set_positions.set_needed_work_size(_nnz);
        set_positions.run(controls, new_rows, new_cols, _rows, _cols, positions, _nnz).wait();

        _rows = std::move(new_rows);
        _cols = std::move(new_cols);
        _nnz = new_nnz;
    }
}
