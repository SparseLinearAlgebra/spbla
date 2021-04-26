#include "matrix_coo.hpp"

namespace clbool {
    matrix_coo::matrix_coo(Controls &controls,
                           index_type nrows,
                           index_type ncols,
                           index_type nnz)
        : matrix_base(nrows, ncols, nnz)
        , _rows(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * _nnz))
        , _cols(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * _nnz))
        {}

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
        : matrix_base(nrows, ncols, nnz)
        , _rows(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * nnz, (void *) rows_indices))
        , _cols(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * nnz, (void *) cols_indices))
     {
        try {

            if (!sorted) {
                sort_arrays(controls, _rows, _cols, _nnz);
            }
            if (!noDuplicates) {

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
                sort_arrays(controls, _rows, _cols, _nnz);
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
}
