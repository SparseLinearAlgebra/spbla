#include "coo_kronecker_product.hpp"

#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/program.hpp"
#include "../cl/headers/coo_kronecker.h"

namespace clbool {

    void kronecker_product(Controls &controls,
                           matrix_coo& matrix_out,
                           const matrix_coo& matrix_a,
                           const matrix_coo& matrix_b) {
        try {
            uint32_t res_size = matrix_a.nnz() * matrix_b.nnz();

            auto p = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                            uint32_t, uint32_t, uint32_t>(coo_kronecker_kernel, coo_kronecker_kernel_length)
                            .set_kernel_name("kronecker")
                            .set_needed_work_size(res_size);

            cl::Buffer res_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * res_size);
            cl::Buffer res_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * res_size);
            p.run(controls, res_rows, res_cols,
                  matrix_a.rows_gpu(), matrix_a.cols_gpu(),
                  matrix_b.rows_gpu(), matrix_b.cols_gpu(),
                  matrix_b.nnz(), matrix_b.nrows(), matrix_b.ncols()
                  );

            matrix_out = matrix_coo(controls, matrix_a.nrows() * matrix_b.nrows(), matrix_a.ncols() * matrix_b.ncols(),
                                    res_size, res_rows, res_cols);

        } catch (const cl::Error &e) {
            std::stringstream exception;
            exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
            throw std::runtime_error(exception.str());
        }

    }
}
