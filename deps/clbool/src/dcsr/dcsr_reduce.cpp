#include "dcsr.hpp"


namespace clbool::dcsr {
    void reduce(Controls &controls, matrix_dcsr &matrix_out, const matrix_dcsr &matrix_in) {

        if (matrix_in.nnz() == 0) {
            matrix_out = matrix_dcsr();
            return;
        }

        cl::Buffer rpt;
        CLB_CREATE_BUF(rpt = utils::create_buffer(controls, matrix_in.nzr() + 1));
        cl::Buffer cols(controls.context, CL_MEM_READ_WRITE, sizeof (uint32_t) * matrix_in.nzr());
        cl::Buffer rows;

        if (&matrix_out == &matrix_in) {
            rows = matrix_in.rows_gpu();
        } else {
            rows = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof (uint32_t) * matrix_in.nzr());
            controls.queue.enqueueCopyBuffer(matrix_in.rows_gpu(), rows, 0, 0, sizeof (uint32_t) * matrix_in.nzr());
        }

        auto reduce_program = kernel<cl::Buffer, cl::Buffer, uint32_t>
                ("reduce", "set_rpt_and_cols");
        reduce_program.set_work_size(matrix_in.nzr());

        reduce_program.run(controls, rpt, cols, matrix_in.nzr());

        matrix_out = matrix_dcsr(rpt, rows, cols, matrix_in.nzr(), 1, matrix_in.nzr(), matrix_in.nzr());
    }
}