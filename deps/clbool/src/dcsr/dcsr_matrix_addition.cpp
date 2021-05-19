#include "dcsr_matrix_addition.hpp"
#include "kernel.hpp"
#include "../common/cl_operations.hpp"

namespace clbool::dcsr {

    void merge(Controls& controls,
               cl::Buffer& rpt_c,
               const cl::Buffer& rpt_a,
               const cl::Buffer& rpt_b,
               uint32_t nzr_a,
               uint32_t nzr_b);

    void reduce_duplicates(Controls &controls,
                           cl::Buffer &data,
                           uint32_t &new_size,
                           uint32_t merged_size);


    // --------------------------------------------------------------------

    // TODO: добавить assert на размеры матрицы
    void matrix_addition(Controls& controls, matrix_dcsr& matrix_out,
                               const matrix_dcsr& a, const matrix_dcsr& c) {

    }

    // nzr_a - количество непустых рядов в матрице A == размер rpt_a
    // nzr_b - количество непустых рядов в матрице B == размер rpt_b
    void build_out_rows(Controls& controls,
                        cl::Buffer& rpt_c,
                        const cl::Buffer& rpt_a,
                        const cl::Buffer& rpt_b,
                        uint32_t nzr_a,
                        uint32_t nzr_b
            ) {
        uint32_t merged_size = nzr_a + nzr_b;
        cl::Buffer rpt(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);
        merge(controls, rpt, rpt_a, rpt_b, nzr_a, nzr_b);

        uint32_t nzr;
        reduce_duplicates(controls, rpt, nzr, merged_size);



    }

    void merge(Controls& controls,
               cl::Buffer& rpt_c,
               const cl::Buffer& rpt_a,
               const cl::Buffer& rpt_b,
               uint32_t nzr_a,
               uint32_t nzr_b) {
        auto merge_program = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>("merge_path1d", "merge");
        merge_program.set_needed_work_size(nzr_a + nzr_b);
        merge_program.run(controls, rpt_c, rpt_a, rpt_b, nzr_a, nzr_b);
    }

    /*
     * data -- sorted array with possible duplicates : 1 1 2 3 4 4 5 6 7 7 ...
     * data is an input and output in the same time
     */
    void reduce_duplicates(Controls &controls,
                           cl::Buffer &data,
                           uint32_t &new_size,
                           uint32_t merged_size
    ) {
        // ------------------------------------ prepare array to count positions ----------------------

        cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

        auto prepare_positions = kernel<cl::Buffer, cl::Buffer, uint32_t>
                ("prepare_positions", "prepare_array_for_rows_positions");
        prepare_positions.set_needed_work_size(merged_size);

        // ------------------------------------ calculate positions, get new_size -----------------------------------

        prefix_sum(controls, positions, new_size, merged_size);

        //  ---------------------------------- sat values to calculated positions --------------------
        cl::Buffer new_data(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_size);

        auto set_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("set_positions", "set_positions1d");
        set_positions.set_needed_work_size(merged_size);
        set_positions.run(controls, new_data, data, positions, merged_size);
        data = std::move(new_data);
    }
}