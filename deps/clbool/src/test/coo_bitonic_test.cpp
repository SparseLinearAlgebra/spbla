#include <algorithm>
#include "../common/cl_includes.hpp"
#include "coo_tests.hpp"
#include "../library_classes/controls.hpp"
#include "../coo/coo_utils.hpp"

void clbool::test::testBitonicSort() {

    Controls controls = utils::create_controls();
    SET_TIMER

    for (uint32_t size = 0; size < 1020000; size += 256) {
        if (size == 0) continue;
//        uint32_t size;
        LOG << "----------------------- size = " << size << " ------------------------";

        cpu_buffer rows_cpu(size);
        cpu_buffer cols_cpu(size);

        coo_utils::fill_random_matrix(rows_cpu, cols_cpu);

        cl::Buffer rows_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
        cl::Buffer cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

        controls.queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_cpu.data());
        controls.queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_cpu.data());


        matrix_coo_cpu_pairs m_cpu;
        coo_utils::form_cpu_matrix(m_cpu, rows_cpu, cols_cpu);

        {
            START_TIMING
            std::sort(m_cpu.begin(), m_cpu.end());
            END_TIMING("sort on CPU: ")
        }
        coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, m_cpu);


        {
            START_TIMING
            sort_arrays(controls, rows_gpu, cols_gpu, size);
            END_TIMING("sort on DEVICE: ")
        }


        utils::compare_buffers(controls, rows_gpu, rows_cpu, size, "rows_gpu");
        utils::compare_buffers(controls, cols_gpu, cols_cpu, size, "cols_gpu");

    }
}

