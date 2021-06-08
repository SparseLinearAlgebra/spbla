#include <algorithm>
#include "../common/cl_includes.hpp"
#include "coo_tests.hpp"
#include "controls.hpp"
#include "../coo/coo_utils.hpp"
#include "../coo/coo_matrix_addition.hpp"


void clbool::test::testReduceDuplicates() {

    Controls controls = utils::create_controls();

    uint32_t size = 10374663;

    // -------------------- create indices ----------------------------

    std::vector<uint32_t> rows_cpu(size);
    std::vector<uint32_t> cols_cpu(size);

    std::vector<uint32_t> rows_from_gpu(size);
    std::vector<uint32_t> cols_from_gpu(size);

    coo_utils::fill_random_matrix(rows_cpu, cols_cpu, 1043);

    // -------------------- create and sort cpu matrix ----------------------------
    matrix_coo_cpu_pairs m_cpu;
    coo_utils::form_cpu_matrix(m_cpu, rows_cpu, cols_cpu);
    std::sort(m_cpu.begin(), m_cpu.end());

    // -------------------- create and sort gpu buffers ----------------------------

    cl::Buffer rows_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
    cl::Buffer cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

    controls.queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_cpu.data());
    controls.queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_cpu.data());

    sort_arrays(controls, rows_gpu, cols_gpu, size);

    // ------------------ now reduce cpu matrix and read result in vectors ------------------------

    std::cout << "\nmatrix cpu before size: " << m_cpu.size() << std::endl;
    m_cpu.erase(std::unique(m_cpu.begin(), m_cpu.end()), m_cpu.end());
    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, m_cpu);
    std::cout << "\nmatrix cpu after size: " << m_cpu.size() << std::endl;

    // ------------------ now reduce gpu buffers and read in vectors ------------------------
    uint32_t new_size;
    reduce_duplicates(controls, rows_gpu, cols_gpu, reinterpret_cast<uint32_t &>(new_size), size);

    rows_from_gpu.resize(new_size);
    cols_from_gpu.resize(new_size);

    controls.queue.enqueueReadBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * new_size, rows_from_gpu.data());
    controls.queue.enqueueReadBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * new_size, cols_from_gpu.data());

    if (rows_from_gpu == rows_cpu && cols_from_gpu == cols_cpu) {
        std::cout << "correct reduce" << std::endl;
    } else {
        std::cerr << "incorrect reduce" << std::endl;
    }
}