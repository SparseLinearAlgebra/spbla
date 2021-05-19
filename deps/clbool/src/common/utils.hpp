#pragma once

#include "controls.hpp"
#include "matrix_coo.hpp"
#include "matrix_dcsr.hpp"
#include "cpu_matrices.hpp"


namespace clbool::utils {
    void reduce(matrix_dcsr_cpu &matrix_out, const matrix_dcsr_cpu &matrix_in);

    void submatrix_cpu(matrix_dcsr_cpu &matrix_out, const matrix_dcsr_cpu &matrix_in,
                       uint32_t i, uint32_t j, uint32_t nrows, uint32_t ncols);

    bool compare_matrices(Controls &controls, const matrix_dcsr &m_gpu, const matrix_dcsr_cpu &m_cpu);

    using cpu_buffer = std::vector<uint32_t>;

    void fill_random_buffer(cpu_buffer &buf, uint32_t max = -1);

// https://stackoverflow.com/a/466242
    unsigned int ceil_to_power2(uint32_t v);

// https://stackoverflow.com/a/2681094
    uint32_t round_to_power2(uint32_t x);

    uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n);

    std::string error_name(cl_int error);

    void printPlatformInfo(const cl::Platform &platform);

    void printDeviceInfo(const cl::Device &device);

    void print_gpu_buffer(Controls &controls, const cl::Buffer &buffer, uint32_t size);

    void print_cpu_buffer(const cpu_buffer &buffer, uint32_t size = -1);

    bool compare_buffers(Controls &controls, const cl::Buffer &buffer_gpu, const cpu_buffer &buffer_cpu, uint32_t size,
                         std::string name = "");

    void program_handler(const cl::Error &e, const cl::Program &program,
                         const cl::Device &device, const std::string &name);

//    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, const coo_utils::matrix_dcsr_cpu &m, uint32_t size);
}