#include "coo_tests.hpp"
#include <random>
#include "../dcsr/dcsr.hpp"
#include <matrices_conversions.hpp>
#include "../coo/coo_utils.hpp"

using namespace clbool::coo_utils;
using namespace clbool::utils;

void clbool::test::test_submatrix() {
    Controls controls = create_controls();
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    SET_TIMER
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 1000; size < 30000; size += 200) {
            for (int iter = 0; iter < 25; ++iter) {
                uint32_t max_size = size;
                uint32_t nnz_max = std::max(10u, max_size * k);

                LOG << " ------------------------------- k = " << k << ", size = " << size
                          << ", iter = " << iter << "-------------------------------------------\n"
                          << "max_size = " << size << ", nnz_max = " << nnz_max;

                std::uniform_int_distribution<uint32_t> rnd_idx{0, max_size};
                uint32_t i1 = rnd_idx(mersenne_engine);
                uint32_t i2 = rnd_idx(mersenne_engine);
                uint32_t i3 = rnd_idx(mersenne_engine);
                uint32_t i4 = rnd_idx(mersenne_engine);

                uint32_t i = std::min(i1, i2);
                uint32_t nrows = std::max(i1, i2) - i;

                uint32_t j = std::min(i3, i4);
                uint32_t ncols = std::max(i3, i4) - j;


                LOG << "i = " << i << ", j = " << j << ", nrows = " << nrows << ", ncols = " << ncols;

                matrix_dcsr_cpu a_cpu = coo_pairs_to_dcsr_cpu(generate_coo_pairs_cpu(nnz_max, max_size));
                matrix_dcsr_cpu c_cpu;


                {
                    START_TIMING
                    submatrix_cpu(c_cpu, a_cpu, i, j, nrows, ncols);
                    END_TIMING("submatrix on CPU: ")
                }

                matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
                matrix_dcsr c_gpu;

                {
                    START_TIMING
                    submatrix(controls, c_gpu, a_gpu, i, j, nrows, ncols);
                    END_TIMING("submatrix on DEVICE: ")
                }

                compare_matrices(controls, c_gpu, c_cpu);
            }
        }
    }
}