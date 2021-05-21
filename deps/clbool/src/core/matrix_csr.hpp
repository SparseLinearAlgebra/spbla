#pragma once

#include "matrix_base.hpp"
#include "controls.hpp"
namespace clbool {

    class matrix_csr : public details::matrix_base {
    private:
        cl::Buffer _rows_pointers_gpu;
        cl::Buffer _cols_indexes_gpu;

        std::vector<uint32_t> _rows_pointers_cpu;
        std::vector<uint32_t> _cols_indexes_cpu;

    public:

        matrix_csr(Controls controls, uint32_t nRows, uint32_t nCols, uint32_t nEntities)
                : matrix_base(nRows, nCols, nEntities)
                // TODO: confirm if we need CL_MEM_READ_WRITE
                , _rows_pointers_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
                , _cols_indexes_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
                , _rows_pointers_cpu(std::vector<uint32_t> (0, n_entities))
                , _cols_indexes_cpu(std::vector<uint32_t> (0, n_entities))
        {}


        matrix_csr(Controls controls, uint32_t nRows, uint32_t nCols, uint32_t nEntities,
                   std::vector<uint32_t> rows_indexes, std::vector<uint32_t> cols_indexes)
                : matrix_base(nRows, nCols, nEntities)
                , _rows_pointers_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
                , _cols_indexes_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
                , _rows_pointers_cpu(std::move(rows_indexes))
                , _cols_indexes_cpu(std::move(cols_indexes))
        {
            try {
                controls.queue.enqueueWriteBuffer(_rows_pointers_gpu, CL_TRUE, 0, sizeof(uint32_t) * _rows_pointers_cpu.size(),
                                                  _rows_pointers_cpu.data());

                controls.queue.enqueueWriteBuffer(_cols_indexes_gpu, CL_TRUE, 0, sizeof(uint32_t) * _cols_indexes_cpu.size(),
                                                  _cols_indexes_cpu.data());

            } catch (const cl::Error& e) {
                std::stringstream exception;
                exception << "\n" << e.what() << " : " << e.err() << "\n";
                throw std::runtime_error(exception.str());
            }
        }

        const auto& rows_pointers_cpu() const {
            return _rows_pointers_cpu;
        }

        const auto& cols_indexes_cpu() const {
            return _cols_indexes_cpu;
        }

    };
}