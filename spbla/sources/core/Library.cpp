/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2021 JetBrains-Research                                          */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#include <core/Library.hpp>
#include <core/Matrix.hpp>
#include <core/Exception.hpp>

#ifdef SPBLA_WITH_CPU_BACKEND
#include <cpu/Backend.hpp>
#endif //SPBLA_WITH_CPU_BACKEND

namespace spbla {

    backend::Backend* Library::mProvider = nullptr;

    void Library::Initialize(spbla_Backend backend, index optionsCount, const char *const *options) {
        bool selected = false;
        OptionsParser optionsParser;

        // todo: log error if it is not parsed
        optionsParser.Parse(optionsCount, options);

        // Default strategy is:
        // - First check if cuda is supported
        // - If not, then check OpenCL implementation
        // - If OpenCL does not supported, then try Cpu backend
        // If user provides hint, then start from the corresponding point

        switch (backend) {
            case spbla_Backend::SPBLA_BACKEND_DEFAULT:
            case spbla_Backend::SPBLA_BACKEND_CUDA: {
#ifdef SPBLA_WITH_CUDA_BACKEND
#endif //SPBLA_WITH_CUDA_BACKEND

                if (selected)
                    break;
            }
            case spbla_Backend::SPBLA_BACKEND_OPENCL: {
#ifdef SPBLA_WITH_OPENCL_BACKEND
#endif //SPBLA_WITH_OPENCL_BACKEND

                if (selected)
                    break;
            }
            case spbla_Backend::SPBLA_BACKEND_CPU:
            default:
#ifdef SPBLA_WITH_CPU_BACKEND
                mProvider = new cpu::Backend();
                mProvider->Initialize(optionsParser);

                if (mProvider->IsInitialized()) {
                    selected = true;
                    break;
                }

                delete mProvider;
                mProvider = nullptr;
#endif //SPBLA_WITH_CPU_BACKEND
        }

        CHECK_RAISE_CRITICAL_ERROR(selected, BackendNotSupported, "Failed to select supported backend for computations");
    }

    void Library::Finalize() {
        // Remember to finalize backend
        mProvider->Finalize();
        // Release memory
        delete mProvider;
        // Note: set to null for convenience
        mProvider = nullptr;
    }

    void Library::Validate() {
        CHECK_RAISE_ERROR(mProvider, InvalidState, "Library backend is not initialized");
    }

    Matrix* Library::CreateMatrix(size_t nrows, size_t ncols) {
        return new Matrix(nrows, ncols, *mProvider);
    }

    void Library::ReleaseMatrix(Matrix *matrix) {
        delete matrix;
    }

}