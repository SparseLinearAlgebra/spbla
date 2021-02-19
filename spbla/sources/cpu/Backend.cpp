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

#include <cpu/Backend.hpp>
#include <cpu/Matrix.hpp>

namespace spbla {
    namespace cpu {

        void Backend::Initialize(uint32_t argc, const char *const *argv) {
            GrB_Info info = GrB_init(GrB_BLOCKING);
            mIsInitialized = info == GrB_SUCCESS;
        }

        bool Backend::IsInitialized() const {
            return mIsInitialized;
        }

        void Backend::Finalize() {
            GrB_Info info = GrB_finalize();
        }

        backend::Matrix *Backend::CreateMatrix(size_t nrows, size_t ncols) {
            return new Matrix(nrows, ncols);
        }

        void Backend::ReleaseMatrix(backend::Matrix *matrix) {
            delete matrix;
        }

        const std::string &Backend::GetName() const {
            static const std::string name =
                    "GraphBlas:SuiteSparse";

            return name;
        }

        const std::string &Backend::GetDescription() const {
            static const std::string description =
                    "SuiteSparse based backend implementation with CPU side computations";

            return description;
        }

        const std::string &Backend::GetAuthorsName() const {
            static const std::string authors =
                    "- SuiteSpare implementation: Dr. Timothy Alden Davis \n"
                    "- Cpu backed for SPBLA: Egor Orachyov";

            return authors;
        }
    }
}
