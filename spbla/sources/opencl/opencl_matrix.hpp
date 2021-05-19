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

#ifndef SPBLA_OPENCL_MATRIX_HPP
#define SPBLA_OPENCL_MATRIX_HPP

#include <memory>
#include <backend/matrix_base.hpp>
#include <library_classes/matrix_dcsr.hpp>
#include "opencl_backend.hpp"

namespace spbla {

    class OpenCLMatrix: public MatrixBase {
    public:
        using MatrixImplType = clbool::matrix_dcsr;

        OpenCLMatrix(size_t nrows, size_t ncols);
        ~OpenCLMatrix() override;

        void setElement(index i, index j) override;
        void build(const index *rows, const index *cols, size_t nvals, bool isSorted, bool noDuplicates) override;
        void extract(index *rows, index *cols, size_t &nvals) override;
        void extractSubMatrix(const MatrixBase &otherBase, index i, index j, index nrows, index ncols, bool checkTime) override;

        void clone(const MatrixBase &otherBase) override;
        void transpose(const MatrixBase &otherBase, bool checkTime) override;
        void reduce(const MatrixBase &otherBase, bool checkTime) override;

        void multiply(const MatrixBase &aBase, const MatrixBase &bBase, bool accumulate, bool checkTime) override;
        void kronecker(const MatrixBase &aBase, const MatrixBase &bBase, bool checkTime) override;
        void eWiseAdd(const MatrixBase &aBase, const MatrixBase &bBase, bool checkTime) override;

        index getNrows() const override;
        index getNcols() const override;
        index getNvals() const override;

    private:

        MatrixImplType mMatrixImpl;
        friend spbla::OpenCLBackend;
        static std::shared_ptr<clbool::Controls> clboolState;

        size_t mNrows = 0;
        size_t mNcols = 0;
        size_t mNvals = 0;

        // TODO как-то оформить в общем духе
        void checkState() {
            if (clboolState == nullptr) {
                throw std::runtime_error("clbool library state is not initialized!");
            }
        }
    };

}

#endif //SPBLA_OPENCL_MATRIX_HPP
