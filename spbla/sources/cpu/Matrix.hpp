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

#ifndef SPBLA_CPU_MATRIX_HPP
#define SPBLA_CPU_MATRIX_HPP

#include <backend/Matrix.hpp>
#include <GraphBLAS.h>

namespace spbla {
    namespace cpu {

        class Matrix final: public backend::Matrix {
        public:
            Matrix(size_t nrows, size_t ncols);
            ~Matrix() override;

            void Build(const index *rowIds, const index *colIds, size_t nvals, bool isSorted) override;
            void Extract(index *rowIds, index *colIds, size_t &nvals) const override;

            void Multiply(const backend::Matrix &matrixA, const backend::Matrix &matrixB) override;
            void EWiseAdd(const backend::Matrix &matrixA, const backend::Matrix &matrixB) override;
            void Kronecker(const backend::Matrix &matrixA, const backend::Matrix &matrixB) override;

            size_t GetNrows() const override;
            size_t GetNcols() const override;
            size_t GetNvals() const override;

        private:

            GrB_Matrix mMatrix = nullptr;
        };

    }
}

#endif //SPBLA_CPU_MATRIX_HPP