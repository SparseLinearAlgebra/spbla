/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020, 2021 JetBrains-Research                                    */
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

#include <testing/testing.hpp>

void testMatrixExtractSubMatrix(spbla_Index m, spbla_Index n, spbla_Index N, float density, spbla_Hints flags) {
    spbla_Matrix r, a;

    auto ta = testing::Matrix::generateSparse(m, n, density);

    ASSERT_EQ(spbla_Matrix_New(&a, m, n), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, SPBLA_HINT_VALUES_SORTED), SPBLA_STATUS_SUCCESS);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            auto k = m / N;
            auto l = n / N;

            ASSERT_EQ(spbla_Matrix_New(&r, k, l), SPBLA_STATUS_SUCCESS);
            ASSERT_EQ(spbla_Matrix_ExtractSubMatrix(r, a, i * k, j * l, k, l, flags), SPBLA_STATUS_SUCCESS);

            auto tr = ta.subMatrix(i * k, j * l, k, l);

            ASSERT_TRUE(tr.areEqual(r));
            ASSERT_EQ(spbla_Matrix_Free(r), SPBLA_STATUS_SUCCESS);
        }
    }

    ASSERT_EQ(spbla_Matrix_Free(a), SPBLA_STATUS_SUCCESS);
}

void testRun(spbla_Index m, spbla_Index n, float step, spbla_Hints setup) {
    // Setup library
    ASSERT_EQ(spbla_Initialize(setup), SPBLA_STATUS_SUCCESS);

    auto N = 10;

    for (size_t i = 0; i < N; i++) {
        testMatrixExtractSubMatrix(m, n, N, 0.01f + step * ((float) i), SPBLA_HINT_NO);
    }

    // Finalize library
    ASSERT_EQ(spbla_Finalize(), SPBLA_STATUS_SUCCESS);
}

#ifdef SPBLA_WITH_CUDA
TEST(spbla_Matrix, SubMatrixExtractSmallCuda) {
    spbla_Index m = 100, n = 200;
    float step = 0.05f;
    testRun(m, n, step, SPBLA_HINT_CUDA_BACKEND);
}

TEST(spbla_Matrix, SubMatrixExtractMediumCuda) {
    spbla_Index m = 400, n = 700;
    float step = 0.05f;
    testRun(m, n, step, SPBLA_HINT_CUDA_BACKEND);
}

TEST(spbla_Matrix, SubMatrixExtractLargeCuda) {
    spbla_Index m = 2000, n = 4000;
    float step = 0.01f;
    testRun(m, n, step, SPBLA_HINT_CUDA_BACKEND);
}
#endif

#ifdef SPBLA_WITH_OPENCL
TEST(spbla_Matrix, SubMatrixExtractSmallOpenCL) {
    spbla_Index m = 100, n = 200;
    float step = 0.05f;
    testRun(m, n, step, SPBLA_HINT_OPENCL_BACKEND);
}

TEST(spbla_Matrix, SubMatrixExtractMediumOpenCL) {
    spbla_Index m = 400, n = 700;
    float step = 0.05f;
    testRun(m, n, step, SPBLA_HINT_OPENCL_BACKEND);
}

TEST(spbla_Matrix, SubMatrixExtractLargeOpenCL) {
    spbla_Index m = 2000, n = 4000;
    float step = 0.01f;
    testRun(m, n, step, SPBLA_HINT_OPENCL_BACKEND);
}
#endif

#ifdef SPBLA_WITH_SEQUENTIAL
TEST(spbla_Matrix, SubMatrixExtractSmallFallback) {
    spbla_Index m = 100, n = 200;
    float step = 0.05f;
    testRun(m, n, step, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, SubMatrixExtractMediumFallback) {
    spbla_Index m = 400, n = 700;
    float step = 0.05f;
    testRun(m, n, step, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, SubMatrixExtractLargeFallback) {
    spbla_Index m = 2000, n = 4000;
    float step = 0.01f;
    testRun(m, n, step, SPBLA_HINT_CPU_BACKEND);
}
#endif

SPBLA_GTEST_MAIN