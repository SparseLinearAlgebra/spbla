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

TEST(spbla_Matrix, CreateDestroy) {
    spbla_Matrix matrix = nullptr;

    ASSERT_EQ(spbla_Initialize(SPBLA_HINT_NO), SPBLA_STATUS_SUCCESS);

    ASSERT_EQ(spbla_Matrix_New(&matrix, 10, 10), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Free(matrix), SPBLA_STATUS_SUCCESS);

    ASSERT_EQ(spbla_Finalize(), SPBLA_STATUS_SUCCESS);
}

// Fills sparse matrix with random data and tests whether the transfer works correctly
void testMatrixFilling(spbla_Index m, spbla_Index n, float density) {
    spbla_Matrix matrix = nullptr;

    testing::Matrix tmatrix = std::move(testing::Matrix::generateSparse(m, n, density));

    ASSERT_EQ(spbla_Matrix_New(&matrix, m, n), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Build(matrix, tmatrix.rowsIndex.data(), tmatrix.colsIndex.data(), tmatrix.nvals, SPBLA_HINT_VALUES_SORTED), SPBLA_STATUS_SUCCESS);

    // Compare test matrix and library one
    ASSERT_EQ(tmatrix.areEqual(matrix), true);

    // Remember to release resources
    ASSERT_EQ(spbla_Matrix_Free(matrix), SPBLA_STATUS_SUCCESS);
}

void testRun(spbla_Index m, spbla_Index n, spbla_Hints setup) {
    ASSERT_EQ(spbla_Initialize(setup), SPBLA_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixFilling(m, n, 0.001f + (0.05f) * ((float) i));
    }

    ASSERT_EQ(spbla_Finalize(), SPBLA_STATUS_SUCCESS);
}

#ifdef SPBLA_WITH_CUDA
TEST(spbla_Matrix, FillingSmallCuda) {
    spbla_Index m = 60, n = 100;
    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
}

TEST(spbla_Matrix, FillingMediumCuda) {
    spbla_Index m = 500, n = 1000;
    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
}

TEST(spbla_Matrix, FillingLargeCuda) {
    spbla_Index m = 1000, n = 2000;
    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
}
#endif

#ifdef SPBLA_WITH_OPENCL
TEST(spbla_Matrix, FillingSmallOpenCL) {
    spbla_Index m = 60, n = 100;
    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
}

TEST(spbla_Matrix, FillingMediumOpenCL) {
    spbla_Index m = 500, n = 1000;
    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
}

TEST(spbla_Matrix, FillingLargeOpenCL) {
    spbla_Index m = 1000, n = 2000;
    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
}
#endif

#ifdef SPBLA_WITH_SEQUENTIAL
TEST(spbla_Matrix, FillingSmallFallback) {
    spbla_Index m = 60, n = 100;
    testRun(m, n, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, FillingMediumFallback) {
    spbla_Index m = 500, n = 1000;
    testRun(m, n, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, FillingLargeFallback) {
    spbla_Index m = 1000, n = 2000;
    testRun(m, n, SPBLA_HINT_CPU_BACKEND);
}
#endif

SPBLA_GTEST_MAIN