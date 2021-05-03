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

void testMatrixAdd(spbla_Index m, spbla_Index n, float density, spbla_Hints flags) {
    spbla_Matrix r, a, b;

    testing::Matrix ta = std::move(testing::Matrix::generateSparse(m, n, density));
    testing::Matrix tb = std::move(testing::Matrix::generateSparse(m, n, density));

    // Allocate input matrices and resize to fill with input data
    ASSERT_EQ(spbla_Matrix_New(&a, m, n), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_New(&b, m, n), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_New(&r, m, n), SPBLA_STATUS_SUCCESS);

    // Transfer input data into input matrices
    ASSERT_EQ(spbla_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, 0), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Build(b, tb.rowsIndex.data(), tb.colsIndex.data(), tb.nvals, 0), SPBLA_STATUS_SUCCESS);

    // Evaluate r = a + b
    ASSERT_EQ(spbla_Matrix_EWiseAdd(r, a, b, flags), SPBLA_STATUS_SUCCESS);

    // Evaluate naive r += a on the cpu to compare results
    testing::MatrixEWiseAddFunctor functor;
    auto tr = std::move(functor(ta, tb));

    // Compare results
    ASSERT_EQ(tr.areEqual(r), true);

    // Deallocate matrices
    ASSERT_EQ(spbla_Matrix_Free(a), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Free(b), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Free(r), SPBLA_STATUS_SUCCESS);
}

void testRun(spbla_Index m, spbla_Index n, spbla_Hints setup) {
    // Setup library
    ASSERT_EQ(spbla_Initialize(setup), SPBLA_STATUS_SUCCESS);

    for (size_t i = 0; i < 5; i++) {
        testMatrixAdd(m, n, 0.1f + (0.05f) * ((float) i), SPBLA_HINT_NO);
    }

    // Finalize library
    ASSERT_EQ(spbla_Finalize(), SPBLA_STATUS_SUCCESS);
}

#ifdef SPBLA_WITH_CUDA
TEST(spbla_Matrix, EWiseAddSmallCuda) {
    spbla_Index m = 60, n = 80;
    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
}

TEST(spbla_Matrix, EWiseAddMediumCuda) {
    spbla_Index m = 500, n = 800;
    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
}

TEST(spbla_Matrix, EWiseAddLargeCuda) {
    spbla_Index m = 2500, n = 1500;
    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
}
#endif

#ifdef SPBLA_WITH_OPENCL
TEST(spbla_Matrix, EWiseAddSmallOpenCL) {
    spbla_Index m = 60, n = 80;
    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
}

TEST(spbla_Matrix, EWiseAddMediumOpenCL) {
    spbla_Index m = 500, n = 800;
    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
}

TEST(spbla_Matrix, EWiseAddLargeOpenCL) {
    spbla_Index m = 2500, n = 1500;
    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
}
#endif

#ifdef SPBLA_WITH_SEQUENTIAL
TEST(spbla_Matrix, EWiseAddSmallFallback) {
    spbla_Index m = 60, n = 80;
    testRun(m, n, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, EWiseAddMediumFallback) {
    spbla_Index m = 500, n = 800;
    testRun(m, n, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, EWiseAddLargeFallback) {
    spbla_Index m = 2500, n = 1500;
    testRun(m, n, SPBLA_HINT_CPU_BACKEND);
}
#endif

SPBLA_GTEST_MAIN