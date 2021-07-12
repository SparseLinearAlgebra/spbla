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

#include <gtest/gtest.h>
#include <testing/testing.hpp>

// Query library version info
TEST(spblaVersion, Query) {
    int major;
    int minor;
    int sub;

    spbla_GetVersion(&major, &minor, &sub);

    std::cout << "Major: " << major << std::endl;
    std::cout << "Minor: " << minor << std::endl;
    std::cout << "Sub: " << sub << std::endl;

    EXPECT_NE(major | minor | sub, 0);
}

// Test spbla library instance creation and destruction
TEST(spbla, Setup) {
    spbla_Status error;

    error = spbla_Initialize(SPBLA_HINT_NO);
    ASSERT_EQ(error, SPBLA_STATUS_SUCCESS);

    error = spbla_Finalize();
    ASSERT_EQ(error, SPBLA_STATUS_SUCCESS);
}

/**
 * Performs transitive closure for directed graph
 *
 * @param A Adjacency matrix of the graph
 * @param T Reference to the handle where to allocate and store result
 *
 * @return Status on this operation
 */
spbla_Status TransitiveClosure(spbla_Matrix A, spbla_Matrix* T) {
    spbla_Matrix_Duplicate(A, T);                              /* Duplicate A to result T */

    spbla_Index total = 0;
    spbla_Index current;

    spbla_Matrix_Nvals(*T, &current);                          /* Query current nvals value */

    while (current != total) {                                 /* Iterate, while new values are added */
        total = current;
        spbla_MxM(*T, *T, *T, SPBLA_HINT_ACCUMULATE);   /* T += T x T */
        spbla_Matrix_Nvals(*T, &current);
    }

    return SPBLA_STATUS_SUCCESS;
}

TEST(spbla, Example) {
    spbla_Matrix A = nullptr;
    spbla_Matrix T = nullptr;

    spbla_Index n = 100;

    spbla_Initialize(SPBLA_HINT_NO);
    spbla_Matrix_New(&A, n, n);

    testing::Matrix ta = testing::Matrix::generateSparse(n , n, 0.2);

    spbla_Matrix_Build(A, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, SPBLA_HINT_VALUES_SORTED);

    TransitiveClosure(A, &T);

    testing::Matrix tr = ta;
    size_t total;

    do {
        total = tr.nvals;

        testing::MatrixMultiplyFunctor functor;
        tr = std::move(functor(tr, tr, tr, true));
    }
    while (tr.nvals != total);

    ASSERT_TRUE(tr.areEqual(T));

    spbla_Matrix_Free(A);
    spbla_Matrix_Free(T);
    spbla_Finalize();
}

TEST(spbla, Logger) {
    const char* logFileName = "testLog.txt";

    ASSERT_EQ(spbla_SetupLogging(logFileName, SPBLA_HINT_LOG_ALL), SPBLA_STATUS_SUCCESS);

    ASSERT_EQ(spbla_Initialize(SPBLA_HINT_RELAXED_FINALIZE), SPBLA_STATUS_SUCCESS);
    spbla_Matrix_New(nullptr, 0, 0);
    spbla_Matrix_New((spbla_Matrix*) 0xffff, 0, 0);
    spbla_MxM((spbla_Matrix) 0xffff, (spbla_Matrix) 0xffff, (spbla_Matrix) nullptr, 0x0);
    ASSERT_EQ(spbla_Finalize(), SPBLA_STATUS_SUCCESS);
}

TEST(spbla, DeviceCaps) {
    spbla_DeviceCaps caps;

    ASSERT_EQ(spbla_Initialize(SPBLA_HINT_RELAXED_FINALIZE), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_GetDeviceCaps(&caps), SPBLA_STATUS_SUCCESS);

    std::cout
        << "name: " << caps.name << std::endl
        << "major: " << caps.major << std::endl
        << "minor: " << caps.minor << std::endl
        << "cuda supported: " << caps.cudaSupported << std::endl
        << "opencl supported: " << caps.openclSupported << std::endl
        << "warp: " << caps.warp << std::endl
        << "globalMemoryKiBs: " << caps.globalMemoryKiBs << std::endl
        << "sharedMemoryPerMultiProcKiBs: " << caps.sharedMemoryPerMultiProcKiBs << std::endl
        << "sharedMemoryPerBlockKiBs: " << caps.sharedMemoryPerBlockKiBs << std::endl;

    ASSERT_EQ(spbla_Finalize(), SPBLA_STATUS_SUCCESS);
}

SPBLA_GTEST_MAIN
