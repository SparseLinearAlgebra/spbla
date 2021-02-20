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

#include <testing/Testing.hpp>

void LibrarySetup(spbla_Backend backend) {
    EXPECT_EQ(spbla_Initialize(backend), SPBLA_INFO_SUCCESS);
    EXPECT_EQ(spbla_Finalize(), SPBLA_INFO_SUCCESS);
}

TEST(spbla_Library, SetupDefault) {
    LibrarySetup(SPBLA_BACKEND_DEFAULT);
}

TEST(spbla_Library, SetupCuda) {
    LibrarySetup(SPBLA_BACKEND_CUDA);
}

TEST(spbla_Library, SetupOpenCL) {
    LibrarySetup(SPBLA_BACKEND_OPENCL);
}

TEST(spbla_Library, SetupCpu) {
    LibrarySetup(SPBLA_BACKEND_CPU);
}

SPBLA_GTEST_MAIN