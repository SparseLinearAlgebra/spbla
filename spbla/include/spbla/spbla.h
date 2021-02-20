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

#ifndef SPBLA_SPBLA_H
#define SPBLA_SPBLA_H

#ifdef __cplusplus
    #include <cinttypes>
#else
    #include <inttypes.h>
#endif

// Preserve C names in shared library
#define SPBLA_EXPORT extern "C"

// Exporting/importing symbols for Microsoft Visual Studio
#if (_MSC_VER && !__INTEL_COMPILER)
    #ifdef SPBLA_EXPORTS
        // Compile the library source code itself
        #define SPBLA_API __declspec(dllexport)
    #else
        // Import symbols from library into user space
        #define SPBLA_API __declspec(dllimport)
    #endif
#else
    // Default case
    #define SPBLA_API
#endif

/**
 * Info flags, returned by the library operations.
 */
typedef enum spbla_Info {
    /** Successful execution of the operation */
    SPBLA_INFO_SUCCESS = 0,
    /** Generic error code */
    SPBLA_INFO_ERROR = 1,
    /** Device side error */
    SPBLA_INFO_DEVICE_ERROR = 2,
    /** Failed to allocate memory on cpy or gpu side */
    SPBLA_INFO_MEM_OP_FAILED = 3,
    /** Passed invalid argument to some function */
    SPBLA_INFO_INVALID_ARGUMENT = 4,
    /** Call of the function is not possible for some context */
    SPBLA_INFO_INVALID_STATE = 5,
    /** Required backed not supported/present in the system */
    SPBLA_INFO_BACKEND_NOT_SUPPORTED = 6,
    /** Required functionality not implemented by the library */
    SPBLA_INFO_NOT_IMPLEMENTED = 7
} spbla_Info;

/**
 * Modes for library backend selection.
 */
typedef enum spbla_Backend {
    /** First supported backend will be selected */
    SPBLA_BACKEND_DEFAULT = 0,
    /** Attempts to initialize cuda based backend */
    SPBLA_BACKEND_CUDA = 1,
    /** Attempts to initialize opencl based backend */
    SPBLA_BACKEND_OPENCL = 2,
    /** Attempts to initialize cpu backend based on SuiteSparse library */
    SPBLA_BACKEND_CPU = 3
} spbla_Backend;

/**
 * Hint flags, passed to the library functions.
 */
typedef enum spbla_Hint {
    /** Pass, if no hints provided */
    SPBLA_HINT_NO = 0x0,
    /** Pass, if matrix input values are sorted in row-col order */
    SPBLA_HINT_VALUES_SORTED = 0x1
} spbla_HINT;

/** Hints mask */
SPBLA_EXPORT typedef uint32_t               spbla_Hints;

/** Index type used to enumerate matrix values */
SPBLA_EXPORT typedef uint32_t               spbla_Index;

/** Sparse matrix type exported by the library */
SPBLA_EXPORT typedef struct spbla_Matrix_t* spbla_Matrix;

/**
 * Initialize the library state before usage.
 * No other library function must be called before this one.
 * spbla_Finalize must be called as the last library function.
 *
 * Initialize function allows to provide hint to the library
 * for preferred backend selection. Default backend selection strategy
 * is following: cuda, opencl, cpu.
 *
 * Initialize function allows to pass init option directly to the
 * backend. This options are formed as array of string.
 * Prefix '-' used for options with value, '--' for options without values.
 *
 * @note Backend options:
 *         --CPU_IGNORE_FINALIZATION - do not finalize cpu backend (for testing purposes only)
 *
 * @param backend Preferred backend type. Pass SPBLA_BACKEND_DEFAULT as default value.
 * @param optionsCount Number of the init options, passed to the backend
 * @param options Actual array of the options, provided by the user
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Initialize(
    spbla_Backend       backend,
    spbla_Index         optionsCount,
    const char* const*  options
);

/**
 * Finalize the library state.
 * Must be called as the last library function in the application.
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Finalize(
);

/**
 *
 * @param matrix
 * @param nrows
 * @param ncols
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Matrix_New(
    spbla_Matrix*       matrix,
    spbla_Index         nrows,
    spbla_Index         ncols
);

/**
 *
 * @param matrix
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Matrix_Free(
    spbla_Matrix*       matrix
);

/**
 *
 * @param matrix
 * @param rows
 * @param cols
 * @param nvals
 * @param hints
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Matrix_Build(
    spbla_Matrix        matrix,
    const spbla_Index*  rows,
    const spbla_Index*  cols,
    spbla_Index         nvals,
    spbla_Hints         hints
);

/**
 *
 * @param matrix
 * @param rows
 * @param cols
 * @param nvals
 * @param hints
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Matrix_Extract(
    spbla_Matrix        matrix,
    spbla_Index*        rows,
    spbla_Index*        cols,
    spbla_Index*        nvals,
    spbla_Hints         hints
);

/**
 *
 * @param matrix
 * @param nrows
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Matrix_Nrows(
    spbla_Matrix        matrix,
    spbla_Index*        nrows
);

/**
 *
 * @param matrix
 * @param ncols
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Matrix_Ncols(
    spbla_Matrix        matrix,
    spbla_Index*        ncols
);

/**
 *
 * @param matrix
 * @param nvals
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Matrix_Nvals(
    spbla_Matrix        matrix,
    spbla_Index*        nvals
);

/**
 *
 * @param result
 * @param left
 * @param right
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Matrix_EWiseAdd(
    spbla_Matrix        result,
    spbla_Matrix        left,
    spbla_Matrix        right
);

/**
 *
 * @param result
 * @param left
 * @param right
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_MxM(
    spbla_Matrix        result,
    spbla_Matrix        left,
    spbla_Matrix        right
);

/**
 *
 * @param result
 * @param left
 * @param right
 *
 * @return Result info
 */
SPBLA_EXPORT SPBLA_API spbla_Info spbla_Kronecker(
    spbla_Matrix        result,
    spbla_Matrix        left,
    spbla_Matrix        right
);

#endif //SPBLA_SPBLA_H
