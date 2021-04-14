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

#ifndef SPBLA_SPBLA_H
#define SPBLA_SPBLA_H

#ifdef __cplusplus
    #include <cinttypes>
#else
    #include <inttypes.h>
#endif

// Preserve C names in shared library
#ifdef __cplusplus
    #define SPBLA_EXPORT extern "C"
#else
    #define SPBLA_EXPORT
#endif

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

/** Possible status codes that can be returned from spbla api */
typedef enum spbla_Status {
    /** Successful execution of the function */
    SPBLA_STATUS_SUCCESS = 0,
    /** Generic error code */
    SPBLA_STATUS_ERROR = 1,
    /** No cuda compatible device in the system */
    SPBLA_STATUS_DEVICE_NOT_PRESENT = 2,
    /** Device side error */
    SPBLA_STATUS_DEVICE_ERROR = 3,
    /** Failed to allocate memory on cpy or gpu side */
    SPBLA_STATUS_MEM_OP_FAILED = 4,
    /** Passed invalid argument to some function */
    SPBLA_STATUS_INVALID_ARGUMENT = 5,
    /** Call of the function is not possible for some context */
    SPBLA_STATUS_INVALID_STATE = 6,
    /** Failed to select supported backend for computations */
    SPBLA_STATUS_BACKEND_ERROR = 7,
    /** Some library feature is not implemented */
    SPBLA_STATUS_NOT_IMPLEMENTED = 8
} spbla_Status;

/** Generic lib hits for matrix processing */
typedef enum spbla_Hint {
    /** No hints passed */
    SPBLA_HINT_NO = 0,
    /** Force Cpu based backend usage */
    SPBLA_HINT_CPU_BACKEND = 1,
    /** Force Cuda based backend usage */
    SPBLA_HINT_CUDA_BACKEND = 2,
    /** Force OpenCL based backend usage */
    SPBLA_HINT_OPENCL_BACKEND = 4,
    /** Use managed gpu memory type instead of default (device) memory */
    SPBLA_HINT_GPU_MEM_MANAGED = 8,
    /** Mark input data as row-col sorted */
    SPBLA_HINT_VALUES_SORTED = 16,
    /** Accumulate result of the operation in the result matrix */
    SPBLA_HINT_ACCUMULATE = 32,
    /** Finalize library state, even if not all resources were explicitly released */
    SPBLA_HINT_RELAXED_FINALIZE = 64,
    /** Logging hint: log includes error message */
    SPBLA_HINT_LOG_ERROR = 128,
    /** Logging hint: log includes warning message */
    SPBLA_HINT_LOG_WARNING = 256,
    /** Logging hint: log includes all types of messages */
    SPBLA_HINT_LOG_ALL = 512,
    /** No duplicates in the build data */
    SPBLA_HINT_NO_DUPLICATES = 1024,
    /** Performs time measurement and logs elapsed operation time */
    SPBLA_HINT_TIME_CHECK = 2048
} spbla_Hint;

/** Hit mask */
typedef uint32_t spbla_Hints;

/** Alias integer type for indexing operations */
typedef uint32_t spbla_Index;

/** Cubool sparse boolean matrix handle */
typedef struct spbla_Matrix_t* spbla_Matrix;

/** Device capabilities */
typedef struct spbla_DeviceCaps {
    char name[256];
    bool cudaSupported;
    bool openclSupported;
    int major;
    int minor;
    int warp;
    unsigned long long globalMemoryKiBs;
    unsigned long long sharedMemoryPerMultiProcKiBs;
    unsigned long long sharedMemoryPerBlockKiBs;
} spbla_DeviceCaps;

/**
 * Query human-readable text info about the project implementation
 * @note It is safe to call this function before the library is initialized.
 *
 * @return Read-only library about info
 */
SPBLA_EXPORT SPBLA_API const char* spbla_GetAbout(
);

/**
 * Query human-readable text info about the project implementation
 * @note It is safe to call this function before the library is initialized.

 * @return Read-only library license info
 */
SPBLA_EXPORT SPBLA_API const char* spbla_GetLicenseInfo(
);

/**
 * Query library version number in form MAJOR.MINOR
 * @note It is safe to call this function before the library is initialized.
 *
 * @param major Major version number part
 * @param minor Minor version number part
 * @param sub Sub version number part
 *
 * @return Error if failed to query version info
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_GetVersion(
    int* major,
    int* minor,
    int* sub
);

/**
 * Allows to setup logging file for all operations, invoked after this function call.
 * Empty hints field is interpreted as `SPBLA_HINT_LOG_ALL` by default.
 *
 * @note It is safe to call this function before the library is initialized.
 *
 * @note Pass `SPBLA_HINT_LOG_ERROR` to include error messages into log
 * @note Pass `SPBLA_HINT_LOG_WARNING` to include warning messages into log
 * @note Pass `SPBLA_HINT_LOG_ALL` to include all messages into log
 *
 * @param logFileName UTF-8 encoded null-terminated file name and path string.
 * @param hints Logging hints to filter messages.
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_SetupLogging(
    const char* logFileName,
    spbla_Hints hints
);

/**
 * Initialize library instance object, which provides context to all library operations and primitives.
 * This function must be called before any other library function is called,
 * except first get-info functions.
 *
 * @note Pass `SPBLA_HINT_RELAXED_FINALIZE` for library setup within python.
 *
 * @param hints Init hints.
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Initialize(
    spbla_Hints hints
);

/**
 * Finalize library state and all objects, which were created on this library context.
 * This function always must be called as the last library function in the application.
 *
 * @note Pass `SPBLA_HINT_RELAXED_FINALIZE` for library init call, if relaxed finalize is required.
 * @note Invalidates all handle to the resources, created within this library instance
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Finalize(
);

/**
 * Query device capabilities/properties if cuda/opencl compatible device is present.
 *
 * @note This function returns no actual info if cuda/opencl backend is not presented.
 * @param deviceCaps Pointer to device caps structure to store result
 *
 * @return Error if cuda/opencl device not present or if failed to query capabilities
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_GetDeviceCaps(
    spbla_DeviceCaps* deviceCaps
);

/**
 * Creates new sparse matrix with specified size.
 *
 * @param matrix Pointer where to store created matrix handle
 * @param nrows Matrix rows count
 * @param ncols Matrix columns count
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_New(
    spbla_Matrix* matrix,
    spbla_Index nrows,
    spbla_Index ncols
);

/**
 * Build sparse matrix from provided pairs array. Pairs are supposed to be stored
 * as (rows[i],cols[i]) for pair with i-th index.
 *
 * @note This function automatically reduces duplicates
 * @note Pass `SPBLA_HINT_VALUES_SORTED` if values already in the row-col order.
 * @note Pass `SPBLA_HINT_NO_DUPLICATES` if values has no duplicates
 *
 * @param matrix Matrix handle to perform operation on
 * @param rows Array of pairs row indices
 * @param cols Array of pairs column indices
 * @param nvals Number of the pairs passed
 * @param hints Hits flags for processing.
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_Build(
    spbla_Matrix matrix,
    const spbla_Index* rows,
    const spbla_Index* cols,
    spbla_Index nvals,
    spbla_Hints hints
);

/**
 * Sets specified (i, j) value of the matrix to True.
 *
 * @note This function automatically reduces duplicates
 *
 * @param matrix Matrix handle to perform operation on
 * @param i Row index
 * @param j Column Index
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_SetElement(
    spbla_Matrix matrix,
    spbla_Index i,
    spbla_Index j
);

/**
 * Sets to the matrix specific debug string marker.
 * This marker will appear in the log messages as string identifier of the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param marker UTF-8 encoded null-terminated string marker name.
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_SetMarker(
    spbla_Matrix matrix,
    const char* marker
);

/**
 * Allows to get matrix debug string marker.
 *
 * @note Pass null marker if you want to retrieve only the required marker buffer size.
 * @note After the function call the actual size of the marker is stored in the size variable.
 *
 * @note size is set to the actual marker length plus null terminator symbol.
 *       For marker "matrix" size variable will be set to 7.
 *
 * @param matrix Matrix handle to perform operation on
 * @param[in,out] marker Where to store null-terminated UTF-8 encoded marker string.
 * @param[in,out] size Size of the provided buffer in bytes to save marker string.
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_Marker(
    spbla_Matrix matrix,
    char* marker,
    spbla_Index* size
);

/**
 * Reads matrix data to the host visible CPU buffer as an array of values pair.
 *
 * The arrays must be provided by the user and the size of this arrays must
 * be greater or equal the values count of the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param[in,out] rows Buffer to store row indices
 * @param[in,out] cols Buffer to store column indices
 * @param[in,out] nvals Total number of the pairs
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_ExtractPairs(
    spbla_Matrix matrix,
    spbla_Index* rows,
    spbla_Index* cols,
    spbla_Index* nvals
);

/**
 * Extracts sub-matrix of the input matrix and stores it into result matrix.
 *
 * @note The result sub-matrix have nrows x ncols dimension,
 *       and includes [i;i+nrow) x [j;j+ncols) cells of the input matrix.
 *
 * @note Result matrix must have compatible size
 *          dim(result) = nrows x ncols
 *
 * @note Provided sub-matrix region must be within the input matrix.
 *
 * @note Pass `SPBLA_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Matrix handle where to store result of the operation
 * @param matrix Input matrix to extract values from
 * @param i First row id to extract
 * @param j First column id to extract
 * @param nrows Number of rows to extract
 * @param ncols Number of columns to extract
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_ExtractSubMatrix(
    spbla_Matrix result,
    spbla_Matrix matrix,
    spbla_Index i,
    spbla_Index j,
    spbla_Index nrows,
    spbla_Index ncols,
    spbla_Hints hints
);

/**
 * Creates new sparse matrix, duplicates content and stores handle in the provided pointer.
 * 
 * @param matrix Matrix handle to perform operation on
 * @param duplicated[out] Pointer to the matrix handle where to create and store created matrix
 * 
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_Duplicate(
    spbla_Matrix matrix,
    spbla_Matrix* duplicated
);

/**
 * Transpose source matrix and store result of this operation in result matrix.
 * Formally: result = matrix ^ T.
 *
 * @note Pass `SPBLA_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Matrix handle to store result of the operation
 * @param matrix The source matrix
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_Transpose(
    spbla_Matrix result,
    spbla_Matrix matrix,
    spbla_Hints hints
);

/**
 * Query number of non-zero values of the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param nvals[out] Pointer to the place where to store number of the non-zero elements of the matrix
 * 
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_Nvals(
    spbla_Matrix matrix,
    spbla_Index* nvals
);

/**
 * Query number of rows in the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param nrows[out] Pointer to the place where to store number of matrix rows
 * 
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_Nrows(
    spbla_Matrix matrix,
    spbla_Index* nrows
);

/**
 * Query number of columns in the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param ncols[out] Pointer to the place where to store number of matrix columns
 * 
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_Ncols(
    spbla_Matrix matrix,
    spbla_Index* ncols
);

/**
 * Deletes sparse matrix object.
 *
 * @param matrix Matrix handle to delete the matrix
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_Free(
    spbla_Matrix matrix
);

/**
 * Reduce the source matrix to the column matrix result (column vector).
 * Formally: result = sum(cols of matrix).

 * @note Matrices must be compatible
 *          dim(matrix) = M x N
 *          dim(result) = M x 1
 *
 * @note Pass `SPBLA_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Matrix hnd where to store result
 * @param matrix Source matrix to reduce
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_Reduce(
    spbla_Matrix result,
    spbla_Matrix matrix,
    spbla_Hints hints
);

/**
 * Performs result = left + right, where '+' is boolean semiring operation.
 *
 * @note Matrices must be compatible
 *          dim(result) = M x N
 *          dim(left) = M x N
 *          dim(right) = M x N
 *
 * @note Pass `SPBLA_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Destination matrix for add-and-assign operation
 * @param left Source matrix to be added
 * @param right Source matrix to be added
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Matrix_EWiseAdd(
    spbla_Matrix result,
    spbla_Matrix left,
    spbla_Matrix right,
    spbla_Hints hints
);

/**
 * Performs result (accum)= left x right evaluation, where source '+' and 'x' are boolean semiring operations.
 * If accum hint passed, the the result of the multiplication is added to the result matrix.
 *
 * @note To perform this operation matrices must be compatible
 *          dim(left) = M x T
 *          dim(right) = T x N
 *          dim(result) = M x N
 *
 * @note Pass `SPBLA_HINT_ACCUMULATE` hint to add result of the left x right operation.
 * @note Pass `SPBLA_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Matrix handle where to store operation result
 * @param left Input left matrix
 * @param right Input right matrix
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_MxM(
    spbla_Matrix result,
    spbla_Matrix left,
    spbla_Matrix right,
    spbla_Hints hints
);

/**
 * Performs result = left `kron` right, where `kron` is a Kronecker product for boolean semiring.
 *
 * @note When the operation is performed, the result matrix has the following dimension
 *          dim(left) = M x N
 *          dim(right) = K x T
 *          dim(result) = MK x NT
 *
 * @note Pass `SPBLA_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Matrix handle where to store operation result
 * @param left Input left matrix
 * @param right Input right matrix
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
SPBLA_EXPORT SPBLA_API spbla_Status spbla_Kronecker(
    spbla_Matrix result,
    spbla_Matrix left,
    spbla_Matrix right,
    spbla_Hints hints
);

#endif //SPBLA_SPBLA_H
