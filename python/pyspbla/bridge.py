"""
spbla C API Python bridge.

Wraps native C API details for accessing
this functionality via Python CTypes library.

Functionality:
- Flags wrapping
- Functions definitions
- Error checking
"""

import ctypes

__all__ = [
    "load_and_configure",
    "get_init_hints",
    "get_build_hints",
    "get_sub_matrix_hints",
    "get_transpose_hints",
    "get_reduce_hints",
    "get_kronecker_hints",
    "get_mxm_hints",
    "get_ewiseadd_hints",
    "check"
]

_hint_no = 0
_hint_cpu_backend = 1
_hint_cuda_backend = 2
_hint_opencl_backend = 4
_hint_gpu_mem_managed = 8
_hint_values_sorted = 16
_hint_accumulate = 32
_hint_relaxed_release = 64
_hint_log_error = 128
_hint_log_warning = 256
_hint_log_all = 512
_hint_no_duplicates = 1024
_hint_time_check = 2048

_backend_name_cpu = "cpu"
_backend_name_cuda = "cuda"
_backend_name_opencl = "opencl"


def get_log_hints(default=True, error=False, warning=False):
    hints = _hint_no

    if default:
        hints |= _hint_log_all
    if error:
        hints |= _hint_log_error
    if warning:
        hints |= _hint_log_warning

    return hints


def get_init_hints(backend_type: str):
    assert backend_type

    hints = _hint_relaxed_release

    if backend_type == _backend_name_cpu:
        hints |= _hint_cpu_backend
    elif backend_type == _backend_name_cuda:
        hints |= _hint_cuda_backend
    elif backend_type == _backend_name_opencl:
        hints |= _hint_opencl_backend

    return hints


def get_sub_matrix_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_transpose_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_reduce_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_kronecker_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_mxm_hints(is_accumulated, time_check):
    hints = _hint_no

    if is_accumulated:
        hints |= _hint_accumulate
    if time_check:
        hints |= _hint_time_check

    return hints


def get_ewiseadd_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_build_hints(is_sorted, no_duplicates):
    hints = _hint_no

    if is_sorted:
        hints |= _hint_values_sorted
    if no_duplicates:
        hints |= _hint_no_duplicates

    return hints


def load_and_configure(cubool_lib_path: str):
    lib = ctypes.cdll.LoadLibrary(cubool_lib_path)

    status_t = ctypes.c_uint
    index_t = ctypes.c_uint
    hints_t = ctypes.c_uint
    matrix_p = ctypes.c_void_p
    p_to_matrix_p = ctypes.POINTER(matrix_p)

    lib.spbla_SetupLogging.restype = status_t
    lib.spbla_SetupLogging.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        hints_t
    ]

    lib.spbla_Initialize.restype = status_t
    lib.spbla_Initialize.argtypes = [
        hints_t
    ]

    lib.spbla_Finalize.restype = status_t
    lib.spbla_Finalize.argtypes = []

    lib.spbla_Matrix_New.restype = status_t
    lib.spbla_Matrix_New.argtypes = [
        p_to_matrix_p,
        ctypes.c_uint,
        ctypes.c_uint
    ]

    lib.spbla_Matrix_Free.restype = status_t
    lib.spbla_Matrix_Free.argtypes = [
        matrix_p
    ]

    lib.spbla_Matrix_Build.restype = status_t
    lib.spbla_Matrix_Build.argtypes = [
        matrix_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_uint,
        hints_t
    ]

    lib.spbla_Matrix_SetElement.restype = status_t
    lib.spbla_Matrix_SetElement.argtypes = [
        matrix_p,
        ctypes.c_uint,
        ctypes.c_uint
    ]

    lib.spbla_Matrix_SetMarker.restype = status_t
    lib.spbla_Matrix_SetMarker.argtypes = [
        matrix_p,
        ctypes.POINTER(ctypes.c_char)
    ]

    lib.spbla_Matrix_Marker.restype = status_t
    lib.spbla_Matrix_Marker.argtypes = [
        matrix_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_uint)
    ]

    lib.spbla_Matrix_ExtractPairs.restype = status_t
    lib.spbla_Matrix_ExtractPairs.argtypes = [
        matrix_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint)
    ]

    lib.spbla_Matrix_ExtractSubMatrix.restype = status_t
    lib.spbla_Matrix_ExtractSubMatrix.argtypes = [
        matrix_p,
        matrix_p,
        index_t,
        index_t,
        index_t,
        index_t,
        hints_t
    ]

    lib.spbla_Matrix_Duplicate.restype = status_t
    lib.spbla_Matrix_Duplicate.argtypes = [
        matrix_p,
        p_to_matrix_p
    ]

    lib.spbla_Matrix_Transpose.restype = status_t
    lib.spbla_Matrix_Transpose.argtypes = [
        matrix_p,
        matrix_p,
        hints_t
    ]

    lib.spbla_Matrix_Nrows.restype = status_t
    lib.spbla_Matrix_Nrows.argtype = [
        matrix_p,
        ctypes.POINTER(ctypes.c_uint)
    ]

    lib.spbla_Matrix_Ncols.restype = status_t
    lib.spbla_Matrix_Ncols.argtype = [
        matrix_p,
        ctypes.POINTER(ctypes.c_uint)
    ]

    lib.spbla_Matrix_Nvals.restype = status_t
    lib.spbla_Matrix_Nvals.spbla_Matrix_Reduce = [
        matrix_p,
        ctypes.POINTER(ctypes.c_size_t)
    ]

    lib.spbla_Matrix_Reduce.restype = status_t
    lib.spbla_Matrix_Reduce.argtype = [
        matrix_p,
        matrix_p,
        hints_t
    ]

    lib.spbla_Matrix_EWiseAdd.restype = status_t
    lib.spbla_Matrix_EWiseAdd.argtypes = [
        matrix_p,
        matrix_p,
        matrix_p,
        hints_t
    ]

    lib.spbla_MxM.restype = status_t
    lib.spbla_MxM.argtypes = [
        matrix_p,
        matrix_p,
        matrix_p,
        hints_t
    ]

    lib.spbla_Kronecker.restype = status_t
    lib.spbla_Kronecker.argtypes = [
        matrix_p,
        matrix_p,
        matrix_p,
        hints_t
    ]

    return lib


"""
/** Possible status codes that can be returned from spbla api */

/** Successful execution of the function */
SPBLA_STATUS_SUCCESS,

/** Generic error code */
SPBLA_STATUS_ERROR,

/** No cuda compatible device in the system */
SPBLA_STATUS_DEVICE_NOT_PRESENT,

/** Device side error */
SPBLA_STATUS_DEVICE_ERROR,

/** Failed to allocate memory on cpy or gpu side */
SPBLA_STATUS_MEM_OP_FAILED,

/** Passed invalid argument to some function */
SPBLA_STATUS_INVALID_ARGUMENT,

/** Call of the function is not possible for some context */
SPBLA_STATUS_INVALID_STATE

/** Failed to select supported backend for computations */
SPBLA_STATUS_BACKEND_ERROR

/** Some library feature is not implemented */
SPBLA_STATUS_NOT_IMPLEMENTED
"""
_status_codes_mappings = {
    0: "SPBLA_STATUS_SUCCESS",
    1: "SPBLA_STATUS_ERROR",
    2: "SPBLA_STATUS_DEVICE_NOT_PRESENT",
    3: "SPBLA_STATUS_DEVICE_ERROR",
    4: "SPBLA_STATUS_MEM_OP_FAILED",
    5: "SPBLA_STATUS_INVALID_ARGUMENT",
    6: "SPBLA_STATUS_INVALID_STATE",
    7: "SPBLA_STATUS_BACKEND_ERROR",
    8: "SPBLA_STATUS_NOT_IMPLEMENTED"
}

_success = 0


def check(status_code):
    if status_code != _success:
        raise Exception(_status_codes_mappings[status_code])
