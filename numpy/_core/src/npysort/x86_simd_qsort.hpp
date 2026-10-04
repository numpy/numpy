#ifndef NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP

#include "common.hpp"

namespace np { namespace qsort_simd {

// ``reverse`` selects descending order; NaNs sort last either way.
#include "x86_simd_qsort.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size, bool reverse))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth, bool reverse))

#include "x86_simd_argsort.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void ArgQSort, (T *arr, npy_intp* arg, npy_intp size, bool reverse))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void ArgQSelect, (T *arr, npy_intp* arg, npy_intp kth, npy_intp size, bool reverse))

#include "x86_simd_qsort_16bit.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size, bool reverse))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth, bool reverse))

} } // np::qsort_simd

#endif // NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP
