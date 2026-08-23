#ifndef NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP

#include "common.hpp"

namespace np { namespace qsort_simd {

#include "x86_simd_qsort.dispatch.h"
// x86 SIMD sort is ascending-only; ``reverse`` must be false (asserted in
// the implementation).  The parameter exists purely so the dispatcher can
// share one function-pointer signature with the Highway path.
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size, bool reverse))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth))

#include "x86_simd_argsort.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void ArgQSort, (T *arr, npy_intp* arg, npy_intp size, bool reverse))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void ArgQSelect, (T *arr, npy_intp* arg, npy_intp kth, npy_intp size))

#include "x86_simd_qsort_16bit.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size, bool reverse))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth))

} } // np::qsort_simd

#endif // NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP
