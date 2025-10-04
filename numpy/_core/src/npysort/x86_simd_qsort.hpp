#ifndef NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP

#include "common.hpp"

namespace np { namespace qsort_simd {

#include "x86_simd_argsort.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void ArgQSort, (T *arr, npy_intp* arg, npy_intp size))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void ArgQSelect, (T *arr, npy_intp* arg, npy_intp kth, npy_intp size))

} } // np::qsort_simd

#endif // NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP
