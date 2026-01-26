#ifndef NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP

#include "common.hpp"

namespace np::highway::qsort_simd {

#include "highway_qsort.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size))

#include "highway_qsort_16bit.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size))

} // np::highway::qsort_simd

#endif // NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP
