#ifndef NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP

#include "hwy/highway.h"

#include "common.hpp"

// This replicates VQSORT_ENABLED from hwy/contrib/sort/shared-inl.h
// without checking the scalar target as this is not built within the dynamic
// dispatched sources.
#if (HWY_COMPILER_MSVC && !HWY_IS_DEBUG_BUILD) ||                   \
    (HWY_ARCH_ARM_V7 && HWY_IS_DEBUG_BUILD) ||                      \
    (HWY_ARCH_ARM_A64 && HWY_COMPILER_GCC_ACTUAL && HWY_IS_ASAN) || \
    (HWY_ARCH_ARM_A64 && HWY_COMPILER_CLANG &&                      \
    (HWY_IS_HWASAN || HWY_IS_MSAN || HWY_IS_TSAN || HWY_IS_ASAN))
#define NPY_DISABLE_HIGHWAY_SORT
#endif

#ifndef NPY_DISABLE_HIGHWAY_SORT
namespace np { namespace highway { namespace qsort_simd {

#ifndef NPY_DISABLE_OPTIMIZATION
    #include "highway_qsort.dispatch.h"
#endif
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth))


#ifndef NPY_DISABLE_OPTIMIZATION
    #include "highway_qsort_16bit.dispatch.h"
#endif
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth))

} } } // np::highway::qsort_simd

#endif // NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP
#endif // NPY_DISABLE_HIGHWAY_SORT
