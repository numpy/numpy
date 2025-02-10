#ifndef NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP

#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "common.hpp"

#if !VQSORT_COMPILER_COMPATIBLE
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
