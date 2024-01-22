/**
 * This module provides the inner loops for the clip ufunc
 */
#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/halffloat.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"
#include "numpy/utils.h"

#include "fast_loop_macros.h"


#include "common.hpp"

namespace {

template <typename T>
inline std::enable_if_t<np::kIsComplex<T>, bool> LessThan(const T &a, const T &b)
{
    auto a_real = a.real();
    auto a_imag = a.imag();
    auto b_real = b.real();
    auto b_imag = b.imag();
    return a_real == b_real ? a_imag < b_imag : a_real < b_real;
}

template <typename T>
inline std::enable_if_t<np::kIsComplex<T>, bool> GreatThan(const T &a, const T &b)
{
    auto a_real = a.real();
    auto a_imag = a.imag();
    auto b_real = b.real();
    auto b_imag = b.imag();
    return a_real == b_real ? a_imag > b_imag : a_real > b_real;
}

template <typename T>
inline T Min(const T &a, const T &b)
{
    if constexpr (std::is_same_v<T, np::Half>) {
        return a.IsNaN() || a < b ? a : b;
    }
    else if constexpr (np::kIsFloat<T>) {
        return std::isnan(a) || a < b  ? a : b;
    }
    else if constexpr (np::kIsComplex<T>) {
        auto a_real = a.real();
        auto a_imag = a.imag();
        // auto b_real = b.real();
        // auto b_imag = b.imag();
        return std::isnan(a_real) || std::isnan(a_imag) || LessThan(a, b) ? a : b;
    }
    else if constexpr (np::kIsTime<T>) {
        return a.IsNaT() || a < b ? a : b;
    }
    else {
        return std::min(a, b);
    }
}

template <typename T>
inline T Max(const T &a, const T &b)
{
    if constexpr (std::is_same_v<T, np::Half>) {
        return a.IsNaN() || a > b ?  a : b;
    }
    else if constexpr (np::kIsFloat<T>) {
        return std::isnan(a) || a > b  ? a : b;
    }
    else if constexpr (np::kIsComplex<T>) {
        auto a_real = a.real();
        auto a_imag = a.imag();
        // auto b_real = b.real();
        // auto b_imag = b.imag();
        return std::isnan(a_real) || std::isnan(a_imag) || GreatThan(a, b) ? a : b;
    }
    else if constexpr (np::kIsTime<T>) {
        return a.IsNaT() || a > b ? a : b;
    }
    else {
        return std::max(a, b);
    }
}

template <typename T>
inline T Clip(const T &x, const T &min, const T &max)
{
    return Min(Max(x, min), max);
}

template <typename T>
inline void Clip(T **args, np::SSize const *dimensions, np::SSize const *steps)
{
    np::SSize n = dimensions[0];
    if (steps[1] == 0 && steps[2] == 0) {
        /* min and max are constant throughout the loop, the most common case
         */
        /* NOTE: it may be possible to optimize these checks for nan */
        T min_val = *args[1];
        T max_val = *args[2];

        T *ip1 = args[0], *op1 = args[3];
        np::SSize is1 = steps[0] / sizeof(T), os1 = steps[3] / sizeof(T);

        /* contiguous, branch to let the compiler optimize */
        if (is1 == 1 && os1 == 1) {
            for (np::SSize i = 0; i < n; i++, ip1++, op1++) {
                *op1 = Clip(*ip1, min_val, max_val);
            }
        }
        else {
            for (np::SSize i = 0; i < n; i++, ip1 += is1, op1 += os1) {
                *op1 = Clip(*ip1, min_val, max_val);
            }
        }
    }
    else {
        T *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];
        np::SSize is1 = steps[0] / sizeof(T), is2 = steps[1] / sizeof(T),
                 is3 = steps[2] / sizeof(T), os1 = steps[3] / sizeof(T);
        for (np::SSize i = 0; i < n;
             i++, ip1 += is1, ip2 += is2, ip3 += is3, op1 += os1)
            *op1 = Clip(*ip1, *ip2, *ip3);
    }
    npy_clear_floatstatus_barrier((char *)dimensions);
}
} // namespace

extern "C" {

NPY_NO_EXPORT void
BOOL_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return Clip((np::Bool**)args, dimensions, steps);
}
NPY_NO_EXPORT void
BYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return Clip((np::Byte**)args, dimensions, steps);
}
NPY_NO_EXPORT void
UBYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    return Clip((np::UByte**)args, dimensions, steps);
}
NPY_NO_EXPORT void
SHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    return Clip((np::Short**)args, dimensions, steps);
}
NPY_NO_EXPORT void
USHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func))
{
    return Clip((np::UShort**)args, dimensions, steps);
}
NPY_NO_EXPORT void
INT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
         void *NPY_UNUSED(func))
{
    return Clip((np::Int**)args, dimensions, steps);
}
NPY_NO_EXPORT void
UINT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return Clip((np::UInt**)args, dimensions, steps);
}
NPY_NO_EXPORT void
LONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return Clip((np::Long**)args, dimensions, steps);
}
NPY_NO_EXPORT void
ULONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    return Clip((np::ULong**)args, dimensions, steps);
}
NPY_NO_EXPORT void
LONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func))
{
    return Clip((np::LongLong**)args, dimensions, steps);
}
NPY_NO_EXPORT void
ULONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func))
{
    return Clip((np::ULongLong**)args, dimensions, steps);
}
NPY_NO_EXPORT void
HALF_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return Clip((np::Half**)args, dimensions, steps);
}
NPY_NO_EXPORT void
FLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    return Clip((np::Float**)args, dimensions, steps);
}
NPY_NO_EXPORT void
DOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func))
{
    return Clip((np::Double**)args, dimensions, steps);
}
NPY_NO_EXPORT void
LONGDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
                void *NPY_UNUSED(func))
{
    return Clip((np::LongDouble**)args, dimensions, steps);
}
NPY_NO_EXPORT void
CFLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func))
{
    return Clip((np::CFloat**)args, dimensions, steps);
}
NPY_NO_EXPORT void
CDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    return Clip((np::CDouble**)args, dimensions, steps);
}
NPY_NO_EXPORT void
CLONGDOUBLE_clip(char **args, npy_intp const *dimensions,
                 npy_intp const *steps, void *NPY_UNUSED(func))
{
    return Clip((np::CLongDouble**)args, dimensions, steps);
}
NPY_NO_EXPORT void
DATETIME_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func))
{
    return Clip((np::DateTime**)args, dimensions, steps);
}
NPY_NO_EXPORT void
TIMEDELTA_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func))
{
    return Clip((np::TimeDelta**)args, dimensions, steps);
}
}
