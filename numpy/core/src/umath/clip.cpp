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

#include "../common/numpy_tag.h"

template <class T>
T
_NPY_MIN(T a, T b, npy::integral_tag const &)
{
    return PyArray_MIN(a, b);
}
template <class T>
T
_NPY_MAX(T a, T b, npy::integral_tag const &)
{
    return PyArray_MAX(a, b);
}

npy_half
_NPY_MIN(npy_half a, npy_half b, npy::half_tag const &)
{
    return npy_half_isnan(a) || npy_half_le(a, b) ? (a) : (b);
}
npy_half
_NPY_MAX(npy_half a, npy_half b, npy::half_tag const &)
{
    return npy_half_isnan(a) || npy_half_ge(a, b) ? (a) : (b);
}

template <class T>
T
_NPY_MIN(T a, T b, npy::floating_point_tag const &)
{
    return npy_isnan(a) ? (a) : PyArray_MIN(a, b);
}
template <class T>
T
_NPY_MAX(T a, T b, npy::floating_point_tag const &)
{
    return npy_isnan(a) ? (a) : PyArray_MAX(a, b);
}

template <class T>
T
_NPY_MIN(T a, T b, npy::complex_tag const &)
{
    return npy_isnan((a).real) || npy_isnan((a).imag) || PyArray_CLT(a, b)
                   ? (a)
                   : (b);
}
template <class T>
T
_NPY_MAX(T a, T b, npy::complex_tag const &)
{
    return npy_isnan((a).real) || npy_isnan((a).imag) || PyArray_CGT(a, b)
                   ? (a)
                   : (b);
}

template <class T>
T
_NPY_MIN(T a, T b, npy::date_tag const &)
{
    return (a) == NPY_DATETIME_NAT   ? (a)
           : (b) == NPY_DATETIME_NAT ? (b)
           : (a) < (b)               ? (a)
                                     : (b);
}
template <class T>
T
_NPY_MAX(T a, T b, npy::date_tag const &)
{
    return (a) == NPY_DATETIME_NAT   ? (a)
           : (b) == NPY_DATETIME_NAT ? (b)
           : (a) > (b)               ? (a)
                                     : (b);
}

/* generic dispatcher */
template <class Tag, class T = typename Tag::type>
T
_NPY_MIN(T const &a, T const &b)
{
    return _NPY_MIN(a, b, Tag{});
}
template <class Tag, class T = typename Tag::type>
T
_NPY_MAX(T const &a, T const &b)
{
    return _NPY_MAX(a, b, Tag{});
}

template <class Tag, class T>
T
_NPY_CLIP(T x, T min, T max)
{
    return _NPY_MIN<Tag>(_NPY_MAX<Tag>((x), (min)), (max));
}

template <class Tag, class T = typename Tag::type>
static void
_npy_clip_(T **args, npy_intp const *dimensions, npy_intp const *steps)
{
    npy_intp n = dimensions[0];
    if (steps[1] == 0 && steps[2] == 0) {
        /* min and max are constant throughout the loop, the most common case
         */
        /* NOTE: it may be possible to optimize these checks for nan */
        T min_val = *args[1];
        T max_val = *args[2];

        T *ip1 = args[0], *op1 = args[3];
        npy_intp is1 = steps[0] / sizeof(T), os1 = steps[3] / sizeof(T);

        /* contiguous, branch to let the compiler optimize */
        if (is1 == 1 && os1 == 1) {
            for (npy_intp i = 0; i < n; i++, ip1++, op1++) {
                *op1 = _NPY_CLIP<Tag>(*ip1, min_val, max_val);
            }
        }
        else {
            for (npy_intp i = 0; i < n; i++, ip1 += is1, op1 += os1) {
                *op1 = _NPY_CLIP<Tag>(*ip1, min_val, max_val);
            }
        }
    }
    else {
        T *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];
        npy_intp is1 = steps[0] / sizeof(T), is2 = steps[1] / sizeof(T),
                 is3 = steps[2] / sizeof(T), os1 = steps[3] / sizeof(T);
        for (npy_intp i = 0; i < n;
             i++, ip1 += is1, ip2 += is2, ip3 += is3, op1 += os1)
            *op1 = _NPY_CLIP<Tag>(*ip1, *ip2, *ip3);
    }
    npy_clear_floatstatus_barrier((char *)dimensions);
}

template <class Tag>
static void
_npy_clip(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    using T = typename Tag::type;
    return _npy_clip_<Tag>((T **)args, dimensions, steps);
}

extern "C" {
NPY_NO_EXPORT void
BOOL_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return _npy_clip<npy::bool_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
BYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return _npy_clip<npy::byte_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
UBYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    return _npy_clip<npy::ubyte_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
SHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    return _npy_clip<npy::short_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
USHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func))
{
    return _npy_clip<npy::ushort_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
INT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
         void *NPY_UNUSED(func))
{
    return _npy_clip<npy::int_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
UINT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return _npy_clip<npy::uint_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
LONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return _npy_clip<npy::long_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
ULONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    return _npy_clip<npy::ulong_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
LONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func))
{
    return _npy_clip<npy::longlong_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
ULONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func))
{
    return _npy_clip<npy::ulonglong_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
HALF_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    return _npy_clip<npy::half_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
FLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    return _npy_clip<npy::float_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
DOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func))
{
    return _npy_clip<npy::double_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
LONGDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
                void *NPY_UNUSED(func))
{
    return _npy_clip<npy::longdouble_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
CFLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func))
{
    return _npy_clip<npy::cfloat_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
CDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    return _npy_clip<npy::cdouble_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
CLONGDOUBLE_clip(char **args, npy_intp const *dimensions,
                 npy_intp const *steps, void *NPY_UNUSED(func))
{
    return _npy_clip<npy::clongdouble_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
DATETIME_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func))
{
    return _npy_clip<npy::datetime_tag>(args, dimensions, steps);
}
NPY_NO_EXPORT void
TIMEDELTA_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func))
{
    return _npy_clip<npy::timedelta_tag>(args, dimensions, steps);
}
}
