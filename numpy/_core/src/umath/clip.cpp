/**
 * This module provides the inner loops for the clip ufunc
 */
#include <type_traits>

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

#define PyArray_CLT(p,q,suffix) (((npy_creal##suffix(p)==npy_creal##suffix(q)) ? (npy_cimag##suffix(p) < npy_cimag##suffix(q)) : \
                               (npy_creal##suffix(p) < npy_creal##suffix(q))))
#define PyArray_CGT(p,q,suffix) (((npy_creal##suffix(p)==npy_creal##suffix(q)) ? (npy_cimag##suffix(p) > npy_cimag##suffix(q)) : \
                               (npy_creal##suffix(p) > npy_creal##suffix(q))))

npy_cdouble
_NPY_MIN(npy_cdouble a, npy_cdouble b, npy::complex_tag const &)
{
    return npy_isnan(npy_creal(a)) || npy_isnan(npy_cimag(a)) || PyArray_CLT(a, b,)
                ? (a)
                : (b);
}

npy_cfloat
_NPY_MIN(npy_cfloat a, npy_cfloat b, npy::complex_tag const &)
{
    return npy_isnan(npy_crealf(a)) || npy_isnan(npy_cimagf(a)) || PyArray_CLT(a, b, f)
                ? (a)
                : (b);
}

npy_clongdouble
_NPY_MIN(npy_clongdouble a, npy_clongdouble b, npy::complex_tag const &)
{
    return npy_isnan(npy_creall(a)) || npy_isnan(npy_cimagl(a)) || PyArray_CLT(a, b, l)
                ? (a)
                : (b);
}

npy_cdouble
_NPY_MAX(npy_cdouble a, npy_cdouble b, npy::complex_tag const &)
{
    return npy_isnan(npy_creal(a)) || npy_isnan(npy_cimag(a)) || PyArray_CGT(a, b,)
                ? (a)
                : (b);
}

npy_cfloat
_NPY_MAX(npy_cfloat a, npy_cfloat b, npy::complex_tag const &)
{
    return npy_isnan(npy_crealf(a)) || npy_isnan(npy_cimagf(a)) || PyArray_CGT(a, b, f)
                ? (a)
                : (b);
}

npy_clongdouble
_NPY_MAX(npy_clongdouble a, npy_clongdouble b, npy::complex_tag const &)
{
    return npy_isnan(npy_creall(a)) || npy_isnan(npy_cimagl(a)) || PyArray_CGT(a, b, l)
                ? (a)
                : (b);
}
#undef PyArray_CLT
#undef PyArray_CGT

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

template <class Tag, class T>
static inline void
_npy_clip_const_minmax_(
    char *ip, npy_intp is, char *op, npy_intp os, npy_intp n, T min_val, T max_val,
    std::false_type /* non-floating point */
)
{
    /* contiguous, branch to let the compiler optimize */
    if (is == sizeof(T) && os == sizeof(T)) {
        for (npy_intp i = 0; i < n; i++, ip += sizeof(T), op += sizeof(T)) {
            *(T *)op = _NPY_CLIP<Tag>(*(T *)ip, min_val, max_val);
        }
    }
    else {
        for (npy_intp i = 0; i < n; i++, ip += is, op += os) {
            *(T *)op = _NPY_CLIP<Tag>(*(T *)ip, min_val, max_val);
        }
    }
}

template <class Tag, class T>
static inline void
_npy_clip_const_minmax_(
    char *ip, npy_intp is, char *op, npy_intp os, npy_intp n, T min_val, T max_val,
    std::true_type  /* floating point */
)
{
    if (!npy_isnan(min_val) && !npy_isnan(max_val)) {
        /*
         * The min/max_val are not NaN so the comparison below will
         * propagate NaNs in the input without further NaN checks.
         */

        /* contiguous, branch to let the compiler optimize */
        if (is == sizeof(T) && os == sizeof(T)) {
            for (npy_intp i = 0; i < n; i++, ip += sizeof(T), op += sizeof(T)) {
                T x = *(T *)ip;
                if (x < min_val) {
                    x = min_val;
                }
                if (x > max_val) {
                    x = max_val;
                }
                *(T *)op = x;
            }
        }
        else {
            for (npy_intp i = 0; i < n; i++, ip += is, op += os) {
                T x = *(T *)ip;
                if (x < min_val) {
                    x = min_val;
                }
                if (x > max_val) {
                    x = max_val;
                }
                *(T *)op = x;
            }
        }
    }
    else {
        /* min_val and/or max_val are nans */
        T x = npy_isnan(min_val) ? min_val : max_val;
        for (npy_intp i = 0; i < n; i++, op += os) {
            *(T *)op = x;
        }
    }
}

template <class Tag, class T = typename Tag::type>
static void
_npy_clip(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    npy_intp n = dimensions[0];
    if (steps[1] == 0 && steps[2] == 0) {
        /* min and max are constant throughout the loop, the most common case */
        T min_val = *(T *)args[1];
        T max_val = *(T *)args[2];

        _npy_clip_const_minmax_<Tag, T>(
            args[0], steps[0], args[3], steps[3], n, min_val, max_val,
            std::is_base_of<npy::floating_point_tag, Tag>{}
        );
    }
    else {
        char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];
        npy_intp is1 = steps[0], is2 = steps[1],
                 is3 = steps[2], os1 = steps[3];
        for (npy_intp i = 0; i < n;
             i++, ip1 += is1, ip2 += is2, ip3 += is3, op1 += os1)
        {
            *(T *)op1 = _NPY_CLIP<Tag>(*(T *)ip1, *(T *)ip2, *(T *)ip3);
        }
    }
    npy_clear_floatstatus_barrier((char *)dimensions);
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
