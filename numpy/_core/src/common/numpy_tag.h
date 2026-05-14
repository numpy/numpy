#ifndef NUMPY_CORE_SRC_COMMON_NUMPY_TAG_H_
#define NUMPY_CORE_SRC_COMMON_NUMPY_TAG_H_

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"

#include <cstring>
#include <type_traits>

/*
 * Per-dtype tags shared by the sort/partition/binsearch and clip
 * implementations.
 *
 * Each tag exposes:
 *
 *   - ``type``        -- the underlying C scalar type
 *   - ``type_value``  -- the corresponding ``NPY_TYPES`` enumerator
 *   - ``less`` / ``less_equal`` / ``greater`` -- the sort-friendly
 *     comparisons that order NaN / NaT to the end (as if largest value
 *     for less and as if smallest for greater).
 *
 * For the four numeric categories that need different NaN/NaT handling,
 * comparisons are implemented once at the ``*_type<T, NPY_TYPES>``
 * template level (with ``if constexpr`` for the only per-scalar variation
 * -- the three real/imag accessors used by the complex types).  Each
 * such template inherits from a tiny empty marker (``integral_tag``,
 * ``floating_point_tag``, ``complex_tag``, ``date_tag``) so that
 * ``clip.cpp`` can dispatch via ordinary overload resolution.
 *
 * Concrete tags (``bool_tag``, ``float_tag``, ...) are then plain aliases
 * for an ``*_type`` instantiation.
 *
 * Distinct tag types matter because several of NumPy's scalar types alias
 * one another at the C level -- most notably ``npy_half`` is a typedef for
 * ``npy_uint16`` -- so the extra ``NPY_TYPES`` template argument is what
 * keeps ``half_tag`` and ``ushort_tag`` apart.
 */

namespace npy {

// Category markers used by clip's overload-set dispatch.  ``half_tag`` is
// its own marker (and the only half tag) so it does not need one here.
struct integral_tag {};
struct floating_point_tag {};
struct complex_tag {};
struct date_tag {};

template <typename T, NPY_TYPES TypeNum>
struct integral_type : integral_tag {
    using type = T;
    static constexpr NPY_TYPES type_value = TypeNum;
    static constexpr int less(T a, T b) { return a < b; }
    static constexpr int less_equal(T a, T b) { return !(b < a); }
    static constexpr int greater(T a, T b) { return a > b; }
};

template <typename T, NPY_TYPES TypeNum>
struct floating_point_type : floating_point_tag {
    using type = T;
    static constexpr NPY_TYPES type_value = TypeNum;
    // NaN sorts to the end:  a is "less than" b if a is non-NaN and
    // either a < b or b is NaN.  ``x != x`` is the IEEE NaN test.
    static constexpr int less(T a, T b) { return a < b || (b != b && a == a); }
    static constexpr int less_equal(T a, T b) { return !less(b, a); }
    // NaN sorts to the end in reverse too: ``a`` is "greater than" ``b``
    // if ``a`` is non-NaN and either ``b < a`` or ``b`` is NaN.
    static constexpr int greater(T a, T b) { return a > b || (b != b && a == a); }
};

// Half is its own per-type tag; no template since there is only one half
// scalar.  It also serves directly as its category marker for clip.
struct half_tag {
    using type = npy_half;
    static constexpr NPY_TYPES type_value = NPY_HALF;

    static constexpr int isnan(npy_half h)
    {
        return ((h & 0x7c00u) == 0x7c00u) && ((h & 0x03ffu) != 0x0000u);
    }

    // Bit-level less-than that assumes neither operand is NaN.
    static constexpr int lt_nonan(npy_half a, npy_half b)
    {
        if (a & 0x8000u) {
            if (b & 0x8000u) {
                return (a & 0x7fffu) > (b & 0x7fffu);
            }
            // Signed zeros compare equal.
            return (a != 0x8000u) || (b != 0x0000u);
        }
        if (b & 0x8000u) {
            return 0;
        }
        return (a & 0x7fffu) < (b & 0x7fffu);
    }

    static constexpr int less(npy_half a, npy_half b)
    {
        if (isnan(b)) {
            return !isnan(a);
        }
        return !isnan(a) && lt_nonan(a, b);
    }
    static constexpr int less_equal(npy_half a, npy_half b) { return !less(b, a); }

    // NaN sorts to the end in reverse too.
    static constexpr int greater(npy_half a, npy_half b)
    {
        if (isnan(b)) {
            return !isnan(a);
        }
        return !isnan(a) && lt_nonan(b, a);
    }
};

template <typename T, NPY_TYPES TypeNum>
struct complex_type : complex_tag {
    using type = T;
    static constexpr NPY_TYPES type_value = TypeNum;

    // In C++ mode the npy_c{float,double,longdouble} types are plain
    // structs with a ``_Val[2]`` member (see numpy/npy_common.h), access
    // directly so this can be a `constexpr` easily.
    static constexpr auto creal(T z) { return z._Val[0]; }
    static constexpr auto cimag(T z) { return z._Val[1]; }

    static constexpr int less(T a, T b)
    {
        const auto ra = creal(a), rb = creal(b);
        const auto ia = cimag(a), ib = cimag(b);
        if (ra < rb) {
            return ia == ia || ib != ib;
        }
        if (ra > rb) {
            return ib != ib && ia == ia;
        }
        if (ra == rb || (ra != ra && rb != rb)) {
            return ia < ib || (ib != ib && ia == ia);
        }
        return rb != rb;
    }
    static constexpr int less_equal(T a, T b) { return !less(b, a); }

    static constexpr int greater(T a, T b)
    {
        const auto ra = creal(a), rb = creal(b);
        const auto ia = cimag(a), ib = cimag(b);
        if (ra > rb || (ra == ra && rb != rb)) {
            return ia == ia || ib != ib;
        }
        if (ra < rb || (ra != ra && rb == rb)) {
            return ib != ib && ia == ia;
        }
        if (ra == rb || (ra != ra && rb != rb)) {
            return ia > ib || (ib != ib && ia == ia);
        }
        return ra != ra;
    }
};

template <typename T, NPY_TYPES TypeNum>
struct datetime_type : date_tag {
    using type = T;
    static constexpr NPY_TYPES type_value = TypeNum;
    static constexpr int less(T a, T b)
    {
        if (a == NPY_DATETIME_NAT) return 0;
        if (b == NPY_DATETIME_NAT) return 1;
        return a < b;
    }
    static constexpr int less_equal(T a, T b) { return !less(b, a); }

    // NaT sorts to the end in reverse too.
    static constexpr int greater(T a, T b)
    {
        if (a == NPY_DATETIME_NAT) return 0;
        if (b == NPY_DATETIME_NAT) return 1;
        return b < a;
    }
};

// String / unicode tags work on runtime-length blocks.  Comparison is
// unsigned (matches strcmp/wcscmp ordering) regardless of whether ``T`` is
// a signed ``char`` on the platform.
template <typename T, NPY_TYPES TypeNum>
struct string_like_type {
    using type = T;
    static constexpr NPY_TYPES type_value = TypeNum;

    static int less(T const *a, T const *b, size_t n)
    {
        using U = std::make_unsigned_t<T>;
        const auto *ua = reinterpret_cast<const U *>(a);
        const auto *ub = reinterpret_cast<const U *>(b);
        for (size_t i = 0; i < n; ++i) {
            if (ua[i] != ub[i]) {
                return ua[i] < ub[i];
            }
        }
        return 0;
    }
    static int less_equal(T const *a, T const *b, size_t n)
    {
        return !less(b, a, n);
    }
    static void swap(T *a, T *b, size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            T t = a[i];
            a[i] = b[i];
            b[i] = t;
        }
    }
    static void copy(T *a, T const *b, size_t n)
    {
        std::memcpy(a, b, n * sizeof(T));
    }

    static int greater(T const *a, T const *b, size_t n)
    {
        return less(b, a, n);
    }
};

// This tag is used to register object sorts, which replaces the old generic sort
// that did not handle NaNs at all. It supposes that any object such that
// obj != obj is NaN-like and should be sorted to the end as in other dtypes.
struct object_tag {
    using type = PyObject *;
    static constexpr NPY_TYPES type_value = NPY_OBJECT;

    static int isnan(PyObject *a) {
        if (a == NULL) {
            return 1;
        }

        /* PyObject_RichCompareBool is not used here because it takes a shortcut
         * for identical objects, hence will return false for NaN != NaN. */
        PyObject *result = PyObject_RichCompare(a, a, Py_NE);
        if (result == NULL) {
            return -1;
        }
        int ret = PyObject_IsTrue(result);
        Py_DECREF(result);
        if (ret < 0) {
            return -1;
        }
        return ret;
    }

    static int less(PyObject *a, PyObject *b)
    {
        /*
         * work around gh-3879, we cannot abort an in-progress quicksort
         * so at least do not raise again
         */
        if (PyErr_Occurred()) {
            return 0;
        }

        int isnan_a = isnan(a);
        int isnan_b = isnan(b); 
        if (isnan_a < 0 || isnan_b < 0) {
            return 0;
        }
        if (isnan_b) {
            return 1;
        }
        if (isnan_a) {
            return 0;
        }

        int ret = PyObject_RichCompareBool(a, b, Py_LT);
        if (ret < 0) {
            return 0;
        }
        return ret;
    }

    static int less_equal(PyObject *a, PyObject *b) {
        return !less(b, a);
    }

    static int greater(PyObject *a, PyObject *b) {
        /*
         * work around gh-3879, we cannot abort an in-progress quicksort
         * so at least do not raise again
         */
        if (PyErr_Occurred()) {
            return 0;
        }

        int isnan_a = isnan(a);
        int isnan_b = isnan(b);
        if (isnan_a < 0 || isnan_b < 0) {
            return 0;
        }
        if (isnan_b) {
            return 1;
        }
        if (isnan_a) {
            return 0;
        }

        int ret = PyObject_RichCompareBool(a, b, Py_GT);
        if (ret < 0) {
            return 0;
        }
        return ret;
    }
};

// Concrete tags consumed by callers.
using bool_tag        = integral_type<npy_bool,        NPY_BOOL>;
using byte_tag        = integral_type<npy_byte,        NPY_BYTE>;
using ubyte_tag       = integral_type<npy_ubyte,       NPY_UBYTE>;
using short_tag       = integral_type<npy_short,       NPY_SHORT>;
using ushort_tag      = integral_type<npy_ushort,      NPY_USHORT>;
using int_tag         = integral_type<npy_int,         NPY_INT>;
using uint_tag        = integral_type<npy_uint,        NPY_UINT>;
using long_tag        = integral_type<npy_long,        NPY_LONG>;
using ulong_tag       = integral_type<npy_ulong,       NPY_ULONG>;
using longlong_tag    = integral_type<npy_longlong,    NPY_LONGLONG>;
using ulonglong_tag   = integral_type<npy_ulonglong,   NPY_ULONGLONG>;
using float_tag       = floating_point_type<npy_float,       NPY_FLOAT>;
using double_tag      = floating_point_type<npy_double,      NPY_DOUBLE>;
using longdouble_tag  = floating_point_type<npy_longdouble,  NPY_LONGDOUBLE>;
using cfloat_tag      = complex_type<npy_cfloat,      NPY_CFLOAT>;
using cdouble_tag     = complex_type<npy_cdouble,     NPY_CDOUBLE>;
using clongdouble_tag = complex_type<npy_clongdouble, NPY_CLONGDOUBLE>;
using datetime_tag    = datetime_type<npy_datetime,   NPY_DATETIME>;
using timedelta_tag   = datetime_type<npy_timedelta,  NPY_TIMEDELTA>;
using string_tag      = string_like_type<npy_char,    NPY_STRING>;
using unicode_tag     = string_like_type<npy_ucs4,    NPY_UNICODE>;

// Type-list helper used by selection.cpp / binsearch.cpp to instantiate one
// function per supported tag.
template <typename... Tags>
struct taglist {
    static constexpr unsigned size = sizeof...(Tags);
};

// Generic comparator dispatch used by the ascending/descending sort.
template <typename Tag, bool reverse, typename... Args>
constexpr int cmp(Args... args)
{
    if constexpr (reverse) {
        return Tag::greater(args...);
    }
    else {
        return Tag::less(args...);
    }
}

}  // namespace npy

#endif  // NUMPY_CORE_SRC_COMMON_NUMPY_TAG_H_
