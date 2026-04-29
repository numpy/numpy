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
 *   - ``less`` / ``less_equal``  -- the sort-friendly comparisons that
 *     propagate NaN / NaT to the high end
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
    static int less(T a, T b) { return a < b; }
    static int less_equal(T a, T b) { return !(b < a); }

    template <bool reverse = false>
    static int cmp(T a, T b) { return reverse ? b < a : a < b; }
};

template <typename T, NPY_TYPES TypeNum>
struct floating_point_type : floating_point_tag {
    using type = T;
    static constexpr NPY_TYPES type_value = TypeNum;
    // NaN sorts to the end:  a is "less than" b if a is non-NaN and
    // either a < b or b is NaN.  ``x != x`` is the IEEE NaN test.
    static int less(T a, T b) { return a < b || (b != b && a == a); }
    static int less_equal(T a, T b) { return !less(b, a); }

    template <bool reverse = false>
    static int cmp(T a, T b)
    {
        if (reverse) {
            // NaN sorts to the end in reverse too
            return b < a || (b != b && a == a);
        }
        else {
            return a < b || (b != b && a == a);
        }
    }
};

// Half is its own per-type tag; no template since there is only one half
// scalar.  It also serves directly as its category marker for clip.
struct half_tag {
    using type = npy_half;
    static constexpr NPY_TYPES type_value = NPY_HALF;

    static int isnan(npy_half h)
    {
        return ((h & 0x7c00u) == 0x7c00u) && ((h & 0x03ffu) != 0x0000u);
    }

    // Bit-level less-than that assumes neither operand is NaN.
    static int lt_nonan(npy_half a, npy_half b)
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

    static int less(npy_half a, npy_half b)
    {
        if (isnan(b)) {
            return !isnan(a);
        }
        return !isnan(a) && lt_nonan(a, b);
    }
    static int less_equal(npy_half a, npy_half b) { return !less(b, a); }

    template <bool reverse = false>
    static int cmp(npy_half a, npy_half b)
    {
        if (reverse) {
            // NaN sorts to the end in reverse too
            if (isnan(b)) {
                return !isnan(a);
            }
            return !isnan(a) && lt_nonan(b, a);
        }
        else {
            if (isnan(b)) {
                return !isnan(a);
            }
            return !isnan(a) && lt_nonan(a, b);
        }
    }
};

template <typename T, NPY_TYPES TypeNum>
struct complex_type : complex_tag {
    using type = T;
    static constexpr NPY_TYPES type_value = TypeNum;

    // Real / imag accessors picked at compile time so ``less`` can be
    // written generically across the three complex scalar types.
    static auto creal(T z)
    {
        if constexpr      (std::is_same_v<T, npy_cfloat>)  return npy_crealf(z);
        else if constexpr (std::is_same_v<T, npy_cdouble>) return npy_creal(z);
        else                                               return npy_creall(z);
    }
    static auto cimag(T z)
    {
        if constexpr      (std::is_same_v<T, npy_cfloat>)  return npy_cimagf(z);
        else if constexpr (std::is_same_v<T, npy_cdouble>) return npy_cimag(z);
        else                                               return npy_cimagl(z);
    }

    static int less(T a, T b)
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
    static int less_equal(T a, T b) { return !less(b, a); }

    template <bool reverse = false>
    static int cmp(T a, T b)
    {
        const auto ra = creal(a), rb = creal(b);
        const auto ia = cimag(a), ib = cimag(b);
        if (reverse) {
            if (ra < rb) {
                return ib != ib && ia == ia;
            }
            if (ra > rb) {
                return ia == ia || ib != ib;
            }
            if (ra == rb || (ra != ra && rb != rb)) {
                return ib < ia || (ia != ia && ib == ib);
            }
            return ra != ra;
        }
        else {
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
    }
};

template <typename T, NPY_TYPES TypeNum>
struct datetime_type : date_tag {
    using type = T;
    static constexpr NPY_TYPES type_value = TypeNum;
    static int less(T a, T b)
    {
        if (a == NPY_DATETIME_NAT) return 0;
        if (b == NPY_DATETIME_NAT) return 1;
        return a < b;
    }
    static int less_equal(T a, T b) { return !less(b, a); }

    template <bool reverse = false>
    static int cmp(T a, T b)
    {
        if (reverse) {
            if (a == NPY_DATETIME_NAT) return 0;
            if (b == NPY_DATETIME_NAT) return 1;
            return b < a;
        }
        else {
            if (a == NPY_DATETIME_NAT) return 0;
            if (b == NPY_DATETIME_NAT) return 1;
            return a < b;
        }
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
  
    template <bool reverse = false>
    static int cmp(T const *a, T const *b, size_t n)
    {
        using U = std::make_unsigned_t<T>;
        const auto *ua = reinterpret_cast<const U *>(a);
        const auto *ub = reinterpret_cast<const U *>(b);
        for (size_t i = 0; i < n; ++i) {
            if (ua[i] != ub[i]) {
                return reverse ? ub[i] < ua[i] : ua[i] < ub[i];
            }
        }
        return 0;
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

}  // namespace npy

#endif  // NUMPY_CORE_SRC_COMMON_NUMPY_TAG_H_
