#ifndef _NPY_COMMON_TAG_H_
#define _NPY_COMMON_TAG_H_

#include "../npysort/npysort_common.h"

namespace npy {

template<typename... tags>
struct taglist {
  static constexpr unsigned size = sizeof...(tags);
};

struct integral_tag {
};
struct floating_point_tag {
};
struct complex_tag {
};
struct date_tag {
};

struct bool_tag : integral_tag {
    using type = npy_bool;
    static constexpr NPY_TYPES type_value = NPY_BOOL;
    static int less(type const& a, type const& b) {
      return BOOL_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct byte_tag : integral_tag {
    using type = npy_byte;
    static constexpr NPY_TYPES type_value = NPY_BYTE;
    static int less(type const& a, type const& b) {
      return BYTE_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct ubyte_tag : integral_tag {
    using type = npy_ubyte;
    static constexpr NPY_TYPES type_value = NPY_UBYTE;
    static int less(type const& a, type const& b) {
      return UBYTE_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct short_tag : integral_tag {
    using type = npy_short;
    static constexpr NPY_TYPES type_value = NPY_SHORT;
    static int less(type const& a, type const& b) {
      return SHORT_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct ushort_tag : integral_tag {
    using type = npy_ushort;
    static constexpr NPY_TYPES type_value = NPY_USHORT;
    static int less(type const& a, type const& b) {
      return USHORT_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct int_tag : integral_tag {
    using type = npy_int;
    static constexpr NPY_TYPES type_value = NPY_INT;
    static int less(type const& a, type const& b) {
      return INT_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct uint_tag : integral_tag {
    using type = npy_uint;
    static constexpr NPY_TYPES type_value = NPY_UINT;
    static int less(type const& a, type const& b) {
      return UINT_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct long_tag : integral_tag {
    using type = npy_long;
    static constexpr NPY_TYPES type_value = NPY_LONG;
    static int less(type const& a, type const& b) {
      return LONG_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct ulong_tag : integral_tag {
    using type = npy_ulong;
    static constexpr NPY_TYPES type_value = NPY_ULONG;
    static int less(type const& a, type const& b) {
      return ULONG_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct longlong_tag : integral_tag {
    using type = npy_longlong;
    static constexpr NPY_TYPES type_value = NPY_LONGLONG;
    static int less(type const& a, type const& b) {
      return LONGLONG_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct ulonglong_tag : integral_tag {
    using type = npy_ulonglong;
    static constexpr NPY_TYPES type_value = NPY_ULONGLONG;
    static int less(type const& a, type const& b) {
      return ULONGLONG_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct half_tag {
    using type = npy_half;
    static constexpr NPY_TYPES type_value = NPY_HALF;
    static int less(type const& a, type const& b) {
      return HALF_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct float_tag : floating_point_tag {
    using type = npy_float;
    static constexpr NPY_TYPES type_value = NPY_FLOAT;
    static int less(type const& a, type const& b) {
      return FLOAT_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct double_tag : floating_point_tag {
    using type = npy_double;
    static constexpr NPY_TYPES type_value = NPY_DOUBLE;
    static int less(type const& a, type const& b) {
      return DOUBLE_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct longdouble_tag : floating_point_tag {
    using type = npy_longdouble;
    static constexpr NPY_TYPES type_value = NPY_LONGDOUBLE;
    static int less(type const& a, type const& b) {
      return LONGDOUBLE_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct cfloat_tag : complex_tag {
    using type = npy_cfloat;
    static constexpr NPY_TYPES type_value = NPY_CFLOAT;
    static int less(type const& a, type const& b) {
      return CFLOAT_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct cdouble_tag : complex_tag {
    using type = npy_cdouble;
    static constexpr NPY_TYPES type_value = NPY_CDOUBLE;
    static int less(type const& a, type const& b) {
      return CDOUBLE_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct clongdouble_tag : complex_tag {
    using type = npy_clongdouble;
    static constexpr NPY_TYPES type_value = NPY_CLONGDOUBLE;
    static int less(type const& a, type const& b) {
      return CLONGDOUBLE_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct datetime_tag : date_tag {
    using type = npy_datetime;
    static constexpr NPY_TYPES type_value = NPY_DATETIME;
    static int less(type const& a, type const& b) {
      return DATETIME_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};
struct timedelta_tag : date_tag {
    using type = npy_timedelta;
    static constexpr NPY_TYPES type_value = NPY_TIMEDELTA;
    static int less(type const& a, type const& b) {
      return TIMEDELTA_LT(a, b);
    }
    static int less_equal(type const& a, type const& b) {
      return !less(b, a);
    }
};

struct string_tag {
    using type = npy_char;
    static constexpr NPY_TYPES type_value = NPY_STRING;
    static int less(type const* a, type const* b, size_t len) {
      return STRING_LT(a, b, len);
    }
    static int less_equal(type const* a, type const* b, size_t len) {
      return !less(b, a, len);
    }
    static void swap(type* a, type* b, size_t len) {
      STRING_SWAP(a, b, len);
    }
    static void copy(type * a, type const* b, size_t len) {
      STRING_COPY(a, b, len);
    }
};

struct unicode_tag {
    using type = npy_ucs4;
    static constexpr NPY_TYPES type_value = NPY_UNICODE;
    static int less(type const* a, type const* b, size_t len) {
      return UNICODE_LT(a, b, len);
    }
    static int less_equal(type const* a, type const* b, size_t len) {
      return !less(b, a, len);
    }
    static void swap(type* a, type* b, size_t len) {
      UNICODE_SWAP(a, b, len);
    }
    static void copy(type * a, type const* b, size_t len) {
      UNICODE_COPY(a, b, len);
    }
};

}  // namespace npy

#endif
