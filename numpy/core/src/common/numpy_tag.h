#ifndef _NPY_COMMON_TAG_H_
#define _NPY_COMMON_TAG_H_

namespace npy {

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
};
struct byte_tag : integral_tag {
    using type = npy_byte;
};
struct ubyte_tag : integral_tag {
    using type = npy_ubyte;
};
struct short_tag : integral_tag {
    using type = npy_short;
};
struct ushort_tag : integral_tag {
    using type = npy_ushort;
};
struct int_tag : integral_tag {
    using type = npy_int;
};
struct uint_tag : integral_tag {
    using type = npy_uint;
};
struct long_tag : integral_tag {
    using type = npy_long;
};
struct ulong_tag : integral_tag {
    using type = npy_ulong;
};
struct longlong_tag : integral_tag {
    using type = npy_longlong;
};
struct ulonglong_tag : integral_tag {
    using type = npy_ulonglong;
};
struct half_tag {
    using type = npy_half;
};
struct float_tag : floating_point_tag {
    using type = npy_float;
};
struct double_tag : floating_point_tag {
    using type = npy_double;
};
struct longdouble_tag : floating_point_tag {
    using type = npy_longdouble;
};
struct cfloat_tag : complex_tag {
    using type = npy_cfloat;
};
struct cdouble_tag : complex_tag {
    using type = npy_cdouble;
};
struct clongdouble_tag : complex_tag {
    using type = npy_clongdouble;
};
struct datetime_tag : date_tag {
    using type = npy_datetime;
};
struct timedelta_tag : date_tag {
    using type = npy_timedelta;
};

}  // namespace npy

#endif
