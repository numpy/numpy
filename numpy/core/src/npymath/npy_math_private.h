/*
 *
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * from: @(#)fdlibm.h 5.1 93/09/24
 * $FreeBSD$
 */

#ifndef _NPY_MATH_PRIVATE_H_
#define _NPY_MATH_PRIVATE_H_

#include <Python.h>
#include <math.h>

#include "npy_config.h"
#include "npy_fpmath.h"

#include "numpy/npy_math.h"
#include "numpy/npy_cpu.h"
#include "numpy/npy_endian.h"
#include "numpy/npy_common.h"

/*
 * The original fdlibm code used statements like:
 *      n0 = ((*(int*)&one)>>29)^1;             * index of high word *
 *      ix0 = *(n0+(int*)&x);                   * high word of x *
 *      ix1 = *((1-n0)+(int*)&x);               * low word of x *
 * to dig two 32 bit words out of the 64 bit IEEE floating point
 * value.  That is non-ANSI, and, moreover, the gcc instruction
 * scheduler gets it wrong.  We instead use the following macros.
 * Unlike the original code, we determine the endianness at compile
 * time, not at run time; I don't see much benefit to selecting
 * endianness at run time.
 */

/*
 * A union which permits us to convert between a double and two 32 bit
 * ints.
 */

/* XXX: not really, but we already make this assumption elsewhere. Will have to
 * fix this at some point */
#define IEEE_WORD_ORDER NPY_BYTE_ORDER

#if IEEE_WORD_ORDER == NPY_BIG_ENDIAN

typedef union
{
  double value;
  struct
  {
    npy_uint32 msw;
    npy_uint32 lsw;
  } parts;
} ieee_double_shape_type;

#endif

#if IEEE_WORD_ORDER == NPY_LITTLE_ENDIAN

typedef union
{
  double value;
  struct
  {
    npy_uint32 lsw;
    npy_uint32 msw;
  } parts;
} ieee_double_shape_type;

#endif

/* Get two 32 bit ints from a double.  */

#define EXTRACT_WORDS(ix0,ix1,d)                                \
do {                                                            \
  ieee_double_shape_type ew_u;                                  \
  ew_u.value = (d);                                             \
  (ix0) = ew_u.parts.msw;                                       \
  (ix1) = ew_u.parts.lsw;                                       \
} while (0)

/* Get the more significant 32 bit int from a double.  */

#define GET_HIGH_WORD(i,d)                                      \
do {                                                            \
  ieee_double_shape_type gh_u;                                  \
  gh_u.value = (d);                                             \
  (i) = gh_u.parts.msw;                                         \
} while (0)

/* Get the less significant 32 bit int from a double.  */

#define GET_LOW_WORD(i,d)                                       \
do {                                                            \
  ieee_double_shape_type gl_u;                                  \
  gl_u.value = (d);                                             \
  (i) = gl_u.parts.lsw;                                         \
} while (0)

/* Set the more significant 32 bits of a double from an int.  */

#define SET_HIGH_WORD(d,v)                                      \
do {                                                            \
  ieee_double_shape_type sh_u;                                  \
  sh_u.value = (d);                                             \
  sh_u.parts.msw = (v);                                         \
  (d) = sh_u.value;                                             \
} while (0)

/* Set the less significant 32 bits of a double from an int.  */

#define SET_LOW_WORD(d,v)                                       \
do {                                                            \
  ieee_double_shape_type sl_u;                                  \
  sl_u.value = (d);                                             \
  sl_u.parts.lsw = (v);                                         \
  (d) = sl_u.value;                                             \
} while (0)

/* Set a double from two 32 bit ints.  */

#define INSERT_WORDS(d,ix0,ix1)                                 \
do {                                                            \
  ieee_double_shape_type iw_u;                                  \
  iw_u.parts.msw = (ix0);                                       \
  iw_u.parts.lsw = (ix1);                                       \
  (d) = iw_u.value;                                             \
} while (0)

/*
 * A union which permits us to convert between a float and a 32 bit
 * int.
 */

typedef union
{
  float value;
  /* FIXME: Assumes 32 bit int.  */
  npy_uint32 word;
} ieee_float_shape_type;

/* Get a 32 bit int from a float.  */

#define GET_FLOAT_WORD(i,d)                                     \
do {                                                            \
  ieee_float_shape_type gf_u;                                   \
  gf_u.value = (d);                                             \
  (i) = gf_u.word;                                              \
} while (0)

/* Set a float from a 32 bit int.  */

#define SET_FLOAT_WORD(d,i)                                     \
do {                                                            \
  ieee_float_shape_type sf_u;                                   \
  sf_u.word = (i);                                              \
  (d) = sf_u.value;                                             \
} while (0)

#ifdef NPY_USE_C99_COMPLEX
#include <complex.h>
#endif

/*
 * Long double support
 */
#if defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE)
    /*
     * Intel extended 80 bits precision. Bit representation is
     *          |  junk  |     s  |eeeeeeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          | 16 bits|  1 bit |    15 bits    |            64 bits            |
     *          |             a[2]                |     a[1]     |    a[0]        |
     *
     * 16 stronger bits of a[2] are junk
     */
    typedef npy_uint32 IEEEl2bitsrep_part;


    union IEEEl2bitsrep {
        npy_longdouble     e;
        IEEEl2bitsrep_part a[3];
    };

    #define LDBL_MANL_INDEX     0
    #define LDBL_MANL_MASK      0xFFFFFFFF
    #define LDBL_MANL_SHIFT     0

    #define LDBL_MANH_INDEX     1
    #define LDBL_MANH_MASK      0xFFFFFFFF
    #define LDBL_MANH_SHIFT     0

    #define LDBL_EXP_INDEX      2
    #define LDBL_EXP_MASK       0x7FFF
    #define LDBL_EXP_SHIFT      0

    #define LDBL_SIGN_INDEX     2
    #define LDBL_SIGN_MASK      0x8000
    #define LDBL_SIGN_SHIFT     15

    #define LDBL_NBIT           0x80000000

    typedef npy_uint32 ldouble_man_t;
    typedef npy_uint32 ldouble_exp_t;
    typedef npy_uint32 ldouble_sign_t;
#elif defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE)
    /*
     * Intel extended 80 bits precision, 16 bytes alignment.. Bit representation is
     *          |  junk  |     s  |eeeeeeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          | 16 bits|  1 bit |    15 bits    |            64 bits            |
     *          |             a[2]                |     a[1]     |    a[0]        |
     *
     * a[3] and 16 stronger bits of a[2] are junk
     */
    typedef npy_uint32 IEEEl2bitsrep_part;

    union IEEEl2bitsrep {
        npy_longdouble     e;
        IEEEl2bitsrep_part a[4];
    };

    #define LDBL_MANL_INDEX     0
    #define LDBL_MANL_MASK      0xFFFFFFFF
    #define LDBL_MANL_SHIFT     0

    #define LDBL_MANH_INDEX     1
    #define LDBL_MANH_MASK      0xFFFFFFFF
    #define LDBL_MANH_SHIFT     0

    #define LDBL_EXP_INDEX      2
    #define LDBL_EXP_MASK       0x7FFF
    #define LDBL_EXP_SHIFT      0

    #define LDBL_SIGN_INDEX     2
    #define LDBL_SIGN_MASK      0x8000
    #define LDBL_SIGN_SHIFT     15

    #define LDBL_NBIT           0x800000000

    typedef npy_uint32 ldouble_man_t;
    typedef npy_uint32 ldouble_exp_t;
    typedef npy_uint32 ldouble_sign_t;
#elif defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE)
    /*
     * Motorola extended 80 bits precision. Bit representation is
     *          |     s  |eeeeeeeeeeeeeee|  junk  |mmmmmmmm................mmmmmmm|
     *          |  1 bit |    15 bits    | 16 bits|            64 bits            |
     *          |             a[0]                |     a[1]     |    a[2]        |
     *
     * 16 low bits of a[0] are junk
     */
    typedef npy_uint32 IEEEl2bitsrep_part;


    union IEEEl2bitsrep {
        npy_longdouble     e;
        IEEEl2bitsrep_part a[3];
    };

    #define LDBL_MANL_INDEX     2
    #define LDBL_MANL_MASK      0xFFFFFFFF
    #define LDBL_MANL_SHIFT     0

    #define LDBL_MANH_INDEX     1
    #define LDBL_MANH_MASK      0xFFFFFFFF
    #define LDBL_MANH_SHIFT     0

    #define LDBL_EXP_INDEX      0
    #define LDBL_EXP_MASK       0x7FFF0000
    #define LDBL_EXP_SHIFT      16

    #define LDBL_SIGN_INDEX     0
    #define LDBL_SIGN_MASK      0x80000000
    #define LDBL_SIGN_SHIFT     31

    #define LDBL_NBIT           0x80000000

    typedef npy_uint32 ldouble_man_t;
    typedef npy_uint32 ldouble_exp_t;
    typedef npy_uint32 ldouble_sign_t;
#elif defined(HAVE_LDOUBLE_IEEE_DOUBLE_BE)
    /* 64 bits IEEE double precision aligned on 16 bytes: used by ppc arch on
     * Mac OS X */

    /*
     * IEEE double precision. Bit representation is
     *          |  s  |eeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          |1 bit|  11 bits  |            52 bits            |
     *          |          a[0]         |         a[1]            |
     */
    typedef npy_uint32 IEEEl2bitsrep_part;

    union IEEEl2bitsrep {
        npy_longdouble     e;
        IEEEl2bitsrep_part a[2];
    };

    #define LDBL_MANL_INDEX     1
    #define LDBL_MANL_MASK      0xFFFFFFFF
    #define LDBL_MANL_SHIFT     0

    #define LDBL_MANH_INDEX     0
    #define LDBL_MANH_MASK      0x000FFFFF
    #define LDBL_MANH_SHIFT     0

    #define LDBL_EXP_INDEX      0
    #define LDBL_EXP_MASK       0x7FF00000
    #define LDBL_EXP_SHIFT      20

    #define LDBL_SIGN_INDEX     0
    #define LDBL_SIGN_MASK      0x80000000
    #define LDBL_SIGN_SHIFT     31

    #define LDBL_NBIT           0

    typedef npy_uint32 ldouble_man_t;
    typedef npy_uint32 ldouble_exp_t;
    typedef npy_uint32 ldouble_sign_t;
#elif defined(HAVE_LDOUBLE_IEEE_DOUBLE_LE)
    /* 64 bits IEEE double precision, Little Endian. */

    /*
     * IEEE double precision. Bit representation is
     *          |  s  |eeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          |1 bit|  11 bits  |            52 bits            |
     *          |          a[1]         |         a[0]            |
     */
    typedef npy_uint32 IEEEl2bitsrep_part;

    union IEEEl2bitsrep {
        npy_longdouble     e;
        IEEEl2bitsrep_part a[2];
    };

    #define LDBL_MANL_INDEX     0
    #define LDBL_MANL_MASK      0xFFFFFFFF
    #define LDBL_MANL_SHIFT     0

    #define LDBL_MANH_INDEX     1
    #define LDBL_MANH_MASK      0x000FFFFF
    #define LDBL_MANH_SHIFT     0

    #define LDBL_EXP_INDEX      1
    #define LDBL_EXP_MASK       0x7FF00000
    #define LDBL_EXP_SHIFT      20

    #define LDBL_SIGN_INDEX     1
    #define LDBL_SIGN_MASK      0x80000000
    #define LDBL_SIGN_SHIFT     31

    #define LDBL_NBIT           0x00000080

    typedef npy_uint32 ldouble_man_t;
    typedef npy_uint32 ldouble_exp_t;
    typedef npy_uint32 ldouble_sign_t;
#elif defined(HAVE_LDOUBLE_IEEE_QUAD_BE)
    /*
     * IEEE quad precision, Big Endian. Bit representation is
     *          |  s  |eeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          |1 bit|  15 bits  |            112 bits           |
     *          |          a[0]         |           a[1]          |
     */
    typedef npy_uint64 IEEEl2bitsrep_part;

    union IEEEl2bitsrep {
        npy_longdouble     e;
        IEEEl2bitsrep_part a[2];
    };

    #define LDBL_MANL_INDEX     1
    #define LDBL_MANL_MASK      0xFFFFFFFFFFFFFFFF
    #define LDBL_MANL_SHIFT     0

    #define LDBL_MANH_INDEX     0
    #define LDBL_MANH_MASK      0x0000FFFFFFFFFFFF
    #define LDBL_MANH_SHIFT     0

    #define LDBL_EXP_INDEX      0
    #define LDBL_EXP_MASK       0x7FFF000000000000
    #define LDBL_EXP_SHIFT      48

    #define LDBL_SIGN_INDEX     0
    #define LDBL_SIGN_MASK      0x8000000000000000
    #define LDBL_SIGN_SHIFT     63

    #define LDBL_NBIT           0

    typedef npy_uint64 ldouble_man_t;
    typedef npy_uint64 ldouble_exp_t;
    typedef npy_uint32 ldouble_sign_t;
#elif defined(HAVE_LDOUBLE_IEEE_QUAD_LE)
    /*
     * IEEE quad precision, Little Endian. Bit representation is
     *          |  s  |eeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          |1 bit|  15 bits  |            112 bits           |
     *          |          a[1]         |           a[0]          |
     */
    typedef npy_uint64 IEEEl2bitsrep_part;

    union IEEEl2bitsrep {
        npy_longdouble     e;
        IEEEl2bitsrep_part a[2];
    };

    #define LDBL_MANL_INDEX     0
    #define LDBL_MANL_MASK      0xFFFFFFFFFFFFFFFF
    #define LDBL_MANL_SHIFT     0

    #define LDBL_MANH_INDEX     1
    #define LDBL_MANH_MASK      0x0000FFFFFFFFFFFF
    #define LDBL_MANH_SHIFT     0

    #define LDBL_EXP_INDEX      1
    #define LDBL_EXP_MASK       0x7FFF000000000000
    #define LDBL_EXP_SHIFT      48

    #define LDBL_SIGN_INDEX     1
    #define LDBL_SIGN_MASK      0x8000000000000000
    #define LDBL_SIGN_SHIFT     63

    #define LDBL_NBIT           0

    typedef npy_uint64 ldouble_man_t;
    typedef npy_uint64 ldouble_exp_t;
    typedef npy_uint32 ldouble_sign_t;
#endif

#if !defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE) && \
    !defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE)
/* Get the sign bit of x. x should be of type IEEEl2bitsrep */
#define GET_LDOUBLE_SIGN(x) \
    (((x).a[LDBL_SIGN_INDEX] & LDBL_SIGN_MASK) >> LDBL_SIGN_SHIFT)

/* Set the sign bit of x to v. x should be of type IEEEl2bitsrep */
#define SET_LDOUBLE_SIGN(x, v) \
  ((x).a[LDBL_SIGN_INDEX] =                                             \
   ((x).a[LDBL_SIGN_INDEX] & ~LDBL_SIGN_MASK) |                         \
   (((IEEEl2bitsrep_part)(v) << LDBL_SIGN_SHIFT) & LDBL_SIGN_MASK))

/* Get the exp bits of x. x should be of type IEEEl2bitsrep */
#define GET_LDOUBLE_EXP(x) \
    (((x).a[LDBL_EXP_INDEX] & LDBL_EXP_MASK) >> LDBL_EXP_SHIFT)

/* Set the exp bit of x to v. x should be of type IEEEl2bitsrep */
#define SET_LDOUBLE_EXP(x, v) \
  ((x).a[LDBL_EXP_INDEX] =                                              \
   ((x).a[LDBL_EXP_INDEX] & ~LDBL_EXP_MASK) |                           \
   (((IEEEl2bitsrep_part)(v) << LDBL_EXP_SHIFT) & LDBL_EXP_MASK))

/* Get the manl bits of x. x should be of type IEEEl2bitsrep */
#define GET_LDOUBLE_MANL(x) \
    (((x).a[LDBL_MANL_INDEX] & LDBL_MANL_MASK) >> LDBL_MANL_SHIFT)

/* Set the manl bit of x to v. x should be of type IEEEl2bitsrep */
#define SET_LDOUBLE_MANL(x, v) \
  ((x).a[LDBL_MANL_INDEX] =                                             \
   ((x).a[LDBL_MANL_INDEX] & ~LDBL_MANL_MASK) |                         \
   (((IEEEl2bitsrep_part)(v) << LDBL_MANL_SHIFT) & LDBL_MANL_MASK))

/* Get the manh bits of x. x should be of type IEEEl2bitsrep */
#define GET_LDOUBLE_MANH(x) \
    (((x).a[LDBL_MANH_INDEX] & LDBL_MANH_MASK) >> LDBL_MANH_SHIFT)

/* Set the manh bit of x to v. x should be of type IEEEl2bitsrep */
#define SET_LDOUBLE_MANH(x, v) \
    ((x).a[LDBL_MANH_INDEX] = \
     ((x).a[LDBL_MANH_INDEX] & ~LDBL_MANH_MASK) |                       \
     (((IEEEl2bitsrep_part)(v) << LDBL_MANH_SHIFT) & LDBL_MANH_MASK))

#endif /* !HAVE_LDOUBLE_DOUBLE_DOUBLE_* */

/*
 * Those unions are used to convert a pointer of npy_cdouble to native C99
 * complex or our own complex type independently on whether C99 complex
 * support is available
 */
#ifdef NPY_USE_C99_COMPLEX

/*
 * Microsoft C defines _MSC_VER
 * Intel compiler does not use MSVC complex types, but defines _MSC_VER by
 * default.
 */
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
typedef union {
        npy_cdouble npy_z;
        _Dcomplex c99_z;
} __npy_cdouble_to_c99_cast;

typedef union {
        npy_cfloat npy_z;
        _Fcomplex c99_z;
} __npy_cfloat_to_c99_cast;

typedef union {
        npy_clongdouble npy_z;
        _Lcomplex c99_z;
} __npy_clongdouble_to_c99_cast;
#else /* !_MSC_VER */
typedef union {
        npy_cdouble npy_z;
        complex double c99_z;
} __npy_cdouble_to_c99_cast;

typedef union {
        npy_cfloat npy_z;
        complex float c99_z;
} __npy_cfloat_to_c99_cast;

typedef union {
        npy_clongdouble npy_z;
        complex long double c99_z;
} __npy_clongdouble_to_c99_cast;
#endif /* !_MSC_VER */

#else /* !NPY_USE_C99_COMPLEX */
typedef union {
        npy_cdouble npy_z;
        npy_cdouble c99_z;
} __npy_cdouble_to_c99_cast;

typedef union {
        npy_cfloat npy_z;
        npy_cfloat c99_z;
} __npy_cfloat_to_c99_cast;

typedef union {
        npy_clongdouble npy_z;
        npy_clongdouble c99_z;
} __npy_clongdouble_to_c99_cast;
#endif /* !NPY_USE_C99_COMPLEX */


#endif /* !_NPY_MATH_PRIVATE_H_ */
