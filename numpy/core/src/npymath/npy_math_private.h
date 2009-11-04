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

//#include "config.h"
#include "numpy/npy_math.h"
#include "numpy/npy_endian.h"

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

/*
 * C99 specifies that complex numbers have the same representation as
 * an array of two elements, where the first element is the real part
 * and the second element is the imaginary part.
 */
#ifdef HAVE_COMPLEX_H
#include "complex.h"
typedef complex npy_complex_double;
typedef union {
        npy_complex_double npy;
        complex c99;
} _double_complex;
#else
typedef struct {
        double x, y;
} npy_complex_double;
typedef union {
        npy_complex_double npy;
        npy_complex_double c99;
} _double_complex;
#endif

#if 0
typedef union {
        npy_complex_float f;
        float a[2];
} _float_complex;
#endif
#if 0
typedef union {
        npy_complex_longdouble f;
        npy_longdouble a[2];
} _long_double_complex;
#endif

/* those can be used as lvalues */
#define REALPART(z)     ((z).a[0])
#define IMAGPART(z)     ((z).a[1])

static inline npy_complex_double cpack(double x, double y)
{
        _double_complex z;

#if 0
        REALPART(z.npy_z) = x;
        IMAGPART(z.npy_z) = y;
#endif
        return z.npy;
}

#endif /* !_NPY_MATH_PRIVATE_H_ */
