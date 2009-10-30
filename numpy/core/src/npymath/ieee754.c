/*
 * Low-level routines related to IEEE-754 format
 */
#include <Python.h>
#include <math.h>

#include "npy_config.h"
#include "numpy/npy_math.h"

#include "npy_math_private.h"

#ifndef HAVE_COPYSIGN
double npy_copysign(double x, double y)
{
    npy_uint32 hx, hy;
    GET_HIGH_WORD(hx, x);
    GET_HIGH_WORD(hy, y);
    SET_HIGH_WORD(x, (hx & 0x7fffffff) | (hy & 0x80000000));
    return x;
}
#endif

#if !defined(HAVE_DECL_SIGNBIT)
#include "_signbit.c"

int _npy_signbit_f(float x)
{
    return _npy_signbit_d((double) x);
}

int _npy_signbit_ld(long double x)
{
    return _npy_signbit_d((double) x);
}
#endif
