/*
 * Copyright (c) 2014 Ryan Juckett
 * http://www.ryanjuckett.com/
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 *
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source
 *    distribution.
 */

/*
 * This file contains a modified version of Ryan Juckett's Dragon4
 * implementation, which has been ported from C++ to C and which has
 * modifications specific to printing floats in numpy.
 */

#ifndef _NPY_DRAGON4_H_
#define _NPY_DRAGON4_H_

#include "Python.h"
#include "structmember.h"
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "numpy/arrayscalars.h"

typedef enum TrimMode
{
    TrimMode_None,         /* don't trim zeros, always leave a decimal point */
    TrimMode_LeaveOneZero, /* trim all but the zero before the decimal point */
    TrimMode_Zeros,        /* trim all trailing zeros, leave decimal point */
    TrimMode_DptZeros,     /* trim trailing zeros & trailing decimal point */
} TrimMode;

PyObject *
Dragon4_Positional_AnySize(void *val, size_t size, npy_bool unique,
                           int precision, int sign, TrimMode trim,
                           int pad_left, int pad_right);

PyObject *
Dragon4_Scientific_AnySize(void *val, size_t size, npy_bool unique,
                           int precision, int sign, TrimMode trim,
                           int pad_left, int exp_digits);

PyObject *
Dragon4_Positional(PyObject *obj, npy_bool unique, int precision, int sign,
                   TrimMode trim, int pad_left, int pad_right);

PyObject *
Dragon4_Scientific(PyObject *obj, npy_bool unique, int precision, int sign,
                   TrimMode trim, int pad_left, int exp_digits);

#endif

