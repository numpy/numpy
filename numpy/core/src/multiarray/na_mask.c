/*
 * This file implements missing value NA mask support for the NumPy array.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API
#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "numpy/npy_3kcompat.h"

/*NUMPY_API
 *
 * Returns true if the array has an NA mask. When
 * NA dtypes are implemented, will also return true
 * if the array's dtype has NA support.
 */
NPY_NO_EXPORT npy_bool
PyArray_HasNASupport(PyArrayObject *arr)
{
    return PyArray_HASMASKNA(arr);
}

/*NUMPY_API
 *
 * Returns false if the array has no NA support. Returns
 * true if the array has NA support AND there is an
 * NA anywhere in the array.
 */
NPY_NO_EXPORT npy_bool
PyArray_ContainsNA(PyArrayObject *arr)
{
    /* Need NA support to contain NA */
    if (!PyArray_HasNASupport(arr)) {
        return 0;
    }

    /* TODO: Loop through NA mask */

    return 0;
}
