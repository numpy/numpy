/*
 * This file implements the missing value NA singleton object for NumPy.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

/*
 * Initializes the numpy.NA singleton object. Just like None in CPython,
 * the type of NA is not exposed, and there is no NpyNA_Check() function.
 *
 * Some behaviors of the NA singleton:
 *   - np.NA(payload=None, dtype=None) creates a zero-dimensional
 *     ndarray containing a single NA. If dtype is an NA dtype, it's
 *     an ndarray without an NA mask, otherwise it does have an NA mask.
 */
NPY_NO_EXPORT void
numpy_na_singleton_init()
{
}

