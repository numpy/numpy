/*
 * This file is for inclusion in cython *.pxd files in order to define
 * the NPY_NO_DEPRECATED_API macro. In order to include it, do
 *
 * cdef extern from "numpy/npy_no_deprecated_api.h": pass
 *
 */
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#endif
