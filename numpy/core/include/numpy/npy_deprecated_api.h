#ifndef _NPY_DEPRECATED_API_H
#define _NPY_DEPRECATED_API_H

/*
 * This header exists to collect all dangerous/deprecated NumPy API.
 *
 * This is an attempt to remove bad API, the proliferation of macros,
 * and namespace pollution currently produced by the NumPy headers.
 */

#ifdef NPY_NO_DEPRECATED_API
#error Should never include npy_deprecated_api directly.
#endif

/* These array flags are deprecated as of NumPy 1.7 */
#define NPY_CONTIGUOUS NPY_ARRAY_C_CONTIGUOUS
#define NPY_FORTRAN NPY_ARRAY_F_CONTIGUOUS

#endif
