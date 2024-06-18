#ifndef LOG1P_COMPLEX_WRAPPERS_H
#define LOG1P_COMPLEX_WRAPPERS_H

// This header is to be included in umathmodule.c,
// so it will be processed by a C compiler.
//
// This file assumes that the numpy header files have
// already been included.

NPY_NO_EXPORT void
nc_log1pf(npy_cfloat *x, npy_cfloat *r);

NPY_NO_EXPORT void
nc_log1p(npy_cdouble *x, npy_cdouble *r);

NPY_NO_EXPORT void
nc_log1pl(npy_clongdouble *x, npy_clongdouble *r);

#endif  // LOG1P_COMPLEX_WRAPPERS_H
