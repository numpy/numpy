#ifndef CLOG1P_WRAPPERS_H
#define CLOG1P_WRAPPERS_H

// This header is to be included in umathmodule.c,
// so it will be processed by a C compiler.
//
// This file assumes that the numpy header files have
// already been included.

NPY_NO_EXPORT void
clog1pf(npy_cfloat *z, npy_cfloat *r);

NPY_NO_EXPORT void
clog1p(npy_cdouble *z, npy_cdouble *r);

NPY_NO_EXPORT void
clog1pl(npy_clongdouble *z, npy_clongdouble *r);

#endif  // CLOG1P_WRAPPERS_H
