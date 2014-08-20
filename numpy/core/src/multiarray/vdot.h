#ifndef _NPY_VDOT_H_
#define _NPY_VDOT_H_

#include "common.h"

NPY_NO_EXPORT void
CFLOAT_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CDOUBLE_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CLONGDOUBLE_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
OBJECT_vdot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

#endif
