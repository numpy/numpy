#ifndef _NPY_ARRAYTYPES_H_
#define _NPY_ARRAYTYPES_H_

#include "common.h"

extern NPY_NO_EXPORT PyArray_Descr LONGLONG_Descr;
extern NPY_NO_EXPORT PyArray_Descr LONG_Descr;
extern NPY_NO_EXPORT PyArray_Descr INT_Descr;

NPY_NO_EXPORT int
set_typeinfo(PyObject *dict);

/* needed for blasfuncs */
NPY_NO_EXPORT void
FLOAT_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CFLOAT_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
DOUBLE_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CDOUBLE_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);


/* for _pyarray_correlate */
NPY_NO_EXPORT int
small_correlate(const char * d_, npy_intp dstride,
                npy_intp nd, enum NPY_TYPES dtype,
                const char * k_, npy_intp kstride,
                npy_intp nk, enum NPY_TYPES ktype,
                char * out_, npy_intp ostride);

#endif
