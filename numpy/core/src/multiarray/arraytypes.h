#ifndef _NPY_ARRAYTYPES_H_
#define _NPY_ARRAYTYPES_H_

#include "common.h"

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyArray_Descr LONGLONG_Descr;
extern NPY_NO_EXPORT PyArray_Descr LONG_Descr;
extern NPY_NO_EXPORT PyArray_Descr INT_Descr;

/* needed for blasfuncs */
NPY_NO_EXPORT void
FLOAT_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CFLOAT_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
DOUBLE_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CDOUBLE_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);
#endif

NPY_NO_EXPORT int
set_typeinfo(PyObject *dict);

#endif
