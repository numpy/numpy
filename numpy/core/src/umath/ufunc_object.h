#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

#endif
