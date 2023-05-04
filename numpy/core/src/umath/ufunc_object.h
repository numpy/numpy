#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_

#include <numpy/ufuncobject.h>

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *NPY_UNUSED(arg));

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *arg);

NPY_NO_EXPORT const char*
ufunc_get_name_cstr(PyUFuncObject *ufunc);

NPY_NO_EXPORT PyObject *
PyUFunc_GetDefaultIdentity(PyUFuncObject *ufunc, npy_bool *reorderable);

/* strings from umathmodule.c that are interned on umath import */
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array_ufunc;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array_prepare;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array_wrap;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_pyvals_name;

#endif
