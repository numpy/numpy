#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT const char*
ufunc_get_name_cstr(PyUFuncObject *ufunc);

/* interned strings (on umath import) */
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_out;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_subok;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_array_prepare;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_array_wrap;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_array_finalize;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_ufunc;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_pyvals_name;

#endif
