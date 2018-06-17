#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT const char*
ufunc_get_name_cstr(PyUFuncObject *ufunc);

/* interned strings (on umath import) */
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_out;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_where;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_axes;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_axis;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_keepdims;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_casting;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_order;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_dtype;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_subok;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_signature;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_sig;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_extobj;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_initial;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_indices;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array_prepare;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array_wrap;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array_finalize;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_ufunc;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_pyvals_name;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str___call__;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_reduce;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_accumulate;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_reduceat;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_outer;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_at;

#endif
