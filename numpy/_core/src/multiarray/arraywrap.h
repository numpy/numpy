#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_


typedef struct {
    PyObject *ufunc;
    PyObject *in;
    PyObject *out;
    int out_i;
} NpyUFuncContext;


NPY_NO_EXPORT PyObject *
npy_apply_wrap(
        PyObject *obj, PyObject *original_out,
        PyObject *wrap, PyObject *wrap_type,
        NpyUFuncContext *context, npy_bool return_scalar, npy_bool force_wrap);


NPY_NO_EXPORT PyObject *
npy_apply_wrap_simple(PyArrayObject *arr_of_subclass, PyArrayObject *towrap);


NPY_NO_EXPORT int
npy_find_array_wrap(
        int nin, PyObject *const *inputs,
        PyObject **out_wrap, PyObject **out_wrap_type);


#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYWRAP_H_ */
