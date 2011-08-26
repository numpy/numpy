#ifndef _NPY_PRIVATE__BOOLEAN_OPS_H_
#define _NPY_PRIVATE__BOOLEAN_OPS_H_

NPY_NO_EXPORT PyArrayObject *
PyArray_ReduceAny(PyArrayObject *arr, PyArrayObject *out,
            npy_bool *axis_flags, int skipna, int keepdims);

NPY_NO_EXPORT PyArrayObject *
PyArray_ReduceAll(PyArrayObject *arr, PyArrayObject *out,
            npy_bool *axis_flags, int skipna, int keepdims);


#endif
