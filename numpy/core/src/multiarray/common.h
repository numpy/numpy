#ifndef _NPY_PRIVATE_COMMON_H_
#define _NPY_PRIVATE_COMMON_H_

#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

NPY_NO_EXPORT PyArray_Descr *
_array_find_type(PyObject *op, PyArray_Descr *minitype, int max);

NPY_NO_EXPORT PyArray_Descr *
_array_small_type(PyArray_Descr *chktype, PyArray_Descr* mintype);

NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op);

NPY_NO_EXPORT PyArray_Descr *
_array_typedescr_fromstr(char *str);

NPY_NO_EXPORT char *
index2ptr(PyArrayObject *mp, intp i);

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret);

NPY_NO_EXPORT int
_IsAligned(PyArrayObject *ap);

NPY_NO_EXPORT Bool
_IsWriteable(PyArrayObject *ap);

NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, intp outstrides, char *src,
                             intp instrides, intp N, int elsize);

NPY_NO_EXPORT void
_strided_byte_swap(void *p, intp stride, intp n, int size);

NPY_NO_EXPORT void
byte_swap_vector(void *p, intp n, int size);

#ifndef Py_UNICODE_WIDE
#include "ucsnarrow.h"
#endif

#endif
