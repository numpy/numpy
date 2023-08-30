#ifndef NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_

#include "numpy/ndarraytypes.h"

NPY_NO_EXPORT PyObject *
arr_place(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_bincount(PyObject *, PyObject *const *, Py_ssize_t, PyObject *);
NPY_NO_EXPORT PyObject *
arr__monotonicity(PyObject *, PyObject *, PyObject *kwds);
NPY_NO_EXPORT PyObject *
arr_interp(PyObject *, PyObject *const *, Py_ssize_t, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_interp_complex(PyObject *, PyObject *const *, Py_ssize_t, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_ravel_multi_index(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_unravel_index(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_add_docstring(PyObject *, PyObject *const *, Py_ssize_t);
NPY_NO_EXPORT PyObject *
io_pack(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
io_unpack(PyObject *, PyObject *, PyObject *);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_ */
