#ifndef _NPY_PRIVATE__COMPILED_BASE_H_
#define _NPY_PRIVATE__COMPILED_BASE_H_
#include <numpy/ndarraytypes.h>

NPY_NO_EXPORT PyObject *
arr_insert(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_bincount(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_digitize(PyObject *, PyObject *, PyObject *kwds);
NPY_NO_EXPORT PyObject *
arr_interp(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_ravel_multi_index(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_unravel_index(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_add_docstring(PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
io_pack(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
io_unpack(PyObject *, PyObject *, PyObject *);

#endif
