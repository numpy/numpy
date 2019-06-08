#ifndef _NPY_ARRAY_STRFUNCS_H_
#define _NPY_ARRAY_STRFUNCS_H_

NPY_NO_EXPORT void
PyArray_SetStringFunction(PyObject *op, int repr);

NPY_NO_EXPORT PyObject *
array_repr(PyArrayObject *self);

NPY_NO_EXPORT PyObject *
array_str(PyArrayObject *self);

NPY_NO_EXPORT PyObject *
array_format(PyArrayObject *self, PyObject *args);

#ifndef NPY_PY3K
    NPY_NO_EXPORT PyObject *
    array_unicode(PyArrayObject *self);
#endif

#endif
