#ifndef _NPY_PRIVATE_REFCOUNT_H_
#define _NPY_PRIVATE_REFCOUNT_H_

NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr);

NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr);

NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp);

NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp);

NPY_NO_EXPORT void
PyArray_FillObjectArray(PyArrayObject *arr, PyObject *obj);

#endif
