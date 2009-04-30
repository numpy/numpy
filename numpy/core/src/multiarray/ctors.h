#ifndef _NPY_ARRAY_CTORS_H_
#define _NPY_ARRAY_CTORS_H_

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(PyTypeObject *subtype, PyArray_Descr *descr, int nd,
                     intp *dims, intp *strides, void *data,
                     int flags, PyObject *obj);

NPY_NO_EXPORT PyObject *PyArray_New(PyTypeObject *, int nd, intp *,
                             int, intp *, void *, int, int, PyObject *);

NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context);

NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny(PyObject *op, PyArray_Descr *descr, int min_depth,
                     int max_depth, int requires, PyObject *context);

NPY_NO_EXPORT PyObject *
PyArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int flags);

NPY_NO_EXPORT PyObject *
PyArray_FromStructInterface(PyObject *input);

NPY_NO_EXPORT PyObject *
PyArray_FromInterface(PyObject *input);

NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode,
                      PyObject *context);

NPY_NO_EXPORT PyObject *
PyArray_EnsureArray(PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_EnsureAnyArray(PyObject *op);

NPY_NO_EXPORT int
PyArray_MoveInto(PyArrayObject *dest, PyArrayObject *src);

NPY_NO_EXPORT int
PyArray_CopyAnyInto(PyArrayObject *dest, PyArrayObject *src);

NPY_NO_EXPORT PyObject *
PyArray_CheckAxis(PyArrayObject *arr, int *axis, int flags);

/* FIXME: remove those from here */
NPY_NO_EXPORT int
_flat_copyinto(PyObject *dst, PyObject *src, NPY_ORDER order);

#endif
