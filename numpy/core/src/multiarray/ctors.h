#ifndef _NPY_ARRAY_CTORS_H_
#define _NPY_ARRAY_CTORS_H_

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(PyTypeObject *subtype, PyArray_Descr *descr, int nd,
                     npy_intp *dims, npy_intp *strides, void *data,
                     int flags, PyObject *obj);

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescrAndBase(
        PyTypeObject *subtype, PyArray_Descr *descr,
        int nd, npy_intp *dims, npy_intp *strides, void *data,
        int flags, PyObject *obj, PyObject *base);

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr_int(PyTypeObject *subtype, PyArray_Descr *descr, int nd,
                         npy_intp *dims, npy_intp *strides, void *data,
                         int flags, PyObject *obj, PyObject *base, int zeroed,
                         int allow_emptystring);

NPY_NO_EXPORT PyObject *PyArray_New(PyTypeObject *, int nd, npy_intp *,
                             int, npy_intp *, void *, int, int, PyObject *);

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

/* TODO: Put the order parameter in PyArray_CopyAnyInto and remove this */
NPY_NO_EXPORT int
PyArray_CopyAsFlat(PyArrayObject *dst, PyArrayObject *src,
                                NPY_ORDER order);

/* FIXME: remove those from here */
NPY_NO_EXPORT void
_array_fill_strides(npy_intp *strides, npy_intp *dims, int nd, size_t itemsize,
                    int inflag, int *objflags);

NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize);

NPY_NO_EXPORT void
_strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size);

NPY_NO_EXPORT void
copy_and_swap(void *dst, void *src, int itemsize, npy_intp numitems,
              npy_intp srcstrides, int swap);

NPY_NO_EXPORT void
byte_swap_vector(void *p, npy_intp n, int size);

NPY_NO_EXPORT int
PyArray_AssignFromSequence(PyArrayObject *self, PyObject *v);

/*
 * Calls arr_of_subclass.__array_wrap__(towrap), in order to make 'towrap'
 * have the same ndarray subclass as 'arr_of_subclass'.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_SubclassWrap(PyArrayObject *arr_of_subclass, PyArrayObject *towrap);


#endif
