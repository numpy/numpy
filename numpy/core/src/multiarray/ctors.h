#ifndef NUMPY_CORE_SRC_MULTIARRAY_CTORS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CTORS_H_

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj);

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescrAndBase(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base);


/* Private options for NewFromDescriptor */
typedef enum {
    /*
     * Indicate the the array should be zeroed, we may use calloc to do so
     * (otherwise much like ).
     */
    _NPY_ARRAY_ZEROED = 1 << 0,
    /* Whether to allow empty strings (implied by ensure dtype identity) */
    _NPY_ARRAY_ALLOW_EMPTY_STRING = 1 << 1,
    /*
     * If we take a view into an existing array and use its dtype, then that
     * dtype must be preserved (for example for subarray and S0, but also
     * possibly for future dtypes that store more metadata).
     */
    _NPY_ARRAY_ENSURE_DTYPE_IDENTITY = 1 << 2,
} _NPY_CREATION_FLAGS;

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr_int(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base, _NPY_CREATION_FLAGS cflags);

NPY_NO_EXPORT PyObject *
PyArray_NewLikeArrayWithShape(
        PyArrayObject *prototype, NPY_ORDER order,
        PyArray_Descr *dtype, int ndim, npy_intp const *dims, int subok);

NPY_NO_EXPORT PyObject *
PyArray_New(
        PyTypeObject *, int nd, npy_intp const *,
        int, npy_intp const*, void *, int, int, PyObject *);

NPY_NO_EXPORT PyObject *
_array_from_array_like(PyObject *op,
        PyArray_Descr *requested_dtype, npy_bool writeable, PyObject *context,
        int never_copy);

NPY_NO_EXPORT PyObject *
PyArray_FromAny_int(PyObject *op, PyArray_Descr *in_descr,
                    PyArray_DTypeMeta *in_DType, int min_depth, int max_depth,
                    int flags, PyObject *context);

NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context);

NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny_int(PyObject *op, PyArray_Descr *in_descr,
                         PyArray_DTypeMeta *in_DType, int min_depth,
                         int max_depth, int requires, PyObject *context);

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
PyArray_FromArrayAttr_int(
        PyObject *op, PyArray_Descr *descr, int never_copy);

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
_array_fill_strides(npy_intp *strides, npy_intp const *dims, int nd, size_t itemsize,
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

/*
 * Calls arr_of_subclass.__array_wrap__(towrap), in order to make 'towrap'
 * have the same ndarray subclass as 'arr_of_subclass'.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_SubclassWrap(PyArrayObject *arr_of_subclass, PyArrayObject *towrap);


#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CTORS_H_ */
