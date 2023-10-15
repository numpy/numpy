#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAY_COERCION_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAY_COERCION_H_


/*
 * We do not want to coerce arrays many times unless absolutely necessary.
 * The same goes for sequences, so everything we have seen, we will have
 * to store somehow. This is a linked list of these objects.
 */
typedef struct coercion_cache_obj {
    PyObject *converted_obj;
    PyObject *arr_or_sequence;
    struct coercion_cache_obj *next;
    npy_bool sequence;
    int depth;  /* the dimension at which this object was found. */
} coercion_cache_obj;

NPY_NO_EXPORT int
_PyArray_MapPyTypeToDType(
        PyArray_DTypeMeta *DType, PyTypeObject *pytype, npy_bool userdef);

NPY_NO_EXPORT PyObject *
PyArray_DiscoverDTypeFromScalarType(PyTypeObject *pytype);

NPY_NO_EXPORT int
PyArray_Pack(PyArray_Descr *descr, char *item, PyObject *value);

NPY_NO_EXPORT PyArray_Descr *
PyArray_AdaptDescriptorToArray(
        PyArrayObject *arr, PyArray_DTypeMeta *dtype, PyArray_Descr *descr);

NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape(
        PyObject *obj, int max_dims,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj **coercion_cache,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        PyArray_Descr **out_descr, int never_copy);

NPY_NO_EXPORT PyObject *
_discover_array_parameters(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames);

/* Would make sense to inline the freeing functions everywhere */
/* Frees the coercion cache object recursively. */
NPY_NO_EXPORT void
npy_free_coercion_cache(coercion_cache_obj *first);

/* unlink a single item and return the next */
NPY_NO_EXPORT coercion_cache_obj *
npy_unlink_coercion_cache(coercion_cache_obj *current);

NPY_NO_EXPORT int
PyArray_AssignFromCache(PyArrayObject *self, coercion_cache_obj *cache);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAY_COERCION_H_ */
