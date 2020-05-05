#ifndef _NPY_ARRAY_COERCION_H
#define _NPY_ARRAY_COERCION_H


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
} coercion_cache_obj;


NPY_NO_EXPORT int
_PyArray_MapPyTypeToDType(
        PyArray_DTypeMeta *DType, PyTypeObject *pytype, npy_bool userdef);

NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape(
        PyObject *obj, int max_dims,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj **coercion_cache,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        PyArray_Descr **out_descr);

NPY_NO_EXPORT int
PyArray_ExtractDTypeAndDescriptor(PyObject *dtype,
        PyArray_Descr **out_descr, PyArray_DTypeMeta **out_DType);

NPY_NO_EXPORT PyObject *
_discover_array_parameters(PyObject *NPY_UNUSED(self),
                           PyObject *args, PyObject *kwargs);


/* Create a new cache object */
NPY_NO_EXPORT int npy_new_coercion_cache(
        PyObject *converted_obj, PyObject *arr_or_sequence, npy_bool sequence,
        coercion_cache_obj ***next_ptr);


/* Frees the coercion cache object. */
NPY_NO_EXPORT void npy_free_coercion_cache(coercion_cache_obj *first);



#endif  /* _NPY_ARRAY_COERCION_H */
