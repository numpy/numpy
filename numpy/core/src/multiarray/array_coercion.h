#ifndef _NPY_ARRAY_COERCION_H
#define _NPY_ARRAY_COERCION_H

#include <numpy/ndarraytypes.h>


NPY_NO_EXPORT int
_PyArray_MapPyTypeToDType(
        PyArray_DTypeMeta *DType, PyTypeObject *pytype, npy_bool userdef);


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

/* Create a new cache object */
NPY_NO_EXPORT int npy_new_coercion_cache(
        PyObject *converted_obj, PyObject *arr_or_sequence, npy_bool sequence,
        coercion_cache_obj ***next_ptr);

/* Frees the coercion cache object. */
NPY_NO_EXPORT void npy_free_coercion_cache(coercion_cache_obj *first);


#endif  /* _NPY_ARRAY_COERCION_H */
