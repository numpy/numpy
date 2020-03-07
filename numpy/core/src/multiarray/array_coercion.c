#include "numpy/arrayobject.h"

#include "descriptor.h"
#include "convert_datatype.h"
#include "dtypemeta.h"

#include "array_coercion.h"
#include "ctors.h"


/*
 * This file defines helpers for some of the ctors.c functions which
 * create an array from Python sequences and types.
 * When creating an array with ``np.array(...)`` we have to do two main things:
 *
 * 1. Find the exact shape of the resulting array
 * 2. Find the correct dtype of the resulting array.
 *
 * In most cases these two things are can be done in a single processing step.
 * There are in principle three different calls that should be distinguished:
 *
 * 1. The user calls ``np.array(..., dtype=np.dtype("<f8"))``
 * 2. The user calls ``np.array(..., dtype="S")``
 * 3. The user calls ``np.array(...)``
 *
 * In the first case, in principle only the shape needs to be found. In the
 * second case, the DType class (e.g. string) is already known but the DType
 * instance (e.g. length of the string) has to be found.
 * In the last case the DType class needs to be found as well. Note that
 * it is not necessary to find the DType class of the entire array, but
 * the DType class needs to be found for each element before the actual
 * dtype instance can be found.
 *
 * Further, there are a few other things to keep in mind when coercing arrays:
 *
 *   * For UFunc promotion, Python scalars need to be handled specially to
 *     allow value based casting.
 *   * It is necessary to decide whether or not a sequence is an element.
 *     For example tuples are considered elements for structured dtypes, but
 *     otherwise are considered sequences.
 *     This means that if a dtype is given (either as a class or instance),
 *     it can effect the dimension discovery part.
 *
 * In the initial version of this implementation, it is assumed that dtype
 * discovery can be implemented sufficiently fast, that it is not necessary
 * to create fast paths that only find the correct shape e.g. when
 * ``dtype=np.dtype("f8")`` is given.
 *
 * One design goal in this code is to avoid multiple conversions of nested
 * array like objects and sequences. Thus a cache is created to store sequences
 * for the internal API which in almost all cases will, after allocating the
 * new array, iterate all objects a second time to fill that array.
 */


static int
update_shape(int curr_ndim, int *max_ndim,
             npy_intp out_shape[NPY_MAXDIMS], int new_ndim,
             const npy_intp new_shape[NPY_MAXDIMS], npy_bool sequence)
{
    int success = 0;  /* unsuccessful if array is ragged */
    if (curr_ndim + new_ndim > *max_ndim) {
        success = -1;
        /* Only update check as many dims as possible, max_ndim is unchanged */
        new_ndim = *max_ndim - curr_ndim;
    }
    else if (!sequence && (*max_ndim != curr_ndim + new_ndim)) {
        /*
         * Sequences do not update max_ndim, otherwise shrink and check.
         * This is depth first, so if it is already set, `out_shape` is filled.
         */
        *max_ndim = curr_ndim + new_ndim;
        /* If a shape was already set, this is also ragged */
        if (out_shape[*max_ndim] >= 0) {
            success = -1;
        }
    }
    for (int i = 0; i < new_ndim; i++) {
        npy_intp curr_dim = out_shape[curr_ndim + i];
        npy_intp new_dim = new_shape[i];

        if (curr_dim == -1) {
            out_shape[curr_ndim + i] = new_dim;
        }
        else if (new_dim != curr_dim) {
            /* The array is ragged, and this dimension is unusable already */
            success = -1;
            if (!sequence) {
                /* Remove dimensions that we cannot use: */
                *max_ndim -= new_ndim + i;
            }
            else {
                assert(i == 0);
                /* max_ndim is usually not updated for sequences, so set now: */
                *max_ndim = curr_ndim;
            }
            break;
        }
    }
    return success;
}


/*
 * This cache is necessary for the simple case of an array input mostly.
 * Since that is the main reason, this may be removed if the single array
 * case is handled specifically up-front.
 * This may be better as a different caching mechanism...
 */
NPY_NO_EXPORT int
npy_new_coercion_cache(
        PyObject *converted_obj, PyObject *arr_or_sequence, npy_bool sequence,
        coercion_cache_obj ***next_ptr)
{
    coercion_cache_obj *cache = PyArray_malloc(sizeof(coercion_cache_obj));
    if (cache == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    cache->converted_obj = converted_obj;
    Py_INCREF(arr_or_sequence);
    cache->arr_or_sequence = arr_or_sequence;
    cache->sequence = sequence;
    cache->next = NULL;
    **next_ptr = cache;
    *next_ptr = &(cache->next);
    return 0;
}


NPY_NO_EXPORT void
npy_free_coercion_cache(coercion_cache_obj *next) {
    /* We only need to check from the last used cache pos */
    int cache_pos = 0;
    while (next != NULL) {
        coercion_cache_obj *current = next;
        next = current->next;

        Py_DECREF(current->arr_or_sequence);
        PyArray_free(current);
    }
}



static NPY_INLINE int
handle_promotion(PyArray_Descr**out_descr, PyArray_Descr *descr)
{
    if (*out_descr == NULL) {
        *out_descr = descr;
        return 0;
    }
    PyArray_Descr *new_descr = PyArray_PromoteTypes(*out_descr, descr);
    // TODO: Have to take care of the retry-with-string logic for now :(
    Py_DECREF(descr);
    if (new_descr == NULL) {
        return -1;
    }
    Py_SETREF(*out_descr, new_descr);
    return 0;

}


enum _dtype_discovery_flags {
    IS_RAGGED_ARRAY = 1,
};


/**
 * Discover the dtype and shape for a potentially nested sequence of scalars.
 * Note that in the ufunc machinery, when value based casting is desired it
 * is necessary to first check for the scalar case.
 *
 * @param obj The python object or nested sequence to convert
 * @param max_dims The maximum number of dimensions.
 * @param curr_dims The current number of dimensions (depth in the recursion)
 * @param out_shape The discovered output shape, will be filled
 * @param coercion_cache The coercion cache object to use.
 * @param DType the DType class that should be used, or NULL, if not provided.
 * @param requested_descr The dtype instance passed in by the user, this is
 *        is used only array-likes.
 * @param flags used signal that this is a ragged array, used internally and
 *        can be expanded if necessary.
 */
NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape_Recursive(
        PyObject *obj, int curr_dims, int max_dims, PyArray_Descr**out_descr,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj ***coercion_cache_tail_ptr,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        _dtype_discovery_flags *flags)
{
    /*
     * The first step is to find the DType class if it was not provided
     */
    PyArray_DTypeMeta *DType;
    if (fixed_DType == NULL) {
        DType = discover_dtype_from_pytype(Py_TYPE(obj));
        if (DType == NULL) {
            // TODO: Is there a need for an error return here?
            return -1;
        }
    }
    else {
        Py_INCREF(fixed_DType);
        DType = fixed_DType;
    }

    /*
     * The second step is to ask the DType class to handle the scalar cases
     * or return NotImplemented to signal that this should be assumed to be
     * an array-like or sequence.
     * We do this even when the dtype was provided, to handle the dimension
     * discovery (possibly a fastpath can be added for that at some point).
     */
    if (DType != (PyArray_DTypeMeta *)Py_NotImplemented) {
        assert(Py_TYPE(DType) == &PyArrayDTypeMeta_Type);

        PyArray_Descr *descr = DType->discover_descr_from_pyobject(obj);
        Py_DECREF(DType);
        if (descr == NULL) {
            return -1;
        }
        else if (descr == (PyArray_Descr *) Py_NotImplemented) {
            Py_DECREF(descr);
        }
        else {
            /* This is a scalar */
            if (update_shape(curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE) < 0) {
                goto ragged_array;
            }
            if (handle_promotion(out_descr, descr) < 0) {
                Py_DECREF(descr);
                return -1;
            }
            Py_DECREF(descr);
            return max_dims;
        }
    }
    else {
        /* If no DType was found, this must be an array or a sequence */
        assert(DType == (PyArray_DTypeMeta *)Py_NotImplemented);
        Py_DECREF(DType);
    }

    /*
     * The third step is to first check for any arrays or array-likes.
     */

    /* Check if it's an ndarray */
    PyArrayObject *arr = NULL;
    if (PyArray_Check(obj)) {
        arr = (PyArrayObject *)obj;
        Py_INCREF(arr);
    }
    else {
        arr = _array_from_array_like(obj,  requested_descr, 0, NULL);
        if (arr == NULL) {
            goto fail;
        }
        else if (arr == (PyArrayObject *)Py_NotImplemented) {
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    if (arr) {
        /* This is an array object which will be added to the cache */
        if (npy_new_coercion_cache(obj, arr, 0, coercion_cache_tail_ptr) < 0) {
            goto fail;
        }
        if (update_shape(curr_dims, &max_dims, out_shape,
                PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FALSE) < 0) {
            goto ragged_array;
        }
        if (handle_promotion(out_descr, PyArray_DESCR(arr)) < 0) {
            return -1;
        }
        return max_dims;
    }

    /*
     * The last step is to assume the input should be handled as a sequence
     * and to handle it recursively.
     */
    // TODO: Is the PySequence check necessary or does PySequence_Fast suffice?
    if (!PySequence_Check(obj) || PySequence_Size(obj) < 0) {
        /* clear any PySequence_Size error which corrupts further calls */
        PyErr_Clear();

        /* This branch always leads to a ragged array */
        update_shape(curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE);
        goto ragged_array;
    }

    /* Ensure we have a sequence (required for PyPy) */
    PyObject *seq = PySequence_Fast(obj, "Could not convert object to sequence");
    if (seq == NULL) {
        /* Specifically do not fail on things that look like a dictionary */
        if (PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
            update_shape(curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE);
            goto ragged_array;
        }
        goto fail;
    }
    if (npy_new_coercion_cache(obj, seq, 1, coercion_cache_tail_ptr) < 0) {
        Py_DECREF(seq);
        goto fail;
    }

    npy_intp size = PySequence_Fast_GET_SIZE(seq);
    PyObject **objects = PySequence_Fast_ITEMS(seq);

    if (update_shape(curr_dims, &max_dims,
                     out_shape, 1, &size, NPY_TRUE) < 0) {
        /* But do update, if there this is a ragged case */
        Py_DECREF(seq);
        goto ragged_array;
    }
    if (size == 0) {
        Py_DECREF(seq);
        /* If the sequence is empty, we have to assume thats it... */
        return curr_dims+1;
    }

    /* Recursive call for each sequence item */
    for (Py_ssize_t i = 0; i < size; ++i) {
        max_dims = PyArray_DiscoverDTypeAndShape_Recursive(
                objects[i], max_dims, curr_dims + 1,
                out_dtype, out_shape, use_minimal,
                single_or_no_element, requested_dtype, context,
                stop_at_tuple, string_is_sequence,
                prev_type, prev_dtype, coercion_cache_tail_ptr);
        // NOTE: If there is a ragged array found (NPY_OBJECT) could break
        if (max_dims < 0) {
            Py_DECREF(seq);
            goto fail;
        }
    }
    Py_DECREF(seq);
    return max_dims;



ragged_array:
    /*
     * This is discovered as a ragged array, which means the dtype is
     * guaranteed to be object. A warning will need to be given if an
     * dtype[object] was not requested (checked outside to only warn once).
     */
    *flags &= IS_RAGGED_ARRAY;
    Py_XDECREF(*out_descr);
    *out_descr = PyArray_DescrFromType(NPY_OBJECT);
    return max_dims;
}