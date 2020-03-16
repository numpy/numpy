#include "numpy/arrayobject.h"

#include "Python.h"
#include "descriptor.h"
#include "convert_datatype.h"
#include "dtypemeta.h"

#include "array_coercion.h"
#include "ctors.h"
#include "common.h"
#include "_datetime.h"
#include "npy_import.h"


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
 *     allow value based casting. For this purpose they have a ``bound_value``
 *     slot.
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


/*
 * For finding a DType quickly from a type, it is easiest to have a
 * a mapping of pytype -> dtype.
 * Since a DType must know its type, but the type not the DType, we will
 * store the DType as a weak reference. When a reference is dead we can
 * remove the item from the dictionary.
 * A cleanup should probably be done occasionally if (and only if) a large
 * number of type -> DType mappings are added.
 * This assumes that the mapping is a bifurcation DType <-> type
 * (there is exactly one DType for each type and vise versa).
 * If it is not, it is possible for a python type to stay alive unnecessarily.
 */
PyObject *_global_pytype_to_type_dict = NULL;


enum _dtype_discovery_flags {
    IS_RAGGED_ARRAY = 1,
    GAVE_SUBCLASS_WARNING = 2,
};


/**
 * Add a new mapping from a python type to the DType class. This assumes
 * that the DType class is guaranteed to hold on the python type (this
 * assumption is guaranteed).
 * This function replaces ``_typenum_fromtypeobj``.
 */
NPY_NO_EXPORT int
_PyArray_MapPyTypeToDType(
        PyArray_DTypeMeta *DType, PyTypeObject *pytype, npy_bool userdef)
{
    PyObject *Dtype_obj = (PyObject *)DType;

    if (userdef) {
        /*
         * It seems we did not strictly enforce this in the legacy dtype
         * API, but assume that it is always true. Further, this could be
         * relaxed in the future. In particular we should have a new
         * superclass of ``np.generic`` in order to note enforce the array
         * scalar behaviour.
         */
        if (!PyObject_IsSubclass((PyObject *)pytype, (PyObject *)&PyGenericArrType_Type)) {
            PyErr_Format(PyExc_RuntimeError,
                    "currently it is only possible to register a DType "
                    "for scalars deriving from `np.generic`, got '%S'.",
                    (PyObject *)pytype);
            return -1;
        }
    }

    /* Create the global dictionary if it does not exist */
    if (_global_pytype_to_type_dict == NULL) {
        _global_pytype_to_type_dict = PyDict_New();
        if (_global_pytype_to_type_dict == NULL) {
            return -1;
        }
    }

    int res = PyDict_Contains(_global_pytype_to_type_dict, Dtype_obj);
    if (res < 0) {
        return -1;
    }
    else if (res) {
        PyErr_SetString(PyExc_RuntimeError,
                "Can only map one python type to DType.");
    }

    PyObject *weakref = PyWeakref_NewRef(Dtype_obj, NULL);
    if (weakref == NULL) {
        return -1;
    }
    PyDict_SetItem(_global_pytype_to_type_dict, Dtype_obj, (PyObject *)pytype);
}


/**
 * Find the correct DType class for the given python type.
 *
 * @param obj The python object, mainly type(pyobj) is used, the object
 *            is passed to reuse existing code at this time only.
 * @param flags Flags used to know if warnings were already given.
 * @return New reference to either a DType class, Py_NotImplemented, or NULL
 */
static PyArray_DTypeMeta *
discover_dtype_from_pytype(PyObject *obj, enum _dtype_discovery_flags *flags)
{
    PyObject *pytype = (PyObject *)Py_TYPE(obj);
    PyObject *weakref = PyDict_GetItem(_global_pytype_to_type_dict, pytype);

    if (weakref == Py_NotImplemented) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    else if (weakref == Py_None) {
        /*
         * The weak reference (and thus the mapping) was invalidated, this
         * should not typically happen, but if it does delete it from the
         * mapping.
         */
        int res = PyDict_DelItem(_global_pytype_to_type_dict, pytype);
        weakref = NULL;
        if (res < 0) {
            return NULL;
        }
    }
    else if (weakref != NULL) {
        assert(PyWeakref_CheckRef(weakref));
        PyObject *DType = PyWeakref_GET_OBJECT(weakref);
        Py_INCREF(DType);
        assert(Py_TYPE(DType) == &PyArrayDTypeMeta_Type);
        return (PyArray_DTypeMeta *)DType;
    }
    /*
     * At this point we have not found a clear mapping, but mainly for
     * backward compatibility we have to make some further attempts at
     * interpreting the input correctly.
     */
    PyArray_Descr *legacy_descr;
    if (PyArray_IsScalar(obj, Generic)) {
        legacy_descr = PyArray_DescrFromScalar(obj);
        if (legacy_descr == NULL) {
            return NULL;
        }
    }
    else if (PyBytes_Check(obj)) {
        legacy_descr = PyArray_DescrFromType(NPY_BYTE);
    }
    else if (PyUnicode_Check(obj)) {
        legacy_descr = PyArray_DescrFromType(NPY_UNICODE);
    }
    else {
        legacy_descr = _array_find_python_scalar_type(obj);
    }

    if (legacy_descr != NULL) {
        PyArray_DTypeMeta *DType = (PyArray_DTypeMeta *)Py_TYPE(legacy_descr);
        Py_INCREF(DType);
        Py_DECREF(legacy_descr);
        if (!(*flags & GAVE_SUBCLASS_WARNING)) {
            if (DEPRECATE(
                    "In the future NumPy will not automatically find the "
                    "dtype for  subclasses of builtin python types and numpy "
                    "scalars. Use the appropriate `dtype=...` to create "
                    "this array.") < 0) {
                return NULL;
            }
            *flags &= GAVE_SUBCLASS_WARNING;
        }
        return DType;
    }

    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


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


static int
handle_promotion(
        PyArray_Descr **out_descr, PyArray_Descr *descr,
        PyArray_DTypeMeta *fixed_DType)
{
    /*
     * TODO: Probably needs fast-path for when the dtype is identical to the
     *       previous dtype, which should be super common.
     */

    if (fixed_DType != NULL && Py_TYPE(descr) != (PyTypeObject *)fixed_DType) {
        /*
         * Before doing the actual promotion we have to find the correct
         * datatype.
         */
        PyErr_SetString(PyExc_SystemError,
                "internal NumPy error, hit a code path which is not yet "
                "implemented, but that should be unreachable at this time.");
        return -1;
    }

    if (*out_descr == NULL) {
        *out_descr = descr;
        return 0;
    }
    /* TODO: If the previous descr is identical to descr, we could skip this */
    PyArray_Descr *new_descr = PyArray_PromoteTypes(*out_descr, descr);
    // TODO: Have to take care of the retry-with-string logic for now :(
    Py_DECREF(descr);
    if (new_descr == NULL) {
        return -1;
    }
    Py_SETREF(*out_descr, new_descr);
    return 0;

}


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
        enum _dtype_discovery_flags *flags)
{
    PyArrayObject *arr = NULL;

    /*
     * The first step is to find the DType class if it was not provided
     */
    PyArray_DTypeMeta *DType = NULL;
    PyArray_Descr *descr = NULL;

    if (fixed_DType != NULL) {
        /*
         * Let the given DType handle the conversion, there are three possible
         * result cases here:
         *   1. A descr, which is ready for promotion. (Correct DType)
         *   2. None to indicate that this should be treated as a sequence.
         *   3. NotImplemented to see if this is a known scalar type and
         *      use normal casting logic instead. This can be slow for
         *      parametric types.
         *   4. NULL in case of an error.
         */
        descr = fixed_DType->discover_descr_from_pyobject(fixed_DType, obj);
        if (descr == NULL) {
            return -1;
        }
        else if (descr == (PyArray_Descr *)Py_None) {
            /* Set DType to None to indicate array or sequence */
            Py_DECREF(Py_None);
            goto array_or_sequence;
        }
        else if (descr == (PyArray_Descr *)Py_NotImplemented) {
            Py_DECREF(Py_NotImplemented);
            descr = NULL;
        }
    }
    /*
     * If either a fixed_DType was given but that DType did not know how to
     * interpret the value, or no fixed_DType was given, we have to try
     * and interpret as a scalar.
     */
    if (descr == NULL) {
        DType = discover_dtype_from_pytype(obj, flags);
        if (DType == NULL) {
            return -1;
        }
        else if (DType != (PyArray_DTypeMeta *)Py_None) {
            descr = DType->discover_descr_from_pyobject(DType, obj);
            Py_DECREF(DType);
            DType = NULL;

            if (descr == NULL) {
                return -1;
            }
            /* The following checks represent programming errors */
            if (descr == (PyArray_Descr *)Py_NotImplemented ||
                    descr == (PyArray_Descr *)Py_None) {
                PyErr_Format(PyExc_RuntimeError,
                        "internal error while finding dtype for scalar. "
                        "`%S` failed to return dtype for a scalar of its own "
                        "type. This is an error in its implementation.", DType);
                Py_DECREF(DType);
                return -1;
            }
        }
        else {
            DType = NULL;  /* (None transfers ownership) */
            descr = (PyArray_Descr *)Py_None;
        }
    }

    /*
     * The second step is to ask the DType class to handle the scalar cases
     * or return NotImplemented to signal that this should be assumed to be
     * an array-like or sequence.
     * We do this even when the dtype was provided, to handle the dimension
     * discovery (possibly a fastpath can be added for that at some point).
     */
    if (descr != (PyArray_Descr *)Py_None) {
        assert(Py_TYPE(DType) == &PyArrayDTypeMeta_Type);


        Py_DECREF(DType);
        if (descr == NULL) {
            return -1;
        }
        else if (descr == (PyArray_Descr *)Py_NotImplemented) {
            Py_DECREF(descr);
        }
        else {
            /* This is a scalar */
            if (update_shape(curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE) < 0) {
                goto ragged_array;
            }
            if (handle_promotion(out_descr, descr, fixed_DType) < 0) {
                Py_DECREF(descr);
                return -1;
            }
            Py_DECREF(descr);
            return max_dims;
        }
    }
    else {
        /* If no DType was found, this must be an array or a sequence */
        Py_DECREF(Py_None);
    }

array_or_sequence:
    /*
     * The third step is to first check for any arrays or array-likes.
     */
    if (PyArray_Check(obj)) {
        arr = (PyArrayObject *)obj;
        Py_INCREF(arr);
    }
    else {
        arr = (PyArrayObject *)_array_from_array_like(obj,
                requested_descr, 0, NULL);
        if (arr == NULL) {
            return -1;
        }
        else if (arr == (PyArrayObject *)Py_NotImplemented) {
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    if (arr) {
        /*
         * This is an array object which will be added to the cache.
         * Steals the reference to arr.
         */
        if (npy_new_coercion_cache(obj, (PyObject *)arr, 0, coercion_cache_tail_ptr) < 0) {
            Py_DECREF(arr);
            return -1;
        }
        if (update_shape(curr_dims, &max_dims, out_shape,
                PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FALSE) < 0) {
            Py_DECREF(arr);
            goto ragged_array;
        }
        if (handle_promotion(out_descr, PyArray_DESCR(arr), fixed_DType) < 0) {
            Py_DECREF(arr);
            return -1;
        }
        Py_DECREF(arr);
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
        return -1;
    }
    if (npy_new_coercion_cache(obj, seq, 1, coercion_cache_tail_ptr) < 0) {
        Py_DECREF(seq);
        return -1;
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
        /* If the sequence is empty, there are no more dimensions */
        return curr_dims+1;
    }

    /* Recursive call for each sequence item */
    for (Py_ssize_t i = 0; i < size; i++) {
        max_dims = PyArray_DiscoverDTypeAndShape_Recursive(
                objects[i], curr_dims + 1, max_dims,
                out_descr, out_shape, coercion_cache_tail_ptr, fixed_DType,
                requested_descr, flags);
        if (max_dims < 0) {
            Py_DECREF(seq);
            return -1;
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


static int
descr_is_legacy_parametric_instance(PyArray_Descr *descr)
{
    if (PyDataType_ISUNSIZED(descr)) {
        return 1;
    }
    /* Flexible descr with generic time unit (which can be adapted) */
    if (PyDataType_ISDATETIME(descr)) {
        PyArray_DatetimeMetaData *meta;
        meta = get_datetime_metadata_from_dtype(descr);
        if (meta->base == NPY_FR_GENERIC) {
            return 1;
        }
    }
    return 0;
}

NPY_NO_EXPORT PyArray_Descr *
PyArray_DiscoverDTypeAndShape(
        PyObject *obj, int max_dims,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj ***coercion_cache_tail_ptr,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr)
{
    /* Valid input of requested descriptor and DType */
    if (requested_descr != NULL && fixed_DType == NULL) {
        fixed_DType = (PyArray_DTypeMeta *)Py_TYPE(requested_descr);
        Py_INCREF(fixed_DType);
        assert(Py_TYPE(fixed_DType) == &PyArrayDTypeMeta_Type);
    }
    if (descr_is_legacy_parametric_instance(requested_descr)) {
        /* TODO: This branch should eventually be moved and/or removed */
        Py_DECREF(requested_descr);
        requested_descr = NULL;
    }
    if (fixed_DType != NULL) {
        assert(fixed_DType != (PyArray_DTypeMeta *)Py_TYPE(requested_descr));
    }

    /*
     * Call the recursive function, the setup for this may need expanding
     * to handle caching better.
     */
    enum _dtype_discovery_flags flags = 0;
    PyArray_Descr *found_descr;

    int res = PyArray_DiscoverDTypeAndShape_Recursive(
            obj, 0, max_dims, &found_descr, out_shape, coercion_cache_tail_ptr,
            fixed_DType, requested_descr, &flags);
    if (res < 0) {
        return NULL;
    }

    if (flags & IS_RAGGED_ARRAY) {
        static PyObject *visibleDeprecationWarning = NULL;
        npy_cache_import(
                "numpy", "VisibleDeprecationWarning",
                &visibleDeprecationWarning);
        if (visibleDeprecationWarning == NULL) {
            return NULL;
        }
        /* NumPy 1.19, 2019-11-01 */
        if (PyErr_WarnEx(visibleDeprecationWarning,
                "Creating an ndarray from ragged nested sequences (which"
                "is a list-or-tuple of lists-or-tuples-or ndarrays with "
                "different lengths or shapes) is deprecated. If you "
                "meant to do this, you must specify 'dtype=object' "
                "when creating the ndarray", 1) < 0)
        {
            return NULL;
        }
    }

    return found_descr;
}