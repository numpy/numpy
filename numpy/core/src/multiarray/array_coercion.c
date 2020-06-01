#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _UMATHMODULE
#define _MULTIARRAYMODULE

#include "Python.h"

#include "lowlevel_strided_loops.h"
#include "numpy/arrayobject.h"

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
 *     allow value based casting.  This requires python complex/float to
 *     have their own DTypes.
 *   * It is necessary to decide whether or not a sequence is an element.
 *     For example tuples are considered elements for structured dtypes, but
 *     otherwise are considered sequences.
 *     This means that if a dtype is given (either as a class or instance),
 *     it can effect the dimension discovery part.
 *     For the "special" NumPy types structured void and "c" (single character)
 *     this is special cased.  For future user-types, this is currently
 *     handled by providing calling an `is_known_scalar` method.  This method
 *     currently ensures that Python numerical types are handled quickly.
 *
 * In the initial version of this implementation, it is assumed that dtype
 * discovery can be implemented sufficiently fast.  That is, it is not
 * necessary to create fast paths that only find the correct shape e.g. when
 * ``dtype=np.dtype("f8")`` is given.
 *
 * The code here avoid multiple conversion of array-like objects (including
 * sequences). These objects are cached after conversion, which will require
 * additional memory, but can drastically speed up coercion from from array
 * like objects.
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


/* Enum to track or signal some things during dtype and shape discovery */
enum _dtype_discovery_flags {
    FOUND_RAGGED_ARRAY = 1,
    GAVE_SUBCLASS_WARNING = 2,
    PROMOTION_FAILED = 4,
    DISCOVER_STRINGS_AS_SEQUENCES = 8,
    DISCOVER_TUPLES_AS_ELEMENTS = 16,
    MAX_DIMS_WAS_REACHED = 32,
};


/**
 * Adds known sequence types to the global type dictionary, note that when
 * a DType is passed in, this lookup may be ignored.
 *
 * @return -1 on error 0 on success
 */
static int
_prime_global_pytype_to_type_dict()
{
    int res;

    /* Add the basic Python sequence types */
    res = PyDict_SetItem(_global_pytype_to_type_dict,
                         (PyObject *)&PyList_Type, Py_None);
    if (res < 0) {
        return -1;
    }
    res = PyDict_SetItem(_global_pytype_to_type_dict,
                         (PyObject *)&PyTuple_Type, Py_None);
    if (res < 0) {
        return -1;
    }
    /* NumPy Arrays are not handled as scalars */
    res = PyDict_SetItem(_global_pytype_to_type_dict,
                         (PyObject *)&PyArray_Type, Py_None);
    if (res < 0) {
        return -1;
    }
    return 0;
}


/**
 * Add a new mapping from a python type to the DType class.
 *
 * This assumes that the DType class is guaranteed to hold on the
 * python type (this assumption is guaranteed).
 * This functionality supercedes ``_typenum_fromtypeobj``.
 *
 * @param DType DType to map the python type to
 * @param pytype Python type to map from
 * @param userdef Whether or not it is user defined. We ensure that user
 *        defined scalars subclass from our scalars (for now).
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
    if (NPY_UNLIKELY(_global_pytype_to_type_dict == NULL)) {
        _global_pytype_to_type_dict = PyDict_New();
        if (_global_pytype_to_type_dict == NULL) {
            return -1;
        }
        if (_prime_global_pytype_to_type_dict() < 0) {
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
        return -1;
    }

    PyObject *weakref = PyWeakref_NewRef(Dtype_obj, NULL);
    if (weakref == NULL) {
        return -1;
    }
    return PyDict_SetItem(_global_pytype_to_type_dict,
            (PyObject *)pytype, weakref);
}


/**
 * Lookup the DType for a registered known python scalar type.
 *
 * @param pytype Python Type to look up
 * @return DType, None if it a known non-scalar, or NULL if an unknown object.
 */
static NPY_INLINE PyArray_DTypeMeta *
discover_dtype_from_pytype(PyTypeObject *pytype)
{
    PyObject *weakref;

    if (pytype == &PyArray_Type) {
        Py_INCREF(Py_None);
        return (PyArray_DTypeMeta *)Py_None;
    }

    weakref = PyDict_GetItem(
            _global_pytype_to_type_dict, (PyObject *)pytype);

    if (weakref == NULL) {
        /* This should not be possible, since types should be hashable */
        assert(!PyErr_Occurred());
        return NULL;
    }
    if (weakref == Py_None) {
        Py_INCREF(Py_None);
        return (PyArray_DTypeMeta *)Py_None;
    }
    assert(PyWeakref_CheckRef(weakref));
    PyObject *DType = PyWeakref_GET_OBJECT(weakref);
    Py_INCREF(DType);
    if (DType == Py_None) {
        /*
         * The weak reference (and thus the mapping) was invalidated, this
         * should not typically happen, but if it does delete it from the
         * mapping.
         */
        int res = PyDict_DelItem(
                _global_pytype_to_type_dict, (PyObject *)pytype);
        if (res < 0) {
            return NULL;
        }
    }
    else {
        assert(PyObject_TypeCheck(DType, (PyTypeObject *)&PyArrayDTypeMeta_Type));
    }
    return (PyArray_DTypeMeta *)DType;
}


/**
 * Find the correct DType class for the given python type. If flags is NULL
 * this is not used to discover a dtype, but only for conversion to an
 * existing dtype. In that case the Python (not NumPy) scalar subclass
 * checks are skipped.
 *
 * @param obj The python object, mainly type(pyobj) is used, the object
 *        is passed to reuse existing code at this time only.
 * @param flags Flags used to know if warnings were already given. If
 *        flags is NULL, this is not
 * @param fixed_DType if not NULL, will be checked first for whether or not
 *        it can/wants to handle the (possible) scalar value.
 * @return New reference to either a DType class, Py_None, or NULL on error.
 */
static NPY_INLINE PyArray_DTypeMeta *
discover_dtype_from_pyobject(
        PyObject *obj, enum _dtype_discovery_flags *flags,
        PyArray_DTypeMeta *fixed_DType)
{
    if (fixed_DType != NULL) {
        /*
         * Let the given DType handle the discovery, there are three possible
         * result cases here:
         *   1. A descr, which is ready for promotion. (Correct DType)
         *   2. None to indicate that this should be treated as a sequence.
         *   3. NotImplemented to see if this is a known scalar type and
         *      use normal casting logic instead. This can be slow especially
         *      for parametric types.
         *   4. NULL in case of an error.
         */
        if ((Py_TYPE(obj) == fixed_DType->scalar_type) ||
                (fixed_DType->is_known_scalar_type != NULL &&
                 fixed_DType->is_known_scalar_type(fixed_DType, Py_TYPE(obj)))) {
            Py_INCREF(fixed_DType);
            return fixed_DType;
        }
    }

    PyArray_DTypeMeta *DType = discover_dtype_from_pytype(Py_TYPE(obj));
    if (DType != NULL) {
        return DType;
    }
    /*
     * At this point we have not found a clear mapping, but mainly for
     * backward compatibility we have to make some further attempts at
     * interpreting the input as a known scalar type.
     */
    PyArray_Descr *legacy_descr;
    if (PyArray_IsScalar(obj, Generic)) {
        legacy_descr = PyArray_DescrFromScalar(obj);
        if (legacy_descr == NULL) {
            return NULL;
        }
    }
    else if (flags == NULL) {
        Py_INCREF(Py_None);
        return (PyArray_DTypeMeta *)Py_None;
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
        DType = NPY_DTYPE(legacy_descr);
        Py_INCREF(DType);
        Py_DECREF(legacy_descr);
        /* TODO: Enable warning about subclass handling */
        if (0 && !((*flags) & GAVE_SUBCLASS_WARNING)) {
            if (DEPRECATE_FUTUREWARNING(
                    "in the future NumPy will not automatically find the "
                    "dtype for subclasses of scalars known to NumPy (i.e. "
                    "python types). Use the appropriate `dtype=...` to create "
                    "this array. This will use the `object` dtype or raise "
                    "an error in the future.") < 0) {
                return NULL;
            }
            *flags |= GAVE_SUBCLASS_WARNING;
        }
        return DType;
    }
    Py_INCREF(Py_None);
    return (PyArray_DTypeMeta *)Py_None;
}


static NPY_INLINE PyArray_Descr *
cast_descriptor_to_fixed_dtype(
        PyArray_Descr *descr, PyArray_DTypeMeta *fixed_DType)
{
    if (fixed_DType == NULL) {
        /* Nothing to do, we only need to promote the new dtype */
        Py_INCREF(descr);
        return descr;
    }

    if (!fixed_DType->parametric) {
        /*
         * Don't actually do anything, the default is always the result
         * of any cast.
         */
        return fixed_DType->default_descr(fixed_DType);
    }
    if (PyObject_TypeCheck((PyObject *)descr, (PyTypeObject *)fixed_DType)) {
        Py_INCREF(descr);
        return descr;
    }
    /*
     * TODO: When this is implemented for all dtypes, the special cases
     *       can be removed...
     */
    if (fixed_DType->legacy && fixed_DType->parametric &&
            NPY_DTYPE(descr)->legacy) {
        PyArray_Descr *flex_dtype = PyArray_DescrFromType(fixed_DType->type_num);
        return PyArray_AdaptFlexibleDType(descr, flex_dtype);
    }

    PyErr_SetString(PyExc_NotImplementedError,
            "Must use casting to find the correct dtype, this is "
            "not yet implemented! "
            "(It should not be possible to hit this code currently!)");
    return NULL;
}


/**
 * Discover the correct descriptor from a known DType class and scalar.
 * If the fixed DType can discover a dtype instance/descr all is fine,
 * if it cannot and DType is used instead, a cast will have to be tried.
 *
 * @param fixed_DType A user provided fixed DType, can be NULL
 * @param DType A discovered DType (by discover_dtype_from_pyobject);
 *        this can be identical to `fixed_DType`, if it obj is a
 *        known scalar. Can be `NULL` indicating no known type.
 * @param obj The Python scalar object. At the time of calling this function
 *        it must be known that `obj` should represent a scalar.
 * @param requested_descr The requested descriptor or NULL, if not NULL
 *        it is returned unmodified.
 */
static NPY_INLINE PyArray_Descr *
find_scalar_descriptor(
        PyArray_DTypeMeta *fixed_DType, PyArray_DTypeMeta *DType,
        PyObject *obj, PyArray_Descr *requested_descr)
{
    PyArray_Descr *descr;

    if (requested_descr != NULL) {
        /* We simply assume that this is correct and continue. */
        Py_INCREF(requested_descr);
        return requested_descr;
    }

    if (DType == NULL && fixed_DType == NULL) {
        /* No known DType and no fixed one means we go to object. */
        return PyArray_DescrFromType(NPY_OBJECT);
    }
    else if (DType == NULL) {
        /*
         * If no DType is known/found, give the fixed give one a second
         * chance.  This allows for example string, to call `str(obj)` to
         * figure out the length for arbitrary objects.
         */
        descr = fixed_DType->discover_descr_from_pyobject(fixed_DType, obj);
    }
    else {
        descr = DType->discover_descr_from_pyobject(DType, obj);
    }
    if (descr == NULL) {
        return NULL;
    }
    if (fixed_DType == NULL) {
        return descr;
    }

    Py_SETREF(descr, cast_descriptor_to_fixed_dtype(descr, fixed_DType));
    return descr;
}


/**
 * Assign a single element in an array from a python value.
 *
 * The dtypes SETITEM should only be trusted to generally do the right
 * thing if something is known to be a scalar *and* is of a python type known
 * to the DType (which should include all basic Python math types), but in
 * general a cast may be necessary.
 * This function handles the cast, which is for example hit when assigning
 * a float128 to complex128.
 *
 * At this time, this function does not support arrays (historically we
 * mainly supported arrays through `__float__()`, etc. Such support should
 * possibly be added (although in some cases we know that the input is not
 * an array).
 *
 * @param descr
 * @param item
 * @param value
 * @return 0 on success -1 on failure.
 */
/*
 * TODO: This function should possibly be public API.
 */
NPY_NO_EXPORT int
PyArray_Pack(PyArray_Descr *descr, char *item, PyObject *value)
{
    static PyArrayObject_fields arr_fields = {
            .ob_base.ob_refcnt = 1,
            .ob_base.ob_type = &PyArrayDescr_Type,
            .flags = NPY_ARRAY_BEHAVED,
        };

    if (NPY_UNLIKELY(descr->type_num == NPY_OBJECT)) {
        /*
         * We always have store objects directly, casting will lose some
         * type information. Any other dtype discards the type information.
         * TODO: For a Categorical[object] this path may be necessary?
         */
        return descr->f->setitem(value, item, &arr_fields);
    }

    /* discover_dtype_from_pyobject includes a check for is_known_scalar_type */
    PyArray_DTypeMeta *DType = discover_dtype_from_pyobject(
            value, NULL, NPY_DTYPE(descr));
    if (DType == NULL) {
        return -1;
    }
    if (DType == NPY_DTYPE(descr) || DType == (PyArray_DTypeMeta *)Py_None) {
        /* We can set the element directly (or at least will try to) */
        Py_XDECREF(DType);
        arr_fields.descr = descr;
        return descr->f->setitem(value, item, &arr_fields);
    }
    PyArray_Descr *tmp_descr;
    tmp_descr = DType->discover_descr_from_pyobject(DType, value);
    Py_DECREF(DType);
    if (tmp_descr == NULL) {
        return -1;
    }

    char *data = PyObject_Malloc(tmp_descr->elsize);
    if (data == NULL) {
        PyErr_NoMemory();
        Py_DECREF(tmp_descr);
        return -1;
    }
    if (PyDataType_FLAGCHK(tmp_descr, NPY_NEEDS_INIT)) {
        memset(data, 0, tmp_descr->elsize);
    }
    arr_fields.descr = tmp_descr;
    if (tmp_descr->f->setitem(value, data, &arr_fields) < 0) {
        PyObject_Free(data);
        Py_DECREF(tmp_descr);
        return -1;
    }

    int needs_api = 0;
    PyArray_StridedUnaryOp *stransfer;
    NpyAuxData *transferdata;
    if (PyArray_GetDTypeTransferFunction(
            0, 0, 0, tmp_descr, descr, 1, &stransfer, &transferdata,
            &needs_api) == NPY_FAIL) {
        PyObject_Free(data);
        Py_DECREF(tmp_descr);
        return -1;
    }
    stransfer(item, 0, data, 0, 1, tmp_descr->elsize, transferdata);
    NPY_AUXDATA_FREE(transferdata);
    PyObject_Free(data);
    Py_DECREF(tmp_descr);
    if (PyErr_Occurred()) {
        return -1;
    }
    return 0;
}


static int
update_shape(int curr_ndim, int *max_ndim,
             npy_intp out_shape[NPY_MAXDIMS], int new_ndim,
             const npy_intp new_shape[NPY_MAXDIMS], npy_bool sequence,
             enum _dtype_discovery_flags *flags)
{
    int success = 0;  /* unsuccessful if array is ragged */
    const npy_bool max_dims_reached = *flags & MAX_DIMS_WAS_REACHED;

    if (curr_ndim + new_ndim > *max_ndim) {
        success = -1;
        /* Only update/check as many dims as possible, max_ndim is unchanged */
        new_ndim = *max_ndim - curr_ndim;
    }
    else if (!sequence && (*max_ndim != curr_ndim + new_ndim)) {
        /*
         * Sequences do not update max_ndim, otherwise shrink and check.
         * This is depth first, so if it is already set, `out_shape` is filled.
         */
        *max_ndim = curr_ndim + new_ndim;
        /* If a shape was already set, this is also ragged */
        if (max_dims_reached) {
            success = -1;
        }
    }
    for (int i = 0; i < new_ndim; i++) {
        npy_intp curr_dim = out_shape[curr_ndim + i];
        npy_intp new_dim = new_shape[i];

        if (!max_dims_reached) {
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
    if (!sequence) {
        *flags |= MAX_DIMS_WAS_REACHED;
    }
    return success;
}


#define COERCION_CACHE_CACHE_SIZE 10
static int _coercion_cache_num = 0;
static coercion_cache_obj *_coercion_cache_cache[COERCION_CACHE_CACHE_SIZE];

/*
 * Steals a reference to the object.
 */
static NPY_INLINE int
npy_new_coercion_cache(
        PyObject *converted_obj, PyObject *arr_or_sequence, npy_bool sequence,
        coercion_cache_obj ***next_ptr, int ndim)
{
    coercion_cache_obj *cache;
    if (_coercion_cache_num > 0) {
        _coercion_cache_num--;
        cache = _coercion_cache_cache[_coercion_cache_num];
    }
    else {
        cache = PyObject_MALLOC(sizeof(coercion_cache_obj));
    }
    if (cache == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    cache->converted_obj = converted_obj;
    cache->arr_or_sequence = arr_or_sequence;
    cache->sequence = sequence;
    cache->depth = ndim;
    cache->next = NULL;
    **next_ptr = cache;
    *next_ptr = &(cache->next);
    return 0;
}

/**
 * Unlink coercion cache item.
 *
 * @param current
 * @return next coercion cache object (or NULL)
 */
NPY_NO_EXPORT NPY_INLINE coercion_cache_obj *
npy_unlink_coercion_cache(coercion_cache_obj *current)
{
    coercion_cache_obj *next = current->next;
    Py_DECREF(current->arr_or_sequence);
    if (_coercion_cache_num < COERCION_CACHE_CACHE_SIZE) {
        _coercion_cache_cache[_coercion_cache_num] = current;
        _coercion_cache_num++;
    }
    else {
        PyObject_FREE(current);
    }
    return next;
}

NPY_NO_EXPORT NPY_INLINE void
npy_free_coercion_cache(coercion_cache_obj *next) {
    /* We only need to check from the last used cache pos */
    while (next != NULL) {
        next = npy_unlink_coercion_cache(next);
    }
}

#undef COERCION_CACHE_CACHE_SIZE

/**
 * Do the promotion step and possible casting. This function should
 * never be called if a descriptor was requested. In that case the output
 * dtype is not of importance, so we must not risk promotion errors.
 *
 * @param out_descr The current descriptor.
 * @param descr The newly found descriptor to promote with
 * @param flags dtype discover flags to signal failed promotion.
 * @return -1 on error, 0 on success.
 */
static int
handle_promotion(PyArray_Descr **out_descr, PyArray_Descr *descr,
        PyArray_Descr *requested_descr, enum _dtype_discovery_flags *flags)
{
    if (requested_descr != NULL) {
        /*
         * If the user fixed a descriptor, do not promote, this will just
         * error during assignment if necessary.
         */
        return 0;
    }
    if (*out_descr == NULL) {
        Py_INCREF(descr);
        *out_descr = descr;
        return 0;
    }
    // TODO: Will have to take care of the retry-with-string logic? :(
    PyArray_Descr *new_descr = PyArray_PromoteTypes(*out_descr, descr);
    if (new_descr == NULL) {
        PyErr_Clear();
        *flags |= PROMOTION_FAILED;
        /* Continue with object, since we may need the dimensionality */
        new_descr = PyArray_DescrFromType(NPY_OBJECT);
    }
    Py_SETREF(*out_descr, new_descr);
    return 0;
}


/**
 * Handle a leave node (known scalar) during dtype and shape discovery.
 *
 * @param obj The python object or nested sequence to convert
 * @param max_dims The maximum number of dimensions.
 * @param curr_dims The current number of dimensions (depth in the recursion)
 * @param out_shape The discovered output shape, will be filled
 * @param coercion_cache The coercion cache object to use.
 * @param DType the DType class that should be used, or NULL, if not provided.
 * @param requested_descr The dtype instance passed in by the user, this is
 *        passed to array-likes, and otherwise prevents any form of promotion
 *        (to avoid errors).
 * @param flags used signal that this is a ragged array, used internally and
 *        can be expanded if necessary.
 */
static NPY_INLINE int
handle_scalar(
        PyObject *obj, int curr_dims, int *max_dims,
        PyArray_Descr **out_descr, npy_intp *out_shape,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        enum _dtype_discovery_flags *flags, PyArray_DTypeMeta *DType)
{
    /* This is a scalar, so find the descriptor */
    PyArray_Descr *descr;
    descr = find_scalar_descriptor(fixed_DType, DType, obj, requested_descr);
    if (descr == NULL) {
        return -1;
    }
    if (update_shape(curr_dims, max_dims, out_shape,
            0, NULL, NPY_FALSE, flags) < 0) {
        *flags |= FOUND_RAGGED_ARRAY;
        Py_XSETREF(*out_descr, PyArray_DescrFromType(NPY_OBJECT));
        return *max_dims;
    }
    if (handle_promotion(out_descr, descr, requested_descr, flags) < 0) {
        Py_DECREF(descr);
        return -1;
    }
    Py_DECREF(descr);
    return *max_dims;
}




/**
 * Return the correct descriptor given an array object and a DType class.
 *
 * This is identical to casting the arrays descriptor/dtype to the new
 * DType class
 *
 * @param arr The array object.
 * @param DType The DType class to cast to (or NULL for convenience)
 * @param out_descr The output descriptor will set. The result can be NULL
 *        when the array is of object dtype and has no elements.
 *
 * @return -1 on failure, 0 on success.
 */
static int
find_descriptor_from_array(
        PyArrayObject *arr, PyArray_DTypeMeta *DType, PyArray_Descr **out_descr)
{
    enum _dtype_discovery_flags flags = 0;
    *out_descr = NULL;

    if (NPY_UNLIKELY(DType != NULL && DType->parametric &&
            PyArray_ISOBJECT(arr))) {
        /*
         * We have one special case, if (and only if) the input array is of
         * object DType and the dtype is not fixed already but parametric.
         * Then, we allow inspection of all elements, treating them as
         * elements. We do this recursively, so nested 0-D arrays can work,
         * but nested higher dimensional arrays will lead to an error.
         */
        assert(DType->type_num != NPY_OBJECT);  /* not parametric */

        PyArrayIterObject *iter;
        iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
        if (iter == NULL) {
            return -1;
        }
        int array_is_object = PyArray_ISOBJECT(arr);
        while (iter->index < iter->size) {
            PyArray_DTypeMeta *item_DType;
            /*
             * Note: If the array contains typed objects we may need to use
             *       the dtype to use casting for finding the correct instance.
             */
            PyObject *elem = PyArray_GETITEM(arr, iter->dataptr);
            if (elem == NULL) {
                elem = Py_None;
            }
            item_DType = discover_dtype_from_pyobject(elem, &flags, DType);
            if (item_DType == NULL) {
                return -1;
            }
            if (item_DType == (PyArray_DTypeMeta *)Py_None) {
                Py_SETREF(item_DType, NULL);
            }
            int flat_max_dims = 0;
            if (handle_scalar(elem, 0, &flat_max_dims, out_descr,
                    NULL, DType, NULL, &flags, item_DType) < 0) {
                Py_DECREF(iter);
                Py_XDECREF(item_DType);
                return -1;
            }
            Py_XDECREF(item_DType);
            PyArray_ITER_NEXT(iter);
        }
        Py_DECREF(iter);
    }
    else if (DType != NULL && NPY_UNLIKELY(DType->type_num == NPY_DATETIME) &&
                PyArray_ISSTRING(arr)) {
        /*
         * TODO: This branch should be deprecated IMO, the workaround is
         *       to cast to the object to a string array. Although a specific
         *       function (if there is even any need) would be better.
         *       This is value based casting!
         * Unless of course we actually want to support this kind of thing
         * in general (not just for object dtype)...
         */
        PyArray_DatetimeMetaData meta;
        meta.base = NPY_FR_GENERIC;
        meta.num = 1;

        if (find_string_array_datetime64_type(arr, &meta) < 0) {
            return -1;
        }
        else {
            *out_descr = create_datetime_dtype(NPY_DATETIME, &meta);
            if (*out_descr == NULL) {
                return -1;
            }
        }
    }
    else {
        /*
         * If this is not an object array figure out the dtype cast,
         * or simply use the returned DType.
         */
        *out_descr = cast_descriptor_to_fixed_dtype(
                     PyArray_DESCR(arr), DType);
        if (*out_descr == NULL) {
            return -1;
        }
    }
    return 0;
}

/**
 * Given a dtype or DType object, find the correct descriptor to cast the
 * array to.
 *
 * This function is identical to normal casting using only the dtype, however,
 * it supports inspecting the elements when the array has object dtype
 * (and the given datatype describes a parametric DType class).
 *
 * @param arr
 * @param dtype A dtype instance or class.
 * @return A concrete dtype instance or NULL
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_AdaptDescriptorToArray(PyArrayObject *arr, PyObject *dtype)
{
    /* If the requested dtype is flexible, adapt it */
    PyArray_Descr *new_dtype;
    PyArray_DTypeMeta *new_DType;
    int res;

    res= PyArray_ExtractDTypeAndDescriptor((PyObject *)dtype,
            &new_dtype, &new_DType);
    if (res < 0) {
        return NULL;
    }
    if (new_dtype == NULL) {
        res = find_descriptor_from_array(arr, new_DType, &new_dtype);
        if (res < 0) {
            Py_DECREF(new_DType);
            return NULL;
        }
        if (new_dtype == NULL) {
            /* This is an object array but contained no elements, use default */
            new_dtype = new_DType->default_descr(new_DType);
        }
    }
    return new_dtype;
}


NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape_Recursive(
        PyObject *obj, int curr_dims, int max_dims, PyArray_Descr**out_descr,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj ***coercion_cache_tail_ptr,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        enum _dtype_discovery_flags *flags)
{
    PyArrayObject *arr = NULL;
    PyObject *seq;

    /*
     * The first step is to find the DType class if it was not provided,
     * alternatively we have to find out that this is not a scalar at all
     * (which could fail and lead us to `object` dtype).
     */
    PyArray_DTypeMeta *DType = NULL;

    if (NPY_UNLIKELY(*flags & DISCOVER_STRINGS_AS_SEQUENCES)) {
        /*
         * We currently support that bytes/strings are considered sequences,
         * if the dtype is np.dtype('c'), this should be deprecated probably,
         * but requires hacks right now.
         */
        if (PyBytes_Check(obj) && PyBytes_Size(obj) != 1) {
            goto force_sequence_due_to_char_dtype;
        }
        else if (PyUnicode_Check(obj) && PyUnicode_GetLength(obj) != 1) {
            goto force_sequence_due_to_char_dtype;
        }
    }

    /* If this is a known scalar, find the corresponding DType class */
    DType = discover_dtype_from_pyobject(obj, flags, fixed_DType);
    if (DType == NULL) {
        return -1;
    }
    else if (DType == (PyArray_DTypeMeta *)Py_None) {
        Py_DECREF(Py_None);
    }
    else {
        max_dims = handle_scalar(
                obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                requested_descr, flags, DType);
        Py_DECREF(DType);
        return max_dims;
    }

    /*
     * At this point we expect to find either a sequence, or an array-like.
     * Although it is still possible that this fails and we have to use
     * `object`.
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
         * This is an array object which will be added to the cache, keeps
         * the reference to the array alive (takes ownership).
         */
        if (npy_new_coercion_cache(obj, (PyObject *)arr,
                0, coercion_cache_tail_ptr, curr_dims) < 0) {
            return -1;
        }

        if (update_shape(curr_dims, &max_dims, out_shape,
                PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FALSE, flags) < 0) {
            *flags |= FOUND_RAGGED_ARRAY;
            Py_XSETREF(*out_descr, PyArray_DescrFromType(NPY_OBJECT));
            return max_dims;
        }

        if (requested_descr != NULL) {
            return max_dims;
        }
        /*
         * For arrays we may not just need to cast the dtype to the user
         * provided fixed_DType. If this is an object array, the elements
         * may need to be inspected individually.
         * Note, this finds the descriptor of the array first and only then
         * promotes here (different associativity).
         */
        PyArray_Descr *cast_descr;
        if (find_descriptor_from_array(arr, fixed_DType, &cast_descr) < 0) {
            return -1;
        }
        if (cast_descr == NULL) {
            /* object array with no elements, no need to promote/adjust. */
            return max_dims;
        }
        if (handle_promotion(out_descr, cast_descr, requested_descr, flags) < 0) {
            Py_DECREF(cast_descr);
            return -1;
        }
        Py_DECREF(cast_descr);
        return max_dims;
    }

    /*
     * The last step is to assume the input should be handled as a sequence
     * and to handle it recursively. That is, unless we have hit the
     * dimension limit.
     */
    npy_bool is_sequence = (PySequence_Check(obj) && PySequence_Size(obj) >= 0);
    if (NPY_UNLIKELY(*flags & DISCOVER_TUPLES_AS_ELEMENTS) &&
            PyTuple_Check(obj)) {
        is_sequence = NPY_FALSE;
    }
    if (curr_dims == max_dims || !is_sequence) {
        /* Clear any PySequence_Size error which would corrupts further calls */
        PyErr_Clear();
        max_dims = handle_scalar(
                obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                requested_descr, flags, NULL);
        if (is_sequence) {
            /* Flag as ragged or too deep array */
            *flags |= FOUND_RAGGED_ARRAY;
        }
        return max_dims;
    }
    /* If we stop supporting bytes/str subclasses, more may be required here: */
    assert(!PyBytes_Check(obj) && !PyUnicode_Check(obj));

  force_sequence_due_to_char_dtype:

    /* Ensure we have a sequence (required for PyPy) */
    seq = PySequence_Fast(obj, "Could not convert object to sequence");
    if (seq == NULL) {
        /*
         * Specifically do not fail on things that look like a dictionary,
         * instead treat them as scalar.
         */
        if (PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
            max_dims = handle_scalar(
                    obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                    requested_descr, flags, NULL);
            return max_dims;
        }
        return -1;
    }
    /* The cache takes ownership of the sequence here. */
    if (npy_new_coercion_cache(obj, seq, 1, coercion_cache_tail_ptr, curr_dims) < 0) {
        return -1;
    }

    npy_intp size = PySequence_Fast_GET_SIZE(seq);
    PyObject **objects = PySequence_Fast_ITEMS(seq);

    if (update_shape(curr_dims, &max_dims,
                     out_shape, 1, &size, NPY_TRUE, flags) < 0) {
        /* But do update, if there this is a ragged case */
        *flags |= FOUND_RAGGED_ARRAY;
        return max_dims;
    }
    if (size == 0) {
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
            return -1;
        }
    }
    return max_dims;
}


/**
 * Check the descriptor is a legacy "flexible" DType instance, this is
 * an instance which is (normally) not attached to an array, such as a string
 * of length 0 or a datetime with no unit.
 * These should be largely deprecated, and represent only the DType class
 * for most `dtype` parameters.
 *
 * TODO: This function should eventually recieve a deprecation warning and
 *       be removed.
 *
 * @param descr
 * @return 1 if this is not a concrete dtype instance 0 otherwise
 */
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


/**
 * Finds the DType and shape of an arbitrary nested sequence. This is the
 * general purpose function to find the parameters of the array (but not
 * the array itself) as returned by `np.array()`
 *
 * @param obj Scalar or nested sequences.
 * @param max_dims Maximum number of dimensions (after this scalars are forced)
 * @param out_shape Will be filled with the output shape (more than the actual
 *        shape may be written).
 * @param coercion_cache NULL initialized reference to a cache pointer.
 *        May be set to the first coercion_cache, and has to be freed using
 *        npy_free_coercion_cache.
 * @param fixed_DType A user provided fixed DType class.
 * @param requested_descr A user provided fixed descriptor. This is always
 *        returned as the discovered descriptor, but currently only used
 *        for the ``__array__`` protocol.
 * @param out_descr The discovered output descriptor.
 * @return dimensions of the discovered object or -1 on error.
 */
NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape(
        PyObject *obj, int max_dims,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj **coercion_cache,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        PyArray_Descr **out_descr)
{
    *out_descr = NULL;
    coercion_cache_obj **coercion_cache_head = coercion_cache;
    *coercion_cache = NULL;

    /* Validate input of requested descriptor and DType */
    if (fixed_DType != NULL) {
        assert(PyObject_TypeCheck(
                (PyObject *)fixed_DType, (PyTypeObject *)&PyArrayDTypeMeta_Type));
    }
    if (requested_descr != NULL) {
        assert(!descr_is_legacy_parametric_instance(requested_descr));
        assert(fixed_DType == NPY_DTYPE(requested_descr));
    }

    /*
     * Call the recursive function, the setup for this may need expanding
     * to handle caching better.
     */
    enum _dtype_discovery_flags flags = 0;

    /* Legacy discovery flags */
    if (requested_descr != NULL) {
        if (requested_descr->type_num == NPY_STRING &&
                requested_descr->type == 'c') {
            /* Character dtype variation of string (should be deprecated...) */
            flags |= DISCOVER_STRINGS_AS_SEQUENCES;
        }
        else if (requested_descr->type_num == NPY_VOID &&
                    (requested_descr->names || requested_descr->subarray))  {
            /* Void is a chimera, in that it may or may not be structured... */
            flags |= DISCOVER_TUPLES_AS_ELEMENTS;
        }
    }

    int ndim = PyArray_DiscoverDTypeAndShape_Recursive(
            obj, 0, max_dims, out_descr, out_shape, &coercion_cache,
            fixed_DType, requested_descr, &flags);
    if (ndim < 0) {
        goto fail;
    }

    if (NPY_UNLIKELY(flags & FOUND_RAGGED_ARRAY)) {
        /*
         * If max-dims was reached and the dimensions reduced, this is ragged.
         * Otherwise, we merely reached the maximum dimensions, which is
         * slightly different. This happens for example for `[1, [2, 3]]`
         * where the maximum dimensions is 1, but then a sequence found.
         *
         * In this case we need to inform the user and clean out the cache
         * since it may be too deep.
         */

        /* Handle reaching the maximum depth differently: */
        int too_deep = ndim == max_dims;

        if (fixed_DType == NULL) {
            /* This is discovered as object, but deprecated */
            static PyObject *visibleDeprecationWarning = NULL;
            npy_cache_import(
                    "numpy", "VisibleDeprecationWarning",
                    &visibleDeprecationWarning);
            if (visibleDeprecationWarning == NULL) {
                goto fail;
            }
            if (!too_deep) {
                /* NumPy 1.19, 2019-11-01 */
                if (PyErr_WarnEx(visibleDeprecationWarning,
                        "Creating an ndarray from ragged nested sequences (which "
                        "is a list-or-tuple of lists-or-tuples-or ndarrays with "
                        "different lengths or shapes) is deprecated. If you "
                        "meant to do this, you must specify 'dtype=object' "
                        "when creating the ndarray.", 1) < 0) {
                    goto fail;
                }
            }
            else {
                /* NumPy 1.20, 2020-05-08 */
                /* Note, max_dims should normally always be NPY_MAXDIMS here */
                if (PyErr_WarnFormat(visibleDeprecationWarning, 1,
                        "Creating an ndarray from nested sequences exceeding "
                        "the maximum number of dimensions of %d is deprecated. "
                        "If you mean to do this, you must specify "
                        "'dtype=object' when creating the ndarray.",
                        max_dims) < 0) {
                    goto fail;
                }
            }
            Py_XSETREF(*out_descr, PyArray_DescrNewFromType(NPY_OBJECT));
        }
        else if (fixed_DType->type_num != NPY_OBJECT) {
            /* Only object DType supports ragged cases unify error */
            if (!too_deep) {
                PyObject *shape = PyArray_IntTupleFromIntp(ndim, out_shape);
                PyErr_Format(PyExc_ValueError,
                        "setting an array element with a sequence. The "
                        "requested array has an inhomogeneous shape after "
                        "%d dimensions. The detected shape was "
                        "%R + inhomogeneous part.",
                        ndim, shape);
                Py_DECREF(shape);
                goto fail;
            }
            else {
                PyErr_Format(PyExc_ValueError,
                        "setting an array element with a sequence. The "
                        "requested array would exceed the maximum number of "
                        "dimension of %d.",
                        max_dims);
                goto fail;
            }
        }

        /*
         * If the array is ragged, the cache may be too deep, so clean it.
         * The cache is left at the same depth as the array though.
         */
        coercion_cache_obj **next_ptr = coercion_cache_head;
        coercion_cache_obj *current = *coercion_cache_head;  /* item to check */
        while (current != NULL) {
            if (current->depth > ndim) {
                /* delete "next" cache item and advanced it (unlike later) */
                current = npy_unlink_coercion_cache(current);
                continue;
            }
            /* advance both prev and next, and set prev->next to new item */
            *next_ptr = current;
            next_ptr = &(current->next);
            current = current->next;
        }
        *next_ptr = NULL;
    }
    /* We could check here for max-ndims being reached as well */

    if (requested_descr != NULL) {
        /* The user had given a specific one, we could sanity check, but... */
        Py_INCREF(requested_descr);
        Py_XSETREF(*out_descr, requested_descr);
    }
    else if (NPY_UNLIKELY(*out_descr == NULL)) {
        /*
         * When the object contained no items, we have to use the default.
         * We do this afterwards, to not cause promotion when there is only
         * a single element.
         */
        // TODO: This may be a tiny, unsubstantial behaviour change.
        if (fixed_DType != NULL) {
            if (fixed_DType->default_descr == NULL) {
                Py_INCREF(fixed_DType->singleton);
                *out_descr = fixed_DType->singleton;
            }
            else {
                *out_descr = fixed_DType->default_descr(fixed_DType);
                if (*out_descr == NULL) {
                    goto fail;
                }
            }
        }
        else {
            *out_descr = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
        }
    }
    return ndim;

  fail:
    npy_free_coercion_cache(*coercion_cache_head);
    *coercion_cache = NULL;
    Py_XSETREF(*out_descr, NULL);
    return -1;
}


/**
 * Given either a DType instance or class, (or legacy flexible instance),
 * ands sets output dtype instance and DType class. Both results may be
 * NULL, but if `out_descr` is set `out_DType` will always be the
 * corresponding class.
 *
 * @param dtype
 * @param out_descr
 * @param out_DType
 * @return 0 on success -1 on failure
 */
NPY_NO_EXPORT int
PyArray_ExtractDTypeAndDescriptor(PyObject *dtype,
        PyArray_Descr **out_descr, PyArray_DTypeMeta **out_DType)
{
    *out_DType = NULL;
    *out_descr = NULL;

    if (dtype != NULL) {
        if (PyObject_TypeCheck(dtype, (PyTypeObject *)&PyArrayDTypeMeta_Type)) {
            assert(dtype != (PyObject * )&PyArrayDescr_Type);  /* not np.dtype */
            *out_DType = (PyArray_DTypeMeta *)dtype;
            Py_INCREF(*out_DType);
        }
        else if (PyObject_TypeCheck((PyObject *)Py_TYPE(dtype),
                    (PyTypeObject *)&PyArrayDTypeMeta_Type)) {
            *out_DType = NPY_DTYPE(dtype);
            Py_INCREF(*out_DType);
            if (!descr_is_legacy_parametric_instance((PyArray_Descr *)dtype)) {
                *out_descr = (PyArray_Descr *)dtype;
                Py_INCREF(*out_descr);
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "dtype parameter must be a DType instance or class.");
            return -1;
        }
    }
    return 0;
}


NPY_NO_EXPORT PyObject *
_discover_array_parameters(PyObject *NPY_UNUSED(self),
                           PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"obj", "dtype", NULL};

    PyObject *obj;
    PyObject *dtype = NULL;
    PyArray_Descr *fixed_descriptor = NULL;
    PyArray_DTypeMeta *fixed_DType = NULL;
    npy_intp shape[NPY_MAXDIMS];

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|O:_discover_array_parameters", kwlist,
            &obj, &dtype)) {
        return NULL;
    }

    if (PyArray_ExtractDTypeAndDescriptor(dtype,
            &fixed_descriptor, &fixed_DType) < 0) {
        return NULL;
    }

    coercion_cache_obj *coercion_cache;
    PyArray_Descr *out_dtype = NULL;
    int ndim = PyArray_DiscoverDTypeAndShape(
            obj, NPY_MAXDIMS, shape,
            &coercion_cache,
            fixed_DType, fixed_descriptor, &out_dtype);
    npy_free_coercion_cache(coercion_cache);
    Py_XDECREF(fixed_DType);
    Py_XDECREF(fixed_descriptor);

    if (ndim < 0) {
        return NULL;
    }
    PyObject *shape_tuple = PyArray_IntTupleFromIntp(ndim, shape);
    if (shape_tuple == NULL) {
        return NULL;
    }

    PyObject *res = PyTuple_Pack(2, (PyObject *)out_dtype, shape_tuple);
    Py_DECREF(out_dtype);
    Py_DECREF(shape_tuple);
    return res;
}
