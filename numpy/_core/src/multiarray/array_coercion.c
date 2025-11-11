#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/npy_3kcompat.h"
#include "npy_pycompat.h"

#include "lowlevel_strided_loops.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "descriptor.h"
#include "convert_datatype.h"
#include "dtypemeta.h"
#include "stringdtype/dtype.h"

#include "npy_argparse.h"
#include "abstractdtypes.h"
#include "array_coercion.h"
#include "ctors.h"
#include "common.h"
#include "_datetime.h"
#include "npy_import.h"
#include "refcount.h"

#include "umathmodule.h"

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
 * additional memory, but can drastically speed up coercion from array like
 * objects.
 */


/*
 * For finding a DType quickly from a type, it is easiest to have a
 * a mapping of pytype -> DType.
 * TODO: This mapping means that it is currently impossible to delete a
 *       pair of pytype <-> DType.  To resolve this, it is necessary to
 *       weakly reference the pytype. As long as the pytype is alive, we
 *       want to be able to use `np.array([pytype()])`.
 *       It should be possible to retrofit this without too much trouble
 *       (all type objects support weak references).
 */
PyObject *_global_pytype_to_type_dict = NULL;


/* Enum to track or signal some things during dtype and shape discovery */
enum _dtype_discovery_flags {
    FOUND_RAGGED_ARRAY = 1 << 0,
    GAVE_SUBCLASS_WARNING = 1 << 1,
    PROMOTION_FAILED = 1 << 2,
    DISCOVER_STRINGS_AS_SEQUENCES = 1 << 3,
    DISCOVER_TUPLES_AS_ELEMENTS = 1 << 4,
    MAX_DIMS_WAS_REACHED = 1 << 5,
    DESCRIPTOR_WAS_SET = 1 << 6,
    COPY_WAS_CREATED_BY__ARRAY__ = 1 << 7,
};


/**
 * Adds known sequence types to the global type dictionary, note that when
 * a DType is passed in, this lookup may be ignored.
 *
 * @return -1 on error 0 on success
 */
static int
_prime_global_pytype_to_type_dict(void)
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
 * Add a new mapping from a python type to the DType class. For a user
 * defined legacy dtype, this function does nothing unless the pytype
 * subclass from `np.generic`.
 *
 * This assumes that the DType class is guaranteed to hold on the
 * python type (this assumption is guaranteed).
 * This functionality supersedes ``_typenum_fromtypeobj``.
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

    if (userdef && !PyObject_IsSubclass(
                    (PyObject *)pytype, (PyObject *)&PyGenericArrType_Type)) {
        /*
         * We expect that user dtypes (for now) will subclass some numpy
         * scalar class to allow automatic discovery.
         */
        if (NPY_DT_is_legacy(DType)) {
            /*
             * For legacy user dtypes, discovery relied on subclassing, but
             * arbitrary type objects are supported, so do nothing.
             */
            return 0;
        }
        /*
         * We currently enforce that user DTypes subclass from `np.generic`
         * (this should become a `np.generic` base class and may be lifted
         * entirely).
         */
        PyErr_Format(PyExc_RuntimeError,
                "currently it is only possible to register a DType "
                "for scalars deriving from `np.generic`, got '%S'.",
                (PyObject *)pytype);
        return -1;
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

    int res = PyDict_Contains(_global_pytype_to_type_dict, (PyObject *)pytype);
    if (res < 0) {
        return -1;
    }
    else if (DType == &PyArray_StringDType) {
        // PyArray_StringDType's scalar is str which we allow because it doesn't
        // participate in DType inference, so don't add it to the
        // pytype to type mapping
        return 0;
    }
    else if (res) {
        PyErr_SetString(PyExc_RuntimeError,
                "Can only map one python type to DType.");
        return -1;
    }

    return PyDict_SetItem(_global_pytype_to_type_dict,
            (PyObject *)pytype, Dtype_obj);
}


/**
 * Lookup the DType for a registered known python scalar type.
 *
 * @param pytype Python Type to look up
 * @return DType, None if it is a known non-scalar, or NULL if an unknown object.
 */
static inline PyArray_DTypeMeta *
npy_discover_dtype_from_pytype(PyTypeObject *pytype)
{
    PyObject *DType;

    if (pytype == &PyArray_Type) {
        DType = Py_NewRef(Py_None);
    }
    else if (pytype == &PyFloat_Type) {
        DType = Py_NewRef((PyObject *)&PyArray_PyFloatDType);
    }
    else if (pytype == &PyLong_Type) {
        DType = Py_NewRef((PyObject *)&PyArray_PyLongDType);
    }
    else {
        int res = PyDict_GetItemRef(_global_pytype_to_type_dict,
                                    (PyObject *)pytype, (PyObject **)&DType);

        if (res <= 0) {
            /* the python type is not known or an error was set */
            return NULL;
        }
    }
    assert(DType == Py_None || PyObject_TypeCheck(DType, (PyTypeObject *)&PyArrayDTypeMeta_Type));
    return (PyArray_DTypeMeta *)DType;
}

/*
 * Note: This function never fails, but will return `NULL` for unknown scalars or
 *       known array-likes (e.g. tuple, list, ndarray).
 */
NPY_NO_EXPORT PyObject *
PyArray_DiscoverDTypeFromScalarType(PyTypeObject *pytype)
{
    PyObject *DType = (PyObject *)npy_discover_dtype_from_pytype(pytype);
    if (DType == NULL || DType == Py_None) {
        return NULL;
    }
    return DType;
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
static inline PyArray_DTypeMeta *
discover_dtype_from_pyobject(
        PyObject *obj, enum _dtype_discovery_flags *flags,
        PyArray_DTypeMeta *fixed_DType)
{
    if (fixed_DType != NULL) {
        /*
         * Let the given DType handle the discovery.  This is when the
         * scalar-type matches exactly, or the DType signals that it can
         * handle the scalar-type.  (Even if it cannot handle here it may be
         * asked to attempt to do so later, if no other matching DType exists.)
         */
        if ((Py_TYPE(obj) == fixed_DType->scalar_type) ||
                NPY_DT_CALL_is_known_scalar_type(fixed_DType, Py_TYPE(obj))) {
            Py_INCREF(fixed_DType);
            return fixed_DType;
        }
    }

    PyArray_DTypeMeta *DType = npy_discover_dtype_from_pytype(Py_TYPE(obj));
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
        if ((0) && !((*flags) & GAVE_SUBCLASS_WARNING)) {
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
 */
static inline PyArray_Descr *
find_scalar_descriptor(
        PyArray_DTypeMeta *fixed_DType, PyArray_DTypeMeta *DType,
        PyObject *obj)
{
    PyArray_Descr *descr;

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
        descr = NPY_DT_CALL_discover_descr_from_pyobject(fixed_DType, obj);
    }
    else {
        descr = NPY_DT_CALL_discover_descr_from_pyobject(DType, obj);
    }
    if (descr == NULL) {
        return NULL;
    }
    if (fixed_DType == NULL) {
        return descr;
    }

    Py_SETREF(descr, PyArray_CastDescrToDType(descr, fixed_DType));
    return descr;
}


/*
 * Helper function for casting a raw value from one descriptor to another.
 * This helper uses the normal casting machinery, but e.g. does not care about
 * checking cast safety.
 */
NPY_NO_EXPORT int
npy_cast_raw_scalar_item(
        PyArray_Descr *from_descr, char *from_item,
        PyArray_Descr *to_descr, char *to_item)
{
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetDTypeTransferFunction(
            0, 0, 0, from_descr, to_descr, 0, &cast_info,
            &flags) == NPY_FAIL) {
        return -1;
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier(from_item);
    }

    char *args[2] = {from_item, to_item};
    const npy_intp strides[2] = {0, 0};
    const npy_intp length = 1;
    if (cast_info.func(&cast_info.context,
            args, &length, strides, cast_info.auxdata) < 0) {
        NPY_cast_info_xfree(&cast_info);
        return -1;
    }
    NPY_cast_info_xfree(&cast_info);

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier(to_item);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }

    return 0;
}


/*NUMPY_API
 **
 * Assign a single element in an array from a python value.
 *
 * The dtypes SETITEM should only be trusted to generally do the right
 * thing if something is known to be a scalar *and* is of a python type known
 * to the DType (which should include all basic Python math types), but in
 * general a cast may be necessary.
 * This function handles the cast, which is for example hit when assigning
 * a float128 to complex128.
 *
 * TODO: This function probably needs to be passed an "owner" for the sake of
 *       future HPy (non CPython) support
 *
 * NOTE: We do support 0-D exact NumPy arrays correctly via casting here.
 *       There be dragons, because we must NOT support generic array-likes.
 *       The problem is that some (e.g. astropy's Quantity and our masked
 *       arrays) have divergent behaviour for `__array__` as opposed to
 *       `__float__`.  And they rely on that.
 *       That is arguably bad as it limits the things that work seamlessly
 *       because `__float__`, etc. cannot even begin to cover all of casting.
 *       However, we have no choice.  We simply CANNOT support array-likes
 *       here without finding a solution for this first.
 *       And the only plausible one I see currently, is expanding protocols
 *       in some form, either to indicate that we want a scalar or to indicate
 *       that we want the unsafe version that `__array__` currently gives
 *       for both objects.
 *
 *       If we ever figure out how to expand this to other array-likes, care
 *       may need to be taken. `PyArray_FromAny`/`PyArray_AssignFromCache`
 *       uses this function but know if the input is an array, array-like,
 *       or scalar.  Relaxing things here should be OK, but looks a bit
 *       like possible recursion, so it may make sense to make a "scalars only"
 *       version of this function.
 *
 * @param descr
 * @param item
 * @param value
 * @return 0 on success -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_Pack(PyArray_Descr *descr, void *item, PyObject *value)
{
    if (NPY_UNLIKELY(descr->type_num == NPY_OBJECT)) {
        /*
         * We always have store objects directly, casting will lose some
         * type information. Any other dtype discards the type information.
         * TODO: For a Categorical[object] this path may be necessary?
         */
        return NPY_DT_CALL_setitem(descr, value, item);
    }

    /* discover_dtype_from_pyobject includes a check for is_known_scalar_type */
    PyArray_DTypeMeta *DType = discover_dtype_from_pyobject(
            value, NULL, NPY_DTYPE(descr));
    if (DType == NULL) {
        return -1;
    }
    if (DType == (PyArray_DTypeMeta *)Py_None && PyArray_CheckExact(value)
            && PyArray_NDIM((PyArrayObject *)value) == 0) {
        /*
         * WARNING: Do NOT relax the above `PyArray_CheckExact`, unless you
         *          read the function doc NOTE carefully and understood it.
         *
         * NOTE: The ndim == 0 check should probably be an error, but
         *       unfortunately. `arr.__float__()` works for 1 element arrays
         *       so in some contexts we need to let it handled like a scalar.
         *       (If we manage to deprecate the above, we can do that.)
         */
        Py_DECREF(DType);

        PyArrayObject *arr = (PyArrayObject *)value;
        if (PyArray_DESCR(arr) == descr && !PyDataType_REFCHK(descr)) {
            /* light-weight fast-path for when the descrs obviously matches */
            memcpy(item, PyArray_BYTES(arr), descr->elsize);
            return 0;  /* success (it was an array-like) */
        }
        return npy_cast_raw_scalar_item(
                PyArray_DESCR(arr), PyArray_BYTES(arr), descr, item);

    }
    if (DType == NPY_DTYPE(descr) || DType == (PyArray_DTypeMeta *)Py_None) {
        /* We can set the element directly (or at least will try to) */
        Py_XDECREF(DType);
        return NPY_DT_CALL_setitem(descr, value, item);
    }
    PyArray_Descr *tmp_descr;
    tmp_descr = NPY_DT_CALL_discover_descr_from_pyobject(DType, value);
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
    if (NPY_DT_CALL_setitem(tmp_descr, value, data) < 0) {
        PyObject_Free(data);
        Py_DECREF(tmp_descr);
        return -1;
    }
    int res = npy_cast_raw_scalar_item(tmp_descr, data, descr, item);

    if (PyDataType_REFCHK(tmp_descr)) {
        if (PyArray_ClearBuffer(tmp_descr, data, 0, 1, 1) < 0) {
            res = -1;
        }
    }

    PyObject_Free(data);
    Py_DECREF(tmp_descr);
    return res;
}


static int
update_shape(int curr_ndim, int *max_ndim,
             npy_intp out_shape[], int new_ndim,
             const npy_intp new_shape[], npy_bool sequence,
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
                *max_ndim -= new_ndim - i;
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

#ifndef Py_GIL_DISABLED
#define COERCION_CACHE_CACHE_SIZE 5
static int _coercion_cache_num = 0;
static coercion_cache_obj *_coercion_cache_cache[COERCION_CACHE_CACHE_SIZE];
#else
#define COERCION_CACHE_CACHE_SIZE 0
#endif

/*
 * Steals a reference to the object.
 */
static inline int
npy_new_coercion_cache(
        PyObject *converted_obj, PyObject *arr_or_sequence, npy_bool sequence,
        coercion_cache_obj ***next_ptr, int ndim)
{
    coercion_cache_obj *cache;
#if COERCION_CACHE_CACHE_SIZE > 0
    if (_coercion_cache_num > 0) {
        _coercion_cache_num--;
        cache = _coercion_cache_cache[_coercion_cache_num];
    }
    else
#endif
    {
        cache = PyMem_Malloc(sizeof(coercion_cache_obj));
    }
    if (cache == NULL) {
        Py_DECREF(arr_or_sequence);
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
 * @param current This coercion cache object
 * @return next Next coercion cache object (or NULL)
 */
NPY_NO_EXPORT coercion_cache_obj *
npy_unlink_coercion_cache(coercion_cache_obj *current)
{
    coercion_cache_obj *next = current->next;
    Py_DECREF(current->arr_or_sequence);
#if COERCION_CACHE_CACHE_SIZE > 0
    if (_coercion_cache_num < COERCION_CACHE_CACHE_SIZE) {
        _coercion_cache_cache[_coercion_cache_num] = current;
        _coercion_cache_num++;
    }
    else
#endif
    {
        PyMem_Free(current);
    }
    return next;
}

NPY_NO_EXPORT void
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
 * @param fixed_DType The user provided (fixed) DType or NULL
 * @param flags dtype discover flags to signal failed promotion.
 * @return -1 on error, 0 on success.
 */
static inline int
handle_promotion(PyArray_Descr **out_descr, PyArray_Descr *descr,
        PyArray_DTypeMeta *fixed_DType, enum _dtype_discovery_flags *flags)
{
    assert(!(*flags & DESCRIPTOR_WAS_SET));

    if (*out_descr == NULL) {
        Py_INCREF(descr);
        *out_descr = descr;
        return 0;
    }
    PyArray_Descr *new_descr = PyArray_PromoteTypes(descr, *out_descr);
    if (NPY_UNLIKELY(new_descr == NULL)) {
        if (fixed_DType != NULL || PyErr_ExceptionMatches(PyExc_FutureWarning)) {
            /*
             * If a DType is fixed, promotion must not fail. Do not catch
             * FutureWarning (raised for string+numeric promotions). We could
             * only catch TypeError here or even always raise the error.
             */
            return -1;
        }
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
 * @param curr_dims The current number of dimensions (depth in the recursion)
 * @param max_dims The maximum number of dimensions.
 * @param out_shape The discovered output shape, will be filled
 * @param fixed_DType The user provided (fixed) DType or NULL
 * @param flags used signal that this is a ragged array, used internally and
 *        can be expanded if necessary.
 * @param DType the DType class that should be used, or NULL, if not provided.
 *
 * @return 0 on success -1 on error
 */
static inline int
handle_scalar(
        PyObject *obj, int curr_dims, int *max_dims,
        PyArray_Descr **out_descr, npy_intp *out_shape,
        PyArray_DTypeMeta *fixed_DType,
        enum _dtype_discovery_flags *flags, PyArray_DTypeMeta *DType)
{
    PyArray_Descr *descr;

    if (update_shape(curr_dims, max_dims, out_shape,
            0, NULL, NPY_FALSE, flags) < 0) {
        *flags |= FOUND_RAGGED_ARRAY;
        return *max_dims;
    }
    if (*flags & DESCRIPTOR_WAS_SET) {
        /* no need to do any promotion */
        return *max_dims;
    }
    /* This is a scalar, so find the descriptor */
    descr = find_scalar_descriptor(fixed_DType, DType, obj);
    if (descr == NULL) {
        return -1;
    }
    if (handle_promotion(out_descr, descr, fixed_DType, flags) < 0) {
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

    if (DType == NULL) {
        *out_descr = PyArray_DESCR(arr);
        Py_INCREF(*out_descr);
        return 0;
    }

    if (NPY_UNLIKELY(NPY_DT_is_parametric(DType) && PyArray_ISOBJECT(arr))) {
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
        while (iter->index < iter->size) {
            PyArray_DTypeMeta *item_DType;
            /*
             * Note: If the array contains typed objects we may need to use
             *       the dtype to use casting for finding the correct instance.
             */
            PyObject *elem = PyArray_GETITEM(arr, iter->dataptr);
            if (elem == NULL) {
                Py_DECREF(iter);
                return -1;
            }
            item_DType = discover_dtype_from_pyobject(elem, &flags, DType);
            if (item_DType == NULL) {
                Py_DECREF(iter);
                Py_DECREF(elem);
                return -1;
            }
            if (item_DType == (PyArray_DTypeMeta *)Py_None) {
                Py_SETREF(item_DType, NULL);
            }
            int flat_max_dims = 0;
            if (handle_scalar(elem, 0, &flat_max_dims, out_descr,
                    NULL, DType, &flags, item_DType) < 0) {
                Py_DECREF(iter);
                Py_DECREF(elem);
                Py_XDECREF(*out_descr);
                Py_XDECREF(item_DType);
                return -1;
            }
            Py_XDECREF(item_DType);
            Py_DECREF(elem);
            PyArray_ITER_NEXT(iter);
        }
        Py_DECREF(iter);
    }
    else if (NPY_UNLIKELY(DType->type_num == NPY_DATETIME) &&
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
        *out_descr = PyArray_CastDescrToDType(PyArray_DESCR(arr), DType);
        if (*out_descr == NULL) {
            return -1;
        }
    }
    return 0;
}

/**
 * Given a dtype or DType object, find the correct descriptor to cast the
 * array to.  In some places, this function is used with dtype=NULL which
 * means that legacy behavior is used: The dtype instances "S0", "U0", and
 * "V0" are converted to mean the DType classes instead.
 * When dtype != NULL, this path is ignored, and the function does nothing
 * unless descr == NULL. If both descr and dtype are null, it returns the
 * descriptor for the array.
 *
 * This function is identical to normal casting using only the dtype, however,
 * it supports inspecting the elements when the array has object dtype
 * (and the given datatype describes a parametric DType class).
 *
 * @param arr The array object.
 * @param dtype NULL or a dtype class
 * @param descr A dtype instance, if the dtype is NULL the dtype class is
 *              found and e.g. "S0" is converted to denote only String.
 * @return A concrete dtype instance or NULL
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_AdaptDescriptorToArray(
        PyArrayObject *arr, PyArray_DTypeMeta *dtype, PyArray_Descr *descr)
{
    /* If the requested dtype is flexible, adapt it */
    PyArray_Descr *new_descr;
    int res;

    if (dtype != NULL && descr != NULL) {
        /* descr was given and no special logic, return (call not necessary) */
        Py_INCREF(descr);
        return descr;
    }
    if (dtype == NULL) {
        res = PyArray_ExtractDTypeAndDescriptor(descr, &new_descr, &dtype);
        if (res < 0) {
            return NULL;
        }
        if (new_descr != NULL) {
            Py_DECREF(dtype);
            return new_descr;
        }
    }
    else {
        assert(descr == NULL);  /* gueranteed above */
        Py_INCREF(dtype);
    }

    res = find_descriptor_from_array(arr, dtype, &new_descr);
    if (res < 0) {
        Py_DECREF(dtype);
        return NULL;
    }
    if (new_descr == NULL) {
        /* This is an object array but contained no elements, use default */
        new_descr = NPY_DT_CALL_default_descr(dtype);
    }
    Py_XDECREF(dtype);
    return new_descr;
}


/**
 * Recursion helper for `PyArray_DiscoverDTypeAndShape`.  See its
 * documentation for additional details.
 *
 * @param obj The current (possibly nested) object
 * @param curr_dims The current depth, i.e. initially 0 and increasing.
 * @param max_dims Maximum number of dimensions, modified during discovery.
 * @param out_descr dtype instance (or NULL) to promoted and update.
 * @param out_shape The current shape (updated)
 * @param coercion_cache_tail_ptr The tail of the linked list of coercion
 *        cache objects, which hold on to converted sequences and arrays.
 *        This is a pointer to the `->next` slot of the previous cache so
 *        that we can append a new cache object (and update this pointer).
 *        (Initially it is a pointer to the user-provided head pointer).
 * @param fixed_DType User provided fixed DType class
 * @param flags Discovery flags (reporting and behaviour flags, see def.)
 * @param copy Specifies the copy behavior. -1 is corresponds to copy=None,
 *        0 to copy=False, and 1 to copy=True in the Python API.
 * @return The updated number of maximum dimensions (i.e. scalars will set
 *         this to the current dimensions).
 */
NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape_Recursive(
        PyObject *obj, int curr_dims, int max_dims, PyArray_Descr**out_descr,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj ***coercion_cache_tail_ptr,
        PyArray_DTypeMeta *fixed_DType, enum _dtype_discovery_flags *flags,
        int copy)
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
                flags, DType);
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
        PyArray_Descr *requested_descr = NULL;
        if (*flags & DESCRIPTOR_WAS_SET) {
            /* __array__ may be passed the requested descriptor if provided */
            requested_descr = *out_descr;
        }
        int was_copied_by__array__ = 0;
        arr = (PyArrayObject *)_array_from_array_like(obj,
                requested_descr, 0, NULL, copy, &was_copied_by__array__);
        if (arr == NULL) {
            return -1;
        }
        else if (arr == (PyArrayObject *)Py_NotImplemented) {
            Py_DECREF(arr);
            arr = NULL;
        }
        if (was_copied_by__array__ == 1) {
            *flags |= COPY_WAS_CREATED_BY__ARRAY__;
        }
    }
    if (arr != NULL) {
        /*
         * This is an array object which will be added to the cache, keeps
         * the reference to the array alive (takes ownership).
         */
        if (npy_new_coercion_cache(obj, (PyObject *)arr,
                0, coercion_cache_tail_ptr, curr_dims) < 0) {
            return -1;
        }

        if (curr_dims == 0) {
            /*
             * Special case for reverse broadcasting, ignore max_dims if this
             * is a single array-like object; needed for PyArray_CopyObject.
             */
            memcpy(out_shape, PyArray_SHAPE(arr),
                   PyArray_NDIM(arr) * sizeof(npy_intp));
            max_dims = PyArray_NDIM(arr);
        }
        else if (update_shape(curr_dims, &max_dims, out_shape,
                PyArray_NDIM(arr), PyArray_SHAPE(arr), NPY_FALSE, flags) < 0) {
            *flags |= FOUND_RAGGED_ARRAY;
            return max_dims;
        }

        if (*flags & DESCRIPTOR_WAS_SET) {
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
        if (handle_promotion(out_descr, cast_descr, fixed_DType, flags) < 0) {
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
    npy_bool is_sequence = PySequence_Check(obj);
    if (is_sequence) {
        is_sequence = PySequence_Size(obj) >= 0;
        if (NPY_UNLIKELY(!is_sequence)) {
            /* NOTE: This should likely just raise all errors */
            if (PyErr_ExceptionMatches(PyExc_RecursionError) ||
                    PyErr_ExceptionMatches(PyExc_MemoryError)) {
                /*
                 * Consider these unrecoverable errors, continuing execution
                 * might crash the interpreter.
                 */
                return -1;
            }
            PyErr_Clear();
        }
    }
    if (NPY_UNLIKELY(*flags & DISCOVER_TUPLES_AS_ELEMENTS) &&
            PyTuple_Check(obj)) {
        is_sequence = NPY_FALSE;
    }
    if (curr_dims == max_dims || !is_sequence) {
        /* Clear any PySequence_Size error which would corrupts further calls */
        max_dims = handle_scalar(
                obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                flags, NULL);
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
    seq = PySequence_Fast(obj, "Could not convert object to sequence"); // noqa: borrowed-ref - manual fix needed
    if (seq == NULL) {
        /*
         * Specifically do not fail on things that look like a dictionary,
         * instead treat them as scalar.
         */
        if (PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
            max_dims = handle_scalar(
                    obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                    flags, NULL);
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
        /* If the sequence is empty, this must be the last dimension */
        *flags |= MAX_DIMS_WAS_REACHED;
        return curr_dims + 1;
    }

    /* Allow keyboard interrupts. See gh issue 18117. */
    if (PyErr_CheckSignals() < 0) {
        return -1;
    }

    /*
     * For a sequence we need to make a copy of the final aggregate anyway.
     * There's no need to pass explicit `copy=True`, so we switch
     * to `copy=None` (copy if needed).
     */
    if (copy == 1) {
        copy = -1;
    }

    /* Recursive call for each sequence item */
    for (Py_ssize_t i = 0; i < size; i++) {
        max_dims = PyArray_DiscoverDTypeAndShape_Recursive(
                objects[i], curr_dims + 1, max_dims,
                out_descr, out_shape, coercion_cache_tail_ptr, fixed_DType,
                flags, copy);

        if (max_dims < 0) {
            return -1;
        }
    }
    return max_dims;
}


/**
 * Finds the DType and shape of an arbitrary nested sequence. This is the
 * general purpose function to find the parameters of the array (but not
 * the array itself) as returned by `np.array()`
 *
 * Note: Before considering to make part of this public, we should consider
 *       whether things such as `out_descr != NULL` should be supported in
 *       a public API.
 *
 * @param obj Scalar or nested sequences.
 * @param max_dims Maximum number of dimensions (after this scalars are forced)
 * @param out_shape Will be filled with the output shape (more than the actual
 *        shape may be written).
 * @param coercion_cache NULL initialized reference to a cache pointer.
 *        May be set to the first coercion_cache, and has to be freed using
 *        npy_free_coercion_cache.
 *        This should be stored in a thread-safe manner (i.e. function static)
 *        and is designed to be consumed by `PyArray_AssignFromCache`.
 *        If not consumed, must be freed using `npy_free_coercion_cache`.
 * @param fixed_DType A user provided fixed DType class.
 * @param requested_descr A user provided fixed descriptor. This is always
 *        returned as the discovered descriptor, but currently only used
 *        for the ``__array__`` protocol.
 * @param out_descr Set to the discovered output descriptor. This may be
 *        non NULL but only when fixed_DType/requested_descr are not given.
 *        If non NULL, it is the first dtype being promoted and used if there
 *        are no elements.
 *        The result may be unchanged (remain NULL) when converting a
 *        sequence with no elements. In this case it is callers responsibility
 *        to choose a default.
 * @param copy Specifies the copy behavior. -1 is corresponds to copy=None,
 *        0 to copy=False, and 1 to copy=True in the Python API.
 * @param was_copied_by__array__ Set to 1 if it can be assumed that a copy was
 *        made by implementor.
 * @return dimensions of the discovered object or -1 on error.
 *         WARNING: If (and only if) the output is a single array, the ndim
 *         returned _can_ exceed the maximum allowed number of dimensions.
 *         It might be nice to deprecate this? But it allows things such as
 *         `arr1d[...] = np.array([[1,2,3,4]])`
 */
NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape(
        PyObject *obj, int max_dims,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj **coercion_cache,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        PyArray_Descr **out_descr, int copy, int *was_copied_by__array__)
{
    coercion_cache_obj **coercion_cache_head = coercion_cache;
    *coercion_cache = NULL;
    enum _dtype_discovery_flags flags = 0;

    /*
     * Support a passed in descriptor (but only if nothing was specified).
     */
    assert(*out_descr == NULL || fixed_DType == NULL);
    /* Validate input of requested descriptor and DType */
    if (fixed_DType != NULL) {
        assert(PyObject_TypeCheck(
                (PyObject *)fixed_DType, (PyTypeObject *)&PyArrayDTypeMeta_Type));
    }

    if (requested_descr != NULL) {
        if (fixed_DType != NULL) {
            assert(fixed_DType == NPY_DTYPE(requested_descr));
        }
        /* The output descriptor must be the input. */
        Py_INCREF(requested_descr);
        *out_descr = requested_descr;
        flags |= DESCRIPTOR_WAS_SET;
    }

    /*
     * Call the recursive function, the setup for this may need expanding
     * to handle caching better.
     */

    /* Legacy discovery flags */
    if (requested_descr != NULL) {
        if (requested_descr->type_num == NPY_STRING &&
                requested_descr->type == 'c') {
            /* Character dtype variation of string (should be deprecated...) */
            flags |= DISCOVER_STRINGS_AS_SEQUENCES;
        }
        else if (requested_descr->type_num == NPY_VOID &&
                    (((_PyArray_LegacyDescr *)requested_descr)->names
                     || ((_PyArray_LegacyDescr *)requested_descr)->subarray))  {
            /* Void is a chimera, in that it may or may not be structured... */
            flags |= DISCOVER_TUPLES_AS_ELEMENTS;
        }
    }

    int ndim = PyArray_DiscoverDTypeAndShape_Recursive(
            obj, 0, max_dims, out_descr, out_shape, &coercion_cache,
            fixed_DType, &flags, copy);
    if (ndim < 0) {
        goto fail;
    }

    if (was_copied_by__array__ != NULL && flags & COPY_WAS_CREATED_BY__ARRAY__) {
        *was_copied_by__array__ = 1;
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

        if (fixed_DType == NULL || fixed_DType->type_num != NPY_OBJECT) {
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
            }
            else {
                PyErr_Format(PyExc_ValueError,
                        "setting an array element with a sequence. The "
                        "requested array would exceed the maximum number of "
                        "dimension of %d.",
                        max_dims);
            }
            goto fail;
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
        /* descriptor was provided, we did not accidentally change it */
        assert(*out_descr == requested_descr);
    }
    else if (NPY_UNLIKELY(*out_descr == NULL)) {
        /*
         * When the object contained no elements (sequence of length zero),
         * the no descriptor may have been found. When a DType was requested
         * we use it to define the output dtype.
         * Otherwise, out_descr will remain NULL and the caller has to set
         * the correct default.
         */
        if (fixed_DType != NULL) {
            *out_descr = NPY_DT_CALL_default_descr(fixed_DType);
            if (*out_descr == NULL) {
                goto fail;
            }
        }
    }
    return ndim;

  fail:
    npy_free_coercion_cache(*coercion_cache_head);
    *coercion_cache_head = NULL;
    Py_XSETREF(*out_descr, NULL);
    return -1;
}


/*
 * Python API function to expose the dtype+shape discovery functionality
 * directly.
 */
NPY_NO_EXPORT PyObject *
_discover_array_parameters(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj;
    npy_dtype_info dt_info = {NULL, NULL};
    npy_intp shape[NPY_MAXDIMS];

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments(
            "_discover_array_parameters", args, len_args, kwnames,
            "", NULL, &obj,
            "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
            NULL, NULL, NULL) < 0) {
        /* fixed is last to parse, so never necessary to clean up */
        return NULL;
    }

    coercion_cache_obj *coercion_cache = NULL;
    PyObject *out_dtype = NULL;
    int ndim = PyArray_DiscoverDTypeAndShape(
            obj, NPY_MAXDIMS, shape,
            &coercion_cache,
            dt_info.dtype, dt_info.descr, (PyArray_Descr **)&out_dtype, 0, NULL);
    Py_XDECREF(dt_info.dtype);
    Py_XDECREF(dt_info.descr);
    if (ndim < 0) {
        return NULL;
    }
    npy_free_coercion_cache(coercion_cache);
    if (out_dtype == NULL) {
        /* Empty sequence, report this as None. */
        out_dtype = Py_None;
        Py_INCREF(Py_None);
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
