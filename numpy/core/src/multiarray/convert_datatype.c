#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"
#include "lowlevel_strided_loops.h"

#include "npy_pycompat.h"
#include "numpy/npy_math.h"

#include "array_coercion.h"
#include "can_cast_table.h"
#include "common.h"
#include "ctors.h"
#include "descriptor.h"
#include "dtypemeta.h"
#include "common_dtype.h"
#include "scalartypes.h"
#include "mapping.h"
#include "legacy_dtype_implementation.h"

#include "abstractdtypes.h"
#include "convert_datatype.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "array_method.h"
#include "usertypes.h"
#include "dtype_transfer.h"
#include "dtype_traversal.h"
#include "arrayobject.h"


/*
 * Required length of string when converting from unsigned integer type.
 * Array index is integer size in bytes.
 * - 3 chars needed for cast to max value of 255 or 127
 * - 5 chars needed for cast to max value of 65535 or 32767
 * - 10 chars needed for cast to max value of 4294967295 or 2147483647
 * - 20 chars needed for cast to max value of 18446744073709551615
 *   or 9223372036854775807
 */
NPY_NO_EXPORT npy_intp REQUIRED_STR_LEN[] = {0, 3, 5, 10, 10, 20, 20, 20, 20};

/*
 * Whether or not legacy value-based promotion/casting is used.
 */
NPY_NO_EXPORT int npy_promotion_state = NPY_USE_LEGACY_PROMOTION;
NPY_NO_EXPORT PyObject *NO_NEP50_WARNING_CTX = NULL;
NPY_NO_EXPORT PyObject *npy_DTypePromotionError = NULL;
NPY_NO_EXPORT PyObject *npy_UFuncNoLoopError = NULL;


static PyObject *
PyArray_GetGenericToVoidCastingImpl(void);

static PyObject *
PyArray_GetVoidToGenericCastingImpl(void);

static PyObject *
PyArray_GetGenericToObjectCastingImpl(void);

static PyObject *
PyArray_GetObjectToGenericCastingImpl(void);


/*
 * Return 1 if promotion warnings should be given and 0 if they are currently
 * suppressed in the local context.
 */
NPY_NO_EXPORT int
npy_give_promotion_warnings(void)
{
    PyObject *val;

    npy_cache_import(
            "numpy.core._ufunc_config", "NO_NEP50_WARNING",
            &NO_NEP50_WARNING_CTX);
    if (NO_NEP50_WARNING_CTX == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 1;
    }

    if (PyContextVar_Get(NO_NEP50_WARNING_CTX, Py_False, &val) < 0) {
        /* Errors should not really happen, but if it does assume we warn. */
        PyErr_WriteUnraisable(NULL);
        return 1;
    }
    Py_DECREF(val);
    /* only when the no-warnings context is false, we give warnings */
    return val == Py_False;
}


NPY_NO_EXPORT PyObject *
npy__get_promotion_state(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(arg)) {
    if (npy_promotion_state == NPY_USE_WEAK_PROMOTION) {
        return PyUnicode_FromString("weak");
    }
    else if (npy_promotion_state == NPY_USE_WEAK_PROMOTION_AND_WARN) {
        return PyUnicode_FromString("weak_and_warn");
    }
    else if (npy_promotion_state == NPY_USE_LEGACY_PROMOTION) {
        return PyUnicode_FromString("legacy");
    }
    PyErr_SetString(PyExc_SystemError, "invalid promotion state!");
    return NULL;
}


NPY_NO_EXPORT PyObject *
npy__set_promotion_state(PyObject *NPY_UNUSED(mod), PyObject *arg)
{
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError,
                "_set_promotion_state() argument or NPY_PROMOTION_STATE "
                "must be a string.");
        return NULL;
    }
    if (PyUnicode_CompareWithASCIIString(arg, "weak") == 0) {
        npy_promotion_state = NPY_USE_WEAK_PROMOTION;
    }
    else if (PyUnicode_CompareWithASCIIString(arg, "weak_and_warn") == 0) {
        npy_promotion_state = NPY_USE_WEAK_PROMOTION_AND_WARN;
    }
    else if (PyUnicode_CompareWithASCIIString(arg, "legacy") == 0) {
        npy_promotion_state = NPY_USE_LEGACY_PROMOTION;
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "_set_promotion_state() argument or NPY_PROMOTION_STATE must be "
                "'weak', 'legacy', or 'weak_and_warn' but got '%.100S'", arg);
        return NULL;
    }
    Py_RETURN_NONE;
}

/**
 * Fetch the casting implementation from one DType to another.
 *
 * @params from
 * @params to
 *
 * @returns A castingimpl (PyArrayDTypeMethod *), None or NULL with an
 *          error set.
 */
NPY_NO_EXPORT PyObject *
PyArray_GetCastingImpl(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to)
{
    PyObject *res;
    if (from == to) {
        res = (PyObject *)NPY_DT_SLOTS(from)->within_dtype_castingimpl;
    }
    else {
        res = PyDict_GetItemWithError(NPY_DT_SLOTS(from)->castingimpls, (PyObject *)to);
    }
    if (res != NULL || PyErr_Occurred()) {
        Py_XINCREF(res);
        return res;
    }
    /*
     * The following code looks up CastingImpl based on the fact that anything
     * can be cast to and from objects or structured (void) dtypes.
     *
     * The last part adds casts dynamically based on legacy definition
     */
    if (from->type_num == NPY_OBJECT) {
        res = PyArray_GetObjectToGenericCastingImpl();
    }
    else if (to->type_num == NPY_OBJECT) {
        res = PyArray_GetGenericToObjectCastingImpl();
    }
    else if (from->type_num == NPY_VOID) {
        res = PyArray_GetVoidToGenericCastingImpl();
    }
    else if (to->type_num == NPY_VOID) {
        res = PyArray_GetGenericToVoidCastingImpl();
    }
    else if (from->type_num < NPY_NTYPES && to->type_num < NPY_NTYPES) {
        /* All builtin dtypes have their casts explicitly defined. */
        PyErr_Format(PyExc_RuntimeError,
                "builtin cast from %S to %S not found, this should not "
                "be possible.", from, to);
        return NULL;
    }
    else {
        /* Reject non-legacy dtypes (they need to use the new API) */
        if (!NPY_DT_is_legacy(from) || !NPY_DT_is_legacy(to)) {
            Py_RETURN_NONE;
        }
        if (from != to) {
            /* A cast function must have been registered */
            PyArray_VectorUnaryFunc *castfunc = PyArray_GetCastFunc(
                    from->singleton, to->type_num);
            if (castfunc == NULL) {
                PyErr_Clear();
                /* Remember that this cast is not possible */
                if (PyDict_SetItem(NPY_DT_SLOTS(from)->castingimpls,
                            (PyObject *) to, Py_None) < 0) {
                    return NULL;
                }
                Py_RETURN_NONE;
            }
        }

        /* PyArray_AddLegacyWrapping_CastingImpl find the correct casting level: */
        /*
         * TODO: Possibly move this to the cast registration time. But if we do
         *       that, we have to also update the cast when the casting safety
         *       is registered.
         */
        if (PyArray_AddLegacyWrapping_CastingImpl(from, to, -1) < 0) {
            return NULL;
        }
        return PyArray_GetCastingImpl(from, to);
    }

    if (res == NULL) {
        return NULL;
    }
    if (from == to) {
        PyErr_Format(PyExc_RuntimeError,
                "Internal NumPy error, within-DType cast missing for %S!", from);
        Py_DECREF(res);
        return NULL;
    }
    if (PyDict_SetItem(NPY_DT_SLOTS(from)->castingimpls,
                (PyObject *)to, res) < 0) {
        Py_DECREF(res);
        return NULL;
    }
    return res;
}


/**
 * Fetch the (bound) casting implementation from one DType to another.
 *
 * @params from
 * @params to
 *
 * @returns A bound casting implementation or None (or NULL for error).
 */
static PyObject *
PyArray_GetBoundCastingImpl(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to)
{
    PyObject *method = PyArray_GetCastingImpl(from, to);
    if (method == NULL || method == Py_None) {
        return method;
    }

    /* TODO: Create better way to wrap method into bound method */
    PyBoundArrayMethodObject *res;
    res = PyObject_New(PyBoundArrayMethodObject, &PyBoundArrayMethod_Type);
    if (res == NULL) {
        return NULL;
    }
    res->method = (PyArrayMethodObject *)method;
    res->dtypes = PyMem_Malloc(2 * sizeof(PyArray_DTypeMeta *));
    if (res->dtypes == NULL) {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(from);
    res->dtypes[0] = from;
    Py_INCREF(to);
    res->dtypes[1] = to;

    return (PyObject *)res;
}


NPY_NO_EXPORT PyObject *
_get_castingimpl(PyObject *NPY_UNUSED(module), PyObject *args)
{
    PyArray_DTypeMeta *from, *to;
    if (!PyArg_ParseTuple(args, "O!O!:_get_castingimpl",
            &PyArrayDTypeMeta_Type, &from, &PyArrayDTypeMeta_Type, &to)) {
        return NULL;
    }
    return PyArray_GetBoundCastingImpl(from, to);
}


/**
 * Find the minimal cast safety level given two cast-levels as input.
 * Supports the NPY_CAST_IS_VIEW check, and should be preferred to allow
 * extending cast-levels if necessary.
 * It is not valid for one of the arguments to be -1 to indicate an error.
 *
 * @param casting1
 * @param casting2
 * @return The minimal casting error (can be -1).
 */
NPY_NO_EXPORT NPY_CASTING
PyArray_MinCastSafety(NPY_CASTING casting1, NPY_CASTING casting2)
{
    if (casting1 < 0 || casting2 < 0) {
        return -1;
    }
    /* larger casting values are less safe */
    if (casting1 > casting2) {
        return casting1;
    }
    return casting2;
}


/*NUMPY_API
 * For backward compatibility
 *
 * Cast an array using typecode structure.
 * steals reference to dtype --- cannot be NULL
 *
 * This function always makes a copy of arr, even if the dtype
 * doesn't change.
 */
NPY_NO_EXPORT PyObject *
PyArray_CastToType(PyArrayObject *arr, PyArray_Descr *dtype, int is_f_order)
{
    PyObject *out;

    if (dtype == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "dtype is NULL in PyArray_CastToType");
        return NULL;
    }

    Py_SETREF(dtype, PyArray_AdaptDescriptorToArray(arr, NULL, dtype));
    if (dtype == NULL) {
        return NULL;
    }

    out = PyArray_NewFromDescr(Py_TYPE(arr), dtype,
                               PyArray_NDIM(arr),
                               PyArray_DIMS(arr),
                               NULL, NULL,
                               is_f_order,
                               (PyObject *)arr);

    if (out == NULL) {
        return NULL;
    }

    if (PyArray_CopyInto((PyArrayObject *)out, arr) < 0) {
        Py_DECREF(out);
        return NULL;
    }

    return out;
}

/*NUMPY_API
 * Get a cast function to cast from the input descriptor to the
 * output type_number (must be a registered data-type).
 * Returns NULL if un-successful.
 */
NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num)
{
    PyArray_VectorUnaryFunc *castfunc = NULL;

    if (type_num < NPY_NTYPES_ABI_COMPATIBLE) {
        castfunc = descr->f->cast[type_num];
    }
    else {
        PyObject *obj = descr->f->castdict;
        if (obj && PyDict_Check(obj)) {
            PyObject *key;
            PyObject *cobj;

            key = PyLong_FromLong(type_num);
            cobj = PyDict_GetItem(obj, key);
            Py_DECREF(key);
            if (cobj && PyCapsule_CheckExact(cobj)) {
                castfunc = PyCapsule_GetPointer(cobj, NULL);
                if (castfunc == NULL) {
                    return NULL;
                }
            }
        }
    }
    if (PyTypeNum_ISCOMPLEX(descr->type_num) &&
            !PyTypeNum_ISCOMPLEX(type_num) &&
            PyTypeNum_ISNUMBER(type_num) &&
            !PyTypeNum_ISBOOL(type_num)) {
        static PyObject *cls = NULL;
        npy_cache_import("numpy.exceptions", "ComplexWarning", &cls);
        if (cls == NULL) {
            return NULL;
        }
        int ret = PyErr_WarnEx(cls,
                "Casting complex values to real discards "
                "the imaginary part", 1);
        if (ret < 0) {
            return NULL;
        }
    }
    if (castfunc) {
        return castfunc;
    }

    PyErr_SetString(PyExc_ValueError,
            "No cast function available.");
    return NULL;
}


/*
 * Must be broadcastable.
 * This code is very similar to PyArray_CopyInto/PyArray_MoveInto
 * except casting is done --- NPY_BUFSIZE is used
 * as the size of the casting buffer.
 */

/*NUMPY_API
 * Cast to an already created array.
 */
NPY_NO_EXPORT int
PyArray_CastTo(PyArrayObject *out, PyArrayObject *mp)
{
    /* CopyInto handles the casting now */
    return PyArray_CopyInto(out, mp);
}

/*NUMPY_API
 * Cast to an already created array.  Arrays don't have to be "broadcastable"
 * Only requirement is they have the same number of elements.
 */
NPY_NO_EXPORT int
PyArray_CastAnyTo(PyArrayObject *out, PyArrayObject *mp)
{
    /* CopyAnyInto handles the casting now */
    return PyArray_CopyAnyInto(out, mp);
}


static NPY_CASTING
_get_cast_safety_from_castingimpl(PyArrayMethodObject *castingimpl,
        PyArray_DTypeMeta *dtypes[2], PyArray_Descr *from, PyArray_Descr *to,
        npy_intp *view_offset)
{
    PyArray_Descr *descrs[2] = {from, to};
    PyArray_Descr *out_descrs[2];

    *view_offset = NPY_MIN_INTP;
    NPY_CASTING casting = castingimpl->resolve_descriptors(
            castingimpl, dtypes, descrs, out_descrs, view_offset);
    if (casting < 0) {
        return -1;
    }
    /* The returned descriptors may not match, requiring a second check */
    if (out_descrs[0] != descrs[0]) {
        npy_intp from_offset = NPY_MIN_INTP;
        NPY_CASTING from_casting = PyArray_GetCastInfo(
                descrs[0], out_descrs[0], NULL, &from_offset);
        casting = PyArray_MinCastSafety(casting, from_casting);
        if (from_offset != *view_offset) {
            /* `view_offset` differs: The multi-step cast cannot be a view. */
            *view_offset = NPY_MIN_INTP;
        }
        if (casting < 0) {
            goto finish;
        }
    }
    if (descrs[1] != NULL && out_descrs[1] != descrs[1]) {
        npy_intp from_offset = NPY_MIN_INTP;
        NPY_CASTING from_casting = PyArray_GetCastInfo(
                descrs[1], out_descrs[1], NULL, &from_offset);
        casting = PyArray_MinCastSafety(casting, from_casting);
        if (from_offset != *view_offset) {
            /* `view_offset` differs: The multi-step cast cannot be a view. */
            *view_offset = NPY_MIN_INTP;
        }
        if (casting < 0) {
            goto finish;
        }
    }

  finish:
    Py_DECREF(out_descrs[0]);
    Py_DECREF(out_descrs[1]);
    /*
     * Check for less harmful non-standard returns.  The following two returns
     * should never happen:
     * 1. No-casting must imply a view offset of 0.
     * 2. Equivalent-casting + 0 view offset is (usually) the definition
     *    of a "no" cast.  However, changing the order of fields can also
     *    create descriptors that are not equivalent but views.
     * Note that unsafe casts can have a view offset.  For example, in
     * principle, casting `<i8` to `<i4` is a cast with 0 offset.
     */
    if (*view_offset != 0) {
        assert(casting != NPY_NO_CASTING);
    }
    else {
        assert(casting != NPY_EQUIV_CASTING
               || (PyDataType_HASFIELDS(from) && PyDataType_HASFIELDS(to)));
    }
    return casting;
}


/**
 * Given two dtype instances, find the correct casting safety.
 *
 * Note that in many cases, it may be preferable to fetch the casting
 * implementations fully to have them available for doing the actual cast
 * later.
 *
 * @param from
 * @param to The descriptor to cast to (may be NULL)
 * @param to_dtype If `to` is NULL, must pass the to_dtype (otherwise this
 *        is ignored).
 * @param[out] view_offset
 * @return NPY_CASTING or -1 on error or if the cast is not possible.
 */
NPY_NO_EXPORT NPY_CASTING
PyArray_GetCastInfo(
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype,
        npy_intp *view_offset)
{
    if (to != NULL) {
        to_dtype = NPY_DTYPE(to);
    }
    PyObject *meth = PyArray_GetCastingImpl(NPY_DTYPE(from), to_dtype);
    if (meth == NULL) {
        return -1;
    }
    if (meth == Py_None) {
        Py_DECREF(Py_None);
        return -1;
    }

    PyArrayMethodObject *castingimpl = (PyArrayMethodObject *)meth;
    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(from), to_dtype};
    NPY_CASTING casting = _get_cast_safety_from_castingimpl(castingimpl,
            dtypes, from, to, view_offset);
    Py_DECREF(meth);

    return casting;
}


/**
 * Check whether a cast is safe, see also `PyArray_GetCastInfo` for
 * a similar function.  Unlike GetCastInfo, this function checks the
 * `castingimpl->casting` when available.  This allows for two things:
 *
 * 1. It avoids  calling `resolve_descriptors` in some cases.
 * 2. Strings need to discover the length, but in some cases we know that the
 *    cast is valid (assuming the string length is discovered first).
 *
 * The latter means that a `can_cast` could return True, but the cast fail
 * because the parametric type cannot guess the correct output descriptor.
 * (I.e. if `object_arr.astype("S")` did _not_ inspect the objects, and the
 * user would have to guess the string length.)
 *
 * @param casting the requested casting safety.
 * @param from
 * @param to The descriptor to cast to (may be NULL)
 * @param to_dtype If `to` is NULL, must pass the to_dtype (otherwise this
 *        is ignored).
 * @return 0 for an invalid cast, 1 for a valid and -1 for an error.
 */
NPY_NO_EXPORT int
PyArray_CheckCastSafety(NPY_CASTING casting,
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype)
{
    if (to != NULL) {
        to_dtype = NPY_DTYPE(to);
    }
    PyObject *meth = PyArray_GetCastingImpl(NPY_DTYPE(from), to_dtype);
    if (meth == NULL) {
        return -1;
    }
    if (meth == Py_None) {
        Py_DECREF(Py_None);
        return -1;
    }
    PyArrayMethodObject *castingimpl = (PyArrayMethodObject *)meth;

    if (PyArray_MinCastSafety(castingimpl->casting, casting) == casting) {
        /* No need to check using `castingimpl.resolve_descriptors()` */
        Py_DECREF(meth);
        return 1;
    }

    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(from), to_dtype};
    npy_intp view_offset;
    NPY_CASTING safety = _get_cast_safety_from_castingimpl(castingimpl,
            dtypes, from, to, &view_offset);
    Py_DECREF(meth);
    /* If casting is the smaller (or equal) safety we match */
    if (safety < 0) {
        return -1;
    }
    return PyArray_MinCastSafety(safety, casting) == casting;
}


/*NUMPY_API
 *Check the type coercion rules.
 */
NPY_NO_EXPORT int
PyArray_CanCastSafely(int fromtype, int totype)
{
    /* Identity */
    if (fromtype == totype) {
        return 1;
    }
    /*
     * As a micro-optimization, keep the cast table around.  This can probably
     * be removed as soon as the ufunc loop lookup is modified (presumably
     * before the 1.21 release).  It does no harm, but the main user of this
     * function is the ufunc-loop lookup calling it until a loop matches!
     *
     * (The table extends further, but is not strictly correct for void).
     * TODO: Check this!
     */
    if ((unsigned int)fromtype <= NPY_CLONGDOUBLE &&
            (unsigned int)totype <= NPY_CLONGDOUBLE) {
        return _npy_can_cast_safely_table[fromtype][totype];
    }

    PyArray_DTypeMeta *from = PyArray_DTypeFromTypeNum(fromtype);
    if (from == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 0;
    }
    PyArray_DTypeMeta *to = PyArray_DTypeFromTypeNum(totype);
    if (to == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 0;
    }
    PyObject *castingimpl = PyArray_GetCastingImpl(from, to);
    Py_DECREF(from);
    Py_DECREF(to);

    if (castingimpl == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 0;
    }
    else if (castingimpl == Py_None) {
        Py_DECREF(Py_None);
        return 0;
    }
    NPY_CASTING safety = ((PyArrayMethodObject *)castingimpl)->casting;
    int res = PyArray_MinCastSafety(safety, NPY_SAFE_CASTING) == NPY_SAFE_CASTING;
    Py_DECREF(castingimpl);
    return res;
}



/*NUMPY_API
 * leaves reference count alone --- cannot be NULL
 *
 * PyArray_CanCastTypeTo is equivalent to this, but adds a 'casting'
 * parameter.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastTo(PyArray_Descr *from, PyArray_Descr *to)
{
    return PyArray_CanCastTypeTo(from, to, NPY_SAFE_CASTING);
}


/* Provides an ordering for the dtype 'kind' character codes */
NPY_NO_EXPORT int
dtype_kind_to_ordering(char kind)
{
    switch (kind) {
        /* Boolean kind */
        case 'b':
            return 0;
        /* Unsigned int kind */
        case 'u':
            return 1;
        /* Signed int kind */
        case 'i':
            return 2;
        /* Float kind */
        case 'f':
            return 4;
        /* Complex kind */
        case 'c':
            return 5;
        /* String kind */
        case 'S':
        case 'a':
            return 6;
        /* Unicode kind */
        case 'U':
            return 7;
        /* Void kind */
        case 'V':
            return 8;
        /* Object kind */
        case 'O':
            return 9;
        /*
         * Anything else, like datetime, is special cased to
         * not fit in this hierarchy
         */
        default:
            return -1;
    }
}

/* Converts a type number from unsigned to signed */
static int
type_num_unsigned_to_signed(int type_num)
{
    switch (type_num) {
        case NPY_UBYTE:
            return NPY_BYTE;
        case NPY_USHORT:
            return NPY_SHORT;
        case NPY_UINT:
            return NPY_INT;
        case NPY_ULONG:
            return NPY_LONG;
        case NPY_ULONGLONG:
            return NPY_LONGLONG;
        default:
            return type_num;
    }
}


/*NUMPY_API
 * Returns true if data of type 'from' may be cast to data of type
 * 'to' according to the rule 'casting'.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastTypeTo(PyArray_Descr *from, PyArray_Descr *to,
        NPY_CASTING casting)
{
    PyArray_DTypeMeta *to_dtype = NPY_DTYPE(to);

    /*
     * NOTE: This code supports U and S, this is identical to the code
     *       in `ctors.c` which does not allow these dtypes to be attached
     *       to an array. Unlike the code for `np.array(..., dtype=)`
     *       which uses `PyArray_ExtractDTypeAndDescriptor` it rejects "m8"
     *       as a flexible dtype instance representing a DType.
     */
    /*
     * TODO: We should grow support for `np.can_cast("d", "S")` being
     *       different from `np.can_cast("d", "S0")` here, at least for
     *       the python side API.
     *       The `to = NULL` branch, which considers "S0" to be "flexible"
     *       should probably be deprecated.
     *       (This logic is duplicated in `PyArray_CanCastArrayTo`)
     */
    if (PyDataType_ISUNSIZED(to) && to->subarray == NULL) {
        to = NULL;  /* consider mainly S0 and U0 as S and U */
    }

    int is_valid = PyArray_CheckCastSafety(casting, from, to, to_dtype);
    /* Clear any errors and consider this unsafe (should likely be changed) */
    if (is_valid < 0) {
        PyErr_Clear();
        return 0;
    }
    return is_valid;
}


/* CanCastArrayTo needs this function */
static int min_scalar_type_num(char *valueptr, int type_num,
                                            int *is_small_unsigned);


/*
 * NOTE: This function uses value based casting logic for scalars. It will
 *       require updates when we phase out value-based-casting.
 */
NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data,
                    PyArray_Descr *to, NPY_CASTING casting)
{
    /*
     * If the two dtypes are actually references to the same object
     * or if casting type is forced unsafe then always OK.
     *
     * TODO: Assuming that unsafe casting always works is not actually correct
     */
    if (scal_type == to || casting == NPY_UNSAFE_CASTING ) {
        return 1;
    }

    int valid = PyArray_CheckCastSafety(casting, scal_type, to, NPY_DTYPE(to));
    if (valid == 1) {
        /* This is definitely a valid cast. */
        return 1;
    }
    if (valid < 0) {
        /* Probably must return 0, but just keep trying for now. */
        PyErr_Clear();
    }

    /*
     * If the scalar isn't a number, value-based casting cannot kick in and
     * we must not attempt it.
     * (Additional fast-checks would be possible, but probably unnecessary.)
     */
    if (!PyTypeNum_ISNUMBER(scal_type->type_num)) {
        return 0;
    }

    /*
     * At this point we have to check value-based casting.
     */
    PyArray_Descr *dtype;
    int is_small_unsigned = 0, type_num;
    /* An aligned memory buffer large enough to hold any builtin numeric type */
    npy_longlong value[4];

    int swap = !PyArray_ISNBO(scal_type->byteorder);
    scal_type->f->copyswap(&value, scal_data, swap, NULL);

    type_num = min_scalar_type_num((char *)&value, scal_type->type_num,
                                    &is_small_unsigned);

    /*
     * If we've got a small unsigned scalar, and the 'to' type
     * is not unsigned, then make it signed to allow the value
     * to be cast more appropriately.
     */
    if (is_small_unsigned && !(PyTypeNum_ISUNSIGNED(to->type_num))) {
        type_num = type_num_unsigned_to_signed(type_num);
    }

    dtype = PyArray_DescrFromType(type_num);
    if (dtype == NULL) {
        return 0;
    }
#if 0
    printf("min scalar cast ");
    PyObject_Print(dtype, stdout, 0);
    printf(" to ");
    PyObject_Print(to, stdout, 0);
    printf("\n");
#endif
    npy_bool ret = PyArray_CanCastTypeTo(dtype, to, casting);
    Py_DECREF(dtype);
    return ret;
}


NPY_NO_EXPORT npy_bool
can_cast_pyscalar_scalar_to(
        int flags, PyArray_Descr *to, NPY_CASTING casting)
{
    /*
     * This function only works reliably for legacy (NumPy dtypes).
     * If we end up here for a non-legacy DType, it is a bug.
     */
    assert(NPY_DT_is_legacy(NPY_DTYPE(to)));

    /*
     * Quickly check for the typical numeric cases, where the casting rules
     * can be hardcoded fairly easily.
     */
    if (PyDataType_ISCOMPLEX(to)) {
        return 1;
    }
    else if (PyDataType_ISFLOAT(to)) {
        if (flags & NPY_ARRAY_WAS_PYTHON_COMPLEX) {
            return casting == NPY_UNSAFE_CASTING;
        }
        return 1;
    }
    else if (PyDataType_ISINTEGER(to)) {
        if (!(flags & NPY_ARRAY_WAS_PYTHON_INT)) {
            return casting == NPY_UNSAFE_CASTING;
        }
        return 1;
    }

    /*
     * For all other cases we use the default dtype.
     */
    PyArray_Descr *from;
    if (flags & NPY_ARRAY_WAS_PYTHON_INT) {
        from = PyArray_DescrFromType(NPY_LONG);
    }
    else if (flags & NPY_ARRAY_WAS_PYTHON_FLOAT) {
        from = PyArray_DescrFromType(NPY_DOUBLE);
    }
    else {
        from = PyArray_DescrFromType(NPY_CDOUBLE);
    }
    int res = PyArray_CanCastTypeTo(from, to, casting);
    Py_DECREF(from);
    return res;
}

/*NUMPY_API
 * Returns 1 if the array object may be cast to the given data type using
 * the casting rule, 0 otherwise.  This differs from PyArray_CanCastTo in
 * that it handles scalar arrays (0 dimensions) specially, by checking
 * their value.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastArrayTo(PyArrayObject *arr, PyArray_Descr *to,
                        NPY_CASTING casting)
{
    PyArray_Descr *from = PyArray_DESCR(arr);
    PyArray_DTypeMeta *to_dtype = NPY_DTYPE(to);

    /* NOTE, TODO: The same logic as `PyArray_CanCastTypeTo`: */
    if (PyDataType_ISUNSIZED(to) && to->subarray == NULL) {
        to = NULL;
    }

    if (npy_promotion_state == NPY_USE_LEGACY_PROMOTION) {
        /*
         * If it's a scalar, check the value.  (This only currently matters for
         * numeric types and for `to == NULL` it can't be numeric.)
         */
        if (PyArray_NDIM(arr) == 0 && !PyArray_HASFIELDS(arr) && to != NULL) {
            return can_cast_scalar_to(from, PyArray_DATA(arr), to, casting);
        }
    }
    else {
        /*
         * If it's a scalar, check the value.  (This only currently matters for
         * numeric types and for `to == NULL` it can't be numeric.)
         */
        if (PyArray_FLAGS(arr) & NPY_ARRAY_WAS_PYTHON_LITERAL && to != NULL) {
            return can_cast_pyscalar_scalar_to(
                    PyArray_FLAGS(arr) & NPY_ARRAY_WAS_PYTHON_LITERAL, to,
                    casting);
        }
    }

    /* Otherwise, use the standard rules (same as `PyArray_CanCastTypeTo`) */
    int is_valid = PyArray_CheckCastSafety(casting, from, to, to_dtype);
    /* Clear any errors and consider this unsafe (should likely be changed) */
    if (is_valid < 0) {
        PyErr_Clear();
        return 0;
    }
    return is_valid;
}


NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return "'no'";
        case NPY_EQUIV_CASTING:
            return "'equiv'";
        case NPY_SAFE_CASTING:
            return "'safe'";
        case NPY_SAME_KIND_CASTING:
            return "'same_kind'";
        case NPY_UNSAFE_CASTING:
            return "'unsafe'";
        default:
            return "<unknown>";
    }
}


/**
 * Helper function to set a useful error when casting is not possible.
 *
 * @param src_dtype
 * @param dst_dtype
 * @param casting
 * @param scalar Whether this was a "scalar" cast (includes 0-D array with
 *               PyArray_CanCastArrayTo result).
 */
NPY_NO_EXPORT void
npy_set_invalid_cast_error(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        NPY_CASTING casting, npy_bool scalar)
{
    char *msg;

    if (!scalar) {
        msg = "Cannot cast array data from %R to %R according to the rule %s";
    }
    else {
        msg = "Cannot cast scalar from %R to %R according to the rule %s";
    }
    PyErr_Format(PyExc_TypeError,
            msg, src_dtype, dst_dtype, npy_casting_to_string(casting));
}


/*NUMPY_API
 * See if array scalars can be cast.
 *
 * TODO: For NumPy 2.0, add a NPY_CASTING parameter.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastScalar(PyTypeObject *from, PyTypeObject *to)
{
    int fromtype;
    int totype;

    fromtype = _typenum_fromtypeobj((PyObject *)from, 0);
    totype = _typenum_fromtypeobj((PyObject *)to, 0);
    if (fromtype == NPY_NOTYPE || totype == NPY_NOTYPE) {
        return NPY_FALSE;
    }
    return (npy_bool) PyArray_CanCastSafely(fromtype, totype);
}

/*
 * Internal promote types function which handles unsigned integers which
 * fit in same-sized signed integers specially.
 */
static PyArray_Descr *
promote_types(PyArray_Descr *type1, PyArray_Descr *type2,
                        int is_small_unsigned1, int is_small_unsigned2)
{
    if (is_small_unsigned1) {
        int type_num1 = type1->type_num;
        int type_num2 = type2->type_num;
        int ret_type_num;

        if (type_num2 < NPY_NTYPES && !(PyTypeNum_ISBOOL(type_num2) ||
                                        PyTypeNum_ISUNSIGNED(type_num2))) {
            /* Convert to the equivalent-sized signed integer */
            type_num1 = type_num_unsigned_to_signed(type_num1);

            ret_type_num = _npy_type_promotion_table[type_num1][type_num2];
            /* The table doesn't handle string/unicode/void, check the result */
            if (ret_type_num >= 0) {
                return PyArray_DescrFromType(ret_type_num);
            }
        }

        return PyArray_PromoteTypes(type1, type2);
    }
    else if (is_small_unsigned2) {
        int type_num1 = type1->type_num;
        int type_num2 = type2->type_num;
        int ret_type_num;

        if (type_num1 < NPY_NTYPES && !(PyTypeNum_ISBOOL(type_num1) ||
                                        PyTypeNum_ISUNSIGNED(type_num1))) {
            /* Convert to the equivalent-sized signed integer */
            type_num2 = type_num_unsigned_to_signed(type_num2);

            ret_type_num = _npy_type_promotion_table[type_num1][type_num2];
            /* The table doesn't handle string/unicode/void, check the result */
            if (ret_type_num >= 0) {
                return PyArray_DescrFromType(ret_type_num);
            }
        }

        return PyArray_PromoteTypes(type1, type2);
    }
    else {
        return PyArray_PromoteTypes(type1, type2);
    }

}


/**
 * This function should possibly become public API eventually.  At this
 * time it is implemented by falling back to `PyArray_AdaptFlexibleDType`.
 * We will use `CastingImpl[from, to].resolve_descriptors(...)` to implement
 * this logic.
 * Before that, the API needs to be reviewed though.
 *
 * WARNING: This function currently does not guarantee that `descr` can
 *          actually be cast to the given DType.
 *
 * @param descr The dtype instance to adapt "cast"
 * @param given_DType The DType class for which we wish to find an instance able
 *        to represent `descr`.
 * @returns Instance of `given_DType`. If `given_DType` is parametric the
 *          descr may be adapted to hold it.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_CastDescrToDType(PyArray_Descr *descr, PyArray_DTypeMeta *given_DType)
{
    if (NPY_DTYPE(descr) == given_DType) {
        Py_INCREF(descr);
        return descr;
    }
    if (!NPY_DT_is_parametric(given_DType)) {
        /*
         * Don't actually do anything, the default is always the result
         * of any cast.
         */
        return NPY_DT_CALL_default_descr(given_DType);
    }
    if (PyObject_TypeCheck((PyObject *)descr, (PyTypeObject *)given_DType)) {
        Py_INCREF(descr);
        return descr;
    }

    PyObject *tmp = PyArray_GetCastingImpl(NPY_DTYPE(descr), given_DType);
    if (tmp == NULL || tmp == Py_None) {
        Py_XDECREF(tmp);
        goto error;
    }
    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(descr), given_DType};
    PyArray_Descr *given_descrs[2] = {descr, NULL};
    PyArray_Descr *loop_descrs[2];

    PyArrayMethodObject *meth = (PyArrayMethodObject *)tmp;
    npy_intp view_offset = NPY_MIN_INTP;
    NPY_CASTING casting = meth->resolve_descriptors(
            meth, dtypes, given_descrs, loop_descrs, &view_offset);
    Py_DECREF(tmp);
    if (casting < 0) {
        goto error;
    }
    Py_DECREF(loop_descrs[0]);
    return loop_descrs[1];

  error:;  /* (; due to compiler limitations) */
    PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
    PyErr_Fetch(&err_type, &err_value, &err_traceback);
    PyErr_Format(PyExc_TypeError,
            "cannot cast dtype %S to %S.", descr, given_DType);
    npy_PyErr_ChainExceptionsCause(err_type, err_value, err_traceback);
    return NULL;
}


/*
 * Helper to find the target descriptor for multiple arrays given an input
 * one that may be a DType class (e.g. "U" or "S").
 * Works with arrays, since that is what `concatenate` works with. However,
 * unlike `np.array(...)` or `arr.astype()` we will never inspect the array's
 * content, which means that object arrays can only be cast to strings if a
 * fixed width is provided (same for string -> generic datetime).
 *
 * As this function uses `PyArray_ExtractDTypeAndDescriptor`, it should
 * eventually be refactored to move the step to an earlier point.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_FindConcatenationDescriptor(
        npy_intp n, PyArrayObject **arrays, PyArray_Descr *requested_dtype)
{
    if (requested_dtype == NULL) {
        return PyArray_ResultType(n, arrays, 0, NULL);
    }

    PyArray_DTypeMeta *common_dtype;
    PyArray_Descr *result = NULL;
    if (PyArray_ExtractDTypeAndDescriptor(
            requested_dtype, &result, &common_dtype) < 0) {
        return NULL;
    }
    if (result != NULL) {
        if (result->subarray != NULL) {
            PyErr_Format(PyExc_TypeError,
                    "The dtype `%R` is not a valid dtype for concatenation "
                    "since it is a subarray dtype (the subarray dimensions "
                    "would be added as array dimensions).", result);
            Py_SETREF(result, NULL);
        }
        goto finish;
    }
    assert(n > 0);  /* concatenate requires at least one array input. */

    /*
     * NOTE: This code duplicates `PyArray_CastToDTypeAndPromoteDescriptors`
     *       to use arrays, copying the descriptors seems not better.
     */
    PyArray_Descr *descr = PyArray_DESCR(arrays[0]);
    result = PyArray_CastDescrToDType(descr, common_dtype);
    if (result == NULL || n == 1) {
        goto finish;
    }
    for (npy_intp i = 1; i < n; i++) {
        descr = PyArray_DESCR(arrays[i]);
        PyArray_Descr *curr = PyArray_CastDescrToDType(descr, common_dtype);
        if (curr == NULL) {
            Py_SETREF(result, NULL);
            goto finish;
        }
        Py_SETREF(result, NPY_DT_SLOTS(common_dtype)->common_instance(result, curr));
        Py_DECREF(curr);
        if (result == NULL) {
            goto finish;
        }
    }

  finish:
    Py_DECREF(common_dtype);
    return result;
}


/*NUMPY_API
 * Produces the smallest size and lowest kind type to which both
 * input types can be cast.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_PromoteTypes(PyArray_Descr *type1, PyArray_Descr *type2)
{
    PyArray_DTypeMeta *common_dtype;
    PyArray_Descr *res;

    /* Fast path for identical inputs (NOTE: This path preserves metadata!) */
    if (type1 == type2
            /*
             * Short-cut for legacy/builtin dtypes except void, since void has
             * no reliable byteorder.  Note: This path preserves metadata!
             */
            && NPY_DT_is_legacy(NPY_DTYPE(type1))
            && PyArray_ISNBO(type1->byteorder) && type1->type_num != NPY_VOID) {
        Py_INCREF(type1);
        return type1;
    }

    common_dtype = PyArray_CommonDType(NPY_DTYPE(type1), NPY_DTYPE(type2));
    if (common_dtype == NULL) {
        return NULL;
    }

    if (!NPY_DT_is_parametric(common_dtype)) {
        /* Note that this path loses all metadata */
        res = NPY_DT_CALL_default_descr(common_dtype);
        Py_DECREF(common_dtype);
        return res;
    }

    /* Cast the input types to the common DType if necessary */
    type1 = PyArray_CastDescrToDType(type1, common_dtype);
    if (type1 == NULL) {
        Py_DECREF(common_dtype);
        return NULL;
    }
    type2 = PyArray_CastDescrToDType(type2, common_dtype);
    if (type2 == NULL) {
        Py_DECREF(type1);
        Py_DECREF(common_dtype);
        return NULL;
    }

    /*
     * And find the common instance of the two inputs
     * NOTE: Common instance preserves metadata (normally and of one input)
     */
    res = NPY_DT_SLOTS(common_dtype)->common_instance(type1, type2);
    Py_DECREF(type1);
    Py_DECREF(type2);
    Py_DECREF(common_dtype);
    return res;
}

/*
 * Produces the smallest size and lowest kind type to which all
 * input types can be cast.
 *
 * Roughly equivalent to functools.reduce(PyArray_PromoteTypes, types)
 * but uses a more complex pairwise approach.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_PromoteTypeSequence(PyArray_Descr **types, npy_intp ntypes)
{
    if (ntypes == 0) {
        PyErr_SetString(PyExc_TypeError, "at least one type needed to promote");
        return NULL;
    }
    return PyArray_ResultType(0, NULL, ntypes, types);
}

/*
 * NOTE: While this is unlikely to be a performance problem, if
 *       it is it could be reverted to a simple positive/negative
 *       check as the previous system used.
 *
 * The is_small_unsigned output flag indicates whether it's an unsigned integer,
 * and would fit in a signed integer of the same bit size.
 */
static int min_scalar_type_num(char *valueptr, int type_num,
                                            int *is_small_unsigned)
{
    switch (type_num) {
        case NPY_BOOL: {
            return NPY_BOOL;
        }
        case NPY_UBYTE: {
            npy_ubyte value = *(npy_ubyte *)valueptr;
            if (value <= NPY_MAX_BYTE) {
                *is_small_unsigned = 1;
            }
            return NPY_UBYTE;
        }
        case NPY_BYTE: {
            npy_byte value = *(npy_byte *)valueptr;
            if (value >= 0) {
                *is_small_unsigned = 1;
                return NPY_UBYTE;
            }
            break;
        }
        case NPY_USHORT: {
            npy_ushort value = *(npy_ushort *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }

            if (value <= NPY_MAX_SHORT) {
                *is_small_unsigned = 1;
            }
            break;
        }
        case NPY_SHORT: {
            npy_short value = *(npy_short *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_USHORT, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_ULONG:
#endif
        case NPY_UINT: {
            npy_uint value = *(npy_uint *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }

            if (value <= NPY_MAX_INT) {
                *is_small_unsigned = 1;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_LONG:
#endif
        case NPY_INT: {
            npy_int value = *(npy_int *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_UINT, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            break;
        }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
        case NPY_ULONG: {
            npy_ulong value = *(npy_ulong *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            else if (value <= NPY_MAX_UINT) {
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }

            if (value <= NPY_MAX_LONG) {
                *is_small_unsigned = 1;
            }
            break;
        }
        case NPY_LONG: {
            npy_long value = *(npy_long *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_ULONG, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
            break;
        }
#endif
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        case NPY_ULONG:
#endif
        case NPY_ULONGLONG: {
            npy_ulonglong value = *(npy_ulonglong *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            else if (value <= NPY_MAX_UINT) {
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            else if (value <= NPY_MAX_ULONG) {
                if (value <= NPY_MAX_LONG) {
                    *is_small_unsigned = 1;
                }
                return NPY_ULONG;
            }
#endif

            if (value <= NPY_MAX_LONGLONG) {
                *is_small_unsigned = 1;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        case NPY_LONG:
#endif
        case NPY_LONGLONG: {
            npy_longlong value = *(npy_longlong *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_ULONGLONG, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            else if (value >= NPY_MIN_LONG) {
                return NPY_LONG;
            }
#endif
            break;
        }
        /*
         * Float types aren't allowed to be demoted to integer types,
         * but precision loss is allowed.
         */
        case NPY_HALF: {
            return NPY_HALF;
        }
        case NPY_FLOAT: {
            float value = *(float *)valueptr;
            if ((value > -65000 && value < 65000) || !npy_isfinite(value)) {
                return NPY_HALF;
            }
            break;
        }
        case NPY_DOUBLE: {
            double value = *(double *)valueptr;
            if ((value > -65000 && value < 65000) || !npy_isfinite(value)) {
                return NPY_HALF;
            }
            else if (value > -3.4e38 && value < 3.4e38) {
                return NPY_FLOAT;
            }
            break;
        }
        case NPY_LONGDOUBLE: {
            npy_longdouble value = *(npy_longdouble *)valueptr;
            if ((value > -65000 && value < 65000) || !npy_isfinite(value)) {
                return NPY_HALF;
            }
            else if (value > -3.4e38 && value < 3.4e38) {
                return NPY_FLOAT;
            }
            else if (value > -1.7e308 && value < 1.7e308) {
                return NPY_DOUBLE;
            }
            break;
        }
        /*
         * The code to demote complex to float is disabled for now,
         * as forcing complex by adding 0j is probably desirable.
         */
        case NPY_CFLOAT: {
            /*
            npy_cfloat value = *(npy_cfloat *)valueptr;
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_FLOAT, is_small_unsigned);
            }
            */
            break;
        }
        case NPY_CDOUBLE: {
            npy_cdouble value = *(npy_cdouble *)valueptr;
            /*
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_DOUBLE, is_small_unsigned);
            }
            */
            if (value.real > -3.4e38 && value.real < 3.4e38 &&
                     value.imag > -3.4e38 && value.imag < 3.4e38) {
                return NPY_CFLOAT;
            }
            break;
        }
        case NPY_CLONGDOUBLE: {
            npy_clongdouble value = *(npy_clongdouble *)valueptr;
            /*
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_LONGDOUBLE, is_small_unsigned);
            }
            */
            if (value.real > -3.4e38 && value.real < 3.4e38 &&
                     value.imag > -3.4e38 && value.imag < 3.4e38) {
                return NPY_CFLOAT;
            }
            else if (value.real > -1.7e308 && value.real < 1.7e308 &&
                     value.imag > -1.7e308 && value.imag < 1.7e308) {
                return NPY_CDOUBLE;
            }
            break;
        }
    }

    return type_num;
}


NPY_NO_EXPORT PyArray_Descr *
PyArray_MinScalarType_internal(PyArrayObject *arr, int *is_small_unsigned)
{
    PyArray_Descr *dtype = PyArray_DESCR(arr);
    *is_small_unsigned = 0;
    /*
     * If the array isn't a numeric scalar, just return the array's dtype.
     */
    if (PyArray_NDIM(arr) > 0 || !PyTypeNum_ISNUMBER(dtype->type_num)) {
        Py_INCREF(dtype);
        return dtype;
    }
    else {
        char *data = PyArray_BYTES(arr);
        int swap = !PyArray_ISNBO(dtype->byteorder);
        /* An aligned memory buffer large enough to hold any type */
        npy_longlong value[4];
        dtype->f->copyswap(&value, data, swap, NULL);

        return PyArray_DescrFromType(
                        min_scalar_type_num((char *)&value,
                                dtype->type_num, is_small_unsigned));

    }
}

/*NUMPY_API
 * If arr is a scalar (has 0 dimensions) with a built-in number data type,
 * finds the smallest type size/kind which can still represent its data.
 * Otherwise, returns the array's data type.
 *
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_MinScalarType(PyArrayObject *arr)
{
    int is_small_unsigned;
    return PyArray_MinScalarType_internal(arr, &is_small_unsigned);
}

/*
 * Provides an ordering for the dtype 'kind' character codes, to help
 * determine when to use the min_scalar_type function. This groups
 * 'kind' into boolean, integer, floating point, and everything else.
 */
static int
dtype_kind_to_simplified_ordering(char kind)
{
    switch (kind) {
        /* Boolean kind */
        case 'b':
            return 0;
        /* Unsigned int kind */
        case 'u':
        /* Signed int kind */
        case 'i':
            return 1;
        /* Float kind */
        case 'f':
        /* Complex kind */
        case 'c':
            return 2;
        /* Anything else */
        default:
            return 3;
    }
}


/*
 * Determine if there is a mix of scalars and arrays/dtypes.
 * If this is the case, the scalars should be handled as the minimum type
 * capable of holding the value when the maximum "category" of the scalars
 * surpasses the maximum "category" of the arrays/dtypes.
 * If the scalars are of a lower or same category as the arrays, they may be
 * demoted to a lower type within their category (the lowest type they can
 * be cast to safely according to scalar casting rules).
 *
 * If any new style dtype is involved (non-legacy), always returns 0.
 */
NPY_NO_EXPORT int
should_use_min_scalar(npy_intp narrs, PyArrayObject **arr,
                      npy_intp ndtypes, PyArray_Descr **dtypes)
{
    int use_min_scalar = 0;

    if (narrs > 0) {
        int all_scalars;
        int max_scalar_kind = -1;
        int max_array_kind = -1;

        all_scalars = (ndtypes > 0) ? 0 : 1;

        /* Compute the maximum "kinds" and whether everything is scalar */
        for (npy_intp i = 0; i < narrs; ++i) {
            if (!NPY_DT_is_legacy(NPY_DTYPE(PyArray_DESCR(arr[i])))) {
                return 0;
            }
            if (PyArray_NDIM(arr[i]) == 0) {
                int kind = dtype_kind_to_simplified_ordering(
                                    PyArray_DESCR(arr[i])->kind);
                if (kind > max_scalar_kind) {
                    max_scalar_kind = kind;
                }
            }
            else {
                int kind = dtype_kind_to_simplified_ordering(
                                    PyArray_DESCR(arr[i])->kind);
                if (kind > max_array_kind) {
                    max_array_kind = kind;
                }
                all_scalars = 0;
            }
        }
        /*
         * If the max scalar kind is bigger than the max array kind,
         * finish computing the max array kind
         */
        for (npy_intp i = 0; i < ndtypes; ++i) {
            if (!NPY_DT_is_legacy(NPY_DTYPE(dtypes[i]))) {
                return 0;
            }
            int kind = dtype_kind_to_simplified_ordering(dtypes[i]->kind);
            if (kind > max_array_kind) {
                max_array_kind = kind;
            }
        }

        /* Indicate whether to use the min_scalar_type function */
        if (!all_scalars && max_array_kind >= max_scalar_kind) {
            use_min_scalar = 1;
        }
    }
    return use_min_scalar;
}


NPY_NO_EXPORT int
should_use_min_scalar_weak_literals(int narrs, PyArrayObject **arr) {
    int all_scalars = 1;
    int max_scalar_kind = -1;
    int max_array_kind = -1;

    for (int i = 0; i < narrs; i++) {
        if (PyArray_FLAGS(arr[i]) & NPY_ARRAY_WAS_PYTHON_INT) {
            /* A Python integer could be `u` so is effectively that: */
            int new = dtype_kind_to_simplified_ordering('u');
            if (new > max_scalar_kind) {
                max_scalar_kind = new;
            }
        }
        /* For the new logic, only complex or not matters: */
        else if (PyArray_FLAGS(arr[i]) & NPY_ARRAY_WAS_PYTHON_FLOAT) {
            max_scalar_kind = dtype_kind_to_simplified_ordering('f');
        }
        else if (PyArray_FLAGS(arr[i]) & NPY_ARRAY_WAS_PYTHON_COMPLEX) {
            max_scalar_kind = dtype_kind_to_simplified_ordering('f');
        }
        else {
            all_scalars = 0;
            int kind = dtype_kind_to_simplified_ordering(
                    PyArray_DESCR(arr[i])->kind);
            if (kind > max_array_kind) {
                max_array_kind = kind;
            }
        }
    }
    if (!all_scalars && max_array_kind >= max_scalar_kind) {
        return 1;
    }

    return 0;
}


/*NUMPY_API
 *
 * Produces the result type of a bunch of inputs, using the same rules
 * as `np.result_type`.
 *
 * NOTE: This function is expected to through a transitional period or
 *       change behaviour.  DTypes should always be strictly enforced for
 *       0-D arrays, while "weak DTypes" will be used to represent Python
 *       integers, floats, and complex in all cases.
 *       (Within this function, these are currently flagged on the array
 *       object to work through `np.result_type`, this may change.)
 *
 *       Until a time where this transition is complete, we probably cannot
 *       add new "weak DTypes" or allow users to create their own.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_ResultType(
        npy_intp narrs, PyArrayObject *arrs[],
        npy_intp ndtypes, PyArray_Descr *descrs[])
{
    PyArray_Descr *result = NULL;

    if (narrs + ndtypes <= 1) {
        /* If the input is a single value, skip promotion. */
        if (narrs == 1) {
            result = PyArray_DTYPE(arrs[0]);
        }
        else if (ndtypes == 1) {
            result = descrs[0];
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "no arrays or types available to calculate result type");
            return NULL;
        }
        return NPY_DT_CALL_ensure_canonical(result);
    }

    void **info_on_heap = NULL;
    void *_info_on_stack[NPY_MAXARGS * 2];
    PyArray_DTypeMeta **all_DTypes;
    PyArray_Descr **all_descriptors;

    if (narrs + ndtypes > NPY_MAXARGS) {
        info_on_heap = PyMem_Malloc(2 * (narrs+ndtypes) * sizeof(PyObject *));
        if (info_on_heap == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        all_DTypes = (PyArray_DTypeMeta **)info_on_heap;
        all_descriptors = (PyArray_Descr **)(info_on_heap + narrs + ndtypes);
    }
    else {
        all_DTypes = (PyArray_DTypeMeta **)_info_on_stack;
        all_descriptors = (PyArray_Descr **)(_info_on_stack + narrs + ndtypes);
    }

    /* Copy all dtypes into a single array defining non-value-based behaviour */
    for (npy_intp i=0; i < ndtypes; i++) {
        all_DTypes[i] = NPY_DTYPE(descrs[i]);
        Py_INCREF(all_DTypes[i]);
        all_descriptors[i] = descrs[i];
    }

    int at_least_one_scalar = 0;
    int all_pyscalar = ndtypes == 0;
    for (npy_intp i=0, i_all=ndtypes; i < narrs; i++, i_all++) {
        /* Array descr is also the correct "default" for scalars: */
        if (PyArray_NDIM(arrs[i]) == 0) {
            at_least_one_scalar = 1;
        }

        /*
         * If the original was a Python scalar/literal, we use only the
         * corresponding abstract DType (and no descriptor) below.
         * Otherwise, we propagate the descriptor as well.
         */
        all_descriptors[i_all] = NULL;  /* no descriptor for py-scalars */
        if (PyArray_FLAGS(arrs[i]) & NPY_ARRAY_WAS_PYTHON_INT) {
            /* This could even be an object dtype here for large ints */
            all_DTypes[i_all] = &PyArray_PyIntAbstractDType;
            if (PyArray_TYPE(arrs[i]) != NPY_LONG) {
                /* Not a "normal" scalar, so we cannot avoid the legacy path */
                all_pyscalar = 0;
            }
        }
        else if (PyArray_FLAGS(arrs[i]) & NPY_ARRAY_WAS_PYTHON_FLOAT) {
            all_DTypes[i_all] = &PyArray_PyFloatAbstractDType;
        }
        else if (PyArray_FLAGS(arrs[i]) & NPY_ARRAY_WAS_PYTHON_COMPLEX) {
            all_DTypes[i_all] = &PyArray_PyComplexAbstractDType;
        }
        else {
            all_descriptors[i_all] = PyArray_DTYPE(arrs[i]);
            all_DTypes[i_all] = NPY_DTYPE(all_descriptors[i_all]);
            all_pyscalar = 0;
        }
        Py_INCREF(all_DTypes[i_all]);
    }

    PyArray_DTypeMeta *common_dtype = PyArray_PromoteDTypeSequence(
            narrs+ndtypes, all_DTypes);
    for (npy_intp i=0; i < narrs+ndtypes; i++) {
        Py_DECREF(all_DTypes[i]);
    }
    if (common_dtype == NULL) {
        goto error;
    }

    if (NPY_DT_is_abstract(common_dtype)) {
        /* (ab)use default descriptor to define a default */
        PyArray_Descr *tmp_descr = NPY_DT_CALL_default_descr(common_dtype);
        if (tmp_descr == NULL) {
            goto error;
        }
        Py_INCREF(NPY_DTYPE(tmp_descr));
        Py_SETREF(common_dtype, NPY_DTYPE(tmp_descr));
        Py_DECREF(tmp_descr);
    }

    /*
     * NOTE: Code duplicates `PyArray_CastToDTypeAndPromoteDescriptors`, but
     *       supports special handling of the abstract values.
     */
    if (NPY_DT_is_parametric(common_dtype)) {
        for (npy_intp i = 0; i < ndtypes+narrs; i++) {
            if (all_descriptors[i] == NULL) {
                continue;  /* originally a python scalar/literal */
            }
            PyArray_Descr *curr = PyArray_CastDescrToDType(
                    all_descriptors[i], common_dtype);
            if (curr == NULL) {
                goto error;
            }
            if (result == NULL) {
                result = curr;
                continue;
            }
            Py_SETREF(result, NPY_DT_SLOTS(common_dtype)->common_instance(result, curr));
            Py_DECREF(curr);
            if (result == NULL) {
                goto error;
            }
        }
    }
    if (result == NULL) {
        /*
         * If the DType is not parametric, or all were weak scalars,
         * a result may not yet be set.
         */
        result = NPY_DT_CALL_default_descr(common_dtype);
        if (result == NULL) {
            goto error;
        }
    }

    /*
     * Unfortunately, when 0-D "scalar" arrays are involved and mixed, we *may*
     * have to use the value-based logic.
     * `PyArray_CheckLegacyResultType` may behave differently based on the
     * current value of `npy_legacy_promotion`:
     * 1. It does nothing (we use the "new" behavior)
     * 2. It does nothing, but warns if there the result would differ.
     * 3. It replaces the result based on the legacy value-based logic.
     */
    if (at_least_one_scalar && !all_pyscalar && result->type_num < NPY_NTYPES) {
        if (PyArray_CheckLegacyResultType(
                &result, narrs, arrs, ndtypes, descrs) < 0) {
            Py_DECREF(common_dtype);
            Py_DECREF(result);
            return NULL;
        }
    }

    Py_DECREF(common_dtype);
    PyMem_Free(info_on_heap);
    return result;

  error:
    Py_XDECREF(result);
    Py_XDECREF(common_dtype);
    PyMem_Free(info_on_heap);
    return NULL;
}


/*
 * Produces the result type of a bunch of inputs, using the UFunc
 * type promotion rules. Use this function when you have a set of
 * input arrays, and need to determine an output array dtype.
 *
 * If all the inputs are scalars (have 0 dimensions) or the maximum "kind"
 * of the scalars is greater than the maximum "kind" of the arrays, does
 * a regular type promotion.
 *
 * Otherwise, does a type promotion on the MinScalarType
 * of all the inputs.  Data types passed directly are treated as array
 * types.
 */
NPY_NO_EXPORT int
PyArray_CheckLegacyResultType(
        PyArray_Descr **new_result,
        npy_intp narrs, PyArrayObject **arr,
        npy_intp ndtypes, PyArray_Descr **dtypes)
{
    PyArray_Descr *ret = NULL;
    if (npy_promotion_state == NPY_USE_WEAK_PROMOTION) {
        return 0;
    }
    if (npy_promotion_state == NPY_USE_WEAK_PROMOTION_AND_WARN
            && !npy_give_promotion_warnings()) {
        return 0;
    }

    npy_intp i;

    /* If there's just one type, results must match */
    if (narrs + ndtypes == 1) {
        return 0;
    }

    int use_min_scalar = should_use_min_scalar(narrs, arr, ndtypes, dtypes);

    /* Loop through all the types, promoting them */
    if (!use_min_scalar) {

        /* Build a single array of all the dtypes */
        PyArray_Descr **all_dtypes = PyArray_malloc(
            sizeof(*all_dtypes) * (narrs + ndtypes));
        if (all_dtypes == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < narrs; ++i) {
            all_dtypes[i] = PyArray_DESCR(arr[i]);
        }
        for (i = 0; i < ndtypes; ++i) {
            all_dtypes[narrs + i] = dtypes[i];
        }
        ret = PyArray_PromoteTypeSequence(all_dtypes, narrs + ndtypes);
        PyArray_free(all_dtypes);
    }
    else {
        int ret_is_small_unsigned = 0;

        for (i = 0; i < narrs; ++i) {
            int tmp_is_small_unsigned;
            PyArray_Descr *tmp = PyArray_MinScalarType_internal(
                arr[i], &tmp_is_small_unsigned);
            if (tmp == NULL) {
                Py_XDECREF(ret);
                return -1;
            }
            /* Combine it with the existing type */
            if (ret == NULL) {
                ret = tmp;
                ret_is_small_unsigned = tmp_is_small_unsigned;
            }
            else {
                PyArray_Descr *tmpret = promote_types(
                    tmp, ret, tmp_is_small_unsigned, ret_is_small_unsigned);
                Py_DECREF(tmp);
                Py_DECREF(ret);
                ret = tmpret;
                if (ret == NULL) {
                    return -1;
                }

                ret_is_small_unsigned = tmp_is_small_unsigned &&
                                        ret_is_small_unsigned;
            }
        }

        for (i = 0; i < ndtypes; ++i) {
            PyArray_Descr *tmp = dtypes[i];
            /* Combine it with the existing type */
            if (ret == NULL) {
                ret = tmp;
                Py_INCREF(ret);
            }
            else {
                PyArray_Descr *tmpret = promote_types(
                    tmp, ret, 0, ret_is_small_unsigned);
                Py_DECREF(ret);
                ret = tmpret;
                if (ret == NULL) {
                    return -1;
                }
            }
        }
        /* None of the above loops ran */
        if (ret == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "no arrays or types available to calculate result type");
        }
    }

    if (ret == NULL) {
        return -1;
    }

    int unchanged_result = PyArray_EquivTypes(*new_result, ret);
    if (unchanged_result) {
        Py_DECREF(ret);
        return 0;
    }
    if (npy_promotion_state == NPY_USE_LEGACY_PROMOTION) {
        Py_SETREF(*new_result, ret);
        return 0;
    }

    assert(npy_promotion_state == NPY_USE_WEAK_PROMOTION_AND_WARN);
    if (PyErr_WarnFormat(PyExc_UserWarning, 1,
            "result dtype changed due to the removal of value-based "
            "promotion from NumPy. Changed from %S to %S.",
            ret, *new_result) < 0) {
        Py_DECREF(ret);
        return -1;
    }
    Py_DECREF(ret);
    return 0;
}

/**
 * Promotion of descriptors (of arbitrary DType) to their correctly
 * promoted instances of the given DType.
 * I.e. the given DType could be a string, which then finds the correct
 * string length, given all `descrs`.
 *
 * @param ndescrs number of descriptors to cast and find the common instance.
 *        At least one must be passed in.
 * @param descrs The descriptors to work with.
 * @param DType The DType of the desired output descriptor.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_CastToDTypeAndPromoteDescriptors(
        npy_intp ndescr, PyArray_Descr *descrs[], PyArray_DTypeMeta *DType)
{
    assert(ndescr > 0);

    PyArray_Descr *result = PyArray_CastDescrToDType(descrs[0], DType);
    if (result == NULL || ndescr == 1) {
        return result;
    }
    if (!NPY_DT_is_parametric(DType)) {
        /* Note that this "fast" path loses all metadata */
        Py_DECREF(result);
        return NPY_DT_CALL_default_descr(DType);
    }

    for (npy_intp i = 1; i < ndescr; i++) {
        PyArray_Descr *curr = PyArray_CastDescrToDType(descrs[i], DType);
        if (curr == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        Py_SETREF(result, NPY_DT_SLOTS(DType)->common_instance(result, curr));
        Py_DECREF(curr);
        if (result == NULL) {
            return NULL;
        }
    }
    return result;
}


/*NUMPY_API
 * Is the typenum valid?
 */
NPY_NO_EXPORT int
PyArray_ValidType(int type)
{
    PyArray_Descr *descr;
    int res=NPY_TRUE;

    descr = PyArray_DescrFromType(type);
    if (descr == NULL) {
        res = NPY_FALSE;
    }
    Py_DECREF(descr);
    return res;
}

/* Backward compatibility only */
/* In both Zero and One

***You must free the memory once you are done with it
using PyDataMem_FREE(ptr) or you create a memory leak***

If arr is an Object array you are getting a
BORROWED reference to Zero or One.
Do not DECREF.
Please INCREF if you will be hanging on to it.

The memory for the ptr still must be freed in any case;
*/

static int
_check_object_rec(PyArray_Descr *descr)
{
    if (PyDataType_HASFIELDS(descr) && PyDataType_REFCHK(descr)) {
        PyErr_SetString(PyExc_TypeError, "Not supported for this data-type.");
        return -1;
    }
    return 0;
}

/*NUMPY_API
  Get pointer to zero of correct type for array.
*/
NPY_NO_EXPORT char *
PyArray_Zero(PyArrayObject *arr)
{
    char *zeroval;
    int ret, storeflags;
    static PyObject * zero_obj = NULL;

    if (_check_object_rec(PyArray_DESCR(arr)) < 0) {
        return NULL;
    }
    zeroval = PyDataMem_NEW(PyArray_DESCR(arr)->elsize);
    if (zeroval == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    if (zero_obj == NULL) {
        zero_obj = PyLong_FromLong((long) 0);
        if (zero_obj == NULL) {
            return NULL;
        }
    }
    if (PyArray_ISOBJECT(arr)) {
        /* XXX this is dangerous, the caller probably is not
           aware that zeroval is actually a static PyObject*
           In the best case they will only use it as-is, but
           if they simply memcpy it into a ndarray without using
           setitem(), refcount errors will occur
        */
        memcpy(zeroval, &zero_obj, sizeof(PyObject *));
        return zeroval;
    }
    storeflags = PyArray_FLAGS(arr);
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_BEHAVED);
    ret = PyArray_SETITEM(arr, zeroval, zero_obj);
    ((PyArrayObject_fields *)arr)->flags = storeflags;
    if (ret < 0) {
        PyDataMem_FREE(zeroval);
        return NULL;
    }
    return zeroval;
}

/*NUMPY_API
  Get pointer to one of correct type for array
*/
NPY_NO_EXPORT char *
PyArray_One(PyArrayObject *arr)
{
    char *oneval;
    int ret, storeflags;
    static PyObject * one_obj = NULL;

    if (_check_object_rec(PyArray_DESCR(arr)) < 0) {
        return NULL;
    }
    oneval = PyDataMem_NEW(PyArray_DESCR(arr)->elsize);
    if (oneval == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    if (one_obj == NULL) {
        one_obj = PyLong_FromLong((long) 1);
        if (one_obj == NULL) {
            return NULL;
        }
    }
    if (PyArray_ISOBJECT(arr)) {
        /* XXX this is dangerous, the caller probably is not
           aware that oneval is actually a static PyObject*
           In the best case they will only use it as-is, but
           if they simply memcpy it into a ndarray without using
           setitem(), refcount errors will occur
        */
        memcpy(oneval, &one_obj, sizeof(PyObject *));
        return oneval;
    }

    storeflags = PyArray_FLAGS(arr);
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_BEHAVED);
    ret = PyArray_SETITEM(arr, oneval, one_obj);
    ((PyArrayObject_fields *)arr)->flags = storeflags;
    if (ret < 0) {
        PyDataMem_FREE(oneval);
        return NULL;
    }
    return oneval;
}

/* End deprecated */

/*NUMPY_API
 * Return the typecode of the array a Python object would be converted to
 *
 * Returns the type number the result should have, or NPY_NOTYPE on error.
 */
NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type)
{
    PyArray_Descr *dtype = NULL;
    int ret;

    if (minimum_type != NPY_NOTYPE && minimum_type >= 0) {
        dtype = PyArray_DescrFromType(minimum_type);
        if (dtype == NULL) {
            return NPY_NOTYPE;
        }
    }
    if (PyArray_DTypeFromObject(op, NPY_MAXDIMS, &dtype) < 0) {
        return NPY_NOTYPE;
    }

    if (dtype == NULL) {
        ret = NPY_DEFAULT_TYPE;
    }
    else if (!NPY_DT_is_legacy(NPY_DTYPE(dtype))) {
        /*
         * TODO: If we keep all type number style API working, by defining
         *       type numbers always. We may be able to allow this again.
         */
        PyErr_Format(PyExc_TypeError,
                "This function currently only supports native NumPy dtypes "
                "and old-style user dtypes, but the dtype was %S.\n"
                "(The function may need to be updated to support arbitrary"
                "user dtypes.)",
                dtype);
        ret = NPY_NOTYPE;
    }
    else {
        ret = dtype->type_num;
    }

    Py_XDECREF(dtype);

    return ret;
}

/* Raises error when len(op) == 0 */

/*NUMPY_API
 *
 * This function is only used in one place within NumPy and should
 * generally be avoided. It is provided mainly for backward compatibility.
 *
 * The user of the function has to free the returned array with PyDataMem_FREE.
 */
NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn)
{
    int i, n;
    PyArray_Descr *common_descr = NULL;
    PyArrayObject **mps = NULL;

    *retn = n = PySequence_Length(op);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "0-length sequence.");
    }
    if (PyErr_Occurred()) {
        *retn = 0;
        return NULL;
    }
    mps = (PyArrayObject **)PyDataMem_NEW(n*sizeof(PyArrayObject *));
    if (mps == NULL) {
        *retn = 0;
        return (void*)PyErr_NoMemory();
    }

    if (PyArray_Check(op)) {
        for (i = 0; i < n; i++) {
            mps[i] = (PyArrayObject *) array_item_asarray((PyArrayObject *)op, i);
        }
        if (!PyArray_ISCARRAY((PyArrayObject *)op)) {
            for (i = 0; i < n; i++) {
                PyObject *obj;
                obj = PyArray_NewCopy(mps[i], NPY_CORDER);
                Py_DECREF(mps[i]);
                mps[i] = (PyArrayObject *)obj;
            }
        }
        return mps;
    }

    for (i = 0; i < n; i++) {
        mps[i] = NULL;
    }

    for (i = 0; i < n; i++) {
        /* Convert everything to an array, this could be optimized away */
        PyObject *tmp = PySequence_GetItem(op, i);
        if (tmp == NULL) {
            goto fail;
        }

        mps[i] = (PyArrayObject *)PyArray_FROM_O(tmp);
        if (mps[i] == NULL) {
            Py_DECREF(tmp);
            goto fail;
        }
        npy_mark_tmp_array_if_pyscalar(tmp, mps[i], NULL);
        Py_DECREF(tmp);
    }

    common_descr = PyArray_ResultType(n, mps, 0, NULL);
    if (common_descr == NULL) {
        goto fail;
    }

    /* Make sure all arrays are contiguous and have the correct dtype. */
    for (i = 0; i < n; i++) {
        int flags = NPY_ARRAY_CARRAY;
        PyArrayObject *tmp = mps[i];

        Py_INCREF(common_descr);
        mps[i] = (PyArrayObject *)PyArray_FromArray(tmp, common_descr, flags);
        Py_DECREF(tmp);
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    Py_DECREF(common_descr);
    return mps;

 fail:
    Py_XDECREF(common_descr);
    *retn = 0;
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return NULL;
}


/**
 * Private function to add a casting implementation by unwrapping a bound
 * array method.
 *
 * @param meth
 * @return 0 on success -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AddCastingImplementation(PyBoundArrayMethodObject *meth)
{
    if (meth->method->nin != 1 || meth->method->nout != 1) {
        PyErr_SetString(PyExc_TypeError,
                "A cast must have one input and one output.");
        return -1;
    }
    if (meth->dtypes[0] == meth->dtypes[1]) {
        /*
         * The method casting between instances of the same dtype is special,
         * since it is common, it is stored explicitly (currently) and must
         * obey additional constraints to ensure convenient casting.
         */
        if (!(meth->method->flags & NPY_METH_SUPPORTS_UNALIGNED)) {
            PyErr_Format(PyExc_TypeError,
                    "A cast where input and output DType (class) are identical "
                    "must currently support unaligned data. (method: %s)",
                    meth->method->name);
            return -1;
        }
        if (NPY_DT_SLOTS(meth->dtypes[0])->within_dtype_castingimpl != NULL) {
            PyErr_Format(PyExc_RuntimeError,
                    "A cast was already added for %S -> %S. (method: %s)",
                    meth->dtypes[0], meth->dtypes[1], meth->method->name);
            return -1;
        }
        Py_INCREF(meth->method);
        NPY_DT_SLOTS(meth->dtypes[0])->within_dtype_castingimpl = meth->method;

        return 0;
    }
    if (PyDict_Contains(NPY_DT_SLOTS(meth->dtypes[0])->castingimpls,
            (PyObject *)meth->dtypes[1])) {
        PyErr_Format(PyExc_RuntimeError,
                "A cast was already added for %S -> %S. (method: %s)",
                meth->dtypes[0], meth->dtypes[1], meth->method->name);
        return -1;
    }
    if (PyDict_SetItem(NPY_DT_SLOTS(meth->dtypes[0])->castingimpls,
            (PyObject *)meth->dtypes[1], (PyObject *)meth->method) < 0) {
        return -1;
    }
    return 0;
}

/**
 * Add a new casting implementation using a PyArrayMethod_Spec.
 *
 * @param spec
 * @param private If private, allow slots not publicly exposed.
 * @return 0 on success -1 on failure
 */
NPY_NO_EXPORT int
PyArray_AddCastingImplementation_FromSpec(PyArrayMethod_Spec *spec, int private)
{
    /* Create a bound method, unbind and store it */
    PyBoundArrayMethodObject *meth = PyArrayMethod_FromSpec_int(spec, private);
    if (meth == NULL) {
        return -1;
    }
    int res = PyArray_AddCastingImplementation(meth);
    Py_DECREF(meth);
    if (res < 0) {
        return -1;
    }
    return 0;
}


NPY_NO_EXPORT NPY_CASTING
legacy_same_dtype_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(loop_descrs[0]);
        if (loop_descrs[1] == NULL) {
            Py_DECREF(loop_descrs[0]);
            return -1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    /* this function only makes sense for non-flexible legacy dtypes: */
    assert(loop_descrs[0]->elsize == loop_descrs[1]->elsize);

    /*
     * Legacy dtypes (except datetime) only have byte-order and elsize as
     * storage parameters.
     */
    if (PyDataType_ISNOTSWAPPED(loop_descrs[0]) ==
                PyDataType_ISNOTSWAPPED(loop_descrs[1])) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }
    return NPY_EQUIV_CASTING;
}


NPY_NO_EXPORT int
legacy_cast_get_strided_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArray_Descr **descrs = context->descriptors;
    int out_needs_api = 0;

    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;

    if (get_wrapped_legacy_cast_function(
            aligned, strides[0], strides[1], descrs[0], descrs[1],
            move_references, out_loop, out_transferdata, &out_needs_api, 0) < 0) {
        return -1;
    }
    if (!out_needs_api) {
        *flags &= ~NPY_METH_REQUIRES_PYAPI;
    }
    return 0;
}


/*
 * Simple dtype resolver for casting between two different (non-parametric)
 * (legacy) dtypes.
 */
NPY_NO_EXPORT NPY_CASTING
simple_cast_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    assert(NPY_DT_is_legacy(dtypes[0]) && NPY_DT_is_legacy(dtypes[1]));

    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return -1;
    }
    if (given_descrs[1] != NULL) {
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
        if (loop_descrs[1] == NULL) {
            Py_DECREF(loop_descrs[0]);
            return -1;
        }
    }
    else {
        loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
    }

    if (self->casting != NPY_NO_CASTING) {
        return self->casting;
    }
    if (PyDataType_ISNOTSWAPPED(loop_descrs[0]) ==
            PyDataType_ISNOTSWAPPED(loop_descrs[1])) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }
    return NPY_EQUIV_CASTING;
}


NPY_NO_EXPORT int
get_byteswap_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArray_Descr **descrs = context->descriptors;
    assert(descrs[0]->kind == descrs[1]->kind);
    assert(descrs[0]->elsize == descrs[1]->elsize);
    int itemsize = descrs[0]->elsize;
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_transferdata = NULL;
    if (descrs[0]->kind == 'c') {
        /*
         * TODO: we have an issue with complex, since the below loops
         *       use the itemsize, the complex alignment would be too small.
         *       Using aligned = 0, might cause slow downs in some cases.
         */
        aligned = 0;
    }

    if (PyDataType_ISNOTSWAPPED(descrs[0]) ==
            PyDataType_ISNOTSWAPPED(descrs[1])) {
        *out_loop = PyArray_GetStridedCopyFn(
                aligned, strides[0], strides[1], itemsize);
    }
    else if (!PyTypeNum_ISCOMPLEX(descrs[0]->type_num)) {
        *out_loop = PyArray_GetStridedCopySwapFn(
                aligned, strides[0], strides[1], itemsize);
    }
    else {
        *out_loop = PyArray_GetStridedCopySwapPairFn(
                aligned, strides[0], strides[1], itemsize);
    }
    if (*out_loop == NULL) {
        return -1;
    }
    return 0;
}


NPY_NO_EXPORT int
complex_to_noncomplex_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    static PyObject *cls = NULL;
    int ret;
    npy_cache_import("numpy.exceptions", "ComplexWarning", &cls);
    if (cls == NULL) {
        return -1;
    }
    ret = PyErr_WarnEx(cls,
            "Casting complex values to real discards "
            "the imaginary part", 1);
    if (ret < 0) {
        return -1;
    }
    return npy_default_get_strided_loop(
            context, aligned, move_references, strides,
            out_loop, out_transferdata, flags);
}


static int
add_numeric_cast(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to)
{
    PyType_Slot slots[7];
    PyArray_DTypeMeta *dtypes[2] = {from, to};
    PyArrayMethod_Spec spec = {
            .name = "numeric_cast",
            .nin = 1,
            .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    npy_intp from_itemsize = from->singleton->elsize;
    npy_intp to_itemsize = to->singleton->elsize;

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &simple_cast_resolve_descriptors;
    /* Fetch the optimized loops (2<<10 is a non-contiguous stride) */
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = PyArray_GetStridedNumericCastFn(
            1, 2<<10, 2<<10, from->type_num, to->type_num);
    slots[2].slot = NPY_METH_contiguous_loop;
    slots[2].pfunc = PyArray_GetStridedNumericCastFn(
            1, from_itemsize, to_itemsize, from->type_num, to->type_num);
    slots[3].slot = NPY_METH_unaligned_strided_loop;
    slots[3].pfunc = PyArray_GetStridedNumericCastFn(
            0, 2<<10, 2<<10, from->type_num, to->type_num);
    slots[4].slot = NPY_METH_unaligned_contiguous_loop;
    slots[4].pfunc = PyArray_GetStridedNumericCastFn(
            0, from_itemsize, to_itemsize, from->type_num, to->type_num);
    if (PyTypeNum_ISCOMPLEX(from->type_num) &&
            !PyTypeNum_ISCOMPLEX(to->type_num) &&
            !PyTypeNum_ISBOOL(to->type_num)) {
        /*
         * The get_loop function must also give a ComplexWarning. We could
         * consider moving this warning into the inner-loop at some point
         * for simplicity (this requires ensuring it is only emitted once).
         */
        slots[5].slot = _NPY_METH_get_loop;
        slots[5].pfunc = &complex_to_noncomplex_get_loop;
        slots[6].slot = 0;
        slots[6].pfunc = NULL;
    }
    else {
        /* Use the default get loop function. */
        slots[5].slot = 0;
        slots[5].pfunc = NULL;
    }

    assert(slots[1].pfunc && slots[2].pfunc && slots[3].pfunc && slots[4].pfunc);

    /* Find the correct casting level, and special case no-cast */
    if (dtypes[0]->singleton->kind == dtypes[1]->singleton->kind
            && from_itemsize == to_itemsize) {
        spec.casting = NPY_EQUIV_CASTING;

        /* When there is no casting (equivalent C-types) use byteswap loops */
        slots[0].slot = NPY_METH_resolve_descriptors;
        slots[0].pfunc = &legacy_same_dtype_resolve_descriptors;
        slots[1].slot = _NPY_METH_get_loop;
        slots[1].pfunc = &get_byteswap_loop;
        slots[2].slot = 0;
        slots[2].pfunc = NULL;

        spec.name = "numeric_copy_or_byteswap";
        spec.flags |= NPY_METH_NO_FLOATINGPOINT_ERRORS;
    }
    else if (_npy_can_cast_safely_table[from->type_num][to->type_num]) {
        spec.casting = NPY_SAFE_CASTING;
    }
    else if (dtype_kind_to_ordering(dtypes[0]->singleton->kind) <=
             dtype_kind_to_ordering(dtypes[1]->singleton->kind)) {
        spec.casting = NPY_SAME_KIND_CASTING;
    }
    else {
        spec.casting = NPY_UNSAFE_CASTING;
    }

    /* Create a bound method, unbind and store it */
    return PyArray_AddCastingImplementation_FromSpec(&spec, 1);
}


/*
 * This registers the castingimpl for all casts between numeric types.
 * Eventually, this function should likely be defined as part of a .c.src
 * file to remove `PyArray_GetStridedNumericCastFn` entirely.
 */
static int
PyArray_InitializeNumericCasts(void)
{
    for (int from = 0; from < NPY_NTYPES; from++) {
        if (!PyTypeNum_ISNUMBER(from) && from != NPY_BOOL) {
            continue;
        }
        PyArray_DTypeMeta *from_dt = PyArray_DTypeFromTypeNum(from);

        for (int to = 0; to < NPY_NTYPES; to++) {
            if (!PyTypeNum_ISNUMBER(to) && to != NPY_BOOL) {
                continue;
            }
            PyArray_DTypeMeta *to_dt = PyArray_DTypeFromTypeNum(to);
            int res = add_numeric_cast(from_dt, to_dt);
            Py_DECREF(to_dt);
            if (res < 0) {
                Py_DECREF(from_dt);
                return -1;
            }
        }
    }
    return 0;
}


static int
cast_to_string_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    /*
     * NOTE: The following code used to be part of PyArray_AdaptFlexibleDType
     *
     * Get a string-size estimate of the input. These
     * are generallly the size needed, rounded up to
     * a multiple of eight.
     */
    npy_intp size = -1;
    switch (given_descrs[0]->type_num) {
        case NPY_BOOL:
        case NPY_UBYTE:
        case NPY_BYTE:
        case NPY_USHORT:
        case NPY_SHORT:
        case NPY_UINT:
        case NPY_INT:
        case NPY_ULONG:
        case NPY_LONG:
        case NPY_ULONGLONG:
        case NPY_LONGLONG:
            assert(given_descrs[0]->elsize <= 8);
            assert(given_descrs[0]->elsize > 0);
            if (given_descrs[0]->kind == 'b') {
                /* 5 chars needed for cast to 'True' or 'False' */
                size = 5;
            }
            else if (given_descrs[0]->kind == 'u') {
                size = REQUIRED_STR_LEN[given_descrs[0]->elsize];
            }
            else if (given_descrs[0]->kind == 'i') {
                /* Add character for sign symbol */
                size = REQUIRED_STR_LEN[given_descrs[0]->elsize] + 1;
            }
            break;
        case NPY_HALF:
        case NPY_FLOAT:
        case NPY_DOUBLE:
            size = 32;
            break;
        case NPY_LONGDOUBLE:
            size = 48;
            break;
        case NPY_CFLOAT:
        case NPY_CDOUBLE:
            size = 2 * 32;
            break;
        case NPY_CLONGDOUBLE:
            size = 2 * 48;
            break;
        case NPY_STRING:
        case NPY_VOID:
            size = given_descrs[0]->elsize;
            break;
        case NPY_UNICODE:
            size = given_descrs[0]->elsize / 4;
            break;
        default:
            PyErr_SetString(PyExc_SystemError,
                    "Impossible cast to string path requested.");
            return -1;
    }
    if (dtypes[1]->type_num == NPY_UNICODE) {
        size *= 4;
    }

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = PyArray_DescrNewFromType(dtypes[1]->type_num);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
        loop_descrs[1]->elsize = size;
    }
    else {
        /* The legacy loop can handle mismatching itemsizes */
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }

    /* Set the input one as well (late for easier error management) */
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return -1;
    }

    if (self->casting == NPY_UNSAFE_CASTING) {
        assert(dtypes[0]->type_num == NPY_UNICODE &&
               dtypes[1]->type_num == NPY_STRING);
        return NPY_UNSAFE_CASTING;
    }

    if (loop_descrs[1]->elsize >= size) {
        return NPY_SAFE_CASTING;
    }
    return NPY_SAME_KIND_CASTING;
}


static int
add_other_to_and_from_string_cast(
        PyArray_DTypeMeta *string, PyArray_DTypeMeta *other)
{
    if (string == other) {
        return 0;
    }

    /* Casting from string, is always a simple legacy-style cast */
    if (other->type_num != NPY_STRING && other->type_num != NPY_UNICODE) {
        if (PyArray_AddLegacyWrapping_CastingImpl(
                string, other, NPY_UNSAFE_CASTING) < 0) {
            return -1;
        }
    }
    /*
     * Casting to strings, is almost the same, but requires a custom resolver
     * to define the correct string length. Right now we use a generic function
     * for this.
     */
    PyArray_DTypeMeta *dtypes[2] = {other, string};
    PyType_Slot slots[] = {
            {_NPY_METH_get_loop, &legacy_cast_get_strided_loop},
            {NPY_METH_resolve_descriptors, &cast_to_string_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
        .name = "legacy_cast_to_string",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS,
        .dtypes = dtypes,
        .slots = slots,
    };
    /* Almost everything can be same-kind cast to string (except unicode) */
    if (other->type_num != NPY_UNICODE) {
        spec.casting = NPY_SAME_KIND_CASTING;  /* same-kind if too short */
    }
    else {
        spec.casting = NPY_UNSAFE_CASTING;
    }

    return PyArray_AddCastingImplementation_FromSpec(&spec, 1);
}


NPY_NO_EXPORT NPY_CASTING
string_to_string_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(loop_descrs[0]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    if (loop_descrs[0]->elsize < loop_descrs[1]->elsize) {
        /* New string is longer: safe but cannot be a view */
        return NPY_SAFE_CASTING;
    }
    else {
        /* New string fits into old: if the byte-order matches can be a view */
        int not_swapped = (PyDataType_ISNOTSWAPPED(loop_descrs[0])
                           == PyDataType_ISNOTSWAPPED(loop_descrs[1]));
        if (not_swapped) {
            *view_offset = 0;
        }

        if (loop_descrs[0]->elsize > loop_descrs[1]->elsize) {
            return NPY_SAME_KIND_CASTING;
        }
        /* The strings have the same length: */
        if (not_swapped) {
            return NPY_NO_CASTING;
        }
        else {
            return NPY_EQUIV_CASTING;
        }
    }
}


NPY_NO_EXPORT int
string_to_string_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    int unicode_swap = 0;
    PyArray_Descr **descrs = context->descriptors;

    assert(NPY_DTYPE(descrs[0]) == NPY_DTYPE(descrs[1]));
    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;
    if (descrs[0]->type_num == NPY_UNICODE) {
        if (PyDataType_ISNOTSWAPPED(descrs[0]) !=
                PyDataType_ISNOTSWAPPED(descrs[1])) {
            unicode_swap = 1;
        }
    }

    if (PyArray_GetStridedZeroPadCopyFn(
            aligned, unicode_swap, strides[0], strides[1],
            descrs[0]->elsize, descrs[1]->elsize,
            out_loop, out_transferdata) == NPY_FAIL) {
        return -1;
    }
    return 0;
}


/*
 * Add string casts. Right now all string casts are just legacy-wrapped ones
 * (except string<->string and unicode<->unicode), but they do require
 * custom type resolution for the string length.
 *
 * A bit like `object`, it could make sense to define a simpler protocol for
 * string casts, however, we also need to remember that the itemsize of the
 * output has to be found.
 */
static int
PyArray_InitializeStringCasts(void)
{
    int result = -1;
    PyArray_DTypeMeta *string = PyArray_DTypeFromTypeNum(NPY_STRING);
    PyArray_DTypeMeta *unicode = PyArray_DTypeFromTypeNum(NPY_UNICODE);
    PyArray_DTypeMeta *other_dt = NULL;

    /* Add most casts as legacy ones */
    for (int other = 0; other < NPY_NTYPES; other++) {
        if (PyTypeNum_ISDATETIME(other) || other == NPY_VOID ||
                other == NPY_OBJECT) {
            continue;
        }
        other_dt = PyArray_DTypeFromTypeNum(other);

        /* The functions skip string == other_dt or unicode == other_dt */
        if (add_other_to_and_from_string_cast(string, other_dt) < 0) {
            goto finish;
        }
        if (add_other_to_and_from_string_cast(unicode, other_dt) < 0) {
            goto finish;
        }

        Py_SETREF(other_dt, NULL);
    }

    /* string<->string and unicode<->unicode have their own specialized casts */
    PyArray_DTypeMeta *dtypes[2];
    PyType_Slot slots[] = {
            {_NPY_METH_get_loop, &string_to_string_get_loop},
            {NPY_METH_resolve_descriptors, &string_to_string_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "string_to_string_cast",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = (NPY_METH_REQUIRES_PYAPI |
                      NPY_METH_NO_FLOATINGPOINT_ERRORS |
                      NPY_METH_SUPPORTS_UNALIGNED),
            .dtypes = dtypes,
            .slots = slots,
    };

    dtypes[0] = string;
    dtypes[1] = string;
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        goto finish;
    }

    dtypes[0] = unicode;
    dtypes[1] = unicode;
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        goto finish;
    }

    result = 0;
  finish:
    Py_DECREF(string);
    Py_DECREF(unicode);
    Py_XDECREF(other_dt);
    return result;
}


/*
 * Small helper function to handle the case of `arr.astype(dtype="V")`.
 * When the output descriptor is not passed, we always use `V<itemsize>`
 * of the other dtype.
 */
static NPY_CASTING
cast_to_void_dtype_class(
        PyArray_Descr **given_descrs, PyArray_Descr **loop_descrs,
        npy_intp *view_offset)
{
    /* `dtype="V"` means unstructured currently (compare final path) */
    loop_descrs[1] = PyArray_DescrNewFromType(NPY_VOID);
    if (loop_descrs[1] == NULL) {
        return -1;
    }
    loop_descrs[1]->elsize = given_descrs[0]->elsize;
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    *view_offset = 0;
    if (loop_descrs[0]->type_num == NPY_VOID &&
            loop_descrs[0]->subarray == NULL && loop_descrs[1]->names == NULL) {
        return NPY_NO_CASTING;
    }
    return NPY_SAFE_CASTING;
}


static NPY_CASTING
nonstructured_to_structured_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    NPY_CASTING casting;

    if (given_descrs[1] == NULL) {
        return cast_to_void_dtype_class(given_descrs, loop_descrs, view_offset);
    }

    if (given_descrs[1]->subarray != NULL) {
        /*
         * We currently consider this at most a safe cast. It would be
         * possible to allow a view if the field has exactly one element.
         */
        casting = NPY_SAFE_CASTING;
        npy_intp sub_view_offset = NPY_MIN_INTP;
        /* Subarray dtype */
        NPY_CASTING base_casting = PyArray_GetCastInfo(
                given_descrs[0], given_descrs[1]->subarray->base, NULL,
                &sub_view_offset);
        if (base_casting < 0) {
            return -1;
        }
        if (given_descrs[1]->elsize == given_descrs[1]->subarray->base->elsize) {
            /* A single field, view is OK if sub-view is */
            *view_offset = sub_view_offset;
        }
        casting = PyArray_MinCastSafety(casting, base_casting);
    }
    else if (given_descrs[1]->names != NULL) {
        /* Structured dtype */
        if (PyTuple_Size(given_descrs[1]->names) == 0) {
            /* TODO: This retained behaviour, but likely should be changed. */
            casting = NPY_UNSAFE_CASTING;
        }
        else {
            /* Considered at most unsafe casting (but this could be changed) */
            casting = NPY_UNSAFE_CASTING;

            Py_ssize_t pos = 0;
            PyObject *key, *tuple;
            while (PyDict_Next(given_descrs[1]->fields, &pos, &key, &tuple)) {
                PyArray_Descr *field_descr = (PyArray_Descr *)PyTuple_GET_ITEM(tuple, 0);
                npy_intp field_view_off = NPY_MIN_INTP;
                NPY_CASTING field_casting = PyArray_GetCastInfo(
                        given_descrs[0], field_descr, NULL, &field_view_off);
                casting = PyArray_MinCastSafety(casting, field_casting);
                if (casting < 0) {
                    return -1;
                }
                if (field_view_off != NPY_MIN_INTP) {
                    npy_intp to_off = PyLong_AsSsize_t(PyTuple_GET_ITEM(tuple, 1));
                    if (error_converting(to_off)) {
                        return -1;
                    }
                    *view_offset = field_view_off - to_off;
                }
            }
            if (PyTuple_Size(given_descrs[1]->names) != 1 || *view_offset < 0) {
                /*
                 * Assume that a view is impossible when there is more than one
                 * field.  (Fields could overlap, but that seems weird...)
                 */
                *view_offset = NPY_MIN_INTP;
            }
        }
    }
    else {
        /* Plain void type. This behaves much like a "view" */
        if (given_descrs[0]->elsize == given_descrs[1]->elsize &&
                !PyDataType_REFCHK(given_descrs[0])) {
            /*
             * A simple view, at the moment considered "safe" (the refcheck is
             * probably not necessary, but more future proof)
             */
            *view_offset = 0;
            casting = NPY_SAFE_CASTING;
        }
        else if (given_descrs[0]->elsize <= given_descrs[1]->elsize) {
            casting = NPY_SAFE_CASTING;
        }
        else {
            casting = NPY_UNSAFE_CASTING;
            /* new elsize is smaller so a view is OK (reject refs for now) */
            if (!PyDataType_REFCHK(given_descrs[0])) {
                *view_offset = 0;
            }
        }
    }

    /* Void dtypes always do the full cast. */
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    return casting;
}


int give_bad_field_error(PyObject *key)
{
    if (!PyErr_Occurred()) {
        PyErr_Format(PyExc_RuntimeError,
                "Invalid or missing field %R, this should be impossible "
                "and indicates a NumPy bug.", key);
    }
    return -1;
}


static int
nonstructured_to_structured_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (context->descriptors[1]->names != NULL) {
        if (get_fields_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    else if (context->descriptors[1]->subarray != NULL) {
        if (get_subarray_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    else {
        /*
         * TODO: This could be a simple zero padded cast, adding a decref
         *       in case of `move_references`. But for now use legacy casts
         *       (which is the behaviour at least up to 1.20).
         */
        int needs_api = 0;
        if (get_wrapped_legacy_cast_function(
                1, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api, 1) < 0) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    return 0;
}


static PyObject *
PyArray_GetGenericToVoidCastingImpl(void)
{
    static PyArrayMethodObject *method = NULL;

    if (method != NULL) {
        Py_INCREF(method);
        return (PyObject *)method;
    }

    method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    if (method == NULL) {
        return PyErr_NoMemory();
    }

    method->name = "any_to_void_cast";
    method->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method->casting = -1;
    method->resolve_descriptors = &nonstructured_to_structured_resolve_descriptors;
    method->get_strided_loop = &nonstructured_to_structured_get_loop;
    method->nin = 1;
    method->nout = 1;

    return (PyObject *)method;
}


static NPY_CASTING
structured_to_nonstructured_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    PyArray_Descr *base_descr;
    /* The structured part may allow a view (and have its own offset): */
    npy_intp struct_view_offset = NPY_MIN_INTP;

    if (given_descrs[0]->subarray != NULL) {
        base_descr = given_descrs[0]->subarray->base;
        /* A view is possible if the subarray has exactly one element: */
        if (given_descrs[0]->elsize == given_descrs[0]->subarray->base->elsize) {
            struct_view_offset = 0;
        }
    }
    else if (given_descrs[0]->names != NULL) {
        if (PyTuple_Size(given_descrs[0]->names) != 1) {
            /* Only allow casting a single field */
            return -1;
        }
        PyObject *key = PyTuple_GetItem(given_descrs[0]->names, 0);
        PyObject *base_tup = PyDict_GetItem(given_descrs[0]->fields, key);
        base_descr = (PyArray_Descr *)PyTuple_GET_ITEM(base_tup, 0);
        struct_view_offset = PyLong_AsSsize_t(PyTuple_GET_ITEM(base_tup, 1));
        if (error_converting(struct_view_offset)) {
            return -1;
        }
    }
    else {
        /*
         * unstructured voids are considered unsafe casts and defined, albeit,
         * at this time they go back to legacy behaviour using getitem/setitem.
         */
        base_descr = NULL;
        struct_view_offset = 0;
    }

    /*
     * The cast is always considered unsafe, so the PyArray_GetCastInfo
     * result currently only matters for the view_offset.
     */
    npy_intp base_view_offset = NPY_MIN_INTP;
    if (base_descr != NULL && PyArray_GetCastInfo(
            base_descr, given_descrs[1], dtypes[1], &base_view_offset) < 0) {
        return -1;
    }
    if (base_view_offset != NPY_MIN_INTP
            && struct_view_offset != NPY_MIN_INTP) {
        *view_offset = base_view_offset + struct_view_offset;
    }

    /* Void dtypes always do the full cast. */
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
        /*
         * Special case strings here, it should be useless (and only actually
         * work for empty arrays).  Possibly this should simply raise for
         * all parametric DTypes.
         */
        if (dtypes[1]->type_num == NPY_STRING) {
            loop_descrs[1]->elsize = given_descrs[0]->elsize;
        }
        else if (dtypes[1]->type_num == NPY_UNICODE) {
            loop_descrs[1]->elsize = given_descrs[0]->elsize * 4;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_UNSAFE_CASTING;
}


static int
structured_to_nonstructured_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (context->descriptors[0]->names != NULL) {
        if (get_fields_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    else if (context->descriptors[0]->subarray != NULL) {
        if (get_subarray_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    else {
        /*
         * In general this is currently defined through legacy behaviour via
         * scalars, and should likely just not be allowed.
         */
        int needs_api = 0;
        if (get_wrapped_legacy_cast_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api, 1) < 0) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    return 0;
}


static PyObject *
PyArray_GetVoidToGenericCastingImpl(void)
{
    static PyArrayMethodObject *method = NULL;

    if (method != NULL) {
        Py_INCREF(method);
        return (PyObject *)method;
    }

    method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    if (method == NULL) {
        return PyErr_NoMemory();
    }

    method->name = "void_to_any_cast";
    method->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method->casting = -1;
    method->resolve_descriptors = &structured_to_nonstructured_resolve_descriptors;
    method->get_strided_loop = &structured_to_nonstructured_get_loop;
    method->nin = 1;
    method->nout = 1;

    return (PyObject *)method;
}


/*
 * Find the correct field casting safety.  See the TODO note below, including
 * in 1.20 (and later) this was based on field names rather than field order
 * which it should be using.
 *
 * NOTE: In theory it would be possible to cache the all the field casting
 *       implementations on the dtype, to avoid duplicate work.
 */
static NPY_CASTING
can_cast_fields_safety(
        PyArray_Descr *from, PyArray_Descr *to, npy_intp *view_offset)
{
    Py_ssize_t field_count = PyTuple_Size(from->names);
    if (field_count != PyTuple_Size(to->names)) {
        return -1;
    }

    NPY_CASTING casting = NPY_NO_CASTING;
    *view_offset = 0;  /* if there are no fields, a view is OK. */
    for (Py_ssize_t i = 0; i < field_count; i++) {
        npy_intp field_view_off = NPY_MIN_INTP;
        PyObject *from_key = PyTuple_GET_ITEM(from->names, i);
        PyObject *from_tup = PyDict_GetItemWithError(from->fields, from_key);
        if (from_tup == NULL) {
            return give_bad_field_error(from_key);
        }
        PyArray_Descr *from_base = (PyArray_Descr *) PyTuple_GET_ITEM(from_tup, 0);

        /* Check whether the field names match */
        PyObject *to_key = PyTuple_GET_ITEM(to->names, i);
        PyObject *to_tup = PyDict_GetItem(to->fields, to_key);
        if (to_tup == NULL) {
            return give_bad_field_error(from_key);
        }
        PyArray_Descr *to_base = (PyArray_Descr *) PyTuple_GET_ITEM(to_tup, 0);

        int cmp = PyUnicode_Compare(from_key, to_key);
        if (error_converting(cmp)) {
            return -1;
        }
        if (cmp != 0) {
            /* Field name mismatch, consider this at most SAFE. */
            casting = PyArray_MinCastSafety(casting, NPY_SAFE_CASTING);
        }

        /* Also check the title (denote mismatch as SAFE only) */
        PyObject *from_title = from_key;
        PyObject *to_title = to_key;
        if (PyTuple_GET_SIZE(from_tup) > 2) {
            from_title = PyTuple_GET_ITEM(from_tup, 2);
        }
        if (PyTuple_GET_SIZE(to_tup) > 2) {
            to_title = PyTuple_GET_ITEM(to_tup, 2);
        }
        cmp = PyObject_RichCompareBool(from_title, to_title, Py_EQ);
        if (error_converting(cmp)) {
            return -1;
        }
        if (!cmp) {
            casting = PyArray_MinCastSafety(casting, NPY_SAFE_CASTING);
        }

        NPY_CASTING field_casting = PyArray_GetCastInfo(
                from_base, to_base, NULL, &field_view_off);
        if (field_casting < 0) {
            return -1;
        }
        casting = PyArray_MinCastSafety(casting, field_casting);

        /* Adjust the "view offset" by the field offsets: */
        if (field_view_off != NPY_MIN_INTP) {
            npy_intp to_off = PyLong_AsSsize_t(PyTuple_GET_ITEM(to_tup, 1));
            if (error_converting(to_off)) {
                return -1;
            }
            npy_intp from_off = PyLong_AsSsize_t(PyTuple_GET_ITEM(from_tup, 1));
            if (error_converting(from_off)) {
                return -1;
            }
            field_view_off = field_view_off - to_off + from_off;
        }

        /*
         * If there is one field, use its field offset.  After that propagate
         * the view offset if they match and set to "invalid" if not.
         */
        if (i == 0) {
            *view_offset = field_view_off;
        }
        else if (*view_offset != field_view_off) {
            *view_offset = NPY_MIN_INTP;
        }
    }

    if (*view_offset != 0 || from->elsize != to->elsize) {
        /* Can never be considered "no" casting. */
        casting = PyArray_MinCastSafety(casting, NPY_EQUIV_CASTING);
    }

    /* The new dtype may have access outside the old one due to padding: */
    if (*view_offset < 0) {
        /* negative offsets would give indirect access before original dtype */
        *view_offset = NPY_MIN_INTP;
    }
    if (from->elsize < to->elsize + *view_offset) {
        /* new dtype has indirect access outside of the original dtype */
        *view_offset = NPY_MIN_INTP;
    }

    return casting;
}


static NPY_CASTING
void_to_void_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    NPY_CASTING casting;

    if (given_descrs[1] == NULL) {
        /* This is weird, since it doesn't return the original descr, but... */
        return cast_to_void_dtype_class(given_descrs, loop_descrs, view_offset);
    }

    if (given_descrs[0]->names != NULL && given_descrs[1]->names != NULL) {
        /* From structured to structured, need to check fields */
        casting = can_cast_fields_safety(
                given_descrs[0], given_descrs[1], view_offset);
        if (casting < 0) {
            return -1;
        }
    }
    else if (given_descrs[0]->names != NULL) {
        return structured_to_nonstructured_resolve_descriptors(
                self, dtypes, given_descrs, loop_descrs, view_offset);
    }
    else if (given_descrs[1]->names != NULL) {
        return nonstructured_to_structured_resolve_descriptors(
                self, dtypes, given_descrs, loop_descrs, view_offset);
    }
    else if (given_descrs[0]->subarray == NULL &&
                given_descrs[1]->subarray == NULL) {
        /* Both are plain void dtypes */
        if (given_descrs[0]->elsize == given_descrs[1]->elsize) {
            casting = NPY_NO_CASTING;
            *view_offset = 0;
        }
        else if (given_descrs[0]->elsize < given_descrs[1]->elsize) {
            casting = NPY_SAFE_CASTING;
        }
        else {
            casting = NPY_SAME_KIND_CASTING;
            *view_offset = 0;
        }
    }
    else {
        /*
         * At this point, one of the dtypes must be a subarray dtype, the
         * other is definitely not a structured one.
         */
        PyArray_ArrayDescr *from_sub = given_descrs[0]->subarray;
        PyArray_ArrayDescr *to_sub = given_descrs[1]->subarray;
        assert(from_sub || to_sub);

        /* If the shapes do not match, this is at most an unsafe cast */
        casting = NPY_UNSAFE_CASTING;
        /*
         * We can use a view in two cases:
         * 1. The shapes and elsizes matches, so any view offset applies to
         *    each element of the subarray identically.
         *    (in practice this probably implies the `view_offset` will be 0)
         * 2. There is exactly one element and the subarray has no effect
         *    (can be tested by checking if the itemsizes of the base matches)
         */
        npy_bool subarray_layout_supports_view = NPY_FALSE;
        if (from_sub && to_sub) {
            int res = PyObject_RichCompareBool(from_sub->shape, to_sub->shape, Py_EQ);
            if (res < 0) {
                return -1;
            }
            else if (res) {
                /* Both are subarrays and the shape matches, could be no cast */
                casting = NPY_NO_CASTING;
                /* May be a view if there is one element or elsizes match */
                if (from_sub->base->elsize == to_sub->base->elsize
                        || given_descrs[0]->elsize == from_sub->base->elsize) {
                    subarray_layout_supports_view = NPY_TRUE;
                }
            }
        }
        else if (from_sub) {
            /* May use a view if "from" has only a single element: */
            if (given_descrs[0]->elsize == from_sub->base->elsize) {
                subarray_layout_supports_view = NPY_TRUE;
            }
        }
        else {
            /* May use a view if "from" has only a single element: */
            if (given_descrs[1]->elsize == to_sub->base->elsize) {
                subarray_layout_supports_view = NPY_TRUE;
            }
        }

        PyArray_Descr *from_base = (from_sub == NULL) ? given_descrs[0] : from_sub->base;
        PyArray_Descr *to_base = (to_sub == NULL) ? given_descrs[1] : to_sub->base;
        /* An offset for  */
        NPY_CASTING field_casting = PyArray_GetCastInfo(
                from_base, to_base, NULL, view_offset);
        if (!subarray_layout_supports_view) {
            *view_offset = NPY_MIN_INTP;
        }
        if (field_casting < 0) {
            return -1;
        }
        casting = PyArray_MinCastSafety(casting, field_casting);
    }

    /* Void dtypes always do the full cast. */
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    return casting;
}


NPY_NO_EXPORT int
void_to_void_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (context->descriptors[0]->names != NULL ||
            context->descriptors[1]->names != NULL) {
        if (get_fields_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    else if (context->descriptors[0]->subarray != NULL ||
             context->descriptors[1]->subarray != NULL) {
        if (get_subarray_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    else {
        /*
         * This is a string-like copy of the two bytes (zero padding if
         * necessary)
         */
        if (PyArray_GetStridedZeroPadCopyFn(
                0, 0, strides[0], strides[1],
                context->descriptors[0]->elsize, context->descriptors[1]->elsize,
                out_loop, out_transferdata) == NPY_FAIL) {
            return -1;
        }
        *flags = PyArrayMethod_MINIMAL_FLAGS;
    }
    return 0;
}


/*
 * This initializes the void to void cast. Voids include structured dtypes,
 * which means that they can cast from and to any other dtype and, in that
 * sense, are special (similar to Object).
 */
static int
PyArray_InitializeVoidToVoidCast(void)
{
    PyArray_DTypeMeta *Void = PyArray_DTypeFromTypeNum(NPY_VOID);
    PyArray_DTypeMeta *dtypes[2] = {Void, Void};
    PyType_Slot slots[] = {
            {_NPY_METH_get_loop, &void_to_void_get_loop},
            {NPY_METH_resolve_descriptors, &void_to_void_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "void_to_void_cast",
            .nin = 1,
            .nout = 1,
            .casting = -1,  /* may not cast at all */
            .flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    int res = PyArray_AddCastingImplementation_FromSpec(&spec, 1);
    Py_DECREF(Void);
    return res;
}


/*
 * Implement object to any casting implementation. Casting from object may
 * require inspecting of all array elements (for parametric dtypes), and
 * the resolver will thus reject all parametric dtypes if the out dtype
 * is not provided.
 */
static NPY_CASTING
object_to_any_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        /*
         * This should not really be called, since object -> parametric casts
         * require inspecting the object array. Allow legacy ones, the path
         * here is that e.g. "M8" input is considered to be the DType class,
         * and by allowing it here, we go back to the "M8" instance.
         */
        if (NPY_DT_is_parametric(dtypes[1])) {
            PyErr_Format(PyExc_TypeError,
                    "casting from object to the parametric DType %S requires "
                    "the specified output dtype instance. "
                    "This may be a NumPy issue, since the correct instance "
                    "should be discovered automatically, however.", dtypes[1]);
            return -1;
        }
        loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    return NPY_UNSAFE_CASTING;
}


/*
 * Casting to object is special since it is generic to all input dtypes.
 */
static PyObject *
PyArray_GetObjectToGenericCastingImpl(void)
{
    static PyArrayMethodObject *method = NULL;

    if (method != NULL) {
        Py_INCREF(method);
        return (PyObject *)method;
    }

    method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    if (method == NULL) {
        return PyErr_NoMemory();
    }

    method->nin = 1;
    method->nout = 1;
    method->name = "object_to_any_cast";
    method->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method->casting = NPY_UNSAFE_CASTING;
    method->resolve_descriptors = &object_to_any_resolve_descriptors;
    method->get_strided_loop = &object_to_any_get_loop;

    return (PyObject *)method;
}



/* Any object is simple (could even use the default) */
static NPY_CASTING
any_to_object_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    return NPY_SAFE_CASTING;
}


/*
 * Casting to object is special since it is generic to all input dtypes.
 */
static PyObject *
PyArray_GetGenericToObjectCastingImpl(void)
{
    static PyArrayMethodObject *method = NULL;

    if (method != NULL) {
        Py_INCREF(method);
        return (PyObject *)method;
    }

    method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    if (method == NULL) {
        return PyErr_NoMemory();
    }

    method->nin = 1;
    method->nout = 1;
    method->name = "any_to_object_cast";
    method->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method->casting = NPY_SAFE_CASTING;
    method->resolve_descriptors = &any_to_object_resolve_descriptors;
    method->get_strided_loop = &any_to_object_get_loop;

    return (PyObject *)method;
}


/*
 * Casts within the object dtype is always just a plain copy/view.
 * For that reason, this function might remain unimplemented.
 */
static int
object_to_object_get_loop(
        PyArrayMethod_Context *NPY_UNUSED(context),
        int NPY_UNUSED(aligned), int move_references,
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS;
    if (move_references) {
        *out_loop = &_strided_to_strided_move_references;
        *out_transferdata = NULL;
    }
    else {
        *out_loop = &_strided_to_strided_copy_references;
        *out_transferdata = NULL;
    }
    return 0;
}


static int
PyArray_InitializeObjectToObjectCast(void)
{
    PyArray_DTypeMeta *Object = PyArray_DTypeFromTypeNum(NPY_OBJECT);
    PyArray_DTypeMeta *dtypes[2] = {Object, Object};
    PyType_Slot slots[] = {
            {_NPY_METH_get_loop, &object_to_object_get_loop},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "object_to_object_cast",
            .nin = 1,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    int res = PyArray_AddCastingImplementation_FromSpec(&spec, 1);
    Py_DECREF(Object);
    return res;
}


NPY_NO_EXPORT int
PyArray_InitializeCasts()
{
    if (PyArray_InitializeNumericCasts() < 0) {
        return -1;
    }
    if (PyArray_InitializeStringCasts() < 0) {
        return -1;
    }
    if (PyArray_InitializeVoidToVoidCast() < 0) {
        return -1;
    }
    if (PyArray_InitializeObjectToObjectCast() < 0) {
        return -1;
    }
    /* Datetime casts are defined in datetime.c */
    if (PyArray_InitializeDatetimeCasts() < 0) {
        return -1;
    }
    return 0;
}
