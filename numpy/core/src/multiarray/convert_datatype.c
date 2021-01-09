#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"
#include "numpy/npy_math.h"

#include "array_coercion.h"
#include "common.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "scalartypes.h"
#include "mapping.h"
#include "legacy_dtype_implementation.h"

#include "convert_datatype.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "array_method.h"
#include "usertypes.h"
#include "dtype_transfer.h"


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


static PyObject *
PyArray_GetGenericToVoidCastingImpl(void);

static PyObject *
PyArray_GetVoidToGenericCastingImpl(void);

static PyObject *
PyArray_GetGenericToObjectCastingImpl(void);

static PyObject *
PyArray_GetObjectToGenericCastingImpl(void);


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
    PyObject *res = PyDict_GetItem(from->castingimpls, (PyObject *)to);
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
                "builtin cast from %S to %s not found, this should not "
                "be possible.", from, to);
        return NULL;
    }
    else {
        if (from->parametric || to->parametric) {
            Py_RETURN_NONE;
        }
        /* Reject non-legacy dtypes (they need to use the new API) */
        if (!from->legacy || !to->legacy) {
            Py_RETURN_NONE;
        }
        if (from != to) {
            /* A cast function must have been registered */
            PyArray_VectorUnaryFunc *castfunc = PyArray_GetCastFunc(
                    from->singleton, to->type_num);
            if (castfunc == NULL) {
                PyErr_Clear();
                /* Remember that this cast is not possible */
                if (PyDict_SetItem(from->castingimpls, (PyObject *) to, Py_None) < 0) {
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
    if (PyDict_SetItem(from->castingimpls, (PyObject *)to, res) < 0) {
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
    NPY_CASTING view = casting1 & casting2 & _NPY_CAST_IS_VIEW;
    casting1 = casting1 & ~_NPY_CAST_IS_VIEW;
    casting2 = casting2 & ~_NPY_CAST_IS_VIEW;
    /* larger casting values are less safe */
    if (casting1 > casting2) {
        return casting1 | view;
    }
    return casting2 | view;
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

    Py_SETREF(dtype, PyArray_AdaptDescriptorToArray(arr, (PyObject *)dtype));
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
        PyObject *cls = NULL, *obj = NULL;
        int ret;
        obj = PyImport_ImportModule("numpy.core");

        if (obj) {
            cls = PyObject_GetAttrString(obj, "ComplexWarning");
            Py_DECREF(obj);
        }
        ret = PyErr_WarnEx(cls,
                "Casting complex values to real discards "
                "the imaginary part", 1);
        Py_XDECREF(cls);
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
 * @return NPY_CASTING or -1 on error or if the cast is not possible.
 */
NPY_NO_EXPORT NPY_CASTING
PyArray_GetCastSafety(
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype)
{
    NPY_CASTING casting;
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
    PyArray_Descr *descrs[2] = {from, to};
    PyArray_Descr *out_descrs[2];

    casting = castingimpl->resolve_descriptors(
            castingimpl, dtypes, descrs, out_descrs);
    Py_DECREF(meth);
    if (casting < 0) {
        return -1;
    }
    /* The returned descriptors may not match, requiring a second check */
    if (out_descrs[0] != descrs[0]) {
        NPY_CASTING from_casting = PyArray_GetCastSafety(
                descrs[0], out_descrs[0], NULL);
        casting = PyArray_MinCastSafety(casting, from_casting);
        if (casting < 0) {
            goto finish;
        }
    }
    if (descrs[1] != NULL && out_descrs[1] != descrs[1]) {
        NPY_CASTING from_casting = PyArray_GetCastSafety(
                descrs[1], out_descrs[1], NULL);
        casting = PyArray_MinCastSafety(casting, from_casting);
        if (casting < 0) {
            goto finish;
        }
    }

  finish:
    Py_DECREF(out_descrs[0]);
    Py_DECREF(out_descrs[1]);
    /* NPY_NO_CASTING has to be used for (NPY_EQUIV_CASTING|_NPY_CAST_IS_VIEW) */
    assert(casting != (NPY_EQUIV_CASTING|_NPY_CAST_IS_VIEW));
    return casting;
}


/*NUMPY_API
 *Check the type coercion rules.
 */
NPY_NO_EXPORT int
PyArray_CanCastSafely(int fromtype, int totype)
{
#if NPY_USE_NEW_CASTINGIMPL
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
#else
    return PyArray_LegacyCanCastSafely(fromtype, totype);
#endif
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
#if NPY_USE_NEW_CASTINGIMPL
    return PyArray_CanCastTypeTo(from, to, NPY_SAFE_CASTING);
#else
    return PyArray_LegacyCanCastTo(from, to);
#endif
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
#if NPY_USE_NEW_CASTINGIMPL
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
     */
    NPY_CASTING safety;
    if (PyDataType_ISUNSIZED(to) && to->subarray == NULL) {
        safety = PyArray_GetCastSafety(from, NULL, NPY_DTYPE(to));
    }
    else {
        safety = PyArray_GetCastSafety(from, to, NPY_DTYPE(to));
    }

    if (safety < 0) {
        PyErr_Clear();
        return 0;
    }
    /* If casting is the smaller (or equal) safety we match */
    return PyArray_MinCastSafety(safety, casting) == casting;
#else
    return PyArray_LegacyCanCastTypeTo(from, to, casting);
#endif
}


/* CanCastArrayTo needs this function */
static int min_scalar_type_num(char *valueptr, int type_num,
                                            int *is_small_unsigned);

NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data,
                    PyArray_Descr *to, NPY_CASTING casting)
{
    int swap;
    int is_small_unsigned = 0, type_num;
    npy_bool ret;
    PyArray_Descr *dtype;

    /* An aligned memory buffer large enough to hold any type */
    npy_longlong value[4];

    /*
     * If the two dtypes are actually references to the same object
     * or if casting type is forced unsafe then always OK.
     */
    if (scal_type == to || casting == NPY_UNSAFE_CASTING ) {
        return 1;
    }

    /*
     * If the scalar isn't a number, or the rule is stricter than
     * NPY_SAFE_CASTING, use the straight type-based rules
     */
    if (!PyTypeNum_ISNUMBER(scal_type->type_num) ||
                            casting < NPY_SAFE_CASTING) {
        return PyArray_CanCastTypeTo(scal_type, to, casting);
    }

    swap = !PyArray_ISNBO(scal_type->byteorder);
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
    ret = PyArray_CanCastTypeTo(dtype, to, casting);
    Py_DECREF(dtype);
    return ret;
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

    /* If it's a scalar, check the value */
    if (PyArray_NDIM(arr) == 0 && !PyArray_HASFIELDS(arr)) {
        return can_cast_scalar_to(from, PyArray_DATA(arr), to, casting);
    }

    /* Otherwise, use the standard rules */
    return PyArray_CanCastTypeTo(from, to, casting);
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

/*
 * Returns a new reference to type if it is already NBO, otherwise
 * returns a copy converted to NBO.
 */
NPY_NO_EXPORT PyArray_Descr *
ensure_dtype_nbo(PyArray_Descr *type)
{
    if (PyArray_ISNBO(type->byteorder)) {
        Py_INCREF(type);
        return type;
    }
    else {
        return PyArray_DescrNewByteorder(type, NPY_NATIVE);
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
    if (!given_DType->parametric) {
        /*
         * Don't actually do anything, the default is always the result
         * of any cast.
         */
        return given_DType->default_descr(given_DType);
    }
    if (PyObject_TypeCheck((PyObject *)descr, (PyTypeObject *)given_DType)) {
        Py_INCREF(descr);
        return descr;
    }

#if NPY_USE_NEW_CASTINGIMPL
    PyObject *tmp = PyArray_GetCastingImpl(NPY_DTYPE(descr), given_DType);
    if (tmp == NULL || tmp == Py_None) {
        Py_XDECREF(tmp);
        goto error;
    }
    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(descr), given_DType};
    PyArray_Descr *given_descrs[2] = {descr, NULL};
    PyArray_Descr *loop_descrs[2];

    PyArrayMethodObject *meth = (PyArrayMethodObject *)tmp;
    NPY_CASTING casting = meth->resolve_descriptors(
            meth, dtypes, given_descrs, loop_descrs);
    Py_DECREF(tmp);
    if (casting < 0) {
        goto error;
    }
    Py_DECREF(loop_descrs[0]);
    return loop_descrs[1];

  error:;  /* (; due to compiler limitations) */
    PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
    PyErr_Fetch(&err_type, &err_value, &err_traceback);
    PyErr_Format(PyExc_ValueError,
            "cannot cast dtype %S to %S.", descr, given_DType);
    npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
    return NULL;

#else  /* NPY_USE_NEW_CASTS */
    if (!given_DType->legacy) {
        PyErr_SetString(PyExc_NotImplementedError,
                "Must use casting to find the correct DType for a parametric "
                "user DType. This is not yet implemented (this error should be "
                "unreachable).");
        return NULL;
    }

    PyArray_Descr *flex_dtype = PyArray_DescrNew(given_DType->singleton);
    return PyArray_AdaptFlexibleDType(descr, flex_dtype);
#endif  /* NPY_USE_NEW_CASTS */
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
        npy_intp n, PyArrayObject **arrays, PyObject *requested_dtype)
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
            Py_DECREF(result);
            return NULL;
        }
        goto finish;
    }
    assert(n > 0);  /* concatenate requires at least one array input. */
    PyArray_Descr *descr = PyArray_DESCR(arrays[0]);
    result = PyArray_CastDescrToDType(descr, common_dtype);
    if (result == NULL || n == 1) {
        goto finish;
    }
    /*
     * This could short-cut a bit, calling `common_instance` directly and/or
     * returning the `default_descr()` directly. Avoiding that (for now) as
     * it would duplicate code from `PyArray_PromoteTypes`.
     */
    for (npy_intp i = 1; i < n; i++) {
        descr = PyArray_DESCR(arrays[i]);
        PyArray_Descr *curr = PyArray_CastDescrToDType(descr, common_dtype);
        if (curr == NULL) {
            Py_SETREF(result, NULL);
            goto finish;
        }
        Py_SETREF(result, PyArray_PromoteTypes(result, curr));
        Py_DECREF(curr);
        if (result == NULL) {
            goto finish;
        }
    }

  finish:
    Py_DECREF(common_dtype);
    return result;
}


/**
 * This function defines the common DType operator.
 *
 * Note that the common DType will not be "object" (unless one of the dtypes
 * is object), even though object can technically represent all values
 * correctly.
 *
 * TODO: Before exposure, we should review the return value (e.g. no error
 *       when no common DType is found).
 *
 * @param dtype1 DType class to find the common type for.
 * @param dtype2 Second DType class.
 * @return The common DType or NULL with an error set
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_CommonDType(PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2)
{
    if (dtype1 == dtype2) {
        Py_INCREF(dtype1);
        return dtype1;
    }

    PyArray_DTypeMeta *common_dtype;

    common_dtype = dtype1->common_dtype(dtype1, dtype2);
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(common_dtype);
        common_dtype = dtype2->common_dtype(dtype2, dtype1);
    }
    if (common_dtype == NULL) {
        return NULL;
    }
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(Py_NotImplemented);
        PyErr_Format(PyExc_TypeError,
                "The DTypes %S and %S do not have a common DType. "
                "For example they cannot be stored in a single array unless "
                "the dtype is `object`.", dtype1, dtype2);
        return NULL;
    }
    return common_dtype;
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
    if (type1 == type2 && PyArray_ISNBO(type1->byteorder)) {
        Py_INCREF(type1);
        return type1;
    }

    common_dtype = PyArray_CommonDType(NPY_DTYPE(type1), NPY_DTYPE(type2));
    if (common_dtype == NULL) {
        return NULL;
    }

    if (!common_dtype->parametric) {
        res = common_dtype->default_descr(common_dtype);
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
    res = common_dtype->common_instance(type1, type2);
    Py_DECREF(type1);
    Py_DECREF(type2);
    Py_DECREF(common_dtype);
    return res;
}

/*
 * Produces the smallest size and lowest kind type to which all
 * input types can be cast.
 *
 * Equivalent to functools.reduce(PyArray_PromoteTypes, types)
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_PromoteTypeSequence(PyArray_Descr **types, npy_intp ntypes)
{
    npy_intp i;
    PyArray_Descr *ret = NULL;
    if (ntypes == 0) {
        PyErr_SetString(PyExc_TypeError, "at least one type needed to promote");
        return NULL;
    }
    ret = types[0];
    Py_INCREF(ret);
    for (i = 1; i < ntypes; ++i) {
        PyArray_Descr *tmp = PyArray_PromoteTypes(types[i], ret);
        Py_DECREF(ret);
        ret = tmp;
        if (ret == NULL) {
            return NULL;
        }
    }
    return ret;
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


/*NUMPY_API
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
 *
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_ResultType(npy_intp narrs, PyArrayObject **arr,
                    npy_intp ndtypes, PyArray_Descr **dtypes)
{
    npy_intp i;

    /* If there's just one type, pass it through */
    if (narrs + ndtypes == 1) {
        PyArray_Descr *ret = NULL;
        if (narrs == 1) {
            ret = PyArray_DESCR(arr[0]);
        }
        else {
            ret = dtypes[0];
        }
        Py_INCREF(ret);
        return ret;
    }

    int use_min_scalar = should_use_min_scalar(narrs, arr, ndtypes, dtypes);

    /* Loop through all the types, promoting them */
    if (!use_min_scalar) {
        PyArray_Descr *ret;

        /* Build a single array of all the dtypes */
        PyArray_Descr **all_dtypes = PyArray_malloc(
            sizeof(*all_dtypes) * (narrs + ndtypes));
        if (all_dtypes == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        for (i = 0; i < narrs; ++i) {
            all_dtypes[i] = PyArray_DESCR(arr[i]);
        }
        for (i = 0; i < ndtypes; ++i) {
            all_dtypes[narrs + i] = dtypes[i];
        }
        ret = PyArray_PromoteTypeSequence(all_dtypes, narrs + ndtypes);
        PyArray_free(all_dtypes);
        return ret;
    }
    else {
        int ret_is_small_unsigned = 0;
        PyArray_Descr *ret = NULL;

        for (i = 0; i < narrs; ++i) {
            int tmp_is_small_unsigned;
            PyArray_Descr *tmp = PyArray_MinScalarType_internal(
                arr[i], &tmp_is_small_unsigned);
            if (tmp == NULL) {
                Py_XDECREF(ret);
                return NULL;
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
                    return NULL;
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
                    return NULL;
                }
            }
        }
        /* None of the above loops ran */
        if (ret == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "no arrays or types available to calculate result type");
        }

        return ret;
    }
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
    else if (!NPY_DTYPE(dtype)->legacy) {
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
 * The user of the function has to free the returned array.
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
        Py_DECREF(tmp);
        if (mps[i] == NULL) {
            goto fail;
        }
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
PyArray_AddCastingImplmentation(PyBoundArrayMethodObject *meth)
{
    if (meth->method->nin != 1 || meth->method->nout != 1) {
        PyErr_SetString(PyExc_TypeError,
                "A cast must have one input and one output.");
        return -1;
    }
    if (meth->dtypes[0] == meth->dtypes[1]) {
        if (!(meth->method->flags & NPY_METH_SUPPORTS_UNALIGNED)) {
            PyErr_Format(PyExc_TypeError,
                    "A cast where input and output DType (class) are identical "
                    "must currently support unaligned data. (method: %s)",
                    meth->method->name);
            return -1;
        }
        if ((meth->method->casting & ~_NPY_CAST_IS_VIEW) != NPY_NO_CASTING) {
            PyErr_Format(PyExc_TypeError,
                    "A cast where input and output DType (class) are identical "
                    "must signal `no-casting`. (method: %s)",
                    meth->method->name);
            return -1;
        }
    }
    if (PyDict_Contains(meth->dtypes[0]->castingimpls,
            (PyObject *)meth->dtypes[1])) {
        PyErr_Format(PyExc_RuntimeError,
                "A cast was already added for %S -> %S. (method: %s)",
                meth->dtypes[0], meth->dtypes[1], meth->method->name);
        return -1;
    }
    if (PyDict_SetItem(meth->dtypes[0]->castingimpls,
            (PyObject *)meth->dtypes[1], (PyObject *)meth->method) < 0) {
        return -1;
    }
    return 0;
}

/**
 * Add a new casting implementation using a PyArrayMethod_Spec.
 *
 * @param spec
 * @param private If private, allow slots not publically exposed.
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
    int res = PyArray_AddCastingImplmentation(meth);
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
        PyArray_Descr *loop_descrs[2])
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = ensure_dtype_nbo(loop_descrs[0]);
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
        return NPY_NO_CASTING | _NPY_CAST_IS_VIEW;
    }
    return NPY_EQUIV_CASTING;
}


NPY_NO_EXPORT int
legacy_cast_get_strided_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArray_Descr **descrs = context->descriptors;
    int out_needs_api = 0;

    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;

    if (PyArray_GetLegacyDTypeTransferFunction(
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
        PyArray_Descr *loop_descrs[2])
{
    assert(dtypes[0]->legacy && dtypes[1]->legacy);

    loop_descrs[0] = ensure_dtype_nbo(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return -1;
    }
    if (given_descrs[1] != NULL) {
        loop_descrs[1] = ensure_dtype_nbo(given_descrs[1]);
        if (loop_descrs[1] == NULL) {
            Py_DECREF(loop_descrs[0]);
            return -1;
        }
    }
    else {
        loop_descrs[1] = dtypes[1]->default_descr(dtypes[1]);
    }

    if (self->casting != NPY_NO_CASTING) {
        return self->casting;
    }
    if (PyDataType_ISNOTSWAPPED(loop_descrs[0]) ==
            PyDataType_ISNOTSWAPPED(loop_descrs[1])) {
        return NPY_NO_CASTING | _NPY_CAST_IS_VIEW;
    }
    return NPY_EQUIV_CASTING;
}


NPY_NO_EXPORT int
get_byteswap_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop, NpyAuxData **out_transferdata,
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
        int aligned, int move_references, npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    static PyObject *cls = NULL;
    int ret;
    npy_cache_import("numpy.core", "ComplexWarning", &cls);
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
            .slots = slots,
            .dtypes = dtypes,
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
        slots[5].slot = NPY_METH_get_loop;
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
    if (dtypes[0]->kind == dtypes[1]->kind && from_itemsize == to_itemsize) {
        spec.casting = NPY_NO_CASTING;

        /* When there is no casting (equivalent C-types) use byteswap loops */
        slots[0].slot = NPY_METH_resolve_descriptors;
        slots[0].pfunc = &legacy_same_dtype_resolve_descriptors;
        slots[1].slot = NPY_METH_get_loop;
        slots[1].pfunc = &get_byteswap_loop;
        slots[2].slot = 0;
        slots[2].pfunc = NULL;

        spec.name = "numeric_copy_or_byteswap";
        spec.flags |= NPY_METH_NO_FLOATINGPOINT_ERRORS;
    }
    else if (_npy_can_cast_safely_table[from->type_num][to->type_num]) {
        spec.casting = NPY_SAFE_CASTING;
    }
    else if (dtype_kind_to_ordering(dtypes[0]->kind) <=
             dtype_kind_to_ordering(dtypes[1]->kind)) {
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
        PyArray_Descr *loop_descrs[2])
{
    /*
     * NOTE: The following code used to be part of PyArray_AdaptFlexibleDType
     *
     * Get a string-size estimate of the input. These
     * are generallly the size needed, rounded up to
     * a multiple of eight.
     */
    npy_intp size = -1;
    switch (dtypes[0]->type_num) {
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
            assert(dtypes[0]->singleton->elsize <= 8);
            assert(dtypes[0]->singleton->elsize > 0);
            if (dtypes[0]->kind == 'b') {
                /* 5 chars needed for cast to 'True' or 'False' */
                size = 5;
            }
            else if (dtypes[0]->kind == 'u') {
                size = REQUIRED_STR_LEN[dtypes[0]->singleton->elsize];
            }
            else if (dtypes[0]->kind == 'i') {
                /* Add character for sign symbol */
                size = REQUIRED_STR_LEN[dtypes[0]->singleton->elsize] + 1;
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
        loop_descrs[1] = ensure_dtype_nbo(given_descrs[1]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }

    /* Set the input one as well (late for easier error management) */
    loop_descrs[0] = ensure_dtype_nbo(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return -1;
    }

    if (self->casting == NPY_UNSAFE_CASTING) {
        assert(dtypes[0]->type_num == NPY_UNICODE &&
               dtypes[1]->type_num == NPY_STRING);
        return NPY_UNSAFE_CASTING;
    }
    assert(self->casting == NPY_SAFE_CASTING);

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
            {NPY_METH_get_loop, &legacy_cast_get_strided_loop},
            {NPY_METH_resolve_descriptors, &cast_to_string_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
        .name = "legacy_cast_to_string",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_REQUIRES_PYAPI,
        .dtypes = dtypes,
        .slots = slots,
    };
    /* Almost everything can be safely cast to string (except unicode) */
    if (other->type_num != NPY_UNICODE) {
        spec.casting = NPY_SAFE_CASTING;
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
        PyArray_Descr *loop_descrs[2])
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = ensure_dtype_nbo(loop_descrs[0]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    if (loop_descrs[0]->elsize == loop_descrs[1]->elsize) {
        if (PyDataType_ISNOTSWAPPED(loop_descrs[0]) ==
                PyDataType_ISNOTSWAPPED(loop_descrs[1])) {
            return NPY_NO_CASTING | _NPY_CAST_IS_VIEW;
        }
        else {
            return NPY_EQUIV_CASTING;
        }
    }
    else if (loop_descrs[0]->elsize <= loop_descrs[1]->elsize) {
        return NPY_SAFE_CASTING;
    }
    return NPY_SAME_KIND_CASTING;
}


NPY_NO_EXPORT int
string_to_string_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop, NpyAuxData **out_transferdata,
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
            {NPY_METH_get_loop, &string_to_string_get_loop},
            {NPY_METH_resolve_descriptors, &string_to_string_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "string_to_string_cast",
            .casting = NPY_NO_CASTING,
            .nin = 1,
            .nout = 1,
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
        PyArray_Descr **given_descrs, PyArray_Descr **loop_descrs)
{
    /* `dtype="V"` means unstructured currently (compare final path) */
    loop_descrs[1] = PyArray_DescrNewFromType(NPY_VOID);
    if (loop_descrs[1] == NULL) {
        return -1;
    }
    loop_descrs[1]->elsize = given_descrs[0]->elsize;
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    return NPY_SAFE_CASTING | _NPY_CAST_IS_VIEW;
}


static NPY_CASTING
nonstructured_to_structured_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2])
{
    NPY_CASTING casting;

    if (given_descrs[1] == NULL) {
        return cast_to_void_dtype_class(given_descrs, loop_descrs);
    }

    if (given_descrs[1]->subarray != NULL) {
        /*
         * We currently consider this at most a safe cast. It would be
         * possible to allow a view if the field has exactly one element.
         */
        casting = NPY_SAFE_CASTING;
        /* Subarray dtype */
        NPY_CASTING base_casting = PyArray_GetCastSafety(
                given_descrs[0], given_descrs[1]->subarray->base, NULL);
        if (base_casting < 0) {
            return -1;
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
            if (PyTuple_Size(given_descrs[1]->names) == 1) {
                /* A view may be acceptable */
                casting |= _NPY_CAST_IS_VIEW;
            }

            Py_ssize_t pos = 0;
            PyObject *key, *tuple;
            while (PyDict_Next(given_descrs[1]->fields, &pos, &key, &tuple)) {
                PyArray_Descr *field_descr = (PyArray_Descr *)PyTuple_GET_ITEM(tuple, 0);
                NPY_CASTING field_casting = PyArray_GetCastSafety(
                        given_descrs[0], field_descr, NULL);
                casting = PyArray_MinCastSafety(casting, field_casting);
                if (casting < 0) {
                    return -1;
                }
            }
        }
    }
    else {
        /* Plain void type. This behaves much like a "view" */
        if (given_descrs[0]->elsize == given_descrs[1]->elsize &&
                !PyDataType_REFCHK(given_descrs[0])) {
            /*
             * A simple view, at the moment considered "safe" (the refcheck is
             * probably not necessary, but more future proof
             */
            casting = NPY_SAFE_CASTING | _NPY_CAST_IS_VIEW;
        }
        else if (given_descrs[0]->elsize <= given_descrs[1]->elsize) {
            casting = NPY_SAFE_CASTING;
        }
        else {
            casting = NPY_UNSAFE_CASTING;
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
        npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (context->descriptors[1]->names != NULL) {
        int needs_api = 0;
        if (get_fields_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else if (context->descriptors[1]->subarray != NULL) {
        int needs_api = 0;
        if (get_subarray_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else {
        /*
         * TODO: This could be a simple zero padded cast, adding a decref
         *       in case of `move_references`. But for now use legacy casts
         *       (which is the behaviour at least up to 1.20).
         */
        int needs_api = 0;
        if (!aligned) {
            /* We need to wrap if aligned is 0. Use a recursive call */

        }
        if (PyArray_GetLegacyDTypeTransferFunction(
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
    method->casting = NPY_SAFE_CASTING;
    method->resolve_descriptors = &nonstructured_to_structured_resolve_descriptors;
    method->get_strided_loop = &nonstructured_to_structured_get_loop;

    return (PyObject *)method;
}


static NPY_CASTING
structured_to_nonstructured_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2])
{
    PyArray_Descr *base_descr;

    if (given_descrs[0]->subarray != NULL) {
        base_descr = given_descrs[0]->subarray->base;
    }
    else if (given_descrs[0]->names != NULL) {
        if (PyTuple_Size(given_descrs[0]->names) != 1) {
            /* Only allow casting a single field */
            return -1;
        }
        PyObject *key = PyTuple_GetItem(given_descrs[0]->names, 0);
        PyObject *base_tup = PyDict_GetItem(given_descrs[0]->fields, key);
        base_descr = (PyArray_Descr *)PyTuple_GET_ITEM(base_tup, 0);
    }
    else {
        /*
         * unstructured voids are considered unsafe casts and defined, albeit,
         * at this time they go back to legacy behaviour using getitem/setitem.
         */
        base_descr = NULL;
    }

    /*
     * The cast is always considered unsafe, so the PyArray_GetCastSafety
     * result currently does not matter.
     */
    if (base_descr != NULL && PyArray_GetCastSafety(
            base_descr, given_descrs[1], dtypes[1]) < 0) {
        return -1;
    }

    /* Void dtypes always do the full cast. */
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = dtypes[1]->default_descr(dtypes[1]);
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
        npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (context->descriptors[0]->names != NULL) {
        int needs_api = 0;
        if (get_fields_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else if (context->descriptors[0]->subarray != NULL) {
        int needs_api = 0;
        if (get_subarray_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else {
        /*
         * In general this is currently defined through legacy behaviour via
         * scalars, and should likely just not be allowed.
         */
        int needs_api = 0;
        if (PyArray_GetLegacyDTypeTransferFunction(
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
    method->casting = NPY_UNSAFE_CASTING;
    method->resolve_descriptors = &structured_to_nonstructured_resolve_descriptors;
    method->get_strided_loop = &structured_to_nonstructured_get_loop;

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
can_cast_fields_safety(PyArray_Descr *from, PyArray_Descr *to)
{
    NPY_CASTING casting = NPY_NO_CASTING | _NPY_CAST_IS_VIEW;

    Py_ssize_t field_count = PyTuple_Size(from->names);
    if (field_count != PyTuple_Size(to->names)) {
        /* TODO: This should be rejected! */
        return NPY_UNSAFE_CASTING;
    }
    for (Py_ssize_t i = 0; i < field_count; i++) {
        PyObject *from_key = PyTuple_GET_ITEM(from->names, i);
        PyObject *from_tup = PyDict_GetItemWithError(from->fields, from_key);
        if (from_tup == NULL) {
            return give_bad_field_error(from_key);
        }
        PyArray_Descr *from_base = (PyArray_Descr*)PyTuple_GET_ITEM(from_tup, 0);

        /*
         * TODO: This should use to_key (order), compare gh-15509 by
         *       by Allan Haldane.  And raise an error on failure.
         *       (Fixing that may also requires fixing/changing promotion.)
         */
        PyObject *to_tup = PyDict_GetItem(to->fields, from_key);
        if (to_tup == NULL) {
            return NPY_UNSAFE_CASTING;
        }
        PyArray_Descr *to_base = (PyArray_Descr*)PyTuple_GET_ITEM(to_tup, 0);

        NPY_CASTING field_casting = PyArray_GetCastSafety(from_base, to_base, NULL);
        if (field_casting < 0) {
            return -1;
        }
        casting = PyArray_MinCastSafety(casting, field_casting);
    }
    if (!(casting & _NPY_CAST_IS_VIEW)) {
        assert((casting & ~_NPY_CAST_IS_VIEW) != NPY_NO_CASTING);
        return casting;
    }

    /*
     * If the itemsize (includes padding at the end), fields, or names
     * do not match, this cannot be a view and also not a "no" cast
     * (identical dtypes).
     * It may be possible that this can be relaxed in some cases.
     */
    if (from->elsize != to->elsize) {
        /*
         * The itemsize may mismatch even if all fields and formats match
         * (due to additional padding).
         */
        return PyArray_MinCastSafety(casting, NPY_EQUIV_CASTING);
    }

    int cmp = PyObject_RichCompareBool(from->fields, to->fields, Py_EQ);
    if (cmp != 1) {
        if (cmp == -1) {
            PyErr_Clear();
        }
        return PyArray_MinCastSafety(casting, NPY_EQUIV_CASTING);
    }
    cmp = PyObject_RichCompareBool(from->names, to->names, Py_EQ);
    if (cmp != 1) {
        if (cmp == -1) {
            PyErr_Clear();
        }
        return PyArray_MinCastSafety(casting, NPY_EQUIV_CASTING);
    }
    return casting;
}


static NPY_CASTING
void_to_void_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2])
{
    NPY_CASTING casting;

    if (given_descrs[1] == NULL) {
        /* This is weird, since it doesn't return the original descr, but... */
        return cast_to_void_dtype_class(given_descrs, loop_descrs);
    }

    if (given_descrs[0]->names != NULL && given_descrs[1]->names != NULL) {
        /* From structured to structured, need to check fields */
        casting = can_cast_fields_safety(given_descrs[0], given_descrs[1]);
    }
    else if (given_descrs[0]->names != NULL) {
        return structured_to_nonstructured_resolve_descriptors(
                self, dtypes, given_descrs, loop_descrs);
    }
    else if (given_descrs[1]->names != NULL) {
        return nonstructured_to_structured_resolve_descriptors(
                self, dtypes, given_descrs, loop_descrs);
    }
    else if (given_descrs[0]->subarray == NULL &&
                given_descrs[1]->subarray == NULL) {
        /* Both are plain void dtypes */
        if (given_descrs[0]->elsize == given_descrs[1]->elsize) {
            casting = NPY_NO_CASTING | _NPY_CAST_IS_VIEW;
        }
        else if (given_descrs[0]->elsize < given_descrs[1]->elsize) {
            casting = NPY_SAFE_CASTING;
        }
        else {
            casting = NPY_SAME_KIND_CASTING;
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
        if (from_sub && to_sub) {
            int res = PyObject_RichCompareBool(from_sub->shape, to_sub->shape, Py_EQ);
            if (res < 0) {
                return -1;
            }
            else if (res) {
                /* Both are subarrays and the shape matches */
                casting = NPY_NO_CASTING | _NPY_CAST_IS_VIEW;
            }
        }
        NPY_CASTING field_casting = PyArray_GetCastSafety(
                given_descrs[0]->subarray->base, given_descrs[1]->subarray->base, NULL);
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
        npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (context->descriptors[0]->names != NULL ||
            context->descriptors[1]->names != NULL) {
        int needs_api = 0;
        if (get_fields_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else if (context->descriptors[0]->subarray != NULL ||
             context->descriptors[1]->subarray != NULL) {
        int needs_api = 0;
        if (get_subarray_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
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
        *flags = 0;
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
            {NPY_METH_get_loop, &void_to_void_get_loop},
            {NPY_METH_resolve_descriptors, &void_to_void_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "void_to_void_cast",
            .casting = NPY_NO_CASTING,
            .nin = 1,
            .nout = 1,
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
        PyArray_Descr *loop_descrs[2])
{
    if (given_descrs[1] == NULL) {
        /*
         * This should not really be called, since object -> parametric casts
         * require inspecting the object array. Allow legacy ones, the path
         * here is that e.g. "M8" input is considered to be the DType class,
         * and by allowing it here, we go back to the "M8" instance.
         */
        if (dtypes[1]->parametric) {
            PyErr_Format(PyExc_TypeError,
                    "casting from object to the parametric DType %S requires "
                    "the specified output dtype instance. "
                    "This may be a NumPy issue, since the correct instance "
                    "should be discovered automatically, however.", dtypes[1]);
            return -1;
        }
        loop_descrs[1] = dtypes[1]->default_descr(dtypes[1]);
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



/* Any object object is simple (could even use the default) */
static NPY_CASTING
any_to_object_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2])
{
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = dtypes[1]->default_descr(dtypes[1]);
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
        npy_intp *NPY_UNUSED(strides),
        PyArray_StridedUnaryOp **out_loop,
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
    /*
     * The object dtype does not support byte order changes, so its cast
     * is always a direct view.
     */
    PyArray_DTypeMeta *Object = PyArray_DTypeFromTypeNum(NPY_OBJECT);
    PyArray_DTypeMeta *dtypes[2] = {Object, Object};
    PyType_Slot slots[] = {
            {NPY_METH_get_loop, &object_to_object_get_loop},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "object_to_object_cast",
            .casting = NPY_NO_CASTING | _NPY_CAST_IS_VIEW,
            .nin = 1,
            .nout = 1,
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
