/*
 * This file implements type resolution for NumPy element-wise ufuncs.
 * This mechanism is still backwards-compatible with the pre-existing
 * legacy mechanism, so performs much slower than is necessary.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>

#include "Python.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"

#include "numpy/ufuncobject.h"
#include "ufunc_type_resolution.h"
#include "ufunc_object.h"
#include "common.h"

#include "mem_overlap.h"
#if defined(HAVE_CBLAS)
#include "cblasfuncs.h"
#endif

static PyObject *
npy_casting_to_py_object(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return PyUString_FromString("no");
        case NPY_EQUIV_CASTING:
            return PyUString_FromString("equiv");
        case NPY_SAFE_CASTING:
            return PyUString_FromString("safe");
        case NPY_SAME_KIND_CASTING:
            return PyUString_FromString("same_kind");
        case NPY_UNSAFE_CASTING:
            return PyUString_FromString("unsafe");
        default:
            return PyInt_FromLong(casting);
    }
}


/**
 * Always returns -1 to indicate the exception was raised, for convenience
 */
static int
raise_binary_type_reso_error(PyUFuncObject *ufunc, PyArrayObject **operands) {
    static PyObject *exc_type = NULL;
    PyObject *exc_value;

    npy_cache_import(
        "numpy.core._exceptions", "_UFuncBinaryResolutionError",
        &exc_type);
    if (exc_type == NULL) {
        return -1;
    }

    /* produce an error object */
    exc_value = Py_BuildValue(
        "O(OO)", ufunc,
        (PyObject *)PyArray_DESCR(operands[0]),
        (PyObject *)PyArray_DESCR(operands[1])
    );
    if (exc_value == NULL){
        return -1;
    }
    PyErr_SetObject(exc_type, exc_value);
    Py_DECREF(exc_value);

    return -1;
}

/** Helper function to raise UFuncNoLoopError
 * Always returns -1 to indicate the exception was raised, for convenience
 */
static int
raise_no_loop_found_error(
        PyUFuncObject *ufunc, PyArray_Descr **dtypes)
{
    static PyObject *exc_type = NULL;
    PyObject *exc_value;
    PyObject *dtypes_tup;
    npy_intp i;

    npy_cache_import(
        "numpy.core._exceptions", "_UFuncNoLoopError",
        &exc_type);
    if (exc_type == NULL) {
        return -1;
    }

    /* convert dtypes to a tuple */
    dtypes_tup = PyTuple_New(ufunc->nargs);
    if (dtypes_tup == NULL) {
        return -1;
    }
    for (i = 0; i < ufunc->nargs; ++i) {
        Py_INCREF(dtypes[i]);
        PyTuple_SET_ITEM(dtypes_tup, i, (PyObject *)dtypes[i]);
    }

    /* produce an error object */
    exc_value = PyTuple_Pack(2, ufunc, dtypes_tup);
    Py_DECREF(dtypes_tup);
    if (exc_value == NULL){
        return -1;
    }
    PyErr_SetObject(exc_type, exc_value);
    Py_DECREF(exc_value);

    return -1;
}

static int
raise_casting_error(
        PyObject *exc_type,
        PyUFuncObject *ufunc,
        NPY_CASTING casting,
        PyArray_Descr *from,
        PyArray_Descr *to,
        npy_intp i)
{
    PyObject *exc_value;
    PyObject *casting_value;

    casting_value = npy_casting_to_py_object(casting);
    if (casting_value == NULL) {
        return -1;
    }

    exc_value = Py_BuildValue(
        "ONOOi",
        ufunc,
        casting_value,
        (PyObject *)from,
        (PyObject *)to,
        i
    );
    if (exc_value == NULL){
        return -1;
    }
    PyErr_SetObject(exc_type, exc_value);
    Py_DECREF(exc_value);

    return -1;
}

/** Helper function to raise UFuncInputCastingError
 * Always returns -1 to indicate the exception was raised, for convenience
 */
static int
raise_input_casting_error(
        PyUFuncObject *ufunc,
        NPY_CASTING casting,
        PyArray_Descr *from,
        PyArray_Descr *to,
        npy_intp i)
{
    static PyObject *exc_type = NULL;
    npy_cache_import(
        "numpy.core._exceptions", "_UFuncInputCastingError",
        &exc_type);
    if (exc_type == NULL) {
        return -1;
    }

    return raise_casting_error(exc_type, ufunc, casting, from, to, i);
}


/** Helper function to raise UFuncOutputCastingError
 * Always returns -1 to indicate the exception was raised, for convenience
 */
static int
raise_output_casting_error(
        PyUFuncObject *ufunc,
        NPY_CASTING casting,
        PyArray_Descr *from,
        PyArray_Descr *to,
        npy_intp i)
{
    static PyObject *exc_type = NULL;
    npy_cache_import(
        "numpy.core._exceptions", "_UFuncOutputCastingError",
        &exc_type);
    if (exc_type == NULL) {
        return -1;
    }

    return raise_casting_error(exc_type, ufunc, casting, from, to, i);
}


/*UFUNC_API
 *
 * Validates that the input operands can be cast to
 * the input types, and the output types can be cast to
 * the output operands where provided.
 *
 * Returns 0 on success, -1 (with exception raised) on validation failure.
 */
NPY_NO_EXPORT int
PyUFunc_ValidateCasting(PyUFuncObject *ufunc,
                            NPY_CASTING casting,
                            PyArrayObject **operands,
                            PyArray_Descr **dtypes)
{
    int i, nin = ufunc->nin, nop = nin + ufunc->nout;

    for (i = 0; i < nop; ++i) {
        if (i < nin) {
            if (!PyArray_CanCastArrayTo(operands[i], dtypes[i], casting)) {
                return raise_input_casting_error(
                    ufunc, casting, PyArray_DESCR(operands[i]), dtypes[i], i);
            }
        } else if (operands[i] != NULL) {
            if (!PyArray_CanCastTypeTo(dtypes[i],
                                    PyArray_DESCR(operands[i]), casting)) {
                return raise_output_casting_error(
                    ufunc, casting, dtypes[i], PyArray_DESCR(operands[i]), i);
            }
        }
    }

    return 0;
}


NPY_NO_EXPORT int
PyUFunc_ValidateCastingDtypes(PyUFuncObject *ufunc,
                              NPY_CASTING casting,
                              PyArray_Descr **op_dtypes,
                              PyArray_Descr **dtypes)
{
    int i, nin = ufunc->nin, nop = nin + ufunc->nout;

    for (i = 0; i < nop; i++) {
        if (i < nin) {
            if (!PyArray_CanCastTypeTo(op_dtypes[i], dtypes[i], casting)) {
                return raise_input_casting_error(
                    ufunc, casting, op_dtypes[i], dtypes[i], i);
            }
        }
        else if (op_dtypes[i] != NULL) {
            if (!PyArray_CanCastTypeTo(dtypes[i], op_dtypes[i], casting)) {
                return raise_output_casting_error(
                    ufunc, casting, dtypes[i], op_dtypes[i], i);
            }
        }
    }

    return 0;
}


/*
 * Returns a new reference to type if it is already NBO, otherwise
 * returns a copy converted to NBO.
 */
static PyArray_Descr *
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

/*UFUNC_API
 *
 * This function applies the default type resolution rules
 * for the provided ufunc.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_DefaultTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    int i, nop = ufunc->nin + ufunc->nout;
    int retval = 0, any_object = 0;
    NPY_CASTING input_casting;

    for (i = 0; i < nop; ++i) {
        if (op_dtypes[i] != NULL &&
                PyTypeNum_ISOBJECT(op_dtypes[i]->type_num)) {
            any_object = 1;
            break;
        }
    }

    /*
     * Decide the casting rules for inputs and outputs.  We want
     * NPY_SAFE_CASTING or stricter, so that the loop selection code
     * doesn't choose an integer loop for float inputs, or a float32
     * loop for float64 inputs.
     */
    input_casting = (casting > NPY_SAFE_CASTING) ? NPY_SAFE_CASTING : casting;

    /* Find the best ufunc inner loop, and fill in the dtypes */
    retval = linear_search_type_resolver(ufunc, op_dtypes,
                    input_casting, casting, any_object,
                    out_dtypes);

    return retval;
}

/*
 * This function applies special type resolution rules for the case
 * where all the functions have the pattern XX->bool, using
 * PyArray_ResultType instead of a linear search to get the best
 * loop.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_SimpleBinaryComparisonTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    int i, type_num1, type_num2;
    const char *ufunc_name = ufunc_get_name_cstr(ufunc);

    if (ufunc->nin != 2 || ufunc->nout != 1) {
        PyErr_Format(PyExc_RuntimeError, "ufunc %s is configured "
                "to use binary comparison type resolution but has "
                "the wrong number of inputs or outputs",
                ufunc_name);
        return -1;
    }

    /*
     * Use the default type resolution if there's a custom data type
     * or object arrays.
     */
    type_num1 = op_dtypes[0]->type_num;
    type_num2 = op_dtypes[1]->type_num;
    if (type_num1 >= NPY_NTYPES || type_num2 >= NPY_NTYPES ||
            type_num1 == NPY_OBJECT || type_num2 == NPY_OBJECT) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting,
                                           op_dtypes, out_dtypes);
    }

    // TODO: DELETED TOO MUCH HERE, IT NEEDS TO FILL out_dtypes!

    /* Output type is always boolean */
    if (out_dtypes[2] != NULL) {
        if (out_dtypes[2]->type_num != NPY_BOOL) {
            PyErr_SetString(PyExc_TypeError, "needs a better message/path");
            return -1;
        }
    }
    else {
        out_dtypes[2] = PyArray_DescrFromType(NPY_BOOL);
        if (out_dtypes[2] == NULL) {  // probably cannot fail?
            return -1;
        }
    }

    /* Check against the casting rules */
    if (PyUFunc_ValidateCastingDtypes(ufunc, casting, op_dtypes, out_dtypes) < 0) {
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    return 0;
}

NPY_NO_EXPORT int
PyUFunc_NegativeTypeResolver(PyUFuncObject *ufunc,
                             NPY_CASTING casting,
                             PyArray_Descr **op_dtypes,
                             PyArray_Descr **out_dtypes)
{
    int ret;
    ret = PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting,
                                                     op_dtypes, out_dtypes);
    if (ret < 0) {
        return ret;
    }

    /* The type resolver would have upcast already */
    if (out_dtypes[0]->type_num == NPY_BOOL) {
        PyErr_Format(PyExc_TypeError,
            "The numpy boolean negative, the `-` operator, is not supported, "
            "use the `~` operator or the logical_not function instead.");
        return -1;
    }

    return ret;
}


/*
 * The ones_like function shouldn't really be a ufunc, but while it
 * still is, this provides type resolution that always forces UNSAFE
 * casting.
 */
NPY_NO_EXPORT int
PyUFunc_OnesLikeTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING NPY_UNUSED(casting),
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    return PyUFunc_SimpleUniformOperationTypeResolver(ufunc,
                        NPY_UNSAFE_CASTING,
                        op_dtypes, out_dtypes);
}

/*
 * This function applies special type resolution rules for the case
 * where all of the types in the signature are the same, eg XX->X or XX->XX.
 * It uses PyArray_ResultType instead of a linear search to get the best
 * loop.
 *
 * Note that a simpler linear search through the functions loop
 * is still done, but switching to a simple array lookup for
 * built-in types would be better at some point.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_SimpleUniformOperationTypeResolver(
        PyUFuncObject *ufunc,
        NPY_CASTING casting,
        PyArray_Descr **op_dtypes,
        PyArray_Descr **out_dtypes)
{
    const char *ufunc_name = ufunc_get_name_cstr(ufunc);

    if (ufunc->nin < 1) {
        PyErr_Format(PyExc_RuntimeError, "ufunc %s is configured "
                "to use uniform operation type resolution but has "
                "no inputs",
                ufunc_name);
        return -1;
    }
    int nop = ufunc->nin + ufunc->nout;

    /*
     * There's a custom data type or an object array
     */
    bool has_custom_or_object = false;
    for (int iop = 0; iop < ufunc->nin; iop++) {
        int type_num = op_dtypes[iop]->type_num;
        if (type_num >= NPY_NTYPES || type_num == NPY_OBJECT) {
            has_custom_or_object = true;
            break;
        }
    }

    if (has_custom_or_object) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting, op_dtypes, out_dtypes);
    }

    if (out_dtypes[0] == NULL) {
        /* PyArray_ResultType forgets to force a byte order when n == 1 */
        if (ufunc->nin == 1){
            out_dtypes[0] = ensure_dtype_nbo(op_dtypes[0]);
        }
        else {
            out_dtypes[0] = PyArray_ResultType(0, NULL, ufunc->nin, op_dtypes);
        }
        if (out_dtypes[0] == NULL) {
            return -1;
        }
    }
    else {
        /* Code used to ensure native byte order on the out_dtype[0] */
    }

    /* All types are the same - copy the first one to the rest */
    for (int iop = 1; iop < nop; iop++) {
        if (out_dtypes[iop] == NULL) {
            out_dtypes[iop] = out_dtypes[0];
            Py_INCREF(out_dtypes[iop]);
        }
    }

    /* Check against the casting rules */
    if (PyUFunc_ValidateCastingDtypes(ufunc, casting, op_dtypes, out_dtypes) < 0) {
        return -1;
    }

    return 0;
}

/*
 * This function applies special type resolution rules for the absolute
 * ufunc. This ufunc converts complex -> float, so isn't covered
 * by the simple unary type resolution.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_AbsoluteTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    /* Use the default for complex types, to find the loop producing float */
    if (PyTypeNum_ISCOMPLEX(op_dtypes[0]->type_num)) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting, op_dtypes,
                                           out_dtypes);
    }
    else {
        return PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting,
                                                          op_dtypes, out_dtypes);
    }
}

/*
 * This function applies special type resolution rules for the isnat
 * ufunc. This ufunc converts datetime/timedelta -> bool, and is not covered
 * by the simple unary type resolution.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_IsNaTTypeResolver(PyUFuncObject *ufunc,
                          NPY_CASTING casting,
                          PyArray_Descr **op_dtypes,
                          PyArray_Descr **out_dtypes)
{
    if (!PyTypeNum_ISDATETIME(op_dtypes[0]->type_num)) {
        PyErr_SetString(PyExc_TypeError,
                "ufunc 'isnat' is only defined for datetime and timedelta.");
        return -1;
    }

    out_dtypes[0] = ensure_dtype_nbo(op_dtypes[0]);
    out_dtypes[1] = PyArray_DescrFromType(NPY_BOOL);

    return 0;
}


NPY_NO_EXPORT int
PyUFunc_IsFiniteTypeResolver(PyUFuncObject *ufunc,
                          NPY_CASTING casting,
                          PyArray_Descr **op_dtypes,
                          PyArray_Descr **out_dtypes)
{
    if (!PyTypeNum_ISDATETIME(op_dtypes[0]->type_num)) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting,
                                           op_dtypes, out_dtypes);
    }

    out_dtypes[0] = ensure_dtype_nbo(op_dtypes[0]);
    out_dtypes[1] = PyArray_DescrFromType(NPY_BOOL);

    return 0;
}


/*
 * Creates a new NPY_TIMEDELTA dtype, copying the datetime metadata
 * from the given dtype.
 *
 * NOTE: This function is copied from datetime.c in multiarray,
 *       because umath and multiarray are not linked together.
 */
static PyArray_Descr *
timedelta_dtype_with_copied_meta(PyArray_Descr *dtype)
{
    PyArray_Descr *ret;
    PyArray_DatetimeMetaData *dst, *src;
    PyArray_DatetimeDTypeMetaData *dst_dtmd, *src_dtmd;

    ret = PyArray_DescrNewFromType(NPY_TIMEDELTA);
    if (ret == NULL) {
        return NULL;
    }

    src_dtmd = ((PyArray_DatetimeDTypeMetaData *)dtype->c_metadata);
    dst_dtmd = ((PyArray_DatetimeDTypeMetaData *)ret->c_metadata);
    src = &(src_dtmd->meta);
    dst = &(dst_dtmd->meta);

    *dst = *src;

    return ret;
}

/*
 * This function applies the type resolution rules for addition.
 * In particular, there are a number of special cases with datetime:
 *    m8[<A>] + m8[<B>] => m8[gcd(<A>,<B>)] + m8[gcd(<A>,<B>)]
 *    m8[<A>] + int     => m8[<A>] + m8[<A>]
 *    int     + m8[<A>] => m8[<A>] + m8[<A>]
 *    M8[<A>] + int     => M8[<A>] + m8[<A>]
 *    int     + M8[<A>] => m8[<A>] + M8[<A>]
 *    M8[<A>] + m8[<B>] => M8[gcd(<A>,<B>)] + m8[gcd(<A>,<B>)]
 *    m8[<A>] + M8[<B>] => m8[gcd(<A>,<B>)] + M8[gcd(<A>,<B>)]
 * TODO: Non-linear time unit cases require highly special-cased loops
 *    M8[<A>] + m8[Y|M|B]
 *    m8[Y|M|B] + M8[<A>]
 */
NPY_NO_EXPORT int
PyUFunc_AdditionTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    type_num1 = op_dtypes[0]->type_num;
    type_num2 = op_dtypes[1]->type_num;

    /* Use the default when datetime and timedelta are not involved */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        return PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting,
                    op_dtypes, out_dtypes);
    }

    if (type_num1 == NPY_TIMEDELTA) {
        /* m8[<A>] + m8[<B>] => m8[gcd(<A>,<B>)] + m8[gcd(<A>,<B>)] */
        if (type_num2 == NPY_TIMEDELTA) {
            out_dtypes[0] = PyArray_PromoteTypes(op_dtypes[0],
                                                op_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);
        }
        /* m8[<A>] + M8[<B>] => m8[gcd(<A>,<B>)] + M8[gcd(<A>,<B>)] */
        else if (type_num2 == NPY_DATETIME) {
            out_dtypes[1] = PyArray_PromoteTypes(op_dtypes[0],
                                                op_dtypes[1]);
            if (out_dtypes[1] == NULL) {
                return -1;
            }
            /* Make a new NPY_TIMEDELTA, and copy the datetime's metadata */
            out_dtypes[0] = timedelta_dtype_with_copied_meta(out_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                Py_DECREF(out_dtypes[1]);
                out_dtypes[1] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[1];
            Py_INCREF(out_dtypes[2]);
        }
        /* m8[<A>] + int => m8[<A>] + m8[<A>] */
        else if (PyTypeNum_ISINTEGER(type_num2) ||
                                    PyTypeNum_ISBOOL(type_num2)) {
            out_dtypes[0] = ensure_dtype_nbo(op_dtypes[0]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_TIMEDELTA;
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else if (type_num1 == NPY_DATETIME) {
        /* M8[<A>] + m8[<B>] => M8[gcd(<A>,<B>)] + m8[gcd(<A>,<B>)] */
        if (type_num2 == NPY_TIMEDELTA) {
            out_dtypes[0] = PyArray_PromoteTypes(op_dtypes[0],
                                                op_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            /* Make a new NPY_TIMEDELTA, and copy the datetime's metadata */
            out_dtypes[1] = timedelta_dtype_with_copied_meta(out_dtypes[0]);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);
        }
        /* M8[<A>] + int => M8[<A>] + m8[<A>] */
        else if (PyTypeNum_ISINTEGER(type_num2) ||
                    PyTypeNum_ISBOOL(type_num2)) {
            out_dtypes[0] = ensure_dtype_nbo(op_dtypes[0]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            /* Make a new NPY_TIMEDELTA, and copy type1's metadata */
            out_dtypes[1] = timedelta_dtype_with_copied_meta(
                                            op_dtypes[0]);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_TIMEDELTA;
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else if (PyTypeNum_ISINTEGER(type_num1) || PyTypeNum_ISBOOL(type_num1)) {
        /* int + m8[<A>] => m8[<A>] + m8[<A>] */
        if (type_num2 == NPY_TIMEDELTA) {
            out_dtypes[0] = ensure_dtype_nbo(op_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num1 = NPY_TIMEDELTA;
        }
        else if (type_num2 == NPY_DATETIME) {
            /* Make a new NPY_TIMEDELTA, and copy type2's metadata */
            out_dtypes[0] = timedelta_dtype_with_copied_meta(
                                            op_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = ensure_dtype_nbo(op_dtypes[1]);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[1];
            Py_INCREF(out_dtypes[2]);

            type_num1 = NPY_TIMEDELTA;
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else {
        return raise_binary_type_reso_error(ufunc, op_dtypes);
    }

    /* Check against the casting rules */
    if (PyUFunc_ValidateCastingDtypes(ufunc, casting, op_dtypes, out_dtypes) < 0) {
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    return 0;
}

/*
 * This function applies the type resolution rules for subtraction.
 * In particular, there are a number of special cases with datetime:
 *    m8[<A>] - m8[<B>] => m8[gcd(<A>,<B>)] - m8[gcd(<A>,<B>)]
 *    m8[<A>] - int     => m8[<A>] - m8[<A>]
 *    int     - m8[<A>] => m8[<A>] - m8[<A>]
 *    M8[<A>] - int     => M8[<A>] - m8[<A>]
 *    M8[<A>] - m8[<B>] => M8[gcd(<A>,<B>)] - m8[gcd(<A>,<B>)]
 * TODO: Non-linear time unit cases require highly special-cased loops
 *    M8[<A>] - m8[Y|M|B]
 */
NPY_NO_EXPORT int
PyUFunc_SubtractionTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    type_num1 = op_dtypes[0]->type_num;
    type_num2 = op_dtypes[1]->type_num;

    /* Use the default when datetime and timedelta are not involved */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        int ret;
        ret = PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting,
                                                         op_dtypes, out_dtypes);
        if (ret < 0) {
            return ret;
        }

        /* The type resolver would have upcast already */
        if (out_dtypes[0]->type_num == NPY_BOOL) {
            PyErr_Format(PyExc_TypeError,
                "numpy boolean subtract, the `-` operator, is deprecated, "
                "use the bitwise_xor, the `^` operator, or the logical_xor "
                "function instead.");
            return -1;
        }
        return ret;
    }

    if (type_num1 == NPY_TIMEDELTA) {
        /* m8[<A>] - m8[<B>] => m8[gcd(<A>,<B>)] - m8[gcd(<A>,<B>)] */
        if (type_num2 == NPY_TIMEDELTA) {
            Py_XSETREF(out_dtypes[0],
                       PyArray_PromoteTypes(op_dtypes[0], op_dtypes[1]));
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            Py_XSETREF(out_dtypes[1], out_dtypes[0]);
            Py_INCREF(out_dtypes[1]);
            Py_XSETREF(out_dtypes[2], out_dtypes[0]);
            Py_INCREF(out_dtypes[2]);
        }
        /* m8[<A>] - int => m8[<A>] - m8[<A>] */
        else if (PyTypeNum_ISINTEGER(type_num2) ||
                                        PyTypeNum_ISBOOL(type_num2)) {
            Py_XSETREF(out_dtypes[0], ensure_dtype_nbo(op_dtypes[0]));
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            Py_XSETREF(out_dtypes[1], out_dtypes[0]);
            Py_INCREF(out_dtypes[1]);
            Py_XSETREF(out_dtypes[2], out_dtypes[0]);
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_TIMEDELTA;
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else if (type_num1 == NPY_DATETIME) {
        /* M8[<A>] - m8[<B>] => M8[gcd(<A>,<B>)] - m8[gcd(<A>,<B>)] */
        if (type_num2 == NPY_TIMEDELTA) {
            out_dtypes[0] = PyArray_PromoteTypes(op_dtypes[0],
                                                op_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            /* Make a new NPY_TIMEDELTA, and copy the datetime's metadata */
            out_dtypes[1] = timedelta_dtype_with_copied_meta(out_dtypes[0]);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);
        }
        /* M8[<A>] - int => M8[<A>] - m8[<A>] */
        else if (PyTypeNum_ISINTEGER(type_num2) ||
                    PyTypeNum_ISBOOL(type_num2)) {
            out_dtypes[0] = ensure_dtype_nbo(op_dtypes[0]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            /* Make a new NPY_TIMEDELTA, and copy type1's metadata */
            out_dtypes[1] = timedelta_dtype_with_copied_meta(
                                            op_dtypes[0]);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_TIMEDELTA;
        }
        /* M8[<A>] - M8[<B>] => M8[gcd(<A>,<B>)] - M8[gcd(<A>,<B>)] */
        else if (type_num2 == NPY_DATETIME) {
            out_dtypes[0] = PyArray_PromoteTypes(op_dtypes[0],
                                                op_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            /* Make a new NPY_TIMEDELTA, and copy type1's metadata */
            out_dtypes[2] = timedelta_dtype_with_copied_meta(out_dtypes[0]);
            if (out_dtypes[2] == NULL) {
                Py_DECREF(out_dtypes[0]);
                return -1;
            }
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else if (PyTypeNum_ISINTEGER(type_num1) || PyTypeNum_ISBOOL(type_num1)) {
        /* int - m8[<A>] => m8[<A>] - m8[<A>] */
        if (type_num2 == NPY_TIMEDELTA) {
            out_dtypes[0] = ensure_dtype_nbo(op_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num1 = NPY_TIMEDELTA;
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else {
        return raise_binary_type_reso_error(ufunc, op_dtypes);
    }

    /* Check against the casting rules */
    if (PyUFunc_ValidateCastingDtypes(ufunc, casting, op_dtypes, out_dtypes) < 0) {
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        printf("raising errors from here\n");
        return -1;
    }

    return 0;
}

/*
 * This function applies the type resolution rules for multiplication.
 * In particular, there are a number of special cases with datetime:
 *    int## * m8[<A>] => int64 * m8[<A>]
 *    m8[<A>] * int## => m8[<A>] * int64
 *    float## * m8[<A>] => float64 * m8[<A>]
 *    m8[<A>] * float## => m8[<A>] * float64
 */
NPY_NO_EXPORT int
PyUFunc_MultiplicationTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    type_num1 = op_dtypes[0]->type_num;
    type_num2 = op_dtypes[1]->type_num;

    /* Use the default when datetime and timedelta are not involved */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        return PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting,
                    op_dtypes, out_dtypes);
    }

    if (type_num1 == NPY_TIMEDELTA) {
        /* m8[<A>] * int## => m8[<A>] * int64 */
        if (PyTypeNum_ISINTEGER(type_num2) || PyTypeNum_ISBOOL(type_num2)) {
            out_dtypes[0] = ensure_dtype_nbo(op_dtypes[0]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = PyArray_DescrNewFromType(NPY_LONGLONG);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_LONGLONG;
        }
        /* m8[<A>] * float## => m8[<A>] * float64 */
        else if (PyTypeNum_ISFLOAT(type_num2)) {
            out_dtypes[0] = ensure_dtype_nbo(op_dtypes[0]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = PyArray_DescrNewFromType(NPY_DOUBLE);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_DOUBLE;
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else if (PyTypeNum_ISINTEGER(type_num1) || PyTypeNum_ISBOOL(type_num1)) {
        /* int## * m8[<A>] => int64 * m8[<A>] */
        if (type_num2 == NPY_TIMEDELTA) {
            out_dtypes[0] = PyArray_DescrNewFromType(NPY_LONGLONG);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = ensure_dtype_nbo(op_dtypes[1]);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[1];
            Py_INCREF(out_dtypes[2]);

            type_num1 = NPY_LONGLONG;
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else if (PyTypeNum_ISFLOAT(type_num1)) {
        /* float## * m8[<A>] => float64 * m8[<A>] */
        if (type_num2 == NPY_TIMEDELTA) {
            out_dtypes[0] = PyArray_DescrNewFromType(NPY_DOUBLE);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = ensure_dtype_nbo(op_dtypes[1]);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[1];
            Py_INCREF(out_dtypes[2]);

            type_num1 = NPY_DOUBLE;
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else {
        return raise_binary_type_reso_error(ufunc, op_dtypes);
    }

    /* Check against the casting rules */
    if (PyUFunc_ValidateCastingDtypes(ufunc, casting, op_dtypes, out_dtypes) < 0) {
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    return 0;
}


/*
 * This function applies the type resolution rules for division.
 * In particular, there are a number of special cases with datetime:
 *    m8[<A>] / m8[<B>] to  m8[gcd(<A>,<B>)] / m8[gcd(<A>,<B>)]  -> float64
 *    m8[<A>] / int##   to m8[<A>] / int64 -> m8[<A>]
 *    m8[<A>] / float## to m8[<A>] / float64 -> m8[<A>]
 */
NPY_NO_EXPORT int
PyUFunc_DivisionTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    type_num1 = op_dtypes[0]->type_num;
    type_num2 = op_dtypes[1]->type_num;

    /* Use the default when datetime and timedelta are not involved */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting,
                                           op_dtypes, out_dtypes);
    }

    if (type_num1 == NPY_TIMEDELTA) {
        /*
         * m8[<A>] / m8[<B>] to
         * m8[gcd(<A>,<B>)] / m8[gcd(<A>,<B>)]  -> float64
         *
         * NOTE: Branch completely ignores fixed dtypes:
         */
        if (type_num2 == NPY_TIMEDELTA) {
            Py_XSETREF(out_dtypes[0],
                       PyArray_PromoteTypes(op_dtypes[0], op_dtypes[1]));
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            Py_XSETREF(out_dtypes[1], out_dtypes[0]);
            Py_INCREF(out_dtypes[1]);

            /*
             * TODO: split function into truediv and floordiv resolvers
             */
            if (strcmp(ufunc->name, "floor_divide") == 0) {
                Py_XSETREF(out_dtypes[2], PyArray_DescrFromType(NPY_LONGLONG));
            }
            else {
                Py_XSETREF(out_dtypes[2], PyArray_DescrFromType(NPY_DOUBLE));
            }
            if (out_dtypes[2] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                Py_DECREF(out_dtypes[1]);
                out_dtypes[1] = NULL;
                return -1;
            }
        }
        /* m8[<A>] / int## => m8[<A>] / int64 */
        else if (PyTypeNum_ISINTEGER(type_num2)) {
            out_dtypes[0] = ensure_dtype_nbo(op_dtypes[0]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = PyArray_DescrFromType(NPY_LONGLONG);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_LONGLONG;
        }
        /* m8[<A>] / float## => m8[<A>] / float64 */
        else if (PyTypeNum_ISFLOAT(type_num2)) {
            Py_XSETREF(out_dtypes[0], ensure_dtype_nbo(op_dtypes[0]));
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            Py_XSETREF(out_dtypes[1], PyArray_DescrNewFromType(NPY_DOUBLE));
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            Py_XSETREF(out_dtypes[2], out_dtypes[0]);
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_DOUBLE;
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else {
        return raise_binary_type_reso_error(ufunc, op_dtypes);
    }

    /* Check against the casting rules */
    if (PyUFunc_ValidateCastingDtypes(ufunc, casting, op_dtypes, out_dtypes) < 0) {
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    return 0;
}


NPY_NO_EXPORT int
PyUFunc_RemainderTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    type_num1 = op_dtypes[0]->type_num;
    type_num2 = op_dtypes[1]->type_num;

    /* Use the default when datetime and timedelta are not involved */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting, op_dtypes, out_dtypes);
    }
    if (type_num1 == NPY_TIMEDELTA) {
        if (type_num2 == NPY_TIMEDELTA) {
            out_dtypes[0] = PyArray_PromoteTypes(op_dtypes[0],
                                                op_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else {
        return raise_binary_type_reso_error(ufunc, op_dtypes);
    }

    /* Check against the casting rules */
    if (PyUFunc_ValidateCastingDtypes(ufunc, casting, op_dtypes, out_dtypes) < 0) {
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    return 0;
}


/*
 * True division should return float64 results when both inputs are integer
 * types. The PyUFunc_DefaultTypeResolver promotes 8 bit integers to float16
 * and 16 bit integers to float32, so that is overridden here by specifying a
 * 'dd->d' signature. Returns -1 on failure.
*/
NPY_NO_EXPORT int
PyUFunc_TrueDivisionTypeResolver(PyUFuncObject *ufunc,
                                 NPY_CASTING casting,
                                 PyArray_Descr **op_dtypes,
                                 PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;

    type_num1 = op_dtypes[0]->type_num;
    type_num2 = op_dtypes[1]->type_num;

    if ((out_dtypes[0] == NULL) &&
                (out_dtypes[1] == NULL) &&
                (out_dtypes[2] == NULL) &&
            (PyTypeNum_ISINTEGER(type_num1) || PyTypeNum_ISBOOL(type_num1)) &&
            (PyTypeNum_ISINTEGER(type_num2) || PyTypeNum_ISBOOL(type_num2))) {
        for (int i = 0; i < 3; i++) {
            out_dtypes[i] = PyArray_DescrFromType(NPY_DOUBLE);
        }
        return PyUFunc_DefaultTypeResolver(ufunc, casting,
                                           op_dtypes, out_dtypes);
    }
    return PyUFunc_DivisionTypeResolver(ufunc, casting, op_dtypes, out_dtypes);
}
/*
 * Function to check and report floor division warning when python2.x is
 * invoked with -3 switch
 * See PEP238 and #7949 for numpy
 * This function will not be hit for py3 or when __future__ imports division.
 * See generate_umath.py for reason
*/
NPY_NO_EXPORT int
PyUFunc_MixedDivisionTypeResolver(PyUFuncObject *ufunc,
                                  NPY_CASTING casting,
                                  PyArray_Descr **op_dtypes,
                                  PyArray_Descr **out_dtypes)
{
 /* Deprecation checks needed only on python 2 */
#if !defined(NPY_PY3K)
    int type_num1, type_num2;

    type_num1 = op_dtypes[0]->type_num;
    type_num2 = op_dtypes[1]->type_num;

    /* If both types are integer, warn the user, same as python does */
    if (Py_DivisionWarningFlag &&
            (PyTypeNum_ISINTEGER(type_num1) || PyTypeNum_ISBOOL(type_num1)) &&
            (PyTypeNum_ISINTEGER(type_num2) || PyTypeNum_ISBOOL(type_num2))) {
        PyErr_Warn(PyExc_DeprecationWarning, "numpy: classic int division");
    }
#endif
    return PyUFunc_DivisionTypeResolver(ufunc, casting, op_dtypes, out_dtypes);
}

static int
find_userloop(PyUFuncObject *ufunc,
                PyArray_Descr **dtypes,
                PyUFuncGenericFunction *out_innerloop,
                void **out_innerloopdata)
{
    npy_intp i, nin = ufunc->nin, j, nargs = nin + ufunc->nout;
    PyUFunc_Loop1d *funcdata;

    /* Use this to try to avoid repeating the same userdef loop search */
    int last_userdef = -1;

    for (i = 0; i < nargs; ++i) {
        int type_num;

        /* no more ufunc arguments to check */
        if (dtypes[i] == NULL) {
            break;
        }

        type_num = dtypes[i]->type_num;
        if (type_num != last_userdef &&
                (PyTypeNum_ISUSERDEF(type_num) || type_num == NPY_VOID)) {
            PyObject *key, *obj;

            last_userdef = type_num;

            key = PyInt_FromLong(type_num);
            if (key == NULL) {
                return -1;
            }
            obj = PyDict_GetItem(ufunc->userloops, key);
            Py_DECREF(key);
            if (obj == NULL) {
                continue;
            }
            for (funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(obj);
                 funcdata != NULL;
                 funcdata = funcdata->next) {
                int *types = funcdata->arg_types;

                for (j = 0; j < nargs; ++j) {
                    if (types[j] != dtypes[j]->type_num) {
                        break;
                    }
                }
                /* It matched */
                if (j == nargs) {
                    *out_innerloop = funcdata->func;
                    *out_innerloopdata = funcdata->data;
                    return 1;
                }
            }
        }
    }

    /* Didn't find a match */
    return 0;
}

NPY_NO_EXPORT int
PyUFunc_DefaultLegacyInnerLoopSelector(PyUFuncObject *ufunc,
                                PyArray_Descr **dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata,
                                int *out_needs_api)
{
    int nargs = ufunc->nargs;
    char *types;
    int i, j;

    /*
     * If there are user-loops search them first.
     * TODO: There needs to be a loop selection acceleration structure,
     *       like a hash table.
     */
    if (ufunc->userloops) {
        switch (find_userloop(ufunc, dtypes,
                    out_innerloop, out_innerloopdata)) {
            /* Error */
            case -1:
                return -1;
            /* Found a loop */
            case 1:
                return 0;
        }
    }

    types = ufunc->types;
    for (i = 0; i < ufunc->ntypes; ++i) {
        /* Copy the types into an int array for matching */
        for (j = 0; j < nargs; ++j) {
            if (types[j] != dtypes[j]->type_num) {
                break;
            }
        }
        if (j == nargs) {
            *out_innerloop = ufunc->functions[i];
            *out_innerloopdata = ufunc->data[i];
            return 0;
        }

        types += nargs;
    }

    return raise_no_loop_found_error(ufunc, dtypes);
}

typedef struct {
    NpyAuxData base;
    PyUFuncGenericFunction unmasked_innerloop;
    void *unmasked_innerloopdata;
    int nargs;
} _ufunc_masker_data;

static NpyAuxData *
ufunc_masker_data_clone(NpyAuxData *data)
{
    _ufunc_masker_data *n;

    /* Allocate a new one */
    n = (_ufunc_masker_data *)PyArray_malloc(sizeof(_ufunc_masker_data));
    if (n == NULL) {
        return NULL;
    }

    /* Copy the data (unmasked data doesn't have object semantics) */
    memcpy(n, data, sizeof(_ufunc_masker_data));

    return (NpyAuxData *)n;
}

/*
 * This function wraps a regular unmasked ufunc inner loop as a
 * masked ufunc inner loop, only calling the function for
 * elements where the mask is True.
 */
static void
unmasked_ufunc_loop_as_masked(
             char **dataptrs, npy_intp *strides,
             char *mask, npy_intp mask_stride,
             npy_intp loopsize,
             NpyAuxData *innerloopdata)
{
    _ufunc_masker_data *data;
    int iargs, nargs;
    PyUFuncGenericFunction unmasked_innerloop;
    void *unmasked_innerloopdata;
    npy_intp subloopsize;

    /* Put the aux data into local variables */
    data = (_ufunc_masker_data *)innerloopdata;
    unmasked_innerloop = data->unmasked_innerloop;
    unmasked_innerloopdata = data->unmasked_innerloopdata;
    nargs = data->nargs;

    /* Process the data as runs of unmasked values */
    do {
        /* Skip masked values */
        mask = npy_memchr(mask, 0, mask_stride, loopsize, &subloopsize, 1);
        for (iargs = 0; iargs < nargs; ++iargs) {
            dataptrs[iargs] += subloopsize * strides[iargs];
        }
        loopsize -= subloopsize;
        /*
         * Process unmasked values (assumes unmasked loop doesn't
         * mess with the 'args' pointer values)
         */
        mask = npy_memchr(mask, 0, mask_stride, loopsize, &subloopsize, 0);
        unmasked_innerloop(dataptrs, &subloopsize, strides,
                                        unmasked_innerloopdata);
        for (iargs = 0; iargs < nargs; ++iargs) {
            dataptrs[iargs] += subloopsize * strides[iargs];
        }
        loopsize -= subloopsize;
    } while (loopsize > 0);
}


/*
 * This function wraps a legacy inner loop so it becomes masked.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_DefaultMaskedInnerLoopSelector(PyUFuncObject *ufunc,
                            PyArray_Descr **dtypes,
                            PyArray_Descr *mask_dtype,
                            npy_intp *NPY_UNUSED(fixed_strides),
                            npy_intp NPY_UNUSED(fixed_mask_stride),
                            PyUFunc_MaskedStridedInnerLoopFunc **out_innerloop,
                            NpyAuxData **out_innerloopdata,
                            int *out_needs_api)
{
    int retcode;
    _ufunc_masker_data *data;

    if (ufunc->legacy_inner_loop_selector == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "the ufunc default masked inner loop selector doesn't "
                "yet support wrapping the new inner loop selector, it "
                "still only wraps the legacy inner loop selector");
        return -1;
    }

    if (mask_dtype->type_num != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError,
                "only boolean masks are supported in ufunc inner loops "
                "presently");
        return -1;
    }

    /* Create a new NpyAuxData object for the masker data */
    data = (_ufunc_masker_data *)PyArray_malloc(sizeof(_ufunc_masker_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    memset(data, 0, sizeof(_ufunc_masker_data));
    data->base.free = (NpyAuxData_FreeFunc *)&PyArray_free;
    data->base.clone = &ufunc_masker_data_clone;
    data->nargs = ufunc->nin + ufunc->nout;

    /* Get the unmasked ufunc inner loop */
    retcode = ufunc->legacy_inner_loop_selector(ufunc, dtypes,
                    &data->unmasked_innerloop, &data->unmasked_innerloopdata,
                    out_needs_api);
    if (retcode < 0) {
        PyArray_free(data);
        return retcode;
    }

    /* Return the loop function + aux data */
    *out_innerloop = &unmasked_ufunc_loop_as_masked;
    *out_innerloopdata = (NpyAuxData *)data;
    return 0;
}

static int
ufunc_loop_matches(PyUFuncObject *self,
                    PyArray_Descr **op_dtypes,
                    NPY_CASTING input_casting,
                    NPY_CASTING output_casting,
                    int any_object,
                    int *types, PyArray_Descr **dtypes,
                    int *out_no_castable_output,
                    char *out_err_src_typecode,
                    char *out_err_dst_typecode)
{
    npy_intp i, nin = self->nin, nop = nin + self->nout;

    /*
     * First check if all the inputs can be safely cast
     * to the types for this function
     */
    for (i = 0; i < nin; ++i) {
        PyArray_Descr *tmp;

        /*
         * If no inputs are objects and there are more than one
         * loop, don't allow conversion to object.  The rationale
         * behind this is mostly performance.  Except for custom
         * ufuncs built with just one object-parametered inner loop,
         * only the types that are supported are implemented.  Trying
         * the object version of logical_or on float arguments doesn't
         * seem right.
         */
        if (types[i] == NPY_OBJECT && !any_object && self->ntypes > 1) {
            return 0;
        }

        /*
         * If type num is NPY_VOID and struct dtypes have been passed in,
         * use struct dtype object. Otherwise create new dtype object
         * from type num.
         */
        if (types[i] == NPY_VOID && dtypes != NULL) {
            tmp = dtypes[i];
            Py_INCREF(tmp);
        }
        else {
            tmp = PyArray_DescrFromType(types[i]);
        }
        if (tmp == NULL) {
            return -1;
        }

#if NPY_UF_DBG_TRACING
        printf("Checking type for op %d, type %d: ", (int)i, (int)types[i]);
        PyObject_Print((PyObject *)tmp, stdout, 0);
        printf(", operand type: ");
        PyObject_Print((PyObject *)op_dtypes[i], stdout, 0);
        printf("\n");
#endif
        if (!PyArray_CanCastTypeTo(op_dtypes[i], tmp, input_casting)) {
            Py_DECREF(tmp);
            return 0;
        }
        Py_DECREF(tmp);
    }

    /*
     * If all the inputs were ok, then check casting back to the
     * outputs.
     */
    for (i = nin; i < nop; ++i) {
        if (op_dtypes[i] != NULL) {
            PyArray_Descr *tmp = PyArray_DescrFromType(types[i]);
            if (tmp == NULL) {
                return -1;
            }
            if (!PyArray_CanCastTypeTo(tmp, op_dtypes[i], output_casting)) {
                if (!(*out_no_castable_output)) {
                    *out_no_castable_output = 1;
                    *out_err_src_typecode = tmp->type;
                    *out_err_dst_typecode = op_dtypes[i]->type;
                }
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);
        }
    }

    return 1;
}

static int
set_ufunc_loop_data_types(PyUFuncObject *self, PyArray_Descr **op_dtypes,
                    PyArray_Descr **out_dtypes, int *type_nums)
{
    int i, nin = self->nin, nop = nin + self->nout;

    /*
     * Fill the dtypes array.
     * For outputs,
     * also search the inputs for a matching type_num to copy
     * instead of creating a new one, similarly to preserve metadata.
     **/
    for (i = 0; i < nop; ++i) {
        /*
         * Copy the dtype from 'op' if the type_num matches,
         * to preserve metadata.
         */
        if (out_dtypes[i] != NULL) {
            /* TODO: Preserving explicit input is new here (likely unused) */
            continue;
        }
        else if (op_dtypes[i] != NULL &&
                 op_dtypes[i]->type_num == type_nums[i]) {
            out_dtypes[i] = ensure_dtype_nbo(op_dtypes[i]);
        }
        /*
         * For outputs, copy the dtype from op[0] if the type_num
         * matches, similarly to preserve metadata.
         */
        else if (i >= nin && op_dtypes[0] != NULL &&
                            op_dtypes[0]->type_num == type_nums[i]) {;
            out_dtypes[i] = ensure_dtype_nbo(op_dtypes[0]);
        }
        /* Otherwise create a plain descr from the type number */
        else {
            out_dtypes[i] = PyArray_DescrFromType(type_nums[i]);
        }

        if (out_dtypes[i] == NULL) {
            goto fail;
        }
    }

    return 0;

fail:
    while (--i >= 0) {
        Py_DECREF(out_dtypes[i]);
        out_dtypes[i] = NULL;
    }
    return -1;
}


/*
 * Does a search through the arguments and the loops
 */
static int
userloop_type_resolver(PyUFuncObject *self,
                       PyArray_Descr **op_dtypes,
                       NPY_CASTING casting,
                       int any_object,
                       PyArray_Descr **out_dtype)
{
    int i, j, nin = self->nin, nop = nin + self->nout;
    PyUFunc_Loop1d *funcdata;

    /* Use this to try to avoid repeating the same userdef loop search */
    int last_userdef = -1;

    int no_castable_output = 0;
    char err_src_typecode = '-', err_dst_typecode = '-';

    for (i = 0; i < nin; ++i) {
        int type_num = op_dtypes[i]->type_num;
        if (type_num != last_userdef && PyTypeNum_ISUSERDEF(type_num)) {
            PyObject *key, *obj;

            last_userdef = type_num;

            key = PyInt_FromLong(type_num);
            if (key == NULL) {
                return -1;
            }
            obj = PyDict_GetItem(self->userloops, key);
            Py_DECREF(key);
            if (obj == NULL) {
                continue;
            }

            for (funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(obj);
                 funcdata != NULL;
                 funcdata = funcdata->next) {
                int *types = funcdata->arg_types;
                int matched = 1;

                for (j = 0; j < nop; ++j) {
                    if (out_dtype[j] != NULL &&
                                types[j] != out_dtype[j]->type_num) {
                        matched = 0;
                        break;
                    }
                }
                if (!matched) {
                    continue;
                }

                switch (ufunc_loop_matches(self, op_dtypes,
                            casting, casting,
                            any_object,
                            types, NULL,
                            &no_castable_output, &err_src_typecode,
                            &err_dst_typecode)) {
                    /* It works */
                    case 1:
                        set_ufunc_loop_data_types(self, op_dtypes,
                                                  out_dtype, types);
                        return 1;
                    /* Didn't match */
                    case 0:
                        PyErr_Format(PyExc_TypeError,
                             "found a user loop for ufunc '%s' "
                             "matching the type-tuple, "
                             "but the inputs and/or outputs could not be "
                             "cast according to the casting rule",
                             ufunc_get_name_cstr(self));
                        return -1;
                    /* Error */
                    case -1:
                        return -1;
                }
            }
        }
    }

    /* Didn't find a match */
    return 0;
}


/*
 * Does a linear search for the inner loop of the ufunc.
 *
 * Note that if an error is returned, the caller must free the non-zero
 * references in out_dtype.  This function does not do its own clean-up.
 */
NPY_NO_EXPORT int
linear_search_type_resolver(PyUFuncObject *self,
                            PyArray_Descr **op_dtypes,
                            NPY_CASTING input_casting,
                            NPY_CASTING casting,
                            int any_object,
                            PyArray_Descr **out_dtypes)
{
    npy_intp i, j, n, nin = self->nin, nop = nin + self->nout;
    int types[NPY_MAXARGS];
    const char *ufunc_name;
    int no_castable_output;

    /* For making a better error message on coercion error */
    char err_dst_typecode = '-', err_src_typecode = '-';

    ufunc_name = ufunc_get_name_cstr(self);

    /* If the ufunc has userloops, search for them. */
    if (self->userloops) {
        switch (userloop_type_resolver(self,
                        op_dtypes, casting,
                        any_object,
                        out_dtypes)) {
            /* Error */
            case -1:
                return -1;
            /* Found matching loop */
            case 1:
                return 0;
        }
    }

    for (i = 0; i < self->ntypes; ++i) {
        char *orig_types = self->types + i*self->nargs;

        /* Copy the types into an int array for matching */
        for (j = 0; j < nop; ++j) {
            types[j] = orig_types[j];
        }

        /* If some types are specified, check if loop matches them */
        for (j = 0; j < nop; ++j) {
            if (out_dtypes[j] != NULL &&
                    types[j] != out_dtypes[j]->type_num) {
                break;
            }
            if (j < nop) {
                /* no match */
                continue;
            }
        }

        switch (ufunc_loop_matches(self, op_dtypes,
                    input_casting, casting,
                    any_object,
                    types, NULL,
                    &no_castable_output, &err_src_typecode,
                    &err_dst_typecode)) {
            case -1:
                /* Error */
                return -1;
            case 0:
                /* Cannot cast inputs */
                continue;
            case 1:
                /* Success */
                set_ufunc_loop_data_types(self, op_dtypes, out_dtypes, types);
                return 0;
        }
    }

    /* If no function was found, throw an error */
    PyErr_Format(PyExc_TypeError,
            "No loop matching the specified signature and casting "
            "was found for ufunc %s", ufunc_name);

    return -1;
}


NPY_NO_EXPORT int
PyUFunc_DivmodTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArray_Descr **op_dtypes,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    type_num1 = op_dtypes[0]->type_num;
    type_num2 = op_dtypes[1]->type_num;

    /* Use the default when datetime and timedelta are not involved */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting,
                                           op_dtypes, out_dtypes);
    }
    if (type_num1 == NPY_TIMEDELTA) {
        if (type_num2 == NPY_TIMEDELTA) {
            out_dtypes[0] = PyArray_PromoteTypes(op_dtypes[0], op_dtypes[1]);
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = PyArray_DescrFromType(NPY_LONGLONG);
            Py_INCREF(out_dtypes[2]);
            out_dtypes[3] = out_dtypes[0];
            Py_INCREF(out_dtypes[3]);
        }
        else {
            return raise_binary_type_reso_error(ufunc, op_dtypes);
        }
    }
    else {
        return raise_binary_type_reso_error(ufunc, op_dtypes);
    }

    /* Check against the casting rules */
    if (PyUFunc_ValidateCastingDtypes(ufunc, casting, op_dtypes, out_dtypes) < 0) {
        for (i = 0; i < 4; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    return 0;
}
