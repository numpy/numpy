/*
 * This file implements universal function dispatching and promotion (which
 * is necessary to happen before dispatching).
 * As such it works on the UFunc object.
 */
#include <Python.h>

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>
#include <common.h>

#include "dispatching.h"
#include "dtypemeta.h"
#include "npy_hashtable.h"
#include "legacy_array_method.h"
#include "ufunc_object.h"
#include "ufunc_type_resolution.h"


/* forward declaration */
static PyObject *
promote_and_get_info_and_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[]);


static int
add_ufunc_loop(PyUFuncObject *ufunc, PyObject *info, int ignore_duplicate)
{
    assert(PyTuple_CheckExact(info) && PyTuple_GET_SIZE(info) == 2);

    if (ufunc->_loops == NULL) {
        ufunc->_loops = PyList_New(0);
        if (ufunc->_loops == NULL) {
            return -1;
        }
    }

    PyObject *DType_tuple = PyTuple_GetItem(info, 0);

    PyObject *loops = ufunc->_loops;
    Py_ssize_t length = PyList_Size(loops);
    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject *item = PyList_GetItem(loops, i);
        PyObject *cur_DType_tuple = PyTuple_GetItem(item, 0);
        int cmp = PyObject_RichCompareBool(cur_DType_tuple, DType_tuple, Py_EQ);
        if (cmp < 0) {
            return -1;
        }
        if (cmp == 0) {
            continue;
        }
        if (ignore_duplicate) {
            return 0;
        }
        PyErr_Format(PyExc_TypeError,
                "A loop/promoter has already been registered with '%s' for %R",
                ufunc_get_name_cstr(ufunc), DType_tuple);
        return -1;
    }

    PyList_Append(loops, info);
    return 0;
}


/**
 * Resolves the implementation to use, this uses typical multiple dispatching
 * methods of finding the best matching implementation or resolver.
 * (Based on `isinstance()`, the knowledge that non-abstract DTypes cannot
 * be subclassed is used, however.)
 *
 * @param ufunc
 * @param op_dtypes The DTypes that are either passed in (defined by an
 *        operand) or defined by the `signature` as also passed in as
 *        `fixed_DTypes`.
 * @param out_info Returns the tuple describing the best implementation
 *        (consisting of dtypes and ArrayMethod or promoter).
 *        WARNING: Returns a borrowed reference!
 * @returns -1 on error 0 on success.  Note that the output can be NULL on
 *          success if nothing is found.
 */
static int
resolve_implementation_info(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyObject **out_info)
{
    int nargs = ufunc->nargs;
    /* Use new style type resolution has to happen... */
    Py_ssize_t size = PySequence_Length(ufunc->_loops);
    PyObject *best_dtypes = NULL;
    PyObject *best_resolver_info = NULL;

    for (Py_ssize_t res_idx = 0; res_idx < size; res_idx++) {
        /* Test all resolvers  */
        PyObject *resolver_info = PySequence_Fast_GET_ITEM(
                ufunc->_loops, res_idx);
        PyObject *curr_dtypes = PyTuple_GET_ITEM(resolver_info, 0);

        /*
         * Test if the current resolver matches, it could make sense to
         * reorder these checks to avoid the IsSubclass check as much as
         * possible.
         * TODO: If we have abstract, there is no point in checking
         *       a resolved ArrayMethod (which is always concrete)
         */

        npy_bool matches = NPY_TRUE;
        for (Py_ssize_t i = 0; i < nargs; i++) {
            PyArray_DTypeMeta *given_dtype = op_dtypes[i];
            PyArray_DTypeMeta *resolver_dtype = (
                    (PyArray_DTypeMeta *)PyTuple_GET_ITEM(curr_dtypes, i));
            assert((PyObject *)given_dtype != Py_None);
            if (given_dtype == NULL) {
                /* Not given, anything matches. */
                continue;
            }
            if (given_dtype == resolver_dtype) {
                continue;
            }
            if (!resolver_dtype->abstract) {
                matches = NPY_FALSE;
                break;
            }
            int subclass = PyObject_IsSubclass(
                    (PyObject *)given_dtype, (PyObject *)resolver_dtype);
            if (subclass < 0) {
                return -1;
            }
            if (!subclass) {
                matches = NPY_FALSE;
                break;
            }
            /*
             * TODO: Should consider allowing reverse subclasses, i.e.
             *       the operation DType passed in to be abstract.  That
             *       definitely is OK for outputs (and potentially useful,
             *       you could enforce e.g. an inexact result).
             *       It might also be useful for some stranger promoters.
             */
        }
        if (!matches) {
            continue;
        }

        /* The resolver matches, but we have to check if it is better */
        if (best_dtypes != NULL) {
            int current_best = -1;  /* -1 neither, 0 current best, 1 new */
            /*
             * If both have concrete and None in the same position and
             * they are identical, we will continue searching using the
             * first best for comparison, in an attempt to find a better
             * one.
             * In all cases, we give up resolution, since it would be
             * necessary to compare to two "best" cases.
             */
            int unambiguous_equivally_good = 1;
            for (Py_ssize_t i = 0; i < nargs; i++) {
                int best;

                /* Whether this (normally output) dtype was specified at all */
                int is_not_specified = (
                        op_dtypes[i] == (PyArray_DTypeMeta *)Py_None);

                PyObject *prev_dtype = PyTuple_GET_ITEM(best_dtypes, i);
                PyObject *new_dtype = PyTuple_GET_ITEM(curr_dtypes, i);

                if (prev_dtype == new_dtype) {
                    /* equivalent, so this entry does not matter */
                    continue;
                }
                if (is_not_specified) {
                    /*
                     * When DType is completely unspecified, prefer abstract
                     * over concrete, assuming it will resolve.
                     * Furthermore, we cannot decide which abstract/None
                     * is "better", only concrete ones which are subclasses
                     * of Abstract ones are defined as worse.
                     */
                    int prev_is_concrete = 0, new_is_concrete = 0;
                    if ((prev_dtype != Py_None) &&
                        (!((PyArray_DTypeMeta *)prev_dtype)->abstract)) {
                        prev_is_concrete = 1;
                    }
                    if ((new_dtype != Py_None) &&
                        (!((PyArray_DTypeMeta *)new_dtype)->abstract)) {
                        new_is_concrete = 1;
                    }
                    if (prev_is_concrete == new_is_concrete) {
                        best = -1;
                    }
                    else if (prev_is_concrete) {
                        unambiguous_equivally_good = 0;
                        best = 1;
                    }
                    else {
                        unambiguous_equivally_good = 0;
                        best = 0;
                    }
                }
                    /* If either is None, the other is strictly more specific */
                else if (prev_dtype == Py_None) {
                    unambiguous_equivally_good = 0;
                    best = 1;
                }
                else if (new_dtype == Py_None) {
                    unambiguous_equivally_good = 0;
                    best = 0;
                }
                    /*
                     * If both are concrete and not identical, this is
                     * ambiguous.
                     */
                else if (!((PyArray_DTypeMeta *)prev_dtype)->abstract &&
                         !((PyArray_DTypeMeta *)new_dtype)->abstract) {
                    /*
                     * Ambiguous unless the are identical (checked above),
                     * but since they are concrete it does not matter which
                     * best to compare.
                     */
                    best = -1;
                }
                else if (!((PyArray_DTypeMeta *)prev_dtype)->abstract) {
                    /* old is not abstract, so better (both not possible) */
                    unambiguous_equivally_good = 0;
                    best = 0;
                }
                else if (!((PyArray_DTypeMeta *)new_dtype)->abstract) {
                    /* new is not abstract, so better (both not possible) */
                    unambiguous_equivally_good = 0;
                    best = 1;
                }
                /*
                 * Both are abstract DTypes, there is a clear order if
                 * one of them is a subclass of the other.
                 * If this fails, reject it completely (could be changed).
                 * The case that it is the same dtype is already caught.
                 */
                else {
                    /* Note the identity check above, so this true subclass */
                    int new_is_subclass = PyObject_IsSubclass(
                            new_dtype, prev_dtype);
                    if (new_is_subclass < 0) {
                        return -1;
                    }
                    /*
                     * Could optimize this away if above is True, but this
                     * catches inconsistent definitions of subclassing.
                     */
                    int prev_is_subclass = PyObject_IsSubclass(
                            prev_dtype, new_dtype);
                    if (prev_is_subclass < 0) {
                        return -1;
                    }
                    if (prev_is_subclass && new_is_subclass) {
                        /* should not happen unless they are identical */
                        PyErr_SetString(PyExc_RuntimeError,
                                "inconsistent subclassing of DTypes; if "
                                "this happens, two dtypes claim to be a "
                                "superclass of the other one.");
                        return -1;
                    }
                    if (!prev_is_subclass && !new_is_subclass) {
                        /* Neither is more precise than the other one */
                        PyErr_SetString(PyExc_TypeError,
                                "inconsistent type resolution hierarchy; "
                                "DTypes of two matching loops do not have "
                                "a clear hierarchy defined. Diamond shape "
                                "inheritance is unsupported for use with "
                                "UFunc type resolution. (You may resolve "
                                "this by inserting an additional common "
                                "subclass). This limitation may be "
                                "partially resolved in the future.");
                        return -1;
                    }
                    if (new_is_subclass) {
                        unambiguous_equivally_good = 0;
                        best = 1;
                    }
                    else {
                        unambiguous_equivally_good = 0;
                        best = 2;
                    }
                }
                if ((current_best != -1) && (current_best != best)) {
                    /*
                     * We need a clear best, this could be tricky, unless
                     * the signature is identical, we would have to compare
                     * against both of the found ones until we find a
                     * better one.
                     * Instead, only support the case where they are
                     * identical.
                     */
                    // TODO: Document the above comment (figure out if OK)
                    current_best = -1;
                    break;
                }
                current_best = best;
            }

            if (current_best == -1) {
                if (unambiguous_equivally_good) {
                    /* unset the best resolver to indicate this */
                    best_resolver_info = NULL;
                    continue;
                }
                PyErr_SetString(PyExc_TypeError,
                        "Could not resolve UFunc loop, two loops "
                        "matched equally well.");
                return -1;
            }
            else if (current_best == 0) {
                /* The new match is not better, continue looking. */
                continue;
            }
        }
        /* The new match is better (or there was no previous match) */
        best_dtypes = curr_dtypes;
        best_resolver_info = resolver_info;
    }
    if (best_dtypes == NULL) {
        /* The non-legacy lookup failed */
        *out_info = NULL;
        return 0;
    }

    if (best_resolver_info == NULL) {
        /*
         * This happens if two were equal, but we kept searching
         * for a better one.
         */
        PyErr_SetString(PyExc_TypeError,
                "Could not resolve UFunc loop, two loops "
                "matched equally well.");
        return -1;
    }

    *out_info = best_resolver_info;
    return 0;
}


static int
call_python_promoter(PyUFuncObject *ufunc, PyObject *promoter,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    int nargs = ufunc->nargs;
    int retval = -1;
    PyObject *dtypes_tuple = NULL;
    PyObject *signature_tuple = NULL;
    PyObject *promoter_result = NULL;

    dtypes_tuple = PyArray_TupleFromItems(nargs, (PyObject **)op_dtypes, 1);
    if (dtypes_tuple == NULL) {
        goto finish;
    }
    signature_tuple = PyArray_TupleFromItems(nargs, (PyObject **)signature, 1);
    if (signature_tuple == NULL) {
        goto finish;
    }

    /* Note: Passing the operands is a crutch and nobody should use them! */
    promoter_result = PyObject_CallFunctionObjArgs(promoter,
            dtypes_tuple, signature_tuple, NULL);
    if (promoter_result == NULL) {
        goto finish;
    }
    if (promoter_result == Py_None) {
        /* TODO: op_dtypes may not be original. */
        raise_no_loop_found_error(ufunc, (PyObject **)op_dtypes);
        goto finish;
    }

    if (!PyTuple_CheckExact(promoter_result) ||
            PyTuple_Size(promoter_result) != nargs) {
        /*
         * My (@seberg's) first intention was to allow returning an ArrayMethod
         * (bound) here, so that arbitrary array methods can be returned.
         * Rejecting this has the advantage that we can side-step settling on
         * the exact method of bound vs. unbound ArrayMethods for now.
         * (It is not necessary to provide a `ufunc.resolve_implementation()`!)
         *
         * Forcing the user to explicitly register the new ArrayMethod before
         * returning (if necessary) should be straight forward.
         */
        PyErr_SetString(PyExc_TypeError,
                "currently a promoter must return a tuple of dtypes with "
                "the correct length.  This may be relaxed in the future. "
                "If a promoter needs to add a new ArrayMethod, it must "
                "register this manually before returning.");
        goto finish;
    }

    /* Copy the dtypes into the result array transferring ownership. */
    for (int i = 0; i < nargs; i++) {
        PyObject *tmp = PyTuple_GET_ITEM(promoter_result, i);
        if (tmp == Py_None) {
            tmp = NULL;
        }
        else if (!PyObject_TypeCheck(promoter_result, &PyArrayDTypeMeta_Type)) {
            PyErr_SetString(PyExc_TypeError,
                    "promoter must return a tuple of DTypes (or None).");
            for (int j = 0; j < i; j++) {
                Py_XDECREF(new_op_dtypes[i]);
            }
            goto finish;
        }
        else {
            Py_INCREF(tmp);
        }
        new_op_dtypes[i] = (PyArray_DTypeMeta *)tmp;
    }
    retval = 0;

  finish:
    Py_XDECREF(dtypes_tuple);
    Py_XDECREF(signature_tuple);
    Py_XDECREF(promoter_result);
    return retval;
}


static PyObject *
call_promoter_and_recurse(PyUFuncObject *ufunc, PyObject *promoter,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArrayObject *const operands[])
{
    int nargs = ufunc->nargs;
    PyObject *resolved_info = NULL;
    PyObject *dtypes_tuple = NULL;
    PyObject *signature_tuple = NULL;
    PyObject *operands_tuple = NULL;

    int promoter_result;
    PyArray_DTypeMeta *new_op_dtypes[NPY_MAXARGS];

    if (PyCapsule_CheckExact(promoter)) {
        /* We could also go the other way and wrap up the python function... */
        promoter_function *promoter_function = PyCapsule_GetPointer(promoter,
                "numpy._ufunc_promoter");
        if (promoter_function == NULL) {
            return NULL;
        }
        promoter_result = promoter_function(ufunc,
                op_dtypes, signature, operands, new_op_dtypes);
    }
    else {
        /* Do not pass operands to python, since nobody should use them. */
        promoter_result = call_python_promoter(ufunc, promoter,
                op_dtypes, signature, new_op_dtypes);
    }
    if (promoter_result < 0) {
        return NULL;
    }

    /*
     * Do a recursive call, the promotion function has to ensure that the
     * new tuple is strictly more precise (thus guaranteeing eventual finishing)
     */
    if (Py_EnterRecursiveCall(" during ufunc promotion.") != 0) {
        goto finish;
    }
    resolved_info = promote_and_get_info_and_ufuncimpl(ufunc,
            operands, signature, new_op_dtypes);

    Py_LeaveRecursiveCall();

  finish:
    Py_XDECREF(dtypes_tuple);
    Py_XDECREF(signature_tuple);
    Py_XDECREF(operands_tuple);
    for (int i = 0; i < nargs; i++) {
        Py_XDECREF(new_op_dtypes[i]);
    }
    return resolved_info;
}


/*
 * Used for the legacy fallback promotion when `signature` or `dtype` is
 * provided.
 * We do not need to pass the type tuple when we use the legacy path
 * for type resolution rather than promotion; the old system did not
 * differentiate between these two concepts.
 */
static int
_make_new_typetup(
        int nop, PyArray_DTypeMeta *signature[], PyObject **out_typetup) {
    *out_typetup = PyTuple_New(nop);
    if (*out_typetup == NULL) {
        return -1;
    }

    int none_count = 0;
    for (int i = 0; i < nop; i++) {
        PyObject *item;
        if (signature[i] == NULL) {
            item = Py_None;
            none_count++;
        }
        else {
            if (!signature[i]->legacy || signature[i]->abstract) {
                /*
                 * The legacy type resolution can't deal with these.
                 * This path will return `None` or so in the future to
                 * set an error later if the legacy type resolution is used.
                 */
                PyErr_SetString(PyExc_RuntimeError,
                        "Internal NumPy error: new DType in signature not yet "
                        "supported. (This should be unreachable code!)");
                Py_SETREF(*out_typetup, NULL);
                return -1;
            }
            item = (PyObject *)signature[i]->singleton;
        }
        Py_INCREF(item);
        PyTuple_SET_ITEM(*out_typetup, i, item);
    }
    if (none_count == nop) {
        /* The whole signature was None, simply ignore type tuple */
        Py_DECREF(*out_typetup);
        *out_typetup = NULL;
    }
    return 0;
}


/*
 * Fills in the operation_DTypes with borrowed references.  This may change
 * the content, since it will use the legacy type resolution, which can special
 * case 0-D arrays (using value-based logic).
 */
static int
legacy_promote_using_legacy_type_resolver(PyUFuncObject *ufunc,
        PyArrayObject *const *ops, PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *operation_DTypes[], int *out_cacheable)
{
    int nargs = ufunc->nargs;
    PyArray_Descr *out_descrs[NPY_MAXARGS];
    memset(out_descrs, 0, nargs * sizeof(*out_descrs));

    PyObject *type_tuple = NULL;
    if (_make_new_typetup(nargs, signature, &type_tuple) < 0) {
        return -1;
    }

    /*
     * We use unsafe casting. This is of course not accurate, but that is OK
     * here, because for promotion/dispatching the casting safety makes no
     * difference.  Whether the actual operands can be casts must be checked
     * during the type resolution step (which may _also_ calls this!).
     */
    if (ufunc->type_resolver(ufunc,
            NPY_UNSAFE_CASTING, (PyArrayObject **)ops, type_tuple,
            out_descrs) < 0) {
        Py_XDECREF(type_tuple);
        return -1;
    }
    Py_XDECREF(type_tuple);

    for (int i = 0; i < nargs; i++) {
        operation_DTypes[i] = NPY_DTYPE(out_descrs[i]);
        Py_INCREF(operation_DTypes[i]);
        Py_DECREF(out_descrs[i]);
    }
    if (ufunc->type_resolver == &PyUFunc_SimpleBinaryComparisonTypeResolver) {
        /*
         * In this one case, the deprecation means that we actually override
         * the signature.
         */
        for (int i = 0; i < nargs; i++) {
            if (signature[i] != NULL && signature[i] != operation_DTypes[i]) {
                Py_INCREF(operation_DTypes[i]);
                Py_SETREF(signature[i], operation_DTypes[i]);
                *out_cacheable = 0;
            }
        }
    }
    return 0;
}


/*
 * Note, this function returns a BORROWED references to info since it adds
 * it to the loops.
 */
NPY_NO_EXPORT PyObject *
add_and_return_legacy_wrapping_ufunc_loop(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *operation_dtypes[], int ignore_duplicate)
{
    PyObject *DType_tuple = PyArray_TupleFromItems(ufunc->nargs,
            (PyObject **)operation_dtypes, 0);
    if (DType_tuple == NULL) {
        return NULL;
    }

    PyArrayMethodObject *method = PyArray_NewLegacyWrappingArrayMethod(
            ufunc, operation_dtypes);
    if (method == NULL) {
        Py_DECREF(DType_tuple);
        return NULL;
    }
    PyObject *info = PyTuple_Pack(2, DType_tuple, method);
    Py_DECREF(DType_tuple);
    Py_DECREF(method);
    if (info == NULL) {
        return NULL;
    }
    if (add_ufunc_loop(ufunc, info, ignore_duplicate) < 0) {
        Py_DECREF(info);
        return NULL;
    }

    return info;
}


/*
 * The central entry-point for the promotion and dispatching machinery.
 * It currently works with the operands (although it would be possible to
 * only work with DType (classes/types).
 */
static NPY_INLINE PyObject *
promote_and_get_info_and_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[])
{
    /*
     * Fetch the dispatching info which consists of the implementation and
     * the DType signature tuple.  There are three steps:
     *
     * 1. Check the cache.
     * 2. Check all registered loops/promoters to find the best match.
     * 3. Fall back to the legacy implementation if no match was found.
     */
    PyObject *info = PyArrayIdentityHash_GetItem(ufunc->_dispatch_cache,
                (PyObject **)op_dtypes);
    if (info == NULL) {
        if (resolve_implementation_info(ufunc, op_dtypes, &info) < 0) {
            return NULL;
        }
        if (info != NULL) {
            if (PyArrayIdentityHash_SetItem(ufunc->_dispatch_cache,
                    (PyObject **)op_dtypes, info, 0) < 0) {
                return NULL;
            }
        }
    }

    if (info == NULL ||
            !PyObject_TypeCheck(PyTuple_GET_ITEM(info, 1), &PyArrayMethod_Type)) {
        /*
         * We have to use promotion. If `info == NULL`, only the legacy
         * promotion is still left to try. Otherwise, we first need to try
         * normal promotion.
         */
        if (info != NULL) {
            PyObject *promoter = PyTuple_GET_ITEM(info, 1);

            info = call_promoter_and_recurse(ufunc,
                    promoter, op_dtypes, signature, ops);
            if (info == NULL && PyErr_Occurred()) {
                return NULL;
            }
        }

        /*
         * Using promotion failed, this should normally be an error.
         * However, we need to give the legacy implementation a chance here.
         * (it will modify `op_dtypes`).
         */
        if ((ufunc->ntypes == 0 && ufunc->userloops == NULL) ||
                ufunc->type_resolver == NULL) {
            /* Not a "legacy" ufunc, so we can just return (nothing found) */
            return NULL;
        }
        PyArray_DTypeMeta *new_op_dtypes[NPY_MAXARGS];
        int cacheable = 1;  /* TODO: only used for comparison deprecation. */
        if (legacy_promote_using_legacy_type_resolver(ufunc,
                ops, signature, new_op_dtypes, &cacheable) < 0) {
            return NULL;
        }
        /* Try the cache one more time now */
        info = PyArrayIdentityHash_GetItem(ufunc->_dispatch_cache,
                (PyObject **)new_op_dtypes);
        if (info == NULL) {
            if (resolve_implementation_info(ufunc, new_op_dtypes, &info) < 0) {
                return NULL;
            }
            if (info == NULL) {
                /*
                 * The loop is probably added to the ufunc already, just not cached,
                 * but simply call but uncached should be very rare...
                 */
                info = add_and_return_legacy_wrapping_ufunc_loop(ufunc,
                        new_op_dtypes, 0);
                if (info == NULL) {
                    return NULL;
                }
            }
            if (cacheable && PyArrayIdentityHash_SetItem(ufunc->_dispatch_cache,
                    (PyObject **)op_dtypes, info, 0) < 0) {
                return NULL;
            }
        }
        else if (!PyObject_TypeCheck(
                PyTuple_GET_ITEM(info, 1), &PyArrayMethod_Type)) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Signature resolved by a legacy DType resolver did "
                    "not point to a loop, indicating an error in the ufunc. "
                    "Please notify the ufunc authors and/or NumPy developers.");
            return NULL;
        }
        return info;
    }

    return info;
}


/*
 * The central entry-point for the promotion and dispatching machinery.
 * It currently works with the operands (although it would be possible to
 * only work with DType (classes/types).
 */
NPY_NO_EXPORT PyArrayMethodObject *
promote_and_get_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[], int force_legacy_promotion)
{
    int nargs = ufunc->nargs;

    /*
     * Get the actual DTypes we operate with by mixing the operand array
     * ones with the passed signature.
     */
    for (int i = 0; i < nargs; i++) {
        if (signature[i] != NULL) {
            /*
             * ignore the operand input, we cannot overwrite signature yet
             * since it is fixed (cannot be promoted!)
             */
            op_dtypes[i] = signature[i];
            assert(i >= ufunc->nin || !signature[i]->abstract);
        }
    }

    if (force_legacy_promotion) {
        /*
         * We must use legacy promotion for value-based logic. Call the old
         * resolver once up-front to get the "actual" loop dtypes.
         * After this (additional) promotion, we can even use normal caching.
         */
        int cacheable = 1;  /* unused, as we modify the original `op_dtypes` */
        if (legacy_promote_using_legacy_type_resolver(ufunc,
                ops, signature, op_dtypes, &cacheable) < 0) {
            return NULL;
        }
    }

    PyObject *info = promote_and_get_info_and_ufuncimpl(ufunc,
            ops, signature, op_dtypes);

    if (info == NULL) {
        if (!PyErr_Occurred()) {
            raise_no_loop_found_error(ufunc, (PyObject **)op_dtypes);
        }
        return NULL;
    }

    PyArrayMethodObject *method = (PyArrayMethodObject *)PyTuple_GET_ITEM(info, 1);

    /* Fill in the signature with the signature that we will be working with */
    PyObject *all_dtypes = PyTuple_GET_ITEM(info, 0);
    for (int i = 0; i < nargs; i++) {
        if (signature[i] == NULL) {
            signature[i] = (PyArray_DTypeMeta *)PyTuple_GET_ITEM(all_dtypes, i);
            Py_INCREF(signature[i]);
        }
        else {
            assert((PyObject *)signature[i] == PyTuple_GET_ITEM(all_dtypes, i));
        }
    }

    return method;
}