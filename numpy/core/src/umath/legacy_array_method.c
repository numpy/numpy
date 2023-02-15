/*
 * This file defines most of the machinery in order to wrap legacy style
 * ufunc loops into new style arraymethods.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"

#include "convert_datatype.h"
#include "array_method.h"
#include "array_coercion.h"
#include "dtype_transfer.h"
#include "legacy_array_method.h"
#include "dtypemeta.h"

#include "ufunc_object.h"


typedef struct {
    NpyAuxData base;
    /* The legacy loop and additional user data: */
    PyUFuncGenericFunction loop;
    void *user_data;
    /* Whether to check for PyErr_Occurred(), must require GIL if used */
    int pyerr_check;
} legacy_array_method_auxdata;


/* Use a free list, since we should normally only need one at a time */
#define NPY_LOOP_DATA_CACHE_SIZE 5
static int loop_data_num_cached = 0;
static  legacy_array_method_auxdata *loop_data_cache[NPY_LOOP_DATA_CACHE_SIZE];


static void
legacy_array_method_auxdata_free(NpyAuxData *data)
{
    if (loop_data_num_cached < NPY_LOOP_DATA_CACHE_SIZE) {
        loop_data_cache[loop_data_num_cached] = (
                (legacy_array_method_auxdata *)data);
        loop_data_num_cached++;
    }
    else {
        PyMem_Free(data);
    }
}

#undef NPY_LOOP_DATA_CACHE_SIZE


NpyAuxData *
get_new_loop_data(
        PyUFuncGenericFunction loop, void *user_data, int pyerr_check)
{
    legacy_array_method_auxdata *data;
    if (NPY_LIKELY(loop_data_num_cached > 0)) {
        loop_data_num_cached--;
        data = loop_data_cache[loop_data_num_cached];
    }
    else {
        data = PyMem_Malloc(sizeof(legacy_array_method_auxdata));
        if (data == NULL) {
            return NULL;
        }
        data->base.free = legacy_array_method_auxdata_free;
        data->base.clone = NULL;  /* no need for cloning (at least for now) */
    }
    data->loop = loop;
    data->user_data = user_data;
    data->pyerr_check = pyerr_check;
    return (NpyAuxData *)data;
}


/*
 * This is a thin wrapper around the legacy loop signature.
 */
static int
generic_wrapped_legacy_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    legacy_array_method_auxdata *ldata = (legacy_array_method_auxdata *)auxdata;

    ldata->loop((char **)data, dimensions, strides, ldata->user_data);
    if (ldata->pyerr_check && PyErr_Occurred()) {
        return -1;
    }
    return 0;
}

/*
 * Signal that the old type-resolution function must be used to resolve
 * the descriptors (mainly/only used for datetimes due to the unit).
 *
 * ArrayMethod's are expected to implement this, but it is too tricky
 * to support properly.  So we simply set an error that should never be seen.
 */
NPY_NO_EXPORT NPY_CASTING
wrapped_legacy_resolve_descriptors(PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[]),
        PyArray_Descr *NPY_UNUSED(given_descrs[]),
        PyArray_Descr *NPY_UNUSED(loop_descrs[]),
        npy_intp *NPY_UNUSED(view_offset))
{
    PyErr_SetString(PyExc_RuntimeError,
            "cannot use legacy wrapping ArrayMethod without calling the ufunc "
            "itself.  If this error is hit, the solution will be to port the "
            "legacy ufunc loop implementation to the new API.");
    return -1;
}

/*
 * Much the same as the default type resolver, but tries a bit harder to
 * preserve metadata.
 */
static NPY_CASTING
simple_legacy_resolve_descriptors(
        PyArrayMethodObject *method,
        PyArray_DTypeMeta **dtypes,
        PyArray_Descr **given_descrs,
        PyArray_Descr **output_descrs,
        npy_intp *NPY_UNUSED(view_offset))
{
    int i = 0;
    int nin = method->nin;
    int nout = method->nout;

    if (nin == 2 && nout == 1 && given_descrs[2] != NULL
            && dtypes[0] == dtypes[2]) {
        /*
         * Could be a reduction, which requires `descr[0] is descr[2]`
         * (identity) at least currently. This is because `op[0] is op[2]`.
         * (If the output descriptor is not passed, the below works.)
         */
        output_descrs[2] = NPY_DT_CALL_ensure_canonical(given_descrs[2]);
        if (output_descrs[2] == NULL) {
            Py_CLEAR(output_descrs[2]);
            return -1;
        }
        Py_INCREF(output_descrs[2]);
        output_descrs[0] = output_descrs[2];
        if (dtypes[1] == dtypes[2]) {
            /* Same for the second one (accumulation is stricter) */
            Py_INCREF(output_descrs[2]);
            output_descrs[1] = output_descrs[2];
        }
        else {
            output_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
            if (output_descrs[1] == NULL) {
                i = 2;
                goto fail;
            }
        }
        return NPY_NO_CASTING;
    }

    for (; i < nin + nout; i++) {
        if (given_descrs[i] != NULL) {
            output_descrs[i] = NPY_DT_CALL_ensure_canonical(given_descrs[i]);
        }
        else if (dtypes[i] == dtypes[0] && i > 0) {
            /* Preserve metadata from the first operand if same dtype */
            Py_INCREF(output_descrs[0]);
            output_descrs[i] = output_descrs[0];
        }
        else {
            output_descrs[i] = NPY_DT_CALL_default_descr(dtypes[i]);
        }
        if (output_descrs[i] == NULL) {
            goto fail;
        }
    }

    return NPY_NO_CASTING;

  fail:
    for (; i >= 0; i--) {
        Py_CLEAR(output_descrs[i]);
    }
    return -1;
}


/*
 * This function grabs the legacy inner-loop.  If this turns out to be slow
 * we could probably cache it (with some care).
 */
NPY_NO_EXPORT int
get_wrapped_legacy_ufunc_loop(PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    assert(aligned);
    assert(!move_references);

    if (context->caller == NULL ||
            !PyObject_TypeCheck(context->caller, &PyUFunc_Type)) {
        PyErr_Format(PyExc_RuntimeError,
                "cannot call %s without its ufunc as caller context.",
                context->method->name);
        return -1;
    }

    PyUFuncObject *ufunc = (PyUFuncObject *)context->caller;
    void *user_data;
    int needs_api = 0;

    PyUFuncGenericFunction loop = NULL;
    /* Note that `needs_api` is not reliable (it was in fact unused normally) */
    if (ufunc->legacy_inner_loop_selector(ufunc,
            context->descriptors, &loop, &user_data, &needs_api) < 0) {
        return -1;
    }
    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;
    if (needs_api) {
        *flags |= NPY_METH_REQUIRES_PYAPI;
    }

    *out_loop = &generic_wrapped_legacy_loop;
    *out_transferdata = get_new_loop_data(
            loop, user_data, (*flags & NPY_METH_REQUIRES_PYAPI) != 0);
    if (*out_transferdata == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}



/*
 * We can shave off a bit of time by just caching the initial and this is
 * trivial for all internal numeric types.  (Wrapped ufuncs never use
 * byte-swapping.)
 */
static int
copy_cached_initial(
        PyArrayMethod_Context *context, npy_bool NPY_UNUSED(reduction_is_empty),
        char *initial)
{
    memcpy(initial, context->method->legacy_initial,
           context->descriptors[0]->elsize);
    return 1;
}


/*
 * The default `get_reduction_initial` attempts to look up the identity
 * from the calling ufunc.  This might fail, so we only call it when necessary.
 *
 * For internal number dtypes, we can easily cache it, so do so after the
 * first call by overriding the function with `copy_cache_initial`.
 * This path is not publicly available.  That could be added, and for a
 * custom initial getter it should be static/compile time data anyway.
 */
static int
get_initial_from_ufunc(
        PyArrayMethod_Context *context, npy_bool reduction_is_empty,
        char *initial)
{
    if (context->caller == NULL
            || !PyObject_TypeCheck(context->caller, &PyUFunc_Type)) {
        /* Impossible in NumPy 1.24;  guard in case it becomes possible. */
        PyErr_SetString(PyExc_ValueError,
                "getting initial failed because it can only done for legacy "
                "ufunc loops when the ufunc is provided.");
        return -1;
    }
    npy_bool reorderable;
    PyObject *identity_obj = PyUFunc_GetDefaultIdentity(
            (PyUFuncObject *)context->caller, &reorderable);
    if (identity_obj == NULL) {
        return -1;
    }
    if (identity_obj == Py_None) {
        /* UFunc has no idenity (should not happen) */
        Py_DECREF(identity_obj);
        return 0;
    }
    if (PyTypeNum_ISUNSIGNED(context->descriptors[1]->type_num)
            && PyLong_CheckExact(identity_obj)) {
        /*
         * This is a bit of a hack until we have truly loop specific
         * identities.  Python -1 cannot be cast to unsigned so convert
         * it to a NumPy scalar, but we use -1 for bitwise functions to
         * signal all 1s.
         * (A builtin identity would not overflow here, although we may
         * unnecessary convert 0 and 1.)
         */
        Py_SETREF(identity_obj, PyObject_CallFunctionObjArgs(
                     (PyObject *)&PyLongArrType_Type, identity_obj, NULL));
        if (identity_obj == NULL) {
            return -1;
        }
    }
    else if (context->descriptors[0]->type_num == NPY_OBJECT
            && !reduction_is_empty) {
        /* Allows `sum([object()])` to work, but use 0 when empty. */
        Py_DECREF(identity_obj);
        return 0;
    }

    int res = PyArray_Pack(context->descriptors[0], initial, identity_obj);
    Py_DECREF(identity_obj);
    if (res < 0) {
        return -1;
    }

    if (PyTypeNum_ISNUMBER(context->descriptors[0]->type_num)) {
        /* For numbers we can cache to avoid going via Python ints */
        memcpy(context->method->legacy_initial, initial,
               context->descriptors[0]->elsize);
        context->method->get_reduction_initial = &copy_cached_initial;
    }

    /* Reduction can use the initial value */
    return 1;
}


/*
 * Get the unbound ArrayMethod which wraps the instances of the ufunc.
 * Note that this function stores the result on the ufunc and then only
 * returns the same one.
 */
NPY_NO_EXPORT PyArrayMethodObject *
PyArray_NewLegacyWrappingArrayMethod(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *signature[])
{
    char method_name[101];
    const char *name = ufunc->name ? ufunc->name : "<unknown>";
    snprintf(method_name, 100, "legacy_ufunc_wrapper_for_%s", name);

    /*
     * Assume that we require the Python API when any of the (legacy) dtypes
     * flags it.
     */
    int any_output_flexible = 0;
    NPY_ARRAYMETHOD_FLAGS flags = 0;
    if (ufunc->nargs == 3 &&
            signature[0]->type_num == NPY_BOOL &&
            signature[1]->type_num == NPY_BOOL &&
            signature[2]->type_num == NPY_BOOL && (
                strcmp(ufunc->name, "logical_or") == 0 ||
                strcmp(ufunc->name, "logical_and") == 0 ||
                strcmp(ufunc->name, "logical_xor") == 0)) {
        /*
         * This is a logical ufunc, and the `??->?` loop`. It is always OK
         * to cast any input to bool, because that cast is defined by
         * truthiness.
         * This allows to ensure two things:
         * 1. `np.all`/`np.any` know that force casting the input is OK
         *    (they must do this since there are no `?l->?`, etc. loops)
         * 2. The logical functions automatically work for any DType
         *    implementing a cast to boolean.
         */
        flags = _NPY_METH_FORCE_CAST_INPUTS;
    }

    get_reduction_initial_function *get_reduction_intial = NULL;
    if (ufunc->nin == 2 && ufunc->nout == 1) {
        npy_bool reorderable = NPY_FALSE;
        PyObject *identity_obj = PyUFunc_GetDefaultIdentity(
                ufunc, &reorderable);
        if (identity_obj == NULL) {
            return NULL;
        }
        /*
         * TODO: For object, "reorderable" is needed(?), because otherwise
         *       we disable multi-axis reductions `arr.sum(0, 1)`. But for
         *       `arr = array([["a", "b"], ["c", "d"]], dtype="object")`
         *       it isn't actually reorderable (order changes result).
         */
        if (reorderable) {
            flags |= NPY_METH_IS_REORDERABLE;
        }
        if (identity_obj != Py_None) {
            get_reduction_intial = &get_initial_from_ufunc;
        }
    }
    for (int i = 0; i < ufunc->nin+ufunc->nout; i++) {
        if (signature[i]->singleton->flags & (
                NPY_ITEM_REFCOUNT | NPY_ITEM_IS_POINTER | NPY_NEEDS_PYAPI)) {
            flags |= NPY_METH_REQUIRES_PYAPI;
        }
        if (NPY_DT_is_parametric(signature[i])) {
            any_output_flexible = 1;
        }
    }

    PyType_Slot slots[4] = {
        {_NPY_METH_get_loop, &get_wrapped_legacy_ufunc_loop},
        {NPY_METH_resolve_descriptors, &simple_legacy_resolve_descriptors},
        {NPY_METH_get_reduction_initial, get_reduction_intial},
        {0, NULL},
    };
    if (any_output_flexible) {
        /* We cannot use the default descriptor resolver. */
        slots[1].pfunc = &wrapped_legacy_resolve_descriptors;
    }

    PyArrayMethod_Spec spec = {
        .name = method_name,
        .nin = ufunc->nin,
        .nout = ufunc->nout,
        .casting = NPY_NO_CASTING,
        .flags = flags,
        .dtypes = signature,
        .slots = slots,
    };

    PyBoundArrayMethodObject *bound_res = PyArrayMethod_FromSpec_int(&spec, 1);
    if (bound_res == NULL) {
        return NULL;
    }
    PyArrayMethodObject *res = bound_res->method;
    Py_INCREF(res);
    Py_DECREF(bound_res);
    return res;
}
