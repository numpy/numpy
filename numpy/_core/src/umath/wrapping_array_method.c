/*
 * This file defines most of the machinery in order to wrap an existing ufunc
 * loop for use with a different set of dtypes.
 *
 * There are two approaches for this, one is to teach the NumPy core about
 * the possibility that the loop descriptors do not match exactly the result
 * descriptors.
 * The other is to handle this fully by "wrapping", so that NumPy core knows
 * nothing about this going on.
 * The slight difficulty here is that `context` metadata needs to be mutated.
 * It also adds a tiny bit of overhead, since we have to "fix" the descriptors
 * and unpack the auxdata.
 *
 * This means that this currently needs to live within NumPy, as it needs both
 * extensive API exposure to do it outside, as well as some thoughts on how to
 * expose the `context` without breaking ABI forward compatibility.
 * (I.e. we probably need to allocate the context and provide a copy function
 * or so.)
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"

#include "common.h"
#include "array_method.h"
#include "legacy_array_method.h"
#include "dtypemeta.h"
#include "dispatching.h"


static NPY_CASTING
wrapping_method_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *dtypes[],
        PyArray_Descr *given_descrs[],
        PyArray_Descr *loop_descrs[],
        npy_intp *view_offset)
{
    int nin = self->nin, nout = self->nout, nargs = nin + nout;
    PyArray_Descr *orig_given_descrs[NPY_MAXARGS];
    PyArray_Descr *orig_loop_descrs[NPY_MAXARGS];

    if (self->translate_given_descrs(
            nin, nout, self->wrapped_dtypes,
            given_descrs, orig_given_descrs) < 0) {
        return -1;
    }
    NPY_CASTING casting = self->wrapped_meth->resolve_descriptors(
            self->wrapped_meth, self->wrapped_dtypes,
            orig_given_descrs, orig_loop_descrs, view_offset);
    for (int i = 0; i < nargs; i++) {
        Py_XDECREF(orig_given_descrs);
    }
    if (casting < 0) {
        return -1;
    }
    int res = self->translate_loop_descrs(
            nin, nout, dtypes, given_descrs, orig_loop_descrs, loop_descrs);
    for (int i = 0; i < nargs; i++) {
        Py_DECREF(orig_given_descrs);
    }
    if (res < 0) {
        return -1;
    }
    return casting;
}


typedef struct {
    NpyAuxData base;
    /* Note that if context is expanded this may become trickier: */
    PyArrayMethod_Context orig_context;
    PyArrayMethod_StridedLoop *orig_loop;
    NpyAuxData *orig_auxdata;
    PyArray_Descr *descriptors[NPY_MAXARGS];
} wrapping_auxdata;


#define WRAPPING_AUXDATA_FREELIST_SIZE 5
static int wrapping_auxdata_freenum = 0;
static wrapping_auxdata *wrapping_auxdata_freelist[WRAPPING_AUXDATA_FREELIST_SIZE] = {NULL};


static void
wrapping_auxdata_free(wrapping_auxdata *wrapping_auxdata)
{
    /* Free auxdata, everything else is borrowed: */
    NPY_AUXDATA_FREE(wrapping_auxdata->orig_auxdata);
    wrapping_auxdata->orig_auxdata = NULL;

    if (wrapping_auxdata_freenum < WRAPPING_AUXDATA_FREELIST_SIZE) {
        wrapping_auxdata_freelist[wrapping_auxdata_freenum] = wrapping_auxdata;
    }
    else {
        PyMem_Free(wrapping_auxdata);
    }
}


static wrapping_auxdata *
get_wrapping_auxdata(void)
{
    wrapping_auxdata *res;
    if (wrapping_auxdata_freenum > 0) {
        wrapping_auxdata_freenum--;
        res = wrapping_auxdata_freelist[wrapping_auxdata_freenum];
    }
    else {
        res = PyMem_Calloc(1, sizeof(wrapping_auxdata));
        if (res < 0) {
            PyErr_NoMemory();
            return NULL;
        }
        res->base.free = (void *)wrapping_auxdata_free;
        res->orig_context.descriptors = res->descriptors;
    }

    return res;
}


static int
wrapping_method_strided_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], wrapping_auxdata *auxdata)
{
    /*
     * If more things get stored on the context, it could be possible that
     * we would have to copy it here.  But currently, we do not.
     */
    return auxdata->orig_loop(
            &auxdata->orig_context, data, dimensions, strides,
            auxdata->orig_auxdata);
}


static int
wrapping_method_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    assert(move_references == 0);  /* only used internally for "decref" funcs */
    int nin = context->method->nin, nout = context->method->nout;

    wrapping_auxdata *auxdata = get_wrapping_auxdata();
    if (auxdata == NULL) {
        return -1;
    }

    auxdata->orig_context.method = context->method->wrapped_meth;
    auxdata->orig_context.caller = context->caller;

    if (context->method->translate_given_descrs(
            nin, nout, context->method->wrapped_dtypes,
            context->descriptors, auxdata->orig_context.descriptors) < 0) {
        NPY_AUXDATA_FREE((NpyAuxData *)auxdata);
        return -1;
    }
    if (context->method->wrapped_meth->get_strided_loop(
            &auxdata->orig_context, aligned, 0, strides,
            &auxdata->orig_loop, &auxdata->orig_auxdata,
            flags) < 0) {
        NPY_AUXDATA_FREE((NpyAuxData *)auxdata);
        return -1;
    }

    *out_loop = (PyArrayMethod_StridedLoop *)&wrapping_method_strided_loop;
    *out_transferdata = (NpyAuxData *)auxdata;
    return 0;
}


/*
 * Wraps the original identity function, needs to translate the descriptors
 * back to the original ones and provide an "original" context (identically to
 * `get_loop`).
 * We assume again that translating the descriptors is quick.
 */
static int
wrapping_method_get_identity_function(
        PyArrayMethod_Context *context, npy_bool reduction_is_empty,
        char *item)
{
    /* Copy the context, and replace descriptors: */
    PyArrayMethod_Context orig_context = *context;
    PyArray_Descr *orig_descrs[NPY_MAXARGS];
    orig_context.descriptors = orig_descrs;
    orig_context.method = context->method->wrapped_meth;

    int nin = context->method->nin, nout = context->method->nout;
    PyArray_DTypeMeta **dtypes = context->method->wrapped_dtypes;

    if (context->method->translate_given_descrs(
            nin, nout, dtypes, context->descriptors, orig_descrs) < 0) {
        return -1;
    }
    int res = context->method->wrapped_meth->get_reduction_initial(
            &orig_context, reduction_is_empty, item);
    for (int i = 0; i < nin + nout; i++) {
        Py_DECREF(orig_descrs);
    }
    return res;
}


/**
 * Allows creating of a fairly lightweight wrapper around an existing ufunc
 * loop.  The idea is mainly for units, as this is currently slightly limited
 * in that it enforces that you cannot use a loop from another ufunc.
 *
 * @param ufunc_obj
 * @param new_dtypes
 * @param wrapped_dtypes
 * @param translate_given_descrs See typedef comment
 * @param translate_loop_descrs See typedef comment
 * @return 0 on success -1 on failure
 */
NPY_NO_EXPORT int
PyUFunc_AddWrappingLoop(PyObject *ufunc_obj,
        PyArray_DTypeMeta *new_dtypes[], PyArray_DTypeMeta *wrapped_dtypes[],
        translate_given_descrs_func *translate_given_descrs,
        translate_loop_descrs_func *translate_loop_descrs)
{
    int res = -1;
    PyUFuncObject *ufunc = (PyUFuncObject *)ufunc_obj;
    PyObject *wrapped_dt_tuple = NULL;
    PyObject *new_dt_tuple = NULL;
    PyArrayMethodObject *meth = NULL;

    if (!PyObject_TypeCheck(ufunc_obj, &PyUFunc_Type)) {
        PyErr_SetString(PyExc_TypeError,
                "ufunc object passed is not a ufunc!");
        return -1;
    }

    wrapped_dt_tuple = PyArray_TupleFromItems(
            ufunc->nargs, (PyObject **)wrapped_dtypes, 1);
    if (wrapped_dt_tuple == NULL) {
        goto finish;
    }

    PyArrayMethodObject *wrapped_meth = NULL;
    PyObject *loops = ufunc->_loops;
    Py_ssize_t length = PyList_Size(loops);
    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject *item = PyList_GetItem(loops, i);
        PyObject *cur_DType_tuple = PyTuple_GetItem(item, 0);
        int cmp = PyObject_RichCompareBool(cur_DType_tuple, wrapped_dt_tuple, Py_EQ);
        if (cmp < 0) {
            goto finish;
        }
        if (cmp == 0) {
            continue;
        }
        wrapped_meth = (PyArrayMethodObject *)PyTuple_GET_ITEM(item, 1);
        if (!PyObject_TypeCheck(wrapped_meth, &PyArrayMethod_Type)) {
            PyErr_SetString(PyExc_TypeError,
                    "Matching loop was not an ArrayMethod.");
            goto finish;
        }
        break;
    }
    if (wrapped_meth == NULL) {
        PyErr_Format(PyExc_TypeError,
                "Did not find the to-be-wrapped loop in the ufunc with given "
                "DTypes. Received wrapping types: %S", wrapped_dt_tuple);
        goto finish;
    }

    PyType_Slot slots[] = {
        {NPY_METH_resolve_descriptors, &wrapping_method_resolve_descriptors},
        {_NPY_METH_get_loop, &wrapping_method_get_loop},
        {NPY_METH_get_reduction_initial,
            &wrapping_method_get_identity_function},
        {0, NULL}
    };

    PyArrayMethod_Spec spec = {
        .name = "wrapped-method",
        .nin = wrapped_meth->nin,
        .nout = wrapped_meth->nout,
        .casting = wrapped_meth->casting,
        .flags = wrapped_meth->flags,
        .dtypes = new_dtypes,
        .slots = slots,
    };
    PyBoundArrayMethodObject *bmeth = PyArrayMethod_FromSpec_int(&spec, 1);
    if (bmeth == NULL) {
        goto finish;
    }

    Py_INCREF(bmeth->method);
    meth = bmeth->method;
    Py_SETREF(bmeth, NULL);

    /* Finalize the "wrapped" part of the new ArrayMethod */
    meth->wrapped_dtypes = PyMem_Malloc(ufunc->nargs * sizeof(PyArray_DTypeMeta *));
    if (meth->wrapped_dtypes == NULL) {
        goto finish;
    }

    Py_INCREF(wrapped_meth);
    meth->wrapped_meth = wrapped_meth;
    meth->translate_given_descrs = translate_given_descrs;
    meth->translate_loop_descrs = translate_loop_descrs;
    for (int i = 0; i < ufunc->nargs; i++) {
        Py_XINCREF(wrapped_dtypes[i]);
        meth->wrapped_dtypes[i] = wrapped_dtypes[i];
    }

    new_dt_tuple = PyArray_TupleFromItems(
            ufunc->nargs, (PyObject **)new_dtypes, 1);
    if (new_dt_tuple == NULL) {
        goto finish;
    }

    PyObject *info = PyTuple_Pack(2, new_dt_tuple, meth);
    if (info == NULL) {
        goto finish;
    }

    res = PyUFunc_AddLoop(ufunc, info, 0);
    Py_DECREF(info);

  finish:
    Py_XDECREF(wrapped_dt_tuple);
    Py_XDECREF(new_dt_tuple);
    Py_XDECREF(meth);
    return res;
}
