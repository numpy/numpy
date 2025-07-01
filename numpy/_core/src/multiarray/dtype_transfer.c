/*
 * This file contains low-level loops for data type transfers.
 * In particular the function PyArray_GetDTypeTransferFunction is
 * implemented here.
 *
 * Copyright (c) 2010 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.
 *
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "lowlevel_strided_loops.h"


#include "convert_datatype.h"
#include "ctors.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "descriptor.h"
#include "array_assign.h"

#include "shape.h"
#include "dtype_transfer.h"
#include "dtype_traversal.h"
#include "alloc.h"
#include "dtypemeta.h"
#include "array_method.h"
#include "array_coercion.h"

#include "umathmodule.h"

#define NPY_LOWLEVEL_BUFFER_BLOCKSIZE  128

/********** PRINTF DEBUG TRACING **************/
#define NPY_DT_DBG_TRACING 0
/* Tracing incref/decref can be very noisy */
#define NPY_DT_REF_DBG_TRACING 0

#if NPY_DT_REF_DBG_TRACING
#define NPY_DT_DBG_REFTRACE(msg, ref) \
    printf("%-12s %20p %s%d%s\n", msg, ref, \
                        ref ? "(refcnt " : "", \
                        ref ? (int)ref->ob_refcnt : 0, \
                        ref ? ((ref->ob_refcnt <= 0) ? \
                                        ") <- BIG PROBLEM!!!!" : ")") : ""); \
    fflush(stdout);
#else
#define NPY_DT_DBG_REFTRACE(msg, ref)
#endif
/**********************************************/

#if NPY_DT_DBG_TRACING
/*
 * Thin wrapper around print that ignores exceptions
 */
static void
_safe_print(PyObject *obj)
{
    if (PyObject_Print(obj, stdout, 0) < 0) {
        PyErr_Clear();
        printf("<error during print>");
    }
}
#endif


/*************************** COPY REFERENCES *******************************/

/* Moves references from src to dst */
NPY_NO_EXPORT int
_strided_to_strided_move_references(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    PyObject *src_ref = NULL, *dst_ref = NULL;
    while (N > 0) {
        memcpy(&src_ref, src, sizeof(src_ref));
        memcpy(&dst_ref, dst, sizeof(dst_ref));

        /* Release the reference in dst */
        NPY_DT_DBG_REFTRACE("dec dst ref", dst_ref);
        Py_XDECREF(dst_ref);
        /* Move the reference */
        NPY_DT_DBG_REFTRACE("move src ref", src_ref);
        memcpy(dst, &src_ref, sizeof(src_ref));
        /* Set the source reference to NULL */
        src_ref = NULL;
        memcpy(src, &src_ref, sizeof(src_ref));

        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

/* Copies references from src to dst */
NPY_NO_EXPORT int
_strided_to_strided_copy_references(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    PyObject *src_ref = NULL, *dst_ref = NULL;
    while (N > 0) {
        memcpy(&src_ref, src, sizeof(src_ref));
        memcpy(&dst_ref, dst, sizeof(dst_ref));

        /* Copy the reference */
        NPY_DT_DBG_REFTRACE("copy src ref", src_ref);
        memcpy(dst, &src_ref, sizeof(src_ref));
        /* Claim the reference */
        Py_XINCREF(src_ref);
        /* Release the reference in dst */
        NPY_DT_DBG_REFTRACE("dec dst ref", dst_ref);
        Py_XDECREF(dst_ref);

        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

/************************** ANY TO OBJECT *********************************/

typedef struct {
    NpyAuxData base;
    PyArray_GetItemFunc *getitem;
    PyArrayObject_fields arr_fields;
    NPY_traverse_info decref_src;
} _any_to_object_auxdata;


static void
_any_to_object_auxdata_free(NpyAuxData *auxdata)
{
    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;

    Py_DECREF(data->arr_fields.descr);
    NPY_traverse_info_xfree(&data->decref_src);
    PyMem_Free(data);
}


static NpyAuxData *
_any_to_object_auxdata_clone(NpyAuxData *auxdata)
{
    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;

    _any_to_object_auxdata *res = PyMem_Malloc(sizeof(_any_to_object_auxdata));

    res->base = data->base;
    res->getitem = data->getitem;
    res->arr_fields = data->arr_fields;
    Py_INCREF(res->arr_fields.descr);

    if (data->decref_src.func != NULL) {
        if (NPY_traverse_info_copy(&res->decref_src, &data->decref_src) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)res);
            return NULL;
        }
    }
    else {
        res->decref_src.func = NULL;
    }
    return (NpyAuxData *)res;
}


static int
_strided_to_strided_any_to_object(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;

    PyObject *dst_ref = NULL;
    char *orig_src = src;
    while (N > 0) {
        memcpy(&dst_ref, dst, sizeof(dst_ref));
        Py_XDECREF(dst_ref);
        dst_ref = data->getitem(src, &data->arr_fields);
        memcpy(dst, &dst_ref, sizeof(PyObject *));

        if (dst_ref == NULL) {
            return -1;
        }
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    if (data->decref_src.func != NULL) {
        /* If necessary, clear the input buffer (`move_references`) */
        if (data->decref_src.func(NULL, data->decref_src.descr,
                orig_src, N, src_stride, data->decref_src.auxdata) < 0) {
            return -1;
        }
    }
    return 0;
}


NPY_NO_EXPORT int
any_to_object_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    /* Python API doesn't use FPEs and this also attempts to hide spurious ones. */
    *flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS;

    *out_loop = _strided_to_strided_any_to_object;
    *out_transferdata = PyMem_Malloc(sizeof(_any_to_object_auxdata));
    if (*out_transferdata == NULL) {
        return -1;
    }
    _any_to_object_auxdata *data = (_any_to_object_auxdata *)*out_transferdata;
    data->base.free = &_any_to_object_auxdata_free;
    data->base.clone = &_any_to_object_auxdata_clone;
    data->arr_fields.base = NULL;
    Py_SET_TYPE(&data->arr_fields, NULL);
    data->arr_fields.descr = context->descriptors[0];
    Py_INCREF(data->arr_fields.descr);
    data->arr_fields.flags = aligned ? NPY_ARRAY_ALIGNED : 0;
    data->arr_fields.nd = 0;

    data->getitem = PyDataType_GetArrFuncs(context->descriptors[0])->getitem;
    NPY_traverse_info_init(&data->decref_src);

    if (move_references && PyDataType_REFCHK(context->descriptors[0])) {
        NPY_ARRAYMETHOD_FLAGS clear_flags;
        if (PyArray_GetClearFunction(
                aligned, strides[0], context->descriptors[0],
                &data->decref_src, &clear_flags) < 0)  {
            NPY_AUXDATA_FREE(*out_transferdata);
            *out_transferdata = NULL;
            return -1;
        }
        *flags = PyArrayMethod_COMBINED_FLAGS(*flags, clear_flags);
    }
    return 0;
}


/************************** OBJECT TO ANY *********************************/

typedef struct {
    NpyAuxData base;
    PyArray_Descr *descr;
    int move_references;
} _object_to_any_auxdata;


static void
_object_to_any_auxdata_free(NpyAuxData *auxdata)
{
    _object_to_any_auxdata *data = (_object_to_any_auxdata *)auxdata;
    Py_DECREF(data->descr);
    PyMem_Free(data);
}

static NpyAuxData *
_object_to_any_auxdata_clone(NpyAuxData *data)
{
    _object_to_any_auxdata *res = PyMem_Malloc(sizeof(*res));
    if (res == NULL) {
        return NULL;
    }
    memcpy(res, data, sizeof(*res));
    Py_INCREF(res->descr);
    return (NpyAuxData *)res;
}


static int
strided_to_strided_object_to_any(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];
    _object_to_any_auxdata *data = (_object_to_any_auxdata *)auxdata;

    PyObject *src_ref;

    while (N > 0) {
        memcpy(&src_ref, src, sizeof(src_ref));
        if (PyArray_Pack(data->descr, dst, src_ref ? src_ref : Py_None) < 0) {
            return -1;
        }

        if (data->move_references && src_ref != NULL) {
            Py_DECREF(src_ref);
            memset(src, 0, sizeof(src_ref));
        }

        N--;
        dst += dst_stride;
        src += src_stride;
    }
    return 0;
}


NPY_NO_EXPORT int
object_to_any_get_loop(
        PyArrayMethod_Context *context,
        int NPY_UNUSED(aligned), int move_references,
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    /* Python API doesn't use FPEs and this also attempts to hide spurious ones. */
    *flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* NOTE: auxdata is only really necessary to flag `move_references` */
    _object_to_any_auxdata *data = PyMem_Malloc(sizeof(*data));
    if (data == NULL) {
        return -1;
    }
    data->base.free = &_object_to_any_auxdata_free;
    data->base.clone = &_object_to_any_auxdata_clone;

    Py_INCREF(context->descriptors[1]);
    data->descr = context->descriptors[1];
    data->move_references = move_references;
    *out_transferdata = (NpyAuxData *)data;
    *out_loop = &strided_to_strided_object_to_any;
    return 0;
}


/************************** ZERO-PADDED COPY ******************************/

/*
 * Does a strided to strided zero-padded copy for the case where
 * dst_itemsize > src_itemsize
 */
static int
_strided_to_strided_zero_pad_copy(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];
    npy_intp src_itemsize = context->descriptors[0]->elsize;
    npy_intp dst_itemsize = context->descriptors[1]->elsize;

    npy_intp zero_size = dst_itemsize-src_itemsize;

    while (N > 0) {
        memcpy(dst, src, src_itemsize);
        memset(dst + src_itemsize, 0, zero_size);
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

/*
 * Does a strided to strided zero-padded copy for the case where
 * dst_itemsize < src_itemsize
 */
static int
_strided_to_strided_truncate_copy(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];
    npy_intp dst_itemsize = context->descriptors[1]->elsize;

    while (N > 0) {
        memcpy(dst, src, dst_itemsize);
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

/*
 * Does a strided to strided zero-padded or truncated copy for the case where
 * unicode swapping is needed.
 */
static int
_strided_to_strided_unicode_copyswap(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];
    npy_intp src_itemsize = context->descriptors[0]->elsize;
    npy_intp dst_itemsize = context->descriptors[1]->elsize;

    npy_intp zero_size = dst_itemsize - src_itemsize;
    npy_intp copy_size = zero_size > 0 ? src_itemsize : dst_itemsize;
    char *_dst;
    npy_intp characters = dst_itemsize / 4;
    int i;

    while (N > 0) {
        memcpy(dst, src, copy_size);
        if (zero_size > 0) {
            memset(dst + src_itemsize, 0, zero_size);
        }
        _dst = dst;
        for (i=0; i < characters; i++) {
            npy_bswap4_unaligned(_dst);
            _dst += 4;
        }
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}


NPY_NO_EXPORT int
PyArray_GetStridedZeroPadCopyFn(int aligned, int unicode_swap,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp src_itemsize, npy_intp dst_itemsize,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata)
{
    *out_transferdata = NULL;
    if ((src_itemsize == dst_itemsize) && !unicode_swap) {
        *out_stransfer = PyArray_GetStridedCopyFn(aligned, src_stride,
                                dst_stride, src_itemsize);
        return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
    }
    else {
        if (unicode_swap) {
            *out_stransfer = &_strided_to_strided_unicode_copyswap;
        }
        else if (src_itemsize < dst_itemsize) {
            *out_stransfer = &_strided_to_strided_zero_pad_copy;
        }
        else {
            *out_stransfer = &_strided_to_strided_truncate_copy;
        }
        return NPY_SUCCEED;
    }
}


/*************************** WRAP DTYPE COPY/SWAP *************************/
/* Wraps the dtype copy swap function */
typedef struct {
    NpyAuxData base;
    PyArray_CopySwapNFunc *copyswapn;
    int swap;
    PyArrayObject *arr;
} _wrap_copy_swap_data;

/* wrap copy swap data free function */
static void _wrap_copy_swap_data_free(NpyAuxData *data)
{
    _wrap_copy_swap_data *d = (_wrap_copy_swap_data *)data;
    Py_DECREF(d->arr);
    PyMem_Free(data);
}

/* wrap copy swap data copy function */
static NpyAuxData *_wrap_copy_swap_data_clone(NpyAuxData *data)
{
    _wrap_copy_swap_data *newdata =
        (_wrap_copy_swap_data *)PyMem_Malloc(sizeof(_wrap_copy_swap_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_wrap_copy_swap_data));
    Py_INCREF(newdata->arr);

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_wrap_copy_swap(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _wrap_copy_swap_data *d = (_wrap_copy_swap_data *)auxdata;

    /* We assume that d->copyswapn should not be able to error. */
    d->copyswapn(dst, dst_stride, src, src_stride, N, d->swap, d->arr);
    return 0;
}

/*
 * This function is used only via `get_wrapped_legacy_cast_function`
 * when we wrap a legacy DType (or explicitly fall back to the legacy
 * wrapping) for an internal cast.
 */
static int
wrap_copy_swap_function(
        PyArray_Descr *dtype, int should_swap,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata)
{
    /* Allocate the data for the copy swap */
    _wrap_copy_swap_data *data = PyMem_Malloc(sizeof(_wrap_copy_swap_data));
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }

    data->base.free = &_wrap_copy_swap_data_free;
    data->base.clone = &_wrap_copy_swap_data_clone;
    data->copyswapn = PyDataType_GetArrFuncs(dtype)->copyswapn;
    data->swap = should_swap;

    /*
     * TODO: This is a hack so the copyswap functions have an array.
     *       The copyswap functions shouldn't need that.
     */
    Py_INCREF(dtype);
    npy_intp shape = 1;
    data->arr = (PyArrayObject *)PyArray_NewFromDescr_int(
            &PyArray_Type, dtype,
            1, &shape, NULL, NULL,
            0, NULL, NULL,
            _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    if (data->arr == NULL) {
        PyMem_Free(data);
        return NPY_FAIL;
    }

    *out_stransfer = &_strided_to_strided_wrap_copy_swap;
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}

/*************************** DTYPE CAST FUNCTIONS *************************/

/* Does a simple aligned cast */
typedef struct {
    NpyAuxData base;
    PyArray_VectorUnaryFunc *castfunc;
    PyArrayObject *aip, *aop;
    npy_bool needs_api;
} _strided_cast_data;

/* strided cast data free function */
static void _strided_cast_data_free(NpyAuxData *data)
{
    _strided_cast_data *d = (_strided_cast_data *)data;
    Py_DECREF(d->aip);
    Py_DECREF(d->aop);
    PyMem_Free(data);
}

/* strided cast data copy function */
static NpyAuxData *_strided_cast_data_clone(NpyAuxData *data)
{
    _strided_cast_data *newdata =
            (_strided_cast_data *)PyMem_Malloc(sizeof(_strided_cast_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_strided_cast_data));
    Py_INCREF(newdata->aip);
    Py_INCREF(newdata->aop);

    return (NpyAuxData *)newdata;
}

static int
_aligned_strided_to_strided_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _strided_cast_data *d = (_strided_cast_data *)auxdata;
    PyArray_VectorUnaryFunc *castfunc = d->castfunc;
    PyArrayObject *aip = d->aip, *aop = d->aop;
    npy_bool needs_api = d->needs_api;

    while (N > 0) {
        castfunc(src, dst, 1, aip, aop);
        /*
         * Since error handling in ufuncs is not ideal (at the time of
         * writing this, an error could be in process before calling this
         * function. For most of NumPy history these checks were completely
         * missing, so this is hopefully OK for the time being (until ufuncs
         * are fixed).
         */
        if (needs_api && PyErr_Occurred()) {
            return -1;
        }
        dst += dst_stride;
        src += src_stride;
        --N;
    }
    return 0;
}

/* This one requires src be of type NPY_OBJECT */
static int
_aligned_strided_to_strided_cast_decref_src(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;
    _strided_cast_data *d = (_strided_cast_data *)data;
    PyArray_VectorUnaryFunc *castfunc = d->castfunc;
    PyArrayObject *aip = d->aip, *aop = d->aop;
    npy_bool needs_api = d->needs_api;
    PyObject *src_ref;

    while (N > 0) {
        castfunc(src, dst, 1, aip, aop);
        /*
         * See comment in `_aligned_strided_to_strided_cast`, an error could
         * in principle be set before `castfunc` is called.
         */
        if (needs_api && PyErr_Occurred()) {
            return -1;
        }
        /* After casting, decrement the source ref and set it to NULL */
        memcpy(&src_ref, src, sizeof(src_ref));
        Py_XDECREF(src_ref);
        memset(src, 0, sizeof(PyObject *));
        NPY_DT_DBG_REFTRACE("dec src ref (cast object -> not object)", src_ref);

        dst += dst_stride;
        src += src_stride;
        --N;
    }
    return 0;
}

static int
_aligned_contig_to_contig_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char * const*args,
        const npy_intp *dimensions, const npy_intp *NPY_UNUSED(strides),
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];

    _strided_cast_data *d = (_strided_cast_data *)auxdata;
    npy_bool needs_api = d->needs_api;

    d->castfunc(src, dst, N, d->aip, d->aop);
    /*
     * See comment in `_aligned_strided_to_strided_cast`, an error could
     * in principle be set before `castfunc` is called.
     */
    if (needs_api && PyErr_Occurred()) {
        return -1;
    }
    return 0;
}


/*
 * Does a datetime->datetime, timedelta->timedelta,
 * datetime->ascii, or ascii->datetime cast
 */
typedef struct {
    NpyAuxData base;
    /* The conversion fraction */
    npy_int64 num, denom;
    /* For the datetime -> string conversion, the dst string length */
    npy_intp src_itemsize, dst_itemsize;
    /*
     * A buffer of size 'src_itemsize + 1', for when the input
     * string is exactly of length src_itemsize with no NULL
     * terminator.
     */
    char *tmp_buffer;
    /*
     * The metadata for when dealing with Months or Years
     * which behave non-linearly with respect to the other
     * units.
     */
    PyArray_DatetimeMetaData src_meta, dst_meta;
} _strided_datetime_cast_data;

/* strided datetime cast data free function */
static void _strided_datetime_cast_data_free(NpyAuxData *data)
{
    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)data;
    PyMem_Free(d->tmp_buffer);
    PyMem_Free(data);
}

/* strided datetime cast data copy function */
static NpyAuxData *_strided_datetime_cast_data_clone(NpyAuxData *data)
{
    _strided_datetime_cast_data *newdata =
            (_strided_datetime_cast_data *)PyMem_Malloc(
                                        sizeof(_strided_datetime_cast_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_strided_datetime_cast_data));
    if (newdata->tmp_buffer != NULL) {
        newdata->tmp_buffer = PyMem_Malloc(newdata->src_itemsize + 1);
        if (newdata->tmp_buffer == NULL) {
            PyMem_Free(newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_datetime_general_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char * const*args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;
    npy_int64 dt;
    npy_datetimestruct dts;

    while (N > 0) {
        memcpy(&dt, src, sizeof(dt));

        if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(&d->src_meta,
                                               dt, &dts) < 0) {
            return -1;
        }
        else {
            if (NpyDatetime_ConvertDatetimeStructToDatetime64(&d->dst_meta,
                                                   &dts, &dt) < 0) {
                return -1;
            }
        }

        memcpy(dst, &dt, sizeof(dt));

        dst += dst_stride;
        src += src_stride;
        --N;
    }
    return 0;
}

static int
_strided_to_strided_datetime_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char * const*args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;
    npy_int64 num = d->num, denom = d->denom;
    npy_int64 dt;

    while (N > 0) {
        memcpy(&dt, src, sizeof(dt));

        if (dt != NPY_DATETIME_NAT) {
            /* Apply the scaling */
            if (dt < 0) {
                dt = (dt * num - (denom - 1)) / denom;
            }
            else {
                dt = dt * num / denom;
            }
        }

        memcpy(dst, &dt, sizeof(dt));

        dst += dst_stride;
        src += src_stride;
        --N;
    }
    return 0;
}

static int
_aligned_strided_to_strided_datetime_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;
    npy_int64 num = d->num, denom = d->denom;
    npy_int64 dt;

    while (N > 0) {
        dt = *(npy_int64 *)src;

        if (dt != NPY_DATETIME_NAT) {
            /* Apply the scaling */
            if (dt < 0) {
                dt = (dt * num - (denom - 1)) / denom;
            }
            else {
                dt = dt * num / denom;
            }
        }

        *(npy_int64 *)dst = dt;

        dst += dst_stride;
        src += src_stride;
        --N;
    }
    return 0;
}

static int
_strided_to_strided_datetime_to_string(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;
    npy_intp dst_itemsize = d->dst_itemsize;
    npy_int64 dt;
    npy_datetimestruct dts;

    while (N > 0) {
        memcpy(&dt, src, sizeof(dt));

        if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(&d->src_meta,
                                               dt, &dts) < 0) {
            return -1;
        }

        /* Initialize the destination to all zeros */
        memset(dst, 0, dst_itemsize);

        if (NpyDatetime_MakeISO8601Datetime(&dts, dst, dst_itemsize,
                                0, 0, d->src_meta.base, -1,
                                NPY_UNSAFE_CASTING) < 0) {
            return -1;
        }

        dst += dst_stride;
        src += src_stride;
        --N;
    }
    return 0;
}

static int
_strided_to_strided_string_to_datetime(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_itemsize = context->descriptors[0]->elsize;
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;
    npy_datetimestruct dts;
    char *tmp_buffer = d->tmp_buffer;
    char *tmp;

    while (N > 0) {
        npy_int64 dt = ~NPY_DATETIME_NAT;

        /* Replicating strnlen with memchr, because Mac OS X lacks it */
        tmp = memchr(src, '\0', src_itemsize);

        /* If the string is all full, use the buffer */
        if (tmp == NULL) {
            memcpy(tmp_buffer, src, src_itemsize);
            tmp_buffer[src_itemsize] = '\0';

            if (NpyDatetime_ParseISO8601Datetime(
                    tmp_buffer, src_itemsize,
                    d->dst_meta.base, NPY_SAME_KIND_CASTING,
                    &dts, NULL, NULL) < 0) {
                return -1;
            }
        }
        /* Otherwise parse the data in place */
        else {
            if (NpyDatetime_ParseISO8601Datetime(
                    src, tmp - src,
                    d->dst_meta.base, NPY_SAME_KIND_CASTING,
                    &dts, NULL, NULL) < 0) {
                return -1;
            }
        }

        /* Convert to the datetime */
        if (dt != NPY_DATETIME_NAT &&
                NpyDatetime_ConvertDatetimeStructToDatetime64(&d->dst_meta,
                                               &dts, &dt) < 0) {
            return -1;
        }

        memcpy(dst, &dt, sizeof(dt));

        dst += dst_stride;
        src += src_stride;
        --N;
    }
    return 0;
}

/*
 * Assumes src_dtype and dst_dtype are both datetimes or both timedeltas
 */
NPY_NO_EXPORT int
get_nbo_cast_datetime_transfer_function(int aligned,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata)
{
    PyArray_DatetimeMetaData *src_meta, *dst_meta;
    npy_int64 num = 0, denom = 0;
    _strided_datetime_cast_data *data;

    src_meta = get_datetime_metadata_from_dtype(src_dtype);
    if (src_meta == NULL) {
        return NPY_FAIL;
    }
    dst_meta = get_datetime_metadata_from_dtype(dst_dtype);
    if (dst_meta == NULL) {
        return NPY_FAIL;
    }

    get_datetime_conversion_factor(src_meta, dst_meta, &num, &denom);

    if (num == 0) {
        return NPY_FAIL;
    }

    /* Allocate the data for the casting */
    data = (_strided_datetime_cast_data *)PyMem_Malloc(
                                    sizeof(_strided_datetime_cast_data));
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }
    data->base.free = &_strided_datetime_cast_data_free;
    data->base.clone = &_strided_datetime_cast_data_clone;
    data->num = num;
    data->denom = denom;
    data->tmp_buffer = NULL;

    /*
     * Special case the datetime (but not timedelta) with the nonlinear
     * units (years and months). For timedelta, an average
     * years and months value is used.
     */
    if (src_dtype->type_num == NPY_DATETIME &&
            (src_meta->base == NPY_FR_Y ||
             src_meta->base == NPY_FR_M ||
             dst_meta->base == NPY_FR_Y ||
             dst_meta->base == NPY_FR_M)) {
        memcpy(&data->src_meta, src_meta, sizeof(data->src_meta));
        memcpy(&data->dst_meta, dst_meta, sizeof(data->dst_meta));
        *out_stransfer = &_strided_to_strided_datetime_general_cast;
    }
    else if (aligned) {
        *out_stransfer = &_aligned_strided_to_strided_datetime_cast;
    }
    else {
        *out_stransfer = &_strided_to_strided_datetime_cast;
    }
    *out_transferdata = (NpyAuxData *)data;

#if NPY_DT_DBG_TRACING
    printf("Dtype transfer from ");
    _safe_print((PyObject *)src_dtype);
    printf(" to ");
    _safe_print((PyObject *)dst_dtype);
    printf("\n");
    printf("has conversion fraction %lld/%lld\n", num, denom);
#endif


    return NPY_SUCCEED;
}

NPY_NO_EXPORT int
get_nbo_datetime_to_string_transfer_function(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer, NpyAuxData **out_transferdata)
{
    PyArray_DatetimeMetaData *src_meta;
    _strided_datetime_cast_data *data;

    src_meta = get_datetime_metadata_from_dtype(src_dtype);
    if (src_meta == NULL) {
        return NPY_FAIL;
    }

    /* Allocate the data for the casting */
    data = (_strided_datetime_cast_data *)PyMem_Malloc(
                                    sizeof(_strided_datetime_cast_data));
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }
    data->base.free = &_strided_datetime_cast_data_free;
    data->base.clone = &_strided_datetime_cast_data_clone;
    data->dst_itemsize = dst_dtype->elsize;
    data->tmp_buffer = NULL;

    memcpy(&data->src_meta, src_meta, sizeof(data->src_meta));

    *out_stransfer = &_strided_to_strided_datetime_to_string;
    *out_transferdata = (NpyAuxData *)data;

#if NPY_DT_DBG_TRACING
    printf("Dtype transfer from ");
    _safe_print((PyObject *)src_dtype);
    printf(" to ");
    _safe_print((PyObject *)dst_dtype);
    printf("\n");
#endif

    return NPY_SUCCEED;
}


NPY_NO_EXPORT int
get_datetime_to_unicode_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyArray_Descr *str_dtype;

    /* Get an ASCII string data type, adapted to match the UNICODE one */
    str_dtype = PyArray_DescrNewFromType(NPY_STRING);
    if (str_dtype == NULL) {
        return NPY_FAIL;
    }
    str_dtype->elsize = dst_dtype->elsize / 4;

    /* ensured in resolve_descriptors for simplicity */
    assert(PyDataType_ISNOTSWAPPED(src_dtype));

    /* Get the NBO datetime to string aligned contig function */
    if (get_nbo_datetime_to_string_transfer_function(
            src_dtype, str_dtype,
            out_stransfer, out_transferdata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        return NPY_FAIL;
    }

    int res = wrap_aligned_transferfunction(
            aligned, 0,  /* no need to ensure contiguous */
            src_stride, dst_stride,
            src_dtype, dst_dtype,
            src_dtype, str_dtype,
            out_stransfer, out_transferdata, out_needs_api);
    Py_DECREF(str_dtype);
    if (res < 0) {
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}

NPY_NO_EXPORT int
get_nbo_string_to_datetime_transfer_function(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer, NpyAuxData **out_transferdata)
{
    PyArray_DatetimeMetaData *dst_meta;
    _strided_datetime_cast_data *data;

    dst_meta = get_datetime_metadata_from_dtype(dst_dtype);
    if (dst_meta == NULL) {
        return NPY_FAIL;
    }

    /* Allocate the data for the casting */
    data = (_strided_datetime_cast_data *)PyMem_Malloc(
                                    sizeof(_strided_datetime_cast_data));
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }
    data->base.free = &_strided_datetime_cast_data_free;
    data->base.clone = &_strided_datetime_cast_data_clone;
    data->src_itemsize = src_dtype->elsize;
    data->tmp_buffer = PyMem_Malloc(data->src_itemsize + 1);
    if (data->tmp_buffer == NULL) {
        PyErr_NoMemory();
        PyMem_Free(data);
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }

    memcpy(&data->dst_meta, dst_meta, sizeof(data->dst_meta));

    *out_stransfer = &_strided_to_strided_string_to_datetime;
    *out_transferdata = (NpyAuxData *)data;

#if NPY_DT_DBG_TRACING
    printf("Dtype transfer from ");
    _safe_print((PyObject *)src_dtype);
    printf(" to ");
    _safe_print((PyObject *)dst_dtype);
    printf("\n");
#endif

    return NPY_SUCCEED;
}

NPY_NO_EXPORT int
get_unicode_to_datetime_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyArray_Descr *str_dtype;

    /* Get an ASCII string data type, adapted to match the UNICODE one */
    str_dtype = PyArray_DescrNewFromType(NPY_STRING);
    if (str_dtype == NULL) {
        return NPY_FAIL;
    }
    assert(src_dtype->type_num == NPY_UNICODE);
    str_dtype->elsize = src_dtype->elsize / 4;

    /* Get the string to NBO datetime aligned function */
    if (get_nbo_string_to_datetime_transfer_function(
            str_dtype, dst_dtype,
            out_stransfer, out_transferdata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        return NPY_FAIL;
    }

    int res = wrap_aligned_transferfunction(
            aligned, 0,  /* no need to ensure contiguous */
            src_stride, dst_stride,
            src_dtype, dst_dtype,
            str_dtype, dst_dtype,
            out_stransfer, out_transferdata, out_needs_api);
    Py_DECREF(str_dtype);

    if (res < 0) {
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}


NPY_NO_EXPORT int
get_legacy_dtype_cast_function(
        int aligned, npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArrayMethod_StridedLoop **out_stransfer, NpyAuxData **out_transferdata,
        int *out_needs_api, int *out_needs_wrap)
{
    _strided_cast_data *data;
    PyArray_VectorUnaryFunc *castfunc;
    PyArray_Descr *tmp_dtype;
    npy_intp shape = 1;
    npy_intp src_itemsize = src_dtype->elsize;
    npy_intp dst_itemsize = dst_dtype->elsize;

    *out_needs_wrap = !aligned ||
                      !PyArray_ISNBO(src_dtype->byteorder) ||
                      !PyArray_ISNBO(dst_dtype->byteorder);

    /* Check the data types whose casting functions use API calls */
    switch (src_dtype->type_num) {
        case NPY_OBJECT:
        case NPY_STRING:
        case NPY_UNICODE:
        case NPY_VOID:
            if (out_needs_api) {
                *out_needs_api = 1;
            }
            break;
    }
    switch (dst_dtype->type_num) {
        case NPY_OBJECT:
        case NPY_STRING:
        case NPY_UNICODE:
        case NPY_VOID:
            if (out_needs_api) {
                *out_needs_api = 1;
            }
            break;
    }

    if (PyDataType_FLAGCHK(src_dtype, NPY_NEEDS_PYAPI) ||
            PyDataType_FLAGCHK(dst_dtype, NPY_NEEDS_PYAPI)) {
        if (out_needs_api) {
            *out_needs_api = 1;
        }
    }

    /* Get the cast function */
    castfunc = PyArray_GetCastFunc(src_dtype, dst_dtype->type_num);
    if (!castfunc) {
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }

    /* Allocate the data for the casting */
    data = (_strided_cast_data *)PyMem_Malloc(sizeof(_strided_cast_data));
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }
    data->base.free = &_strided_cast_data_free;
    data->base.clone = &_strided_cast_data_clone;
    data->castfunc = castfunc;
    data->needs_api = *out_needs_api;
    /*
     * TODO: This is a hack so the cast functions have an array.
     *       The cast functions shouldn't need that.  Also, since we
     *       always handle byte order conversions, this array should
     *       have native byte order.
     */
    if (PyArray_ISNBO(src_dtype->byteorder)) {
        tmp_dtype = src_dtype;
        Py_INCREF(tmp_dtype);
    }
    else {
        tmp_dtype = PyArray_DescrNewByteorder(src_dtype, NPY_NATIVE);
        if (tmp_dtype == NULL) {
            PyMem_Free(data);
            return NPY_FAIL;
        }
    }
    data->aip = (PyArrayObject *)PyArray_NewFromDescr_int(
            &PyArray_Type, tmp_dtype,
            1, &shape, NULL, NULL,
            0, NULL, NULL,
            _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    if (data->aip == NULL) {
        PyMem_Free(data);
        return NPY_FAIL;
    }
    /*
     * TODO: This is a hack so the cast functions have an array.
     *       The cast functions shouldn't need that.  Also, since we
     *       always handle byte order conversions, this array should
     *       have native byte order.
     */
    if (PyArray_ISNBO(dst_dtype->byteorder)) {
        tmp_dtype = dst_dtype;
        Py_INCREF(tmp_dtype);
    }
    else {
        tmp_dtype = PyArray_DescrNewByteorder(dst_dtype, NPY_NATIVE);
        if (tmp_dtype == NULL) {
            Py_DECREF(data->aip);
            PyMem_Free(data);
            return NPY_FAIL;
        }
    }
    data->aop = (PyArrayObject *)PyArray_NewFromDescr_int(
            &PyArray_Type, tmp_dtype,
            1, &shape, NULL, NULL,
            0, NULL, NULL,
            _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    if (data->aop == NULL) {
        Py_DECREF(data->aip);
        PyMem_Free(data);
        return NPY_FAIL;
    }

    /* If it's aligned and all native byte order, we're all done */
    if (move_references && src_dtype->type_num == NPY_OBJECT) {
        *out_stransfer = _aligned_strided_to_strided_cast_decref_src;
    }
    else {
        /*
         * Use the contig version if the strides are contiguous or
         * we're telling the caller to wrap the return, because
         * the wrapping uses a contiguous buffer.
         */
        if ((src_stride == src_itemsize && dst_stride == dst_itemsize) ||
                        *out_needs_wrap) {
            *out_stransfer = _aligned_contig_to_contig_cast;
        }
        else {
            *out_stransfer = _aligned_strided_to_strided_cast;
        }
    }
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}


/**************************** COPY 1 TO N CONTIGUOUS ************************/

/* Copies 1 element to N contiguous elements */
typedef struct {
    NpyAuxData base;
    npy_intp N;
    NPY_cast_info wrapped;
    /* If finish->func is non-NULL the source needs a decref */
    NPY_traverse_info decref_src;
} _one_to_n_data;

/* transfer data free function */
static void _one_to_n_data_free(NpyAuxData *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    NPY_cast_info_xfree(&d->wrapped);
    NPY_traverse_info_xfree(&d->decref_src);
    PyMem_Free(data);
}

/* transfer data copy function */
static NpyAuxData *_one_to_n_data_clone(NpyAuxData *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    _one_to_n_data *newdata;

    /* Allocate the data, and populate it */
    newdata = (_one_to_n_data *)PyMem_Malloc(sizeof(_one_to_n_data));
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base.free = &_one_to_n_data_free;
    newdata->base.clone = &_one_to_n_data_clone;
    newdata->N = d->N;
    /* Initialize in case of error, or if it is unused */
    NPY_traverse_info_init(&newdata->decref_src);

    if (NPY_cast_info_copy(&newdata->wrapped, &d->wrapped) < 0) {
        _one_to_n_data_free((NpyAuxData *)newdata);
        return NULL;
    }
    if (d->decref_src.func == NULL) {
        return (NpyAuxData *)newdata;
    }

    if (NPY_traverse_info_copy(&newdata->decref_src, &d->decref_src) < 0) {
        _one_to_n_data_free((NpyAuxData *)newdata);
        return NULL;
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_one_to_n(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _one_to_n_data *d = (_one_to_n_data *)auxdata;

    const npy_intp subN = d->N;
    npy_intp sub_strides[2] = {0, d->wrapped.descriptors[1]->elsize};

    while (N > 0) {
        char *sub_args[2] = {src, dst};
        if (d->wrapped.func(&d->wrapped.context,
                sub_args, &subN, sub_strides, d->wrapped.auxdata) < 0) {
            return -1;
        }

        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

static int
_strided_to_strided_one_to_n_with_finish(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _one_to_n_data *d = (_one_to_n_data *)auxdata;

    const npy_intp subN = d->N;
    const npy_intp one_item = 1, zero_stride = 0;
    npy_intp sub_strides[2] = {0, d->wrapped.descriptors[1]->elsize};

    while (N > 0) {
        char *sub_args[2] = {src, dst};
        if (d->wrapped.func(&d->wrapped.context,
                sub_args, &subN, sub_strides, d->wrapped.auxdata) < 0) {
            return -1;
        }

        if (d->decref_src.func(NULL, d->decref_src.descr,
                src, one_item, zero_stride, d->decref_src.auxdata) < 0) {
            return -1;
        }

        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}


static int
get_one_to_n_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            npy_intp N,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    _one_to_n_data *data = PyMem_Malloc(sizeof(_one_to_n_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }

    data->base.free = &_one_to_n_data_free;
    data->base.clone = &_one_to_n_data_clone;
    data->N = N;
    NPY_traverse_info_init(&data->decref_src);  /* In case of error */

    /*
     * move_references is set to 0, handled in the wrapping transfer fn,
     * src_stride is set to zero, because its 1 to N copying,
     * and dst_stride is set to contiguous, because subarrays are always
     * contiguous.
     */
    if (PyArray_GetDTypeTransferFunction(aligned,
                    0, dst_dtype->elsize,
                    src_dtype, dst_dtype,
                    0,
                    &data->wrapped,
                    out_flags) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE((NpyAuxData *)data);
        return NPY_FAIL;
    }

    /* If the src object will need a DECREF, set src_dtype */
    if (move_references && PyDataType_REFCHK(src_dtype)) {
        NPY_ARRAYMETHOD_FLAGS clear_flags;
        if (PyArray_GetClearFunction(
                aligned, src_stride, src_dtype,
                &data->decref_src, &clear_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
        *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, clear_flags);
    }

    if (data->decref_src.func == NULL) {
        *out_stransfer = &_strided_to_strided_one_to_n;
    }
    else {
        *out_stransfer = &_strided_to_strided_one_to_n_with_finish;
    }
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}

/**************************** COPY N TO N CONTIGUOUS ************************/

/* Copies N contiguous elements to N contiguous elements */
typedef struct {
    NpyAuxData base;
    NPY_cast_info wrapped;
    npy_intp N;
    npy_intp strides[2];  /* avoid look up on the dtype (dst can be NULL) */
} _n_to_n_data;

/* transfer data free function */
static void _n_to_n_data_free(NpyAuxData *data)
{
    _n_to_n_data *d = (_n_to_n_data *)data;
    NPY_cast_info_xfree(&d->wrapped);
    PyMem_Free(data);
}

/* transfer data copy function */
static NpyAuxData *_n_to_n_data_clone(NpyAuxData *data)
{
    _n_to_n_data *d = (_n_to_n_data *)data;
    _n_to_n_data *newdata;

    /* Allocate the data, and populate it */
    newdata = (_n_to_n_data *)PyMem_Malloc(sizeof(_n_to_n_data));
    if (newdata == NULL) {
        return NULL;
    }
    *newdata = *d;

    if (NPY_cast_info_copy(&newdata->wrapped, &d->wrapped) < 0) {
        _n_to_n_data_free((NpyAuxData *)newdata);
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_1_to_1(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    _n_to_n_data *d = (_n_to_n_data *)auxdata;
    return d->wrapped.func(&d->wrapped.context,
            args, dimensions, strides, d->wrapped.auxdata);
}

static int
_strided_to_strided_n_to_n(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _n_to_n_data *d = (_n_to_n_data *)auxdata;
    npy_intp subN = d->N;

    while (N > 0) {
        char *sub_args[2] = {src, dst};
        if (d->wrapped.func(&d->wrapped.context,
                sub_args, &subN, d->strides, d->wrapped.auxdata) < 0) {
            return -1;
        }
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

static int
_contig_to_contig_n_to_n(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *NPY_UNUSED(strides),
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];

    _n_to_n_data *d = (_n_to_n_data *)auxdata;
    /* Make one large transfer including both outer and inner iteration: */
    npy_intp subN = N * d->N;

    char *sub_args[2] = {src, dst};
    if (d->wrapped.func(&d->wrapped.context,
            sub_args, &subN, d->strides, d->wrapped.auxdata) < 0) {
        return -1;
    }
    return 0;
}


/*
 * Note that this function is currently both used for structured dtype
 * casting as well as a decref function (with `dst_dtype == NULL`)
 */
static int
get_n_to_n_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            npy_intp N,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    _n_to_n_data *data = PyMem_Malloc(sizeof(_n_to_n_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->base.free = &_n_to_n_data_free;
    data->base.clone = &_n_to_n_data_clone;
    data->N = N;

    if (N != 1) {
        /*
         * If N == 1, we can use the original strides,
         * otherwise fields are contiguous
         */
        src_stride = src_dtype->elsize;
        dst_stride = dst_dtype != NULL ? dst_dtype->elsize : 0;
        /* Store the wrapped strides for easier access */
        data->strides[0] = src_stride;
        data->strides[1] = dst_stride;
    }

    /*
     * src_stride and dst_stride are set to contiguous, because
     * subarrays are always contiguous.
     */
    if (PyArray_GetDTypeTransferFunction(aligned,
                    src_stride, dst_stride,
                    src_dtype, dst_dtype,
                    move_references,
                    &data->wrapped,
                    out_flags) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE((NpyAuxData *)data);
        return NPY_FAIL;
    }

    if (N == 1) {
        /*
         * No need for wrapping, we can just copy directly. In principle
         * this step could be optimized away entirely, but it requires
         * replacing the context (to have the unpacked dtypes).
         */
        *out_stransfer = &_strided_to_strided_1_to_1;
    }
    else if (src_stride == N * src_stride &&
             dst_stride == N * dst_stride) {
        /* The subarrays can be coalesced (probably very rarely) */
        *out_stransfer = &_contig_to_contig_n_to_n;
    }
    else {
        *out_stransfer = &_strided_to_strided_n_to_n;
    }
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}

/********************** COPY WITH SUBARRAY BROADCAST ************************/

typedef struct {
    npy_intp offset, count;
} _subarray_broadcast_offsetrun;

/* Copies element with subarray broadcasting */
typedef struct {
    NpyAuxData base;
    NPY_cast_info wrapped;
    NPY_traverse_info decref_src;
    NPY_traverse_info decref_dst;  /* The use-case should probably be deprecated */
    npy_intp src_N, dst_N;
    /* This gets a run-length encoded representation of the transfer */
    npy_intp run_count;
    _subarray_broadcast_offsetrun offsetruns[];
} _subarray_broadcast_data;


/* transfer data free function */
static void _subarray_broadcast_data_free(NpyAuxData *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;
    NPY_cast_info_xfree(&d->wrapped);
    NPY_traverse_info_xfree(&d->decref_src);
    NPY_traverse_info_xfree(&d->decref_dst);
    PyMem_Free(data);
}

/* transfer data copy function */
static NpyAuxData *_subarray_broadcast_data_clone(NpyAuxData *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;

    npy_intp offsetruns_size = d->run_count*sizeof(_subarray_broadcast_offsetrun);
    npy_intp structsize = sizeof(_subarray_broadcast_data) + offsetruns_size;

    /* Allocate the data and populate it */
    _subarray_broadcast_data *newdata = PyMem_Malloc(structsize);
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base.free = &_subarray_broadcast_data_free;
    newdata->base.clone = &_subarray_broadcast_data_clone;
    newdata->src_N = d->src_N;
    newdata->dst_N = d->dst_N;
    newdata->run_count = d->run_count;
    memcpy(newdata->offsetruns, d->offsetruns, offsetruns_size);

    NPY_traverse_info_init(&newdata->decref_src);
    NPY_traverse_info_init(&newdata->decref_dst);

    if (NPY_cast_info_copy(&newdata->wrapped, &d->wrapped) < 0) {
        _subarray_broadcast_data_free((NpyAuxData *)newdata);
        return NULL;
    }
    if (d->decref_src.func != NULL) {
        if (NPY_traverse_info_copy(&newdata->decref_src, &d->decref_src) < 0) {
            _subarray_broadcast_data_free((NpyAuxData *) newdata);
            return NULL;
        }
    }
    if (d->decref_dst.func != NULL) {
        if (NPY_traverse_info_copy(&newdata->decref_dst, &d->decref_dst) < 0) {
            _subarray_broadcast_data_free((NpyAuxData *) newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_subarray_broadcast(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _subarray_broadcast_data *d = (_subarray_broadcast_data *)auxdata;
    npy_intp run, run_count = d->run_count;
    npy_intp loop_index, offset, count;

    npy_intp src_subitemsize = d->wrapped.descriptors[0]->elsize;
    npy_intp dst_subitemsize = d->wrapped.descriptors[1]->elsize;

    npy_intp sub_strides[2] = {src_subitemsize, dst_subitemsize};

    while (N > 0) {
        loop_index = 0;
        for (run = 0; run < run_count; ++run) {
            offset = d->offsetruns[run].offset;
            count = d->offsetruns[run].count;
            char *dst_ptr = dst + loop_index*dst_subitemsize;
            char *sub_args[2] = {src + offset, dst_ptr};
            if (offset != -1) {
                if (d->wrapped.func(&d->wrapped.context,
                        sub_args, &count, sub_strides, d->wrapped.auxdata) < 0) {
                    return -1;
                }
            }
            else {
                memset(dst_ptr, 0, count*dst_subitemsize);
            }
            loop_index += count;
        }

        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}


static int
_strided_to_strided_subarray_broadcast_withrefs(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _subarray_broadcast_data *d = (_subarray_broadcast_data *)auxdata;
    npy_intp run, run_count = d->run_count;
    npy_intp loop_index, offset, count;

    npy_intp src_subitemsize = d->wrapped.descriptors[0]->elsize;
    npy_intp dst_subitemsize = d->wrapped.descriptors[1]->elsize;

    npy_intp sub_strides[2] = {src_subitemsize, dst_subitemsize};

    while (N > 0) {
        loop_index = 0;
        for (run = 0; run < run_count; ++run) {
            offset = d->offsetruns[run].offset;
            count = d->offsetruns[run].count;
            char *dst_ptr = dst + loop_index*dst_subitemsize;
            char *sub_args[2] = {src + offset, dst_ptr};
            if (offset != -1) {
                if (d->wrapped.func(&d->wrapped.context,
                        sub_args, &count, sub_strides, d->wrapped.auxdata) < 0) {
                    return -1;
                }
            }
            else {
                if (d->decref_dst.func != NULL) {
                    if (d->decref_dst.func(NULL, d->decref_dst.descr,
                            dst_ptr, count, dst_subitemsize,
                            d->decref_dst.auxdata) < 0) {
                        return -1;
                    }
                }
                memset(dst_ptr, 0, count*dst_subitemsize);
            }
            loop_index += count;
        }

        if (d->decref_src.func != NULL) {
            if (d->decref_src.func(NULL, d->decref_src.descr,
                    src, d->src_N, src_subitemsize,
                    d->decref_src.auxdata) < 0) {
                return -1;
            }
        }

        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}


static int
get_subarray_broadcast_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            npy_intp src_size, npy_intp dst_size,
                            PyArray_Dims src_shape, PyArray_Dims dst_shape,
                            int move_references,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    _subarray_broadcast_data *data;
    npy_intp structsize, loop_index, run, run_size,
             src_index, dst_index, i, ndim;

    structsize = sizeof(_subarray_broadcast_data) +
                        dst_size*sizeof(_subarray_broadcast_offsetrun);

    /* Allocate the data and populate it */
    data = (_subarray_broadcast_data *)PyMem_Malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->base.free = &_subarray_broadcast_data_free;
    data->base.clone = &_subarray_broadcast_data_clone;
    data->src_N = src_size;
    data->dst_N = dst_size;

    NPY_traverse_info_init(&data->decref_src);
    NPY_traverse_info_init(&data->decref_dst);

    /*
     * move_references is set to 0, handled in the wrapping transfer fn,
     * src_stride and dst_stride are set to contiguous, as N will always
     * be 1 when it's called.
     */
    if (PyArray_GetDTypeTransferFunction(aligned,
                    src_dtype->elsize, dst_dtype->elsize,
                    src_dtype, dst_dtype,
                    0,
                    &data->wrapped,
                    out_flags) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE((NpyAuxData *)data);
        return NPY_FAIL;
    }

    /* If the src object will need a DECREF */
    if (move_references && PyDataType_REFCHK(src_dtype)) {
        if (PyArray_GetClearFunction(aligned,
                        src_dtype->elsize, src_dtype,
                        &data->decref_src, out_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
    }

    /* If the dst object needs a DECREF to set it to NULL */
    if (PyDataType_REFCHK(dst_dtype)) {
        if (PyArray_GetClearFunction(aligned,
                        dst_dtype->elsize, dst_dtype,
                        &data->decref_dst, out_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
    }

    /* Calculate the broadcasting and set the offsets */
    _subarray_broadcast_offsetrun *offsetruns = data->offsetruns;
    ndim = (src_shape.len > dst_shape.len) ? src_shape.len : dst_shape.len;
    for (loop_index = 0; loop_index < dst_size; ++loop_index) {
        npy_intp src_factor = 1;

        dst_index = loop_index;
        src_index = 0;
        for (i = ndim-1; i >= 0; --i) {
            npy_intp coord = 0, shape;

            /* Get the dst coord of this index for dimension i */
            if (i >= ndim - dst_shape.len) {
                shape = dst_shape.ptr[i-(ndim-dst_shape.len)];
                coord = dst_index % shape;
                dst_index /= shape;
            }

            /* Translate it into a src coord and update src_index */
            if (i >= ndim - src_shape.len) {
                shape = src_shape.ptr[i-(ndim-src_shape.len)];
                if (shape == 1) {
                    coord = 0;
                }
                else {
                    if (coord < shape) {
                        src_index += src_factor*coord;
                        src_factor *= shape;
                    }
                    else {
                        /* Out of bounds, flag with -1 */
                        src_index = -1;
                        break;
                    }
                }
            }
        }
        /* Set the offset */
        if (src_index == -1) {
            offsetruns[loop_index].offset = -1;
        }
        else {
            offsetruns[loop_index].offset = src_index;
        }
    }

    /* Run-length encode the result */
    run = 0;
    run_size = 1;
    for (loop_index = 1; loop_index < dst_size; ++loop_index) {
        if (offsetruns[run].offset == -1) {
            /* Stop the run when there's a valid index again */
            if (offsetruns[loop_index].offset != -1) {
                offsetruns[run].count = run_size;
                run++;
                run_size = 1;
                offsetruns[run].offset = offsetruns[loop_index].offset;
            }
            else {
                run_size++;
            }
        }
        else {
            /* Stop the run when there's a valid index again */
            if (offsetruns[loop_index].offset !=
                            offsetruns[loop_index-1].offset + 1) {
                offsetruns[run].count = run_size;
                run++;
                run_size = 1;
                offsetruns[run].offset = offsetruns[loop_index].offset;
            }
            else {
                run_size++;
            }
        }
    }
    offsetruns[run].count = run_size;
    run++;
    data->run_count = run;

    /* Multiply all the offsets by the src item size */
    while (run--) {
        if (offsetruns[run].offset != -1) {
            offsetruns[run].offset *= src_dtype->elsize;
        }
    }

    if (data->decref_src.func == NULL &&
            data->decref_dst.func == NULL) {
        *out_stransfer = &_strided_to_strided_subarray_broadcast;
    }
    else {
        *out_stransfer = &_strided_to_strided_subarray_broadcast_withrefs;
    }
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}

/*
 * Handles subarray transfer.  To call this, at least one of the dtype's
 * subarrays must be non-NULL
 */
NPY_NO_EXPORT int
get_subarray_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    PyArray_Dims src_shape = {NULL, -1}, dst_shape = {NULL, -1};
    npy_intp src_size = 1, dst_size = 1;

    /* Get the subarray shapes and sizes */
    if (PyDataType_HASSUBARRAY(src_dtype)) {
        if (!(PyArray_IntpConverter(PyDataType_SUBARRAY(src_dtype)->shape,
                                            &src_shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return NPY_FAIL;
        }
        src_size = PyArray_MultiplyList(src_shape.ptr, src_shape.len);
        src_dtype = PyDataType_SUBARRAY(src_dtype)->base;
    }
    if (PyDataType_HASSUBARRAY(dst_dtype)) {
        if (!(PyArray_IntpConverter(PyDataType_SUBARRAY(dst_dtype)->shape,
                                            &dst_shape))) {
            npy_free_cache_dim_obj(src_shape);
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return NPY_FAIL;
        }
        dst_size = PyArray_MultiplyList(dst_shape.ptr, dst_shape.len);
        dst_dtype = PyDataType_SUBARRAY(dst_dtype)->base;
    }

    /*
     * Copy the src value to all the dst values, the size one can be
     * special cased for speed.
     */
    if ((dst_size == 1 && src_size == 1) || (
            src_shape.len == dst_shape.len && PyArray_CompareLists(
                    src_shape.ptr, dst_shape.ptr, src_shape.len))) {

        npy_free_cache_dim_obj(src_shape);
        npy_free_cache_dim_obj(dst_shape);

        return get_n_to_n_transfer_function(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        src_size,
                        out_stransfer, out_transferdata,
                        out_flags);
    }
    /* Copy the src value to all the dst values */
    else if (src_size == 1) {
        npy_free_cache_dim_obj(src_shape);
        npy_free_cache_dim_obj(dst_shape);

        return get_one_to_n_transfer_function(aligned,
                src_stride, dst_stride,
                src_dtype, dst_dtype,
                move_references,
                dst_size,
                out_stransfer, out_transferdata,
                out_flags);
    }
    /*
     * Copy the subarray with broadcasting, truncating, and zero-padding
     * as necessary.
     */
    else {
        int ret = get_subarray_broadcast_transfer_function(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        src_size, dst_size,
                        src_shape, dst_shape,
                        move_references,
                        out_stransfer, out_transferdata,
                        out_flags);

        npy_free_cache_dim_obj(src_shape);
        npy_free_cache_dim_obj(dst_shape);
        return ret;
    }
}

/**************************** COPY FIELDS *******************************/
typedef struct {
    npy_intp src_offset, dst_offset;
    NPY_cast_info info;
} _single_field_transfer;

typedef struct {
    NpyAuxData base;
    npy_intp field_count;
    NPY_traverse_info decref_src;
    _single_field_transfer fields[];
} _field_transfer_data;


/* transfer data free function */
static void _field_transfer_data_free(NpyAuxData *data)
{
    _field_transfer_data *d = (_field_transfer_data *)data;
    NPY_traverse_info_xfree(&d->decref_src);

    for (npy_intp i = 0; i < d->field_count; ++i) {
        NPY_cast_info_xfree(&d->fields[i].info);
    }
    PyMem_Free(d);
}

/* transfer data copy function */
static NpyAuxData *_field_transfer_data_clone(NpyAuxData *data)
{
    _field_transfer_data *d = (_field_transfer_data *)data;

    npy_intp field_count = d->field_count;
    npy_intp structsize = sizeof(_field_transfer_data) +
                    field_count * sizeof(_single_field_transfer);

    /* Allocate the data and populate it */
    _field_transfer_data *newdata = PyMem_Malloc(structsize);
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base = d->base;
    newdata->field_count = 0;
    if (NPY_traverse_info_copy(&newdata->decref_src, &d->decref_src) < 0) {
        PyMem_Free(newdata);
        return NULL;
    }

    /* Copy all the fields transfer data */
    for (npy_intp i = 0; i < field_count; ++i) {
        if (NPY_cast_info_copy(&newdata->fields[i].info, &d->fields[i].info) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)newdata);
            return NULL;
        }
        newdata->fields[i].src_offset = d->fields[i].src_offset;
        newdata->fields[i].dst_offset = d->fields[i].dst_offset;
        newdata->field_count++;
    }

    return (NpyAuxData *)newdata;
}


static int
_strided_to_strided_field_transfer(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _field_transfer_data *d = (_field_transfer_data *)auxdata;
    npy_intp i, field_count = d->field_count;
    const npy_intp blocksize = NPY_LOWLEVEL_BUFFER_BLOCKSIZE;

    /* Do the transfer a block at a time */
    for (;;) {
        if (N > blocksize) {
            for (i = 0; i < field_count; ++i) {
                _single_field_transfer field = d->fields[i];
                char *fargs[2] = {src + field.src_offset, dst + field.dst_offset};
                if (field.info.func(&field.info.context,
                        fargs, &blocksize, strides, field.info.auxdata) < 0) {
                    return -1;
                }
            }
            if (d->decref_src.func != NULL && d->decref_src.func(
                    NULL, d->decref_src.descr, src, blocksize, src_stride,
                    d->decref_src.auxdata) < 0) {
                return -1;
            }
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            for (i = 0; i < field_count; ++i) {
                _single_field_transfer field = d->fields[i];
                char *fargs[2] = {src + field.src_offset, dst + field.dst_offset};
                if (field.info.func(&field.info.context,
                        fargs, &N, strides, field.info.auxdata) < 0) {
                    return -1;
                }
            }
            if (d->decref_src.func != NULL && d->decref_src.func(
                    NULL, d->decref_src.descr, src, blocksize, src_stride,
                    d->decref_src.auxdata) < 0) {
                return -1;
            }
            return 0;
        }
    }
}

/*
 * Handles fields transfer.  To call this, at least one of the dtypes
 * must have fields. Does not take care of object<->structure conversion
 */
NPY_NO_EXPORT int
get_fields_transfer_function(int NPY_UNUSED(aligned),
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    PyObject *key, *tup, *title;
    PyArray_Descr *src_fld_dtype, *dst_fld_dtype;
    npy_int i;
    size_t structsize;
    Py_ssize_t field_count;
    int src_offset, dst_offset;
    _field_transfer_data *data;

    /*
     * There are three cases to take care of: 1. src is non-structured,
     * 2. dst is non-structured, or 3. both are structured.
     */

    /* 1. src is non-structured. Copy the src value to all the fields of dst */
    if (!PyDataType_HASFIELDS(src_dtype)) {
        field_count = PyTuple_GET_SIZE(PyDataType_NAMES(dst_dtype));

        /* Allocate the field-data structure and populate it */
        structsize = sizeof(_field_transfer_data) +
                        (field_count + 1) * sizeof(_single_field_transfer);
        data = PyMem_Malloc(structsize);
        if (data == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
        data->base.free = &_field_transfer_data_free;
        data->base.clone = &_field_transfer_data_clone;
        data->field_count = 0;
        NPY_traverse_info_init(&data->decref_src);

        *out_flags = PyArrayMethod_MINIMAL_FLAGS;
        for (i = 0; i < field_count; ++i) {
            key = PyTuple_GET_ITEM(PyDataType_NAMES(dst_dtype), i);
            tup = PyDict_GetItem(PyDataType_FIELDS(dst_dtype), key);
            if (!PyArg_ParseTuple(tup, "Oi|O", &dst_fld_dtype,
                                                    &dst_offset, &title)) {
                PyMem_Free(data);
                return NPY_FAIL;
            }
            NPY_ARRAYMETHOD_FLAGS field_flags;
            if (PyArray_GetDTypeTransferFunction(0,
                                    src_stride, dst_stride,
                                    src_dtype, dst_fld_dtype,
                                    0,
                                    &data->fields[i].info,
                                    &field_flags) != NPY_SUCCEED) {
                NPY_AUXDATA_FREE((NpyAuxData *)data);
                return NPY_FAIL;
            }
            *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, field_flags);
            data->fields[i].src_offset = 0;
            data->fields[i].dst_offset = dst_offset;
            data->field_count++;
        }

        /*
         * If references should be decrefd in src, add a clear function.
         */
        if (move_references && PyDataType_REFCHK(src_dtype)) {
            NPY_ARRAYMETHOD_FLAGS clear_flags;
            if (PyArray_GetClearFunction(
                    0, src_stride, src_dtype, &data->decref_src,
                    &clear_flags) < 0) {
                NPY_AUXDATA_FREE((NpyAuxData *)data);
                return NPY_FAIL;
            }
            *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, clear_flags);
        }

        *out_stransfer = &_strided_to_strided_field_transfer;
        *out_transferdata = (NpyAuxData *)data;

        return NPY_SUCCEED;
    }

    /* 2. dst is non-structured. Allow transfer from single-field src to dst */
    if (!PyDataType_HASFIELDS(dst_dtype)) {
        if (PyTuple_GET_SIZE(PyDataType_NAMES(src_dtype)) != 1) {
            PyErr_SetString(PyExc_ValueError,
                    "Can't cast from structure to non-structure, except if the "
                    "structure only has a single field.");
            return NPY_FAIL;
        }

        /* Allocate the field-data structure and populate it */
        structsize = sizeof(_field_transfer_data) +
                        1 * sizeof(_single_field_transfer);
        data = PyMem_Malloc(structsize);
        if (data == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
        data->base.free = &_field_transfer_data_free;
        data->base.clone = &_field_transfer_data_clone;
        NPY_traverse_info_init(&data->decref_src);

        key = PyTuple_GET_ITEM(PyDataType_NAMES(src_dtype), 0);
        tup = PyDict_GetItem(PyDataType_FIELDS(src_dtype), key);
        if (!PyArg_ParseTuple(tup, "Oi|O",
                              &src_fld_dtype, &src_offset, &title)) {
            PyMem_Free(data);
            return NPY_FAIL;
        }

        if (PyArray_GetDTypeTransferFunction(0,
                                             src_stride, dst_stride,
                                             src_fld_dtype, dst_dtype,
                                             move_references,
                                             &data->fields[0].info,
                                             out_flags) != NPY_SUCCEED) {
            PyMem_Free(data);
            return NPY_FAIL;
        }
        data->fields[0].src_offset = src_offset;
        data->fields[0].dst_offset = 0;
        data->field_count = 1;

        *out_stransfer = &_strided_to_strided_field_transfer;
        *out_transferdata = (NpyAuxData *)data;

        return NPY_SUCCEED;
    }

    /* 3. Otherwise both src and dst are structured arrays */
    field_count = PyTuple_GET_SIZE(PyDataType_NAMES(dst_dtype));

    /* Match up the fields to copy (field-by-field transfer) */
    if (PyTuple_GET_SIZE(PyDataType_NAMES(src_dtype)) != field_count) {
        PyErr_SetString(PyExc_ValueError, "structures must have the same size");
        return NPY_FAIL;
    }

    /* Allocate the field-data structure and populate it */
    structsize = sizeof(_field_transfer_data) +
                    field_count * sizeof(_single_field_transfer);
    data = PyMem_Malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->base.free = &_field_transfer_data_free;
    data->base.clone = &_field_transfer_data_clone;
    data->field_count = 0;
    NPY_traverse_info_init(&data->decref_src);

    *out_flags = PyArrayMethod_MINIMAL_FLAGS;
    /* set up the transfer function for each field */
    for (i = 0; i < field_count; ++i) {
        key = PyTuple_GET_ITEM(PyDataType_NAMES(dst_dtype), i);
        tup = PyDict_GetItem(PyDataType_FIELDS(dst_dtype), key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &dst_fld_dtype,
                                                &dst_offset, &title)) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
        key = PyTuple_GET_ITEM(PyDataType_NAMES(src_dtype), i);
        tup = PyDict_GetItem(PyDataType_FIELDS(src_dtype), key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &src_fld_dtype,
                                                &src_offset, &title)) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }

        NPY_ARRAYMETHOD_FLAGS field_flags;
        if (PyArray_GetDTypeTransferFunction(0,
                                             src_stride, dst_stride,
                                             src_fld_dtype, dst_fld_dtype,
                                             move_references,
                                             &data->fields[i].info,
                                             &field_flags) != NPY_SUCCEED) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
        *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, field_flags);
        data->fields[i].src_offset = src_offset;
        data->fields[i].dst_offset = dst_offset;
        data->field_count++;
    }

    *out_stransfer = &_strided_to_strided_field_transfer;
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}


/************************* MASKED TRANSFER WRAPPER *************************/

typedef struct {
    NpyAuxData base;
    /* The transfer function being wrapped (could likely be stored directly) */
    NPY_cast_info wrapped;
    /* The src decref function if necessary */
    NPY_traverse_info decref_src;
} _masked_wrapper_transfer_data;

/* transfer data free function */
static void
_masked_wrapper_transfer_data_free(NpyAuxData *data)
{
    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)data;
    NPY_cast_info_xfree(&d->wrapped);
    NPY_traverse_info_xfree(&d->decref_src);
    PyMem_Free(data);
}

/* transfer data copy function */
static NpyAuxData *
_masked_wrapper_transfer_data_clone(NpyAuxData *data)
{
    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)data;
    _masked_wrapper_transfer_data *newdata;

    /* Allocate the data and populate it */
    newdata = PyMem_Malloc(sizeof(*newdata));
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base = d->base;

    if (NPY_cast_info_copy(&newdata->wrapped, &d->wrapped) < 0) {
        PyMem_Free(newdata);
        return NULL;
    }
    if (d->decref_src.func != NULL) {
        if (NPY_traverse_info_copy(&newdata->decref_src, &d->decref_src) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_masked_wrapper_clear_function(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        npy_bool *mask, npy_intp mask_stride,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)auxdata;
    npy_intp subloopsize;

    while (N > 0) {
        /* Skip masked values, still calling decref for move_references */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 1);
        if (d->decref_src.func(NULL, d->decref_src.descr,
                src, subloopsize, src_stride, d->decref_src.auxdata) < 0) {
            return -1;
        }
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
        if (N <= 0) {
            break;
        }

        /* Process unmasked values */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 0);
        char *wrapped_args[2] = {src, dst};
        if (d->wrapped.func(&d->wrapped.context,
                wrapped_args, &subloopsize, strides, d->wrapped.auxdata) < 0) {
            return -1;
        }
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
    }
    return 0;
}

static int
_strided_masked_wrapper_transfer_function(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        npy_bool *mask, npy_intp mask_stride,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)auxdata;
    npy_intp subloopsize;

    while (N > 0) {
        /* Skip masked values */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 1);
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
        if (N <= 0) {
            break;
        }

        /* Process unmasked values */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 0);
        char *wrapped_args[2] = {src, dst};
        if (d->wrapped.func(&d->wrapped.context,
                wrapped_args, &subloopsize, strides, d->wrapped.auxdata) < 0) {
            return -1;
        }
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
    }
    return 0;
}



/* A no-op function (currently only used for cleanup purposes really) */
static int
_cast_no_op(
        PyArrayMethod_Context *NPY_UNUSED(context),
        char *const *NPY_UNUSED(args), const npy_intp *NPY_UNUSED(dimensions),
        const npy_intp *NPY_UNUSED(strides), NpyAuxData *NPY_UNUSED(auxdata))
{
    /* Do nothing */
    return 0;
}


/*
 * ********************* Generalized Multistep Cast ************************
 *
 * New general purpose multiple step cast function when resolve descriptors
 * implies that multiple cast steps are necessary.
 */

typedef struct {
    NpyAuxData base;
    /* Information for main cast */
    NPY_cast_info main;
    /* Information for input preparation cast */
    NPY_cast_info from;
    /* Information for output finalization cast */
    NPY_cast_info to;
    char *from_buffer;
    char *to_buffer;
} _multistep_castdata;


/* zero-padded data copy function */
static void
_multistep_cast_auxdata_free(NpyAuxData *auxdata)
{
    _multistep_castdata *data = (_multistep_castdata *)auxdata;
    NPY_cast_info_xfree(&data->main);
    if (data->from.func != NULL) {
        NPY_cast_info_xfree(&data->from);
    }
    if (data->to.func != NULL) {
        NPY_cast_info_xfree(&data->to);
    }
    PyMem_Free(data);
}


static NpyAuxData *
_multistep_cast_auxdata_clone(NpyAuxData *auxdata_old);


static NpyAuxData *
_multistep_cast_auxdata_clone_int(_multistep_castdata *castdata, int move_info)
{
    /* Round up the structure size to 16-byte boundary for the buffers */
    Py_ssize_t datasize = (sizeof(_multistep_castdata) + 15) & ~0xf;

    Py_ssize_t from_buffer_offset = datasize;
    if (castdata->from.func != NULL) {
        Py_ssize_t src_itemsize = castdata->main.context.descriptors[0]->elsize;
        datasize += NPY_LOWLEVEL_BUFFER_BLOCKSIZE * src_itemsize;
        datasize = (datasize + 15) & ~0xf;
    }
    Py_ssize_t to_buffer_offset = datasize;
    if (castdata->to.func != NULL) {
        Py_ssize_t dst_itemsize = castdata->main.context.descriptors[1]->elsize;
        datasize += NPY_LOWLEVEL_BUFFER_BLOCKSIZE * dst_itemsize;
    }

    char *char_data = PyMem_Malloc(datasize);
    if (char_data == NULL) {
        return NULL;
    }

    _multistep_castdata *newdata = (_multistep_castdata *)char_data;

    /* Fix up the basic information: */
    newdata->base.free = &_multistep_cast_auxdata_free;
    newdata->base.clone = &_multistep_cast_auxdata_clone;
    /* And buffer information: */
    newdata->from_buffer = char_data + from_buffer_offset;
    newdata->to_buffer = char_data + to_buffer_offset;

    /* Initialize funcs to NULL to signal no-cleanup in case of an error. */
    newdata->from.func = NULL;
    newdata->to.func = NULL;

    if (move_info) {
        NPY_cast_info_move(&newdata->main, &castdata->main);
    }
    else if (NPY_cast_info_copy(&newdata->main, &castdata->main) < 0) {
        goto fail;
    }

    if (castdata->from.func != NULL) {
        if (move_info) {
            NPY_cast_info_move(&newdata->from, &castdata->from);
        }
        else if (NPY_cast_info_copy(&newdata->from, &castdata->from) < 0) {
            goto fail;
        }

        if (PyDataType_FLAGCHK(newdata->main.descriptors[0], NPY_NEEDS_INIT)) {
            memset(newdata->from_buffer, 0, to_buffer_offset - from_buffer_offset);
        }
    }
    if (castdata->to.func != NULL) {
        if (move_info) {
            NPY_cast_info_move(&newdata->to, &castdata->to);
        }
        else if (NPY_cast_info_copy(&newdata->to, &castdata->to) < 0) {
            goto fail;
        }

        if (PyDataType_FLAGCHK(newdata->main.descriptors[1], NPY_NEEDS_INIT)) {
            memset(newdata->to_buffer, 0, datasize - to_buffer_offset);
        }
    }

    return (NpyAuxData *)newdata;

  fail:
    NPY_AUXDATA_FREE((NpyAuxData *)newdata);
    return NULL;
}


static NpyAuxData *
_multistep_cast_auxdata_clone(NpyAuxData *auxdata_old)
{
    return _multistep_cast_auxdata_clone_int(
            (_multistep_castdata *)auxdata_old, 0);
}


static int
_strided_to_strided_multistep_cast(
        /* The context is always stored explicitly in auxdata */
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    _multistep_castdata *castdata = (_multistep_castdata *)auxdata;
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    char *main_src, *main_dst;
    npy_intp main_src_stride, main_dst_stride;

    npy_intp block_size = NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
    while (N > 0) {
        if (block_size > N) {
            block_size = N;
        }

        if (castdata->from.func != NULL) {
            npy_intp out_stride = castdata->from.descriptors[1]->elsize;
            char *const data[2] = {src, castdata->from_buffer};
            npy_intp strides[2] = {src_stride, out_stride};
            if (castdata->from.func(&castdata->from.context,
                    data, &block_size,
                    strides,
                    castdata->from.auxdata) != 0) {
                /* TODO: Internal buffer may require cleanup on error. */
                return -1;
            }
            main_src = castdata->from_buffer;
            main_src_stride = out_stride;
        }
        else {
            main_src = src;
            main_src_stride = src_stride;
        }

        if (castdata->to.func != NULL) {
            main_dst = castdata->to_buffer;
            main_dst_stride = castdata->main.descriptors[1]->elsize;
        }
        else {
            main_dst = dst;
            main_dst_stride = dst_stride;
        }

        char *const data[2] = {main_src, main_dst};
        npy_intp strides[2] = {main_src_stride, main_dst_stride};
        if (castdata->main.func(&castdata->main.context,
                data, &block_size,
                strides,
                castdata->main.auxdata) != 0) {
            /* TODO: Internal buffer may require cleanup on error. */
            return -1;
        }

        if (castdata->to.func != NULL) {
            char *const data[2] = {main_dst, dst};
            npy_intp strides[2] = {main_dst_stride, dst_stride};
            if (castdata->to.func(&castdata->to.context,
                    data, &block_size,
                    strides,
                    castdata->to.auxdata) != 0) {
                return -1;
            }
        }

        N -= block_size;
        src += block_size * src_stride;
        dst += block_size * dst_stride;
    }
    return 0;
}


/*
 * Initialize most of a cast-info structure, this step does not fetch the
 * transferfunction and transferdata.
 */
static inline int
init_cast_info(
        NPY_cast_info *cast_info, NPY_CASTING *casting, npy_intp *view_offset,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype, int main_step)
{
    PyObject *meth = PyArray_GetCastingImpl(
            NPY_DTYPE(src_dtype), NPY_DTYPE(dst_dtype));
    if (meth == NULL) {
        return -1;
    }
    if (meth == Py_None) {
        Py_DECREF(Py_None);
        PyErr_Format(PyExc_TypeError,
                "Cannot cast data from %S to %S.", src_dtype, dst_dtype);
        return -1;
    }
    /* Initialize the context and related data */
    NPY_cast_info_init(cast_info);
    cast_info->auxdata = NULL;

    cast_info->context.caller = NULL;
    cast_info->context.method = (PyArrayMethodObject *)meth;

    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(src_dtype), NPY_DTYPE(dst_dtype)};
    PyArray_Descr *in_descr[2] = {src_dtype, dst_dtype};

    *casting = cast_info->context.method->resolve_descriptors(
            cast_info->context.method, dtypes,
            in_descr, cast_info->descriptors, view_offset);
    if (NPY_UNLIKELY(*casting < 0)) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError,
                    "Cannot cast array data from %R to %R.", src_dtype, dst_dtype);
        }
        Py_DECREF(meth);
        return -1;
    }
    assert(PyArray_DescrCheck(cast_info->descriptors[0]));
    assert(PyArray_DescrCheck(cast_info->descriptors[1]));

    if (!main_step && NPY_UNLIKELY(src_dtype != cast_info->descriptors[0] ||
                                   dst_dtype != cast_info->descriptors[1])) {
        /*
         * We currently do not resolve recursively, but require a non
         * main cast (within the same DType) to be done in a single step.
         * This could be expanded at some point if the need arises.
         */
        PyErr_Format(PyExc_RuntimeError,
                "Required internal cast from %R to %R was not done in a single "
                "step (a secondary cast must currently be between instances of "
                "the same DType class and such a cast must currently return "
                "the input descriptors unmodified).",
                src_dtype, dst_dtype);
        NPY_cast_info_xfree(cast_info);
        return -1;
    }

    return 0;
}


/*
 * When there is a failure in ArrayMethod.get_loop(...) we still have
 * to clean up references, but assume that `auxdata` and `func`
 * have undefined values.
 * NOTE: This should possibly be moved, but is only necessary here
 */
static void
_clear_cast_info_after_get_loop_failure(NPY_cast_info *cast_info)
{
    /* As public API we could choose to clear auxdata != NULL */
    assert(cast_info->auxdata == NULL);
    /* Set func to be non-null so that `NPY_cats_info_xfree` does not skip */
    cast_info->func = &_cast_no_op;
    NPY_cast_info_xfree(cast_info);
}


/*
 * Helper for PyArray_GetDTypeTransferFunction, which fetches a single
 * transfer function from the each casting implementation (ArrayMethod)
 * May set the transfer function to NULL when the cast can be achieved using
 * a view.
 * TODO: Expand the view functionality for general offsets, not just 0:
 *       Partial casts could be skipped also for `view_offset != 0`.
 *
 * The `out_needs_api` flag must be initialized.
 *
 * NOTE: In theory casting errors here could be slightly misleading in case
 *       of a multi-step casting scenario. It should be possible to improve
 *       this in the future.
 *
 * Note about `move_references`: Move references means stealing of
 * references.  It is useful to clear buffers immediately. No matter the
 * input all copies from a buffer must use `move_references`. Move references
 * is thus used:
 *   * For the added initial "from" cast if it was passed in
 *   * Always in the main step if a "from" cast is made (it casts from a buffer)
 *   * Always for the "to" cast, as it always cast from a buffer to the output.
 *
 * Returns -1 on failure, 0 on success
 */
static int
define_cast_for_descrs(
        int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        NPY_cast_info *cast_info, NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    assert(dst_dtype != NULL);  /* Was previously used for decref */

    /* Storage for all cast info in case multi-step casting is necessary */
    _multistep_castdata castdata;
    /* Initialize funcs to NULL to simplify cleanup on error. */
    castdata.main.func = NULL;
    castdata.to.func = NULL;
    castdata.from.func = NULL;
    /* `view_offset` passed to `init_cast_info` but unused for the main cast */
    npy_intp view_offset = NPY_MIN_INTP;
    NPY_CASTING casting = -1;
    *out_flags = PyArrayMethod_MINIMAL_FLAGS;

    if (init_cast_info(
            cast_info, &casting, &view_offset, src_dtype, dst_dtype, 1) < 0) {
        return -1;
    }

    /*
     * Both input and output must be wrapped in case they may be unaligned
     * and the method does not support unaligned data.
     * NOTE: It is probable that most/all legacy loops actually do support
     *       unaligned output, we could move the wrapping there if we wanted
     *       to. It probably isn't speed relevant though and they should be
     *       deleted in any case.
     */
    int must_wrap = (!aligned &&
        (cast_info->context.method->flags & NPY_METH_SUPPORTS_UNALIGNED) == 0);

    /*
     * Wrap the input with an additional cast if necessary.
     */
    if (NPY_UNLIKELY(src_dtype != cast_info->descriptors[0] || must_wrap)) {
        NPY_CASTING from_casting = -1;
        npy_intp from_view_offset = NPY_MIN_INTP;
        /* Cast function may not support the input, wrap if necessary */
        if (init_cast_info(
                &castdata.from, &from_casting, &from_view_offset,
                src_dtype, cast_info->descriptors[0], 0) < 0) {
            goto fail;
        }
        casting = PyArray_MinCastSafety(casting, from_casting);

        /* Prepare the actual cast (if necessary): */
        if (from_view_offset == 0 && !must_wrap) {
            /* This step is not necessary and can be skipped */
            castdata.from.func = &_cast_no_op;  /* avoid NULL */
            NPY_cast_info_xfree(&castdata.from);
        }
        else {
            /* Fetch the cast function and set up */
            PyArrayMethod_Context *context = &castdata.from.context;
            npy_intp strides[2] = {src_stride, cast_info->descriptors[0]->elsize};
            NPY_ARRAYMETHOD_FLAGS flags;
            if (context->method->get_strided_loop(
                    context, aligned, move_references, strides,
                    &castdata.from.func, &castdata.from.auxdata, &flags) < 0) {
                _clear_cast_info_after_get_loop_failure(&castdata.from);
                goto fail;
            }
            assert(castdata.from.func != NULL);

            *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, flags);
            /* The main cast now uses a buffered input: */
            src_stride = strides[1];
            move_references = 1;  /* main cast has to clear the buffer */
        }
    }
    /*
     * Wrap the output with an additional cast if necessary.
     */
    if (NPY_UNLIKELY(dst_dtype != cast_info->descriptors[1] || must_wrap)) {
        NPY_CASTING to_casting = -1;
        npy_intp to_view_offset = NPY_MIN_INTP;
        /* Cast function may not support the output, wrap if necessary */
        if (init_cast_info(
                &castdata.to, &to_casting, &to_view_offset,
                cast_info->descriptors[1], dst_dtype,  0) < 0) {
            goto fail;
        }
        casting = PyArray_MinCastSafety(casting, to_casting);

        /* Prepare the actual cast (if necessary): */
        if (to_view_offset == 0 && !must_wrap) {
            /* This step is not necessary and can be skipped. */
            castdata.to.func = &_cast_no_op;  /* avoid NULL */
            NPY_cast_info_xfree(&castdata.to);
        }
        else {
            /* Fetch the cast function and set up */
            PyArrayMethod_Context *context = &castdata.to.context;
            npy_intp strides[2] = {cast_info->descriptors[1]->elsize, dst_stride};
            NPY_ARRAYMETHOD_FLAGS flags;
            if (context->method->get_strided_loop(
                    context, aligned, 1 /* clear buffer */, strides,
                    &castdata.to.func, &castdata.to.auxdata, &flags) < 0) {
                _clear_cast_info_after_get_loop_failure(&castdata.to);
                goto fail;
            }
            assert(castdata.to.func != NULL);

            *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, flags);
            /* The main cast now uses a buffered input: */
            dst_stride = strides[0];
            if (castdata.from.func != NULL) {
                /* Both input and output are wrapped, now always aligned */
                aligned = 1;
            }
        }
    }

    /* Fetch the main cast function (with updated values) */
    PyArrayMethod_Context *context = &cast_info->context;
    npy_intp strides[2] = {src_stride, dst_stride};
    NPY_ARRAYMETHOD_FLAGS flags;
    if (context->method->get_strided_loop(
            context, aligned, move_references, strides,
            &cast_info->func, &cast_info->auxdata, &flags) < 0) {
        _clear_cast_info_after_get_loop_failure(cast_info);
        goto fail;
    }

    *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, flags);

    if (castdata.from.func == NULL && castdata.to.func == NULL) {
        /* Most of the time, there will be only one step required. */
        return 0;
    }
    /* The full cast passed in is only the "main" step, copy cast_info there */
    NPY_cast_info_move(&castdata.main, cast_info);
    Py_INCREF(src_dtype);
    cast_info->descriptors[0] = src_dtype;
    Py_INCREF(dst_dtype);
    cast_info->descriptors[1] = dst_dtype;
    cast_info->context.method = NULL;

    cast_info->func = &_strided_to_strided_multistep_cast;
    cast_info->auxdata = _multistep_cast_auxdata_clone_int(&castdata, 1);
    if (cast_info->auxdata == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    return 0;

  fail:
    NPY_cast_info_xfree(&castdata.main);
    NPY_cast_info_xfree(&castdata.from);
    NPY_cast_info_xfree(&castdata.to);
    return -1;
}


NPY_NO_EXPORT int
PyArray_GetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            NPY_cast_info *cast_info,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    if (define_cast_for_descrs(aligned,
            src_stride, dst_stride,
            src_dtype, dst_dtype, move_references,
            cast_info, out_flags) < 0) {
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}


/*
 * Internal wrapping of casts that have to be performed in a "single"
 * function (i.e. not by the generic multi-step-cast), but rely on it
 * internally. There are only two occasions where this is used:
 *
 * 1. Void advertises that it handles unaligned casts, but has to wrap the
 *    legacy cast which (probably) does not.
 * 2. Datetime to unicode casts are implemented via bytes "U" vs. "S". If
 *    we relax the chaining rules to allow "recursive" cast chaining where
 *    `resolve_descriptors` can return a descriptor with a different type,
 *    this would become unnecessary.
 *  3. Time <-> Time casts, which currently must support byte swapping, but
 *     have a non-trivial inner-loop (due to units) which does not support
 *     it.
 *
 * When wrapping is performed (guaranteed for `aligned == 0` and if the
 * wrapped dtype is not identical to the input dtype), the wrapped transfer
 * function can assume a contiguous input.
 * Otherwise use `must_wrap` to ensure that wrapping occurs, which guarantees
 * a contiguous, aligned, call of the wrapped function.
 */
NPY_NO_EXPORT int
wrap_aligned_transferfunction(
        int aligned, int must_wrap,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArray_Descr *src_wrapped_dtype, PyArray_Descr *dst_wrapped_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata, int *out_needs_api)
{
    must_wrap = must_wrap | !aligned;

    _multistep_castdata castdata;
    NPY_cast_info_init(&castdata.main);
    NPY_cast_info_init(&castdata.from);
    NPY_cast_info_init(&castdata.to);

    /* Finalize the existing cast information: */
    castdata.main.func = *out_stransfer;
    *out_stransfer = NULL;
    castdata.main.auxdata = *out_transferdata;
    *out_transferdata = NULL;
    castdata.main.context.method = NULL;
    /* These are always legacy casts that only support native-byte-order: */
    Py_INCREF(src_wrapped_dtype);
    castdata.main.descriptors[0] = src_wrapped_dtype;
    if (castdata.main.descriptors[0] == NULL) {
        castdata.main.descriptors[1] = NULL;
        goto fail;
    }
    Py_INCREF(dst_wrapped_dtype);
    castdata.main.descriptors[1] = dst_wrapped_dtype;
    if (castdata.main.descriptors[1] == NULL) {
        goto fail;
    }

    /*
     * Similar to the normal multi-step cast, but we always have to wrap
     * it all up, but we can simply do this via a "recursive" call.
     * TODO: This is slightly wasteful, since it unnecessarily checks casting,
     *       but this whole function is about corner cases, which should rather
     *       have an explicit implementation instead if we want performance.
     */
    if (must_wrap || src_wrapped_dtype != src_dtype) {
        NPY_ARRAYMETHOD_FLAGS flags;
        if (PyArray_GetDTypeTransferFunction(aligned,
                src_stride, castdata.main.descriptors[0]->elsize,
                src_dtype, castdata.main.descriptors[0], 0,
                &castdata.from, &flags) != NPY_SUCCEED) {
            goto fail;
        }
        if (flags & NPY_METH_REQUIRES_PYAPI) {
            *out_needs_api = 1;
        }
    }
    if (must_wrap || dst_wrapped_dtype != dst_dtype) {
        NPY_ARRAYMETHOD_FLAGS flags;
        if (PyArray_GetDTypeTransferFunction(aligned,
                castdata.main.descriptors[1]->elsize, dst_stride,
                castdata.main.descriptors[1], dst_dtype,
                1,  /* clear buffer if it includes references */
                &castdata.to, &flags) != NPY_SUCCEED) {
            goto fail;
        }
        if (flags & NPY_METH_REQUIRES_PYAPI) {
            *out_needs_api = 1;
        }
    }

    *out_transferdata = _multistep_cast_auxdata_clone_int(&castdata, 1);
    if (*out_transferdata == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    *out_stransfer = &_strided_to_strided_multistep_cast;
    return 0;

  fail:
    NPY_cast_info_xfree(&castdata.main);
    NPY_cast_info_xfree(&castdata.from);
    NPY_cast_info_xfree(&castdata.to);

    return -1;
}


/*
 * This function wraps the legacy casts stored on the PyDataType_GetArrFuncs(`dtype)->cast`
 * or registered with `PyArray_RegisterCastFunc`.
 * For casts between two dtypes with the same type (within DType casts)
 * it also wraps the `copyswapn` function.
 *
 * This function is called from `ArrayMethod.get_loop()` when a specialized
 * cast function is missing.
 *
 * In general, the legacy cast functions do not support unaligned access,
 * so an ArrayMethod using this must signal that.  In a few places we do
 * signal support for unaligned access (or byte swapping).
 * In this case `allow_wrapped=1` will wrap it into an additional multi-step
 * cast as necessary.
 */
NPY_NO_EXPORT int
get_wrapped_legacy_cast_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api, int allow_wrapped)
{
    /* Note: We ignore `needs_wrap`; needs-wrap is handled by another cast */
    int needs_wrap = 0;

    if (src_dtype->type_num == dst_dtype->type_num) {
        /*
         * This is a cast within the same dtype. For legacy user-dtypes,
         * it is always valid to handle this using the copy swap function.
         */
        return wrap_copy_swap_function(src_dtype,
                PyDataType_ISNOTSWAPPED(src_dtype) !=
                PyDataType_ISNOTSWAPPED(dst_dtype),
                out_stransfer, out_transferdata);
    }

    if (get_legacy_dtype_cast_function(
            aligned,
            src_stride, dst_stride,
            src_dtype, dst_dtype,
            move_references,
            out_stransfer,
            out_transferdata,
            out_needs_api,
            &needs_wrap) != NPY_SUCCEED) {
        return -1;
    }
    if (!needs_wrap) {
        return 0;
    }
    if (NPY_UNLIKELY(!allow_wrapped)) {
        /*
         * Legacy casts do not support unaligned which requires wrapping.
         * However, normally we ensure that wrapping happens before calling
         * this function, so this path should never happen.
         */
        PyErr_Format(PyExc_RuntimeError,
                "Internal NumPy error, casting %S to %S required wrapping, "
                "probably it incorrectly flagged support for unaligned data. "
                "(aligned passed to discovery is %d)",
                src_dtype, dst_dtype, aligned);
        goto fail;
    }

    /*
     * If we are here, use the legacy code to wrap the above cast (which
     * does not support unaligned data) into copyswapn.
     */
    PyArray_Descr *src_wrapped_dtype = NPY_DT_CALL_ensure_canonical(src_dtype);
    if (src_wrapped_dtype == NULL) {
        goto fail;
    }
    PyArray_Descr *dst_wrapped_dtype = NPY_DT_CALL_ensure_canonical(dst_dtype);
    if (dst_wrapped_dtype == NULL) {
        goto fail;
    }
    int res = wrap_aligned_transferfunction(
            aligned, 1,  /* We assume wrapped is contiguous here */
            src_stride, dst_stride,
            src_dtype, dst_dtype,
            src_wrapped_dtype, dst_wrapped_dtype,
            out_stransfer, out_transferdata, out_needs_api);
    Py_DECREF(src_wrapped_dtype);
    Py_DECREF(dst_wrapped_dtype);
    return res;

  fail:
    NPY_AUXDATA_FREE(*out_transferdata);
    *out_transferdata = NULL;
    return -1;
}


NPY_NO_EXPORT int
PyArray_GetMaskedDTypeTransferFunction(int aligned,
                            npy_intp src_stride,
                            npy_intp dst_stride,
                            npy_intp mask_stride,
                            PyArray_Descr *src_dtype,
                            PyArray_Descr *dst_dtype,
                            PyArray_Descr *mask_dtype,
                            int move_references,
                            NPY_cast_info *cast_info,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    NPY_cast_info_init(cast_info);

    if (mask_dtype->type_num != NPY_BOOL &&
                            mask_dtype->type_num != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError,
                "Only bool and uint8 masks are supported.");
        return NPY_FAIL;
    }

    /* Create the wrapper function's auxdata */
    _masked_wrapper_transfer_data *data;
    data = PyMem_Malloc(sizeof(_masked_wrapper_transfer_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->base.free = &_masked_wrapper_transfer_data_free;
    data->base.clone = &_masked_wrapper_transfer_data_clone;

    /* Fall back to wrapping a non-masked transfer function */
    assert(dst_dtype != NULL);
    if (PyArray_GetDTypeTransferFunction(aligned,
                                src_stride, dst_stride,
                                src_dtype, dst_dtype,
                                move_references,
                                &data->wrapped,
                                out_flags) != NPY_SUCCEED) {
        PyMem_Free(data);
        return NPY_FAIL;
    }

    /* If the src object will need a DECREF, get a function to handle that */
    if (move_references && PyDataType_REFCHK(src_dtype)) {
        NPY_ARRAYMETHOD_FLAGS clear_flags;
        if (PyArray_GetClearFunction(
                aligned, src_stride, src_dtype,
                &data->decref_src, &clear_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
        *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, clear_flags);
        cast_info->func = (PyArrayMethod_StridedLoop *)
                &_strided_masked_wrapper_clear_function;
    }
    else {
        NPY_traverse_info_init(&data->decref_src);
        cast_info->func = (PyArrayMethod_StridedLoop *)
                &_strided_masked_wrapper_transfer_function;
    }
    cast_info->auxdata = (NpyAuxData *)data;
    /* The context is almost unused, but clear it for cleanup. */
    Py_INCREF(src_dtype);
    cast_info->descriptors[0] = src_dtype;
    Py_INCREF(dst_dtype);
    cast_info->descriptors[1] = dst_dtype;
    cast_info->context.caller = NULL;
    cast_info->context.method = NULL;

    return NPY_SUCCEED;
}

NPY_NO_EXPORT int
PyArray_CastRawArrays(npy_intp count,
                      char *src, char *dst,
                      npy_intp src_stride, npy_intp dst_stride,
                      PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                      int move_references)
{
    int aligned;

    /* Make sure the copy is reasonable */
    if (dst_stride == 0 && count > 1) {
        PyErr_SetString(PyExc_ValueError,
                    "NumPy CastRawArrays cannot do a reduction");
        return NPY_FAIL;
    }
    else if (count == 0) {
        return NPY_SUCCEED;
    }

    /* Check data alignment, both uint and true */
    aligned = raw_array_is_aligned(1, &count, dst, &dst_stride,
                                   npy_uint_alignment(dst_dtype->elsize)) &&
              raw_array_is_aligned(1, &count, dst, &dst_stride,
                                   dst_dtype->alignment) &&
              raw_array_is_aligned(1, &count, src, &src_stride,
                                   npy_uint_alignment(src_dtype->elsize)) &&
              raw_array_is_aligned(1, &count, src, &src_stride,
                                   src_dtype->alignment);

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetDTypeTransferFunction(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        &cast_info,
                        &flags) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char*)&cast_info);
    }

    /* Cast */
    char *args[2] = {src, dst};
    npy_intp strides[2] = {src_stride, dst_stride};
    cast_info.func(&cast_info.context, args, &count, strides, cast_info.auxdata);

    /* Cleanup */
    NPY_cast_info_xfree(&cast_info);

    if (flags & NPY_METH_REQUIRES_PYAPI && PyErr_Occurred()) {
        return NPY_FAIL;
    }
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier(*args);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return NPY_FAIL;
        }
    }
    return NPY_SUCCEED;
}

/*
 * Prepares shape and strides for a simple raw array iteration.
 * This sorts the strides into FORTRAN order, reverses any negative
 * strides, then coalesces axes where possible. The results are
 * filled in the output parameters.
 *
 * This is intended for simple, lightweight iteration over arrays
 * where no buffering of any kind is needed, and the array may
 * not be stored as a PyArrayObject.
 *
 * The arrays shape, out_shape, strides, and out_strides must all
 * point to different data.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareOneRawArrayIter(int ndim, npy_intp const *shape,
                            char *data, npy_intp const *strides,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_data, npy_intp *out_strides)
{
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_data = data;
        out_shape[0] = 1;
        out_strides[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        npy_intp stride_entry = strides[0], shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride */
        if (stride_entry >= 0) {
            *out_data = data;
            out_strides[0] = stride_entry;
        }
        else {
            *out_data = data + stride_entry * (shape_entry - 1);
            out_strides[0] = -stride_entry;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    PyArray_CreateSortedStridePerm(ndim, strides, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_strides[i] = strides[iperm];
    }

    /* Reverse any negative strides */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride_entry = out_strides[i], shape_entry = out_shape[i];

        if (stride_entry < 0) {
            data += stride_entry * (shape_entry - 1);
            out_strides[i] = -stride_entry;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_data = data;
            out_shape[0] = 0;
            out_strides[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_strides[i] * out_shape[i] == out_strides[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
    }
    ndim = i+1;

#if 0
    /* DEBUG */
    {
        printf("raw iter ndim %d\n", ndim);
        printf("shape: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_shape[i]);
        }
        printf("\n");
        printf("strides: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_strides[i]);
        }
        printf("\n");
    }
#endif

    *out_data = data;
    *out_ndim = ndim;
    return 0;
}

/*
 * The same as PyArray_PrepareOneRawArrayIter, but for two
 * operands instead of one. Any broadcasting of the two operands
 * should have already been done before calling this function,
 * as the ndim and shape is only specified once for both operands.
 *
 * Only the strides of the first operand are used to reorder
 * the dimensions, no attempt to consider all the strides together
 * is made, as is done in the NpyIter object.
 *
 * You can use this together with NPY_RAW_ITER_START and
 * NPY_RAW_ITER_TWO_NEXT to handle the looping boilerplate of everything
 * but the innermost loop (which is for idim == 0).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareTwoRawArrayIter(int ndim, npy_intp const *shape,
                            char *dataA, npy_intp const *stridesA,
                            char *dataB, npy_intp const *stridesB,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB)
{
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_dataA = dataA;
        *out_dataB = dataB;
        out_shape[0] = 1;
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        npy_intp stride_entryA = stridesA[0], stride_entryB = stridesB[0];
        npy_intp shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride for the first operand */
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
        }
        else {
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    PyArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
    }

    /* Reverse any negative strides of operand A */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride_entryA = out_stridesA[i];
        npy_intp stride_entryB = out_stridesB[i];
        npy_intp shape_entry = out_shape[i];

        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_shape[0] = 0;
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                    out_stridesB[i] * out_shape[i] == out_stridesB[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
    }
    ndim = i+1;

    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_ndim = ndim;
    return 0;
}

/*
 * The same as PyArray_PrepareOneRawArrayIter, but for three
 * operands instead of one. Any broadcasting of the three operands
 * should have already been done before calling this function,
 * as the ndim and shape is only specified once for all operands.
 *
 * Only the strides of the first operand are used to reorder
 * the dimensions, no attempt to consider all the strides together
 * is made, as is done in the NpyIter object.
 *
 * You can use this together with NPY_RAW_ITER_START and
 * NPY_RAW_ITER_THREE_NEXT to handle the looping boilerplate of everything
 * but the innermost loop (which is for idim == 0).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareThreeRawArrayIter(int ndim, npy_intp const *shape,
                            char *dataA, npy_intp const *stridesA,
                            char *dataB, npy_intp const *stridesB,
                            char *dataC, npy_intp const *stridesC,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB,
                            char **out_dataC, npy_intp *out_stridesC)
{
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_dataA = dataA;
        *out_dataB = dataB;
        *out_dataC = dataC;
        out_shape[0] = 1;
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        out_stridesC[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        npy_intp stride_entryA = stridesA[0];
        npy_intp stride_entryB = stridesB[0];
        npy_intp stride_entryC = stridesC[0];
        npy_intp shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride for the first operand */
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            *out_dataC = dataC;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
            out_stridesC[0] = stride_entryC;
        }
        else {
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            *out_dataC = dataC + stride_entryC * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
            out_stridesC[0] = -stride_entryC;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    PyArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
        out_stridesC[i] = stridesC[iperm];
    }

    /* Reverse any negative strides of operand A */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride_entryA = out_stridesA[i];
        npy_intp stride_entryB = out_stridesB[i];
        npy_intp stride_entryC = out_stridesC[i];
        npy_intp shape_entry = out_shape[i];

        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            dataC += stride_entryC * (shape_entry - 1);
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
            out_stridesC[i] = -stride_entryC;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_dataA = dataA;
            *out_dataB = dataB;
            *out_dataC = dataC;
            out_shape[0] = 0;
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            out_stridesC[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
            out_stridesC[i] = out_stridesC[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                    out_stridesB[i] * out_shape[i] == out_stridesB[j] &&
                    out_stridesC[i] * out_shape[i] == out_stridesC[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
            out_stridesC[i] = out_stridesC[j];
        }
    }
    ndim = i+1;

    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_dataC = dataC;
    *out_ndim = ndim;
    return 0;
}
