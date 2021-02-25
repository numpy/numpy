/*
 * This file contains low-level loops for data type transfers.
 * In particular the function PyArray_GetDTypeTransferFunction is
 * implemented here.
 *
 * Copyright (c) 2010 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.

 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_pycompat.h"

#include "convert_datatype.h"
#include "ctors.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "descriptor.h"
#include "array_assign.h"

#include "shape.h"
#include "dtype_transfer.h"
#include "alloc.h"
#include "dtypemeta.h"
#include "array_method.h"
#include "array_coercion.h"

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

/*
 * Returns a transfer function which DECREFs any references in src_type.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
static int
get_decsrcref_transfer_function(int aligned,
                            npy_intp src_stride,
                            PyArray_Descr *src_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api);


/*************************** COPY REFERENCES *******************************/

/* Moves references from src to dst */
NPY_NO_EXPORT int
_strided_to_strided_move_references(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
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
_strided_to_strided_copy_references(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
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
    PyArray_StridedUnaryOp *decref_func;
    NpyAuxData *decref_data;
} _any_to_object_auxdata;


static void
_any_to_object_auxdata_free(NpyAuxData *auxdata)
{
    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;

    Py_DECREF(data->arr_fields.descr);
    NPY_AUXDATA_FREE(data->decref_data);
    PyMem_Free(data);
}


static NpyAuxData *
_any_to_object_auxdata_clone(NpyAuxData *auxdata)
{
    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;

    _any_to_object_auxdata *res = PyMem_Malloc(sizeof(_any_to_object_auxdata));

    memcpy(res, data, sizeof(*data));
    Py_INCREF(res->arr_fields.descr);
    if (res->decref_data != NULL) {
        res->decref_data = NPY_AUXDATA_CLONE(data->decref_data);
        if (res->decref_data == NULL) {
            NPY_AUXDATA_FREE((NpyAuxData *) res);
            return NULL;
        }
    }
    return (NpyAuxData *)res;
}


static int
_strided_to_strided_any_to_object(char *dst, npy_intp dst_stride,
        char *src, npy_intp src_stride,
        npy_intp N, npy_intp src_itemsize,
        NpyAuxData *auxdata)
{
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
    if (data->decref_func != NULL) {
        /* If necessar, clear the input buffer (`move_references`) */
        if (data->decref_func(NULL, 0, orig_src, src_stride, N,
                              src_itemsize, data->decref_data) < 0) {
            return -1;
        }
    }
    return 0;
}


NPY_NO_EXPORT int
any_to_object_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{

    *flags = NPY_METH_REQUIRES_PYAPI;  /* No need for floating point errors */

    *out_loop = _strided_to_strided_any_to_object;
    *out_transferdata = PyMem_Malloc(sizeof(_any_to_object_auxdata));
    if (*out_transferdata == NULL) {
        return -1;
    }
    _any_to_object_auxdata *data = (_any_to_object_auxdata *)*out_transferdata;
    data->base.free = &_any_to_object_auxdata_free;
    data->base.clone = &_any_to_object_auxdata_clone;
    data->arr_fields.base = NULL;
    data->arr_fields.descr = context->descriptors[0];
    Py_INCREF(data->arr_fields.descr);
    data->arr_fields.flags = aligned ? NPY_ARRAY_ALIGNED : 0;
    data->arr_fields.nd = 0;

    data->getitem = context->descriptors[0]->f->getitem;
    data->decref_func = NULL;
    data->decref_data = NULL;

    if (move_references && PyDataType_REFCHK(context->descriptors[0])) {
        int needs_api;
        if (get_decsrcref_transfer_function(
                aligned, strides[0], context->descriptors[0],
                &data->decref_func, &data->decref_data,
                &needs_api) == NPY_FAIL)  {
            NPY_AUXDATA_FREE(*out_transferdata);
            *out_transferdata = NULL;
            return -1;
        }
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
        char *dst, npy_intp dst_stride,
        char *src, npy_intp src_stride,
        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
        NpyAuxData *auxdata)
{
    _object_to_any_auxdata *data = (_object_to_any_auxdata *)auxdata;

    PyObject *src_ref;

    while (N > 0) {
        memcpy(&src_ref, src, sizeof(src_ref));
        if (PyArray_Pack(data->descr, dst, src_ref) < 0) {
            return -1;
        }

        if (data->move_references) {
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
        npy_intp *NPY_UNUSED(strides),
        PyArray_StridedUnaryOp **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_REQUIRES_PYAPI;

    /*
     * TODO: After passing `context`, auxdata can be statically allocated
     *       since `descriptor` is always passed.
     */
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

/* Does a zero-padded copy */
typedef struct {
    NpyAuxData base;
    npy_intp dst_itemsize;
} _strided_zero_pad_data;

/* zero-padded data copy function */
static NpyAuxData *_strided_zero_pad_data_clone(NpyAuxData *data)
{
    _strided_zero_pad_data *newdata =
            (_strided_zero_pad_data *)PyArray_malloc(
                                    sizeof(_strided_zero_pad_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_strided_zero_pad_data));

    return (NpyAuxData *)newdata;
}

/*
 * Does a strided to strided zero-padded copy for the case where
 * dst_itemsize > src_itemsize
 */
static int
_strided_to_strided_zero_pad_copy(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_zero_pad_data *d = (_strided_zero_pad_data *)data;
    npy_intp dst_itemsize = d->dst_itemsize;
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
_strided_to_strided_truncate_copy(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_zero_pad_data *d = (_strided_zero_pad_data *)data;
    npy_intp dst_itemsize = d->dst_itemsize;

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
_strided_to_strided_unicode_copyswap(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_zero_pad_data *d = (_strided_zero_pad_data *)data;
    npy_intp dst_itemsize = d->dst_itemsize;
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
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata)
{
    if ((src_itemsize == dst_itemsize) && !unicode_swap) {
        *out_stransfer = PyArray_GetStridedCopyFn(aligned, src_stride,
                                dst_stride, src_itemsize);
        *out_transferdata = NULL;
        return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
    }
    else {
        _strided_zero_pad_data *d = PyArray_malloc(
                                        sizeof(_strided_zero_pad_data));
        if (d == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
        d->dst_itemsize = dst_itemsize;
        d->base.free = (NpyAuxData_FreeFunc *)&PyArray_free;
        d->base.clone = &_strided_zero_pad_data_clone;

        if (unicode_swap) {
            *out_stransfer = &_strided_to_strided_unicode_copyswap;
        }
        else if (src_itemsize < dst_itemsize) {
            *out_stransfer = &_strided_to_strided_zero_pad_copy;
        }
        else {
            *out_stransfer = &_strided_to_strided_truncate_copy;
        }

        *out_transferdata = (NpyAuxData *)d;
        return NPY_SUCCEED;
    }
}

/***************** WRAP ALIGNED CONTIGUOUS TRANSFER FUNCTION **************/

/* Wraps a transfer function + data in alignment code */
typedef struct {
    NpyAuxData base;
    PyArray_StridedUnaryOp *wrapped,
                *tobuffer, *frombuffer;
    NpyAuxData *wrappeddata, *todata, *fromdata;
    npy_intp src_itemsize, dst_itemsize;
    char *bufferin, *bufferout;
    npy_bool init_dest, out_needs_api;
} _align_wrap_data;

/* transfer data free function */
static void _align_wrap_data_free(NpyAuxData *data)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    NPY_AUXDATA_FREE(d->wrappeddata);
    NPY_AUXDATA_FREE(d->todata);
    NPY_AUXDATA_FREE(d->fromdata);
    PyArray_free(data);
}

/* transfer data copy function */
static NpyAuxData *_align_wrap_data_clone(NpyAuxData *data)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    _align_wrap_data *newdata;
    npy_intp basedatasize, datasize;

    /* Round up the structure size to 16-byte boundary */
    basedatasize = (sizeof(_align_wrap_data)+15)&(-0x10);
    /* Add space for two low level buffers */
    datasize = basedatasize +
                NPY_LOWLEVEL_BUFFER_BLOCKSIZE*d->src_itemsize +
                NPY_LOWLEVEL_BUFFER_BLOCKSIZE*d->dst_itemsize;

    /* Allocate the data, and populate it */
    newdata = (_align_wrap_data *)PyArray_malloc(datasize);
    if (newdata == NULL) {
        return NULL;
    }
    memcpy(newdata, data, basedatasize);
    newdata->bufferin = (char *)newdata + basedatasize;
    newdata->bufferout = newdata->bufferin +
                NPY_LOWLEVEL_BUFFER_BLOCKSIZE*newdata->src_itemsize;
    if (newdata->wrappeddata != NULL) {
        newdata->wrappeddata = NPY_AUXDATA_CLONE(d->wrappeddata);
        if (newdata->wrappeddata == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (newdata->todata != NULL) {
        newdata->todata = NPY_AUXDATA_CLONE(d->todata);
        if (newdata->todata == NULL) {
            NPY_AUXDATA_FREE(newdata->wrappeddata);
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (newdata->fromdata != NULL) {
        newdata->fromdata = NPY_AUXDATA_CLONE(d->fromdata);
        if (newdata->fromdata == NULL) {
            NPY_AUXDATA_FREE(newdata->wrappeddata);
            NPY_AUXDATA_FREE(newdata->todata);
            PyArray_free(newdata);
            return NULL;
        }
    }

    newdata->init_dest = d->init_dest;
    newdata->out_needs_api = d->out_needs_api;

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_contig_align_wrap(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    PyArray_StridedUnaryOp *wrapped = d->wrapped,
            *tobuffer = d->tobuffer,
            *frombuffer = d->frombuffer;
    npy_intp inner_src_itemsize = d->src_itemsize,
             dst_itemsize = d->dst_itemsize;
    NpyAuxData *wrappeddata = d->wrappeddata,
            *todata = d->todata,
            *fromdata = d->fromdata;
    char *bufferin = d->bufferin, *bufferout = d->bufferout;
    npy_bool init_dest = d->init_dest;

    for(;;) {
        if (N > NPY_LOWLEVEL_BUFFER_BLOCKSIZE) {
            if (tobuffer(
                    bufferin, inner_src_itemsize, src, src_stride,
                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE, src_itemsize, todata) < 0) {
                return -1;
            }
            if (init_dest) {
                memset(bufferout, 0,
                       dst_itemsize*NPY_LOWLEVEL_BUFFER_BLOCKSIZE);
            }
            if (wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize,
                        NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                        inner_src_itemsize, wrappeddata) < 0) {
                return -1;
            }
            if (frombuffer(dst, dst_stride, bufferout, dst_itemsize,
                           NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                           dst_itemsize, fromdata) < 0) {
                return -1;
            }
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            if (tobuffer(bufferin, inner_src_itemsize, src, src_stride,
                         N, src_itemsize, todata) < 0) {
                return -1;
            }
            if (init_dest) {
                memset(bufferout, 0, dst_itemsize*N);
            }
            if (wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize,
                        N, inner_src_itemsize, wrappeddata) < 0) {
                return -1;
            }
            if (frombuffer(dst, dst_stride, bufferout, dst_itemsize,
                           N, dst_itemsize, fromdata) < 0) {
                return -1;
            }
            return 0;
        }
    }
}

/*
 * Wraps an aligned contig to contig transfer function between either
 * copies or byte swaps to temporary buffers.
 *
 * src_itemsize/dst_itemsize - The sizes of the src and dst datatypes.
 * tobuffer - copy/swap function from src to an aligned contiguous buffer.
 * todata - data for tobuffer
 * frombuffer - copy/swap function from an aligned contiguous buffer to dst.
 * fromdata - data for frombuffer
 * wrapped - contig to contig transfer function being wrapped
 * wrappeddata - data for wrapped
 * init_dest - 1 means to memset the dest buffer to 0 before calling wrapped.
 * out_needs_api - if NPY_TRUE, check for (and break on) Python API errors.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
wrap_aligned_contig_transfer_function(
            npy_intp src_itemsize, npy_intp dst_itemsize,
            PyArray_StridedUnaryOp *tobuffer, NpyAuxData *todata,
            PyArray_StridedUnaryOp *frombuffer, NpyAuxData *fromdata,
            PyArray_StridedUnaryOp *wrapped, NpyAuxData *wrappeddata,
            int init_dest,
            int out_needs_api,
            PyArray_StridedUnaryOp **out_stransfer,
            NpyAuxData **out_transferdata)
{
    _align_wrap_data *data;
    npy_intp basedatasize, datasize;

    /* Round up the structure size to 16-byte boundary */
    basedatasize = (sizeof(_align_wrap_data)+15)&(-0x10);
    /* Add space for two low level buffers */
    datasize = basedatasize +
                NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_itemsize +
                NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_itemsize;

    /* Allocate the data, and populate it */
    data = (_align_wrap_data *)PyArray_malloc(datasize);
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->base.free = &_align_wrap_data_free;
    data->base.clone = &_align_wrap_data_clone;
    data->tobuffer = tobuffer;
    data->todata = todata;
    data->frombuffer = frombuffer;
    data->fromdata = fromdata;
    data->wrapped = wrapped;
    data->wrappeddata = wrappeddata;
    data->src_itemsize = src_itemsize;
    data->dst_itemsize = dst_itemsize;
    data->bufferin = (char *)data + basedatasize;
    data->bufferout = data->bufferin +
                NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_itemsize;
    data->init_dest = (npy_bool) init_dest;
    data->out_needs_api = (npy_bool) out_needs_api;

    /* Set the function and data */
    *out_stransfer = &_strided_to_strided_contig_align_wrap;
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
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
    PyArray_free(data);
}

/* wrap copy swap data copy function */
static NpyAuxData *_wrap_copy_swap_data_clone(NpyAuxData *data)
{
    _wrap_copy_swap_data *newdata =
        (_wrap_copy_swap_data *)PyArray_malloc(sizeof(_wrap_copy_swap_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_wrap_copy_swap_data));
    Py_INCREF(newdata->arr);

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_wrap_copy_swap(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data)
{
    _wrap_copy_swap_data *d = (_wrap_copy_swap_data *)data;

    /* We assume that d->copyswapn should not be able to error. */
    d->copyswapn(dst, dst_stride, src, src_stride, N, d->swap, d->arr);
    return 0;
}

/* This only gets used for custom data types and for Unicode when swapping */
static int
wrap_copy_swap_function(int aligned,
                npy_intp src_stride, npy_intp dst_stride,
                PyArray_Descr *dtype,
                int should_swap,
                PyArray_StridedUnaryOp **out_stransfer,
                NpyAuxData **out_transferdata)
{
    _wrap_copy_swap_data *data;
    npy_intp shape = 1;

    /* Allocate the data for the copy swap */
    data = (_wrap_copy_swap_data *)PyArray_malloc(sizeof(_wrap_copy_swap_data));
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }

    data->base.free = &_wrap_copy_swap_data_free;
    data->base.clone = &_wrap_copy_swap_data_clone;
    data->copyswapn = dtype->f->copyswapn;
    data->swap = should_swap;

    /*
     * TODO: This is a hack so the copyswap functions have an array.
     *       The copyswap functions shouldn't need that.
     */
    Py_INCREF(dtype);
    data->arr = (PyArrayObject *)PyArray_NewFromDescr_int(
            &PyArray_Type, dtype,
            1, &shape, NULL, NULL,
            0, NULL, NULL,
            0, 1);
    if (data->arr == NULL) {
        PyArray_free(data);
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
    PyArray_free(data);
}

/* strided cast data copy function */
static NpyAuxData *_strided_cast_data_clone(NpyAuxData *data)
{
    _strided_cast_data *newdata =
            (_strided_cast_data *)PyArray_malloc(sizeof(_strided_cast_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_strided_cast_data));
    Py_INCREF(newdata->aip);
    Py_INCREF(newdata->aop);

    return (NpyAuxData *)newdata;
}

static int
_aligned_strided_to_strided_cast(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_cast_data *d = (_strided_cast_data *)data;
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
_aligned_strided_to_strided_cast_decref_src(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
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
_aligned_contig_to_contig_cast(char *dst, npy_intp NPY_UNUSED(dst_stride),
                        char *src, npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(itemsize),
                        NpyAuxData *data)
{
    _strided_cast_data *d = (_strided_cast_data *)data;
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

#if !NPY_USE_NEW_CASTINGIMPL
static int
get_nbo_cast_numeric_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            int src_type_num, int dst_type_num,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata)
{
    /* Emit a warning if complex imaginary is being cast away */
    if (PyTypeNum_ISCOMPLEX(src_type_num) &&
                    !PyTypeNum_ISCOMPLEX(dst_type_num) &&
                    !PyTypeNum_ISBOOL(dst_type_num)) {
        static PyObject *cls = NULL;
        int ret;
        npy_cache_import("numpy.core", "ComplexWarning", &cls);
        if (cls == NULL) {
            return NPY_FAIL;
        }
        ret = PyErr_WarnEx(cls,
                "Casting complex values to real discards "
                "the imaginary part", 1);
        if (ret < 0) {
            return NPY_FAIL;
        }
    }

    *out_stransfer = PyArray_GetStridedNumericCastFn(aligned,
                                src_stride, dst_stride,
                                src_type_num, dst_type_num);
    *out_transferdata = NULL;
    if (*out_stransfer == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "unexpected error in GetStridedNumericCastFn");
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}
#endif

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
    PyArray_free(d->tmp_buffer);
    PyArray_free(data);
}

/* strided datetime cast data copy function */
static NpyAuxData *_strided_datetime_cast_data_clone(NpyAuxData *data)
{
    _strided_datetime_cast_data *newdata =
            (_strided_datetime_cast_data *)PyArray_malloc(
                                        sizeof(_strided_datetime_cast_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_strided_datetime_cast_data));
    if (newdata->tmp_buffer != NULL) {
        newdata->tmp_buffer = PyArray_malloc(newdata->src_itemsize + 1);
        if (newdata->tmp_buffer == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_datetime_general_cast(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)data;
    npy_int64 dt;
    npy_datetimestruct dts;

    while (N > 0) {
        memcpy(&dt, src, sizeof(dt));

        if (convert_datetime_to_datetimestruct(&d->src_meta,
                                               dt, &dts) < 0) {
            return -1;
        }
        else {
            if (convert_datetimestruct_to_datetime(&d->dst_meta,
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
_strided_to_strided_datetime_cast(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)data;
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
_aligned_strided_to_strided_datetime_cast(char *dst,
                        npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)data;
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
_strided_to_strided_datetime_to_string(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data)
{
    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)data;
    npy_intp dst_itemsize = d->dst_itemsize;
    npy_int64 dt;
    npy_datetimestruct dts;

    while (N > 0) {
        memcpy(&dt, src, sizeof(dt));

        if (convert_datetime_to_datetimestruct(&d->src_meta,
                                               dt, &dts) < 0) {
            return -1;
        }

        /* Initialize the destination to all zeros */
        memset(dst, 0, dst_itemsize);

        if (make_iso_8601_datetime(&dts, dst, dst_itemsize,
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
_strided_to_strided_string_to_datetime(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)data;
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

            if (parse_iso_8601_datetime(tmp_buffer, src_itemsize,
                                    d->dst_meta.base, NPY_SAME_KIND_CASTING,
                                    &dts, NULL, NULL) < 0) {
                return -1;
            }
        }
        /* Otherwise parse the data in place */
        else {
            if (parse_iso_8601_datetime(src, tmp - src,
                                    d->dst_meta.base, NPY_SAME_KIND_CASTING,
                                    &dts, NULL, NULL) < 0) {
                return -1;
            }
        }

        /* Convert to the datetime */
        if (dt != NPY_DATETIME_NAT &&
                convert_datetimestruct_to_datetime(&d->dst_meta,
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
                            PyArray_StridedUnaryOp **out_stransfer,
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
    data = (_strided_datetime_cast_data *)PyArray_malloc(
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
        PyArray_StridedUnaryOp **out_stransfer, NpyAuxData **out_transferdata)
{
    PyArray_DatetimeMetaData *src_meta;
    _strided_datetime_cast_data *data;

    src_meta = get_datetime_metadata_from_dtype(src_dtype);
    if (src_meta == NULL) {
        return NPY_FAIL;
    }

    /* Allocate the data for the casting */
    data = (_strided_datetime_cast_data *)PyArray_malloc(
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
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    NpyAuxData *castdata = NULL, *todata = NULL, *fromdata = NULL;
    PyArray_StridedUnaryOp *caststransfer, *tobuffer, *frombuffer;
    PyArray_Descr *str_dtype;

    /* Get an ASCII string data type, adapted to match the UNICODE one */
    str_dtype = PyArray_DescrNewFromType(NPY_STRING);
    if (str_dtype == NULL) {
        return NPY_FAIL;
    }
    str_dtype->elsize = dst_dtype->elsize / 4;

    /* Get the copy/swap operation to dst */
    if (PyArray_GetDTypeCopySwapFn(aligned,
                            src_stride, src_dtype->elsize,
                            src_dtype,
                            &tobuffer, &todata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        return NPY_FAIL;
    }

    /* Get the NBO datetime to string aligned contig function */
    if (get_nbo_datetime_to_string_transfer_function(
            src_dtype, str_dtype,
            &caststransfer, &castdata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        NPY_AUXDATA_FREE(todata);
        return NPY_FAIL;
    }

    /* Get the cast operation to dst */
    if (PyArray_GetDTypeTransferFunction(aligned,
                            str_dtype->elsize, dst_stride,
                            str_dtype, dst_dtype,
                            0,
                            &frombuffer, &fromdata,
                            out_needs_api) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        NPY_AUXDATA_FREE(todata);
        NPY_AUXDATA_FREE(castdata);
        return NPY_FAIL;
    }

    /* Wrap it all up in a new transfer function + data */
    if (wrap_aligned_contig_transfer_function(
                        src_dtype->elsize, str_dtype->elsize,
                        tobuffer, todata,
                        frombuffer, fromdata,
                        caststransfer, castdata,
                        PyDataType_FLAGCHK(str_dtype, NPY_NEEDS_INIT),
                        *out_needs_api,
                        out_stransfer, out_transferdata) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE(castdata);
        NPY_AUXDATA_FREE(todata);
        NPY_AUXDATA_FREE(fromdata);
        return NPY_FAIL;
    }

    Py_DECREF(str_dtype);

    return NPY_SUCCEED;
}

NPY_NO_EXPORT int
get_nbo_string_to_datetime_transfer_function(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArray_StridedUnaryOp **out_stransfer, NpyAuxData **out_transferdata)
{
    PyArray_DatetimeMetaData *dst_meta;
    _strided_datetime_cast_data *data;

    dst_meta = get_datetime_metadata_from_dtype(dst_dtype);
    if (dst_meta == NULL) {
        return NPY_FAIL;
    }

    /* Allocate the data for the casting */
    data = (_strided_datetime_cast_data *)PyArray_malloc(
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
    data->tmp_buffer = PyArray_malloc(data->src_itemsize + 1);
    if (data->tmp_buffer == NULL) {
        PyErr_NoMemory();
        PyArray_free(data);
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
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    NpyAuxData *castdata = NULL, *todata = NULL, *fromdata = NULL;
    PyArray_StridedUnaryOp *caststransfer, *tobuffer, *frombuffer;
    PyArray_Descr *str_dtype;

    /* Get an ASCII string data type, adapted to match the UNICODE one */
    str_dtype = PyArray_DescrNewFromType(NPY_STRING);
    if (str_dtype == NULL) {
        return NPY_FAIL;
    }
    assert(src_dtype->type_num == NPY_UNICODE);
    str_dtype->elsize = src_dtype->elsize / 4;

    /* Get the cast operation from src */
    if (PyArray_GetDTypeTransferFunction(aligned,
                            src_stride, str_dtype->elsize,
                            src_dtype, str_dtype,
                            0,
                            &tobuffer, &todata,
                            out_needs_api) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        return NPY_FAIL;
    }

    /* Get the string to NBO datetime aligned contig function */
    if (get_nbo_string_to_datetime_transfer_function(
            str_dtype, dst_dtype,
            &caststransfer, &castdata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        NPY_AUXDATA_FREE(todata);
        return NPY_FAIL;
    }

    /* Get the copy/swap operation to dst */
    if (PyArray_GetDTypeCopySwapFn(aligned,
                            dst_dtype->elsize, dst_stride,
                            dst_dtype,
                            &frombuffer, &fromdata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        NPY_AUXDATA_FREE(todata);
        NPY_AUXDATA_FREE(castdata);
        return NPY_FAIL;
    }

    /* Wrap it all up in a new transfer function + data */
    if (wrap_aligned_contig_transfer_function(
                        str_dtype->elsize, dst_dtype->elsize,
                        tobuffer, todata,
                        frombuffer, fromdata,
                        caststransfer, castdata,
                        PyDataType_FLAGCHK(dst_dtype, NPY_NEEDS_INIT),
                        *out_needs_api,
                        out_stransfer, out_transferdata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        NPY_AUXDATA_FREE(castdata);
        NPY_AUXDATA_FREE(todata);
        NPY_AUXDATA_FREE(fromdata);
        return NPY_FAIL;
    }

    Py_DECREF(str_dtype);

    return NPY_SUCCEED;
}


NPY_NO_EXPORT int
get_legacy_dtype_cast_function(
        int aligned, npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArray_StridedUnaryOp **out_stransfer, NpyAuxData **out_transferdata,
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
    data = (_strided_cast_data *)PyArray_malloc(sizeof(_strided_cast_data));
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
            PyArray_free(data);
            return NPY_FAIL;
        }
    }
    data->aip = (PyArrayObject *)PyArray_NewFromDescr_int(
            &PyArray_Type, tmp_dtype,
            1, &shape, NULL, NULL,
            0, NULL, NULL,
            0, 1);
    if (data->aip == NULL) {
        PyArray_free(data);
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
            PyArray_free(data);
            return NPY_FAIL;
        }
    }
    data->aop = (PyArrayObject *)PyArray_NewFromDescr_int(
            &PyArray_Type, tmp_dtype,
            1, &shape, NULL, NULL,
            0, NULL, NULL,
            0, 1);
    if (data->aop == NULL) {
        Py_DECREF(data->aip);
        PyArray_free(data);
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


#if !NPY_USE_NEW_CASTINGIMPL
static int
get_nbo_cast_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api,
                            int *out_needs_wrap)
{
    if (PyTypeNum_ISNUMBER(src_dtype->type_num) &&
                    PyTypeNum_ISNUMBER(dst_dtype->type_num)) {
        *out_needs_wrap = !PyArray_ISNBO(src_dtype->byteorder) ||
                          !PyArray_ISNBO(dst_dtype->byteorder);
        return get_nbo_cast_numeric_transfer_function(aligned,
                                    src_stride, dst_stride,
                                    src_dtype->type_num, dst_dtype->type_num,
                                    out_stransfer, out_transferdata);
    }

    if (src_dtype->type_num == NPY_DATETIME ||
            src_dtype->type_num == NPY_TIMEDELTA ||
            dst_dtype->type_num == NPY_DATETIME ||
            dst_dtype->type_num == NPY_TIMEDELTA) {
        /* A parameterized type, datetime->datetime sometimes needs casting */
        if ((src_dtype->type_num == NPY_DATETIME &&
                    dst_dtype->type_num == NPY_DATETIME) ||
                (src_dtype->type_num == NPY_TIMEDELTA &&
                    dst_dtype->type_num == NPY_TIMEDELTA)) {
            *out_needs_wrap = !PyArray_ISNBO(src_dtype->byteorder) ||
                              !PyArray_ISNBO(dst_dtype->byteorder);
            return get_nbo_cast_datetime_transfer_function(aligned,
                                        src_dtype, dst_dtype,
                                        out_stransfer, out_transferdata);
        }

        /*
         * Datetime <-> string conversions can be handled specially.
         * The functions may raise an error if the strings have no
         * space, or can't be parsed properly.
         */
        if (src_dtype->type_num == NPY_DATETIME) {
            switch (dst_dtype->type_num) {
                case NPY_STRING:
                    *out_needs_api = 1;
                    *out_needs_wrap = !PyArray_ISNBO(src_dtype->byteorder);
                    return get_nbo_datetime_to_string_transfer_function(
                            src_dtype, dst_dtype,
                            out_stransfer, out_transferdata);

                case NPY_UNICODE:
                    return get_datetime_to_unicode_transfer_function(
                                        aligned,
                                        src_stride, dst_stride,
                                        src_dtype, dst_dtype,
                                        out_stransfer, out_transferdata,
                                        out_needs_api);
            }
        }
        else if (dst_dtype->type_num == NPY_DATETIME) {
            switch (src_dtype->type_num) {
                case NPY_STRING:
                    *out_needs_api = 1;
                    *out_needs_wrap = !PyArray_ISNBO(dst_dtype->byteorder);
                    return get_nbo_string_to_datetime_transfer_function(
                            src_dtype, dst_dtype,
                            out_stransfer, out_transferdata);

                case NPY_UNICODE:
                    return get_unicode_to_datetime_transfer_function(
                                        aligned,
                                        src_stride, dst_stride,
                                        src_dtype, dst_dtype,
                                        out_stransfer, out_transferdata,
                                        out_needs_api);
            }
        }
    }

    return get_legacy_dtype_cast_function(
            aligned, src_stride, dst_stride, src_dtype, dst_dtype,
            move_references, out_stransfer, out_transferdata,
            out_needs_api, out_needs_wrap);
}
#endif


NPY_NO_EXPORT int
wrap_aligned_contig_transfer_function_with_copyswapn(
        int aligned, npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArray_StridedUnaryOp **out_stransfer, NpyAuxData **out_transferdata,
        int *out_needs_api,
        PyArray_StridedUnaryOp *caststransfer, NpyAuxData *castdata)
{
    NpyAuxData *todata = NULL, *fromdata = NULL;
    PyArray_StridedUnaryOp *tobuffer = NULL, *frombuffer = NULL;
    npy_intp src_itemsize = src_dtype->elsize;
    npy_intp dst_itemsize = dst_dtype->elsize;

    /* Get the copy/swap operation from src */
    PyArray_GetDTypeCopySwapFn(
            aligned, src_stride, src_itemsize, src_dtype, &tobuffer, &todata);

    if (!PyDataType_REFCHK(dst_dtype)) {
        /* Copying from buffer is a simple copy/swap operation */
        PyArray_GetDTypeCopySwapFn(
                aligned, dst_itemsize, dst_stride, dst_dtype,
                &frombuffer, &fromdata);
    }
    else {
        /*
         * Since the buffer is initialized to NULL, need to move the
         * references in order to DECREF the existing data.
         */
         /* Object types cannot be byte swapped */
        assert(PyDataType_ISNOTSWAPPED(dst_dtype));
        /* The loop already needs the python api if this is reached */
        assert(*out_needs_api);

        if (PyArray_GetDTypeTransferFunction(
                aligned, dst_itemsize, dst_stride,
                dst_dtype, dst_dtype, 1,
                &frombuffer, &fromdata, out_needs_api) != NPY_SUCCEED) {
            return NPY_FAIL;
        }
    }

    if (frombuffer == NULL || tobuffer == NULL) {
        NPY_AUXDATA_FREE(castdata);
        NPY_AUXDATA_FREE(todata);
        NPY_AUXDATA_FREE(fromdata);
        return NPY_FAIL;
    }

    *out_stransfer = caststransfer;

    /* Wrap it all up in a new transfer function + data */
    if (wrap_aligned_contig_transfer_function(
                        src_itemsize, dst_itemsize,
                        tobuffer, todata,
                        frombuffer, fromdata,
                        caststransfer, castdata,
                        PyDataType_FLAGCHK(dst_dtype, NPY_NEEDS_INIT),
                        *out_needs_api,
                        out_stransfer, out_transferdata) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE(castdata);
        NPY_AUXDATA_FREE(todata);
        NPY_AUXDATA_FREE(fromdata);
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}


#if !NPY_USE_NEW_CASTINGIMPL
static int
get_cast_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyArray_StridedUnaryOp *caststransfer;
    NpyAuxData *castdata;
    int needs_wrap = 0;

    if (get_nbo_cast_transfer_function(aligned,
                            src_stride, dst_stride,
                            src_dtype, dst_dtype,
                            move_references,
                            &caststransfer,
                            &castdata,
                            out_needs_api,
                            &needs_wrap) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    /*
     * If all native byte order and doesn't need alignment wrapping,
     * return the function
     */
    if (!needs_wrap) {
        *out_stransfer = caststransfer;
        *out_transferdata = castdata;

        return NPY_SUCCEED;
    }
    /* Otherwise, we have to copy and/or swap to aligned temporaries */
    else {
        return wrap_aligned_contig_transfer_function_with_copyswapn(
                aligned, src_stride, dst_stride, src_dtype, dst_dtype,
                out_stransfer, out_transferdata, out_needs_api,
                caststransfer, castdata);
    }
}
#endif

/**************************** COPY 1 TO N CONTIGUOUS ************************/

/* Copies 1 element to N contiguous elements */
typedef struct {
    NpyAuxData base;
    PyArray_StridedUnaryOp *stransfer;
    NpyAuxData *data;
    npy_intp N, dst_itemsize;
    /* If this is non-NULL the source type has references needing a decref */
    PyArray_StridedUnaryOp *stransfer_finish_src;
    NpyAuxData *data_finish_src;
} _one_to_n_data;

/* transfer data free function */
static void _one_to_n_data_free(NpyAuxData *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    NPY_AUXDATA_FREE(d->data);
    NPY_AUXDATA_FREE(d->data_finish_src);
    PyArray_free(data);
}

/* transfer data copy function */
static NpyAuxData *_one_to_n_data_clone(NpyAuxData *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    _one_to_n_data *newdata;

    /* Allocate the data, and populate it */
    newdata = (_one_to_n_data *)PyArray_malloc(sizeof(_one_to_n_data));
    if (newdata == NULL) {
        return NULL;
    }
    memcpy(newdata, data, sizeof(_one_to_n_data));
    if (d->data != NULL) {
        newdata->data = NPY_AUXDATA_CLONE(d->data);
        if (newdata->data == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (d->data_finish_src != NULL) {
        newdata->data_finish_src = NPY_AUXDATA_CLONE(d->data_finish_src);
        if (newdata->data_finish_src == NULL) {
            NPY_AUXDATA_FREE(newdata->data);
            PyArray_free(newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_one_to_n(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    PyArray_StridedUnaryOp *subtransfer = d->stransfer;
    NpyAuxData *subdata = d->data;
    npy_intp subN = d->N, dst_itemsize = d->dst_itemsize;

    while (N > 0) {
        if (subtransfer(
                dst, dst_itemsize, src, 0, subN, src_itemsize, subdata) < 0) {
            return -1;
        }

        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

static int
_strided_to_strided_one_to_n_with_finish(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    PyArray_StridedUnaryOp *subtransfer = d->stransfer,
                *stransfer_finish_src = d->stransfer_finish_src;
    NpyAuxData *subdata = d->data, *data_finish_src = d->data_finish_src;
    npy_intp subN = d->N, dst_itemsize = d->dst_itemsize;

    while (N > 0) {
        if (subtransfer(
                dst, dst_itemsize, src, 0, subN, src_itemsize, subdata) < 0) {
            return -1;
        }

        if (stransfer_finish_src(
                NULL, 0, src, 0, 1, src_itemsize, data_finish_src) < 0) {
            return -1;
        }

        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

/*
 * Wraps a transfer function to produce one that copies one element
 * of src to N contiguous elements of dst.  If stransfer_finish_src is
 * not NULL, it should be a transfer function which just affects
 * src, for example to do a final DECREF operation for references.
 */
static int
wrap_transfer_function_one_to_n(
                            PyArray_StridedUnaryOp *stransfer_inner,
                            NpyAuxData *data_inner,
                            PyArray_StridedUnaryOp *stransfer_finish_src,
                            NpyAuxData *data_finish_src,
                            npy_intp dst_itemsize,
                            npy_intp N,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata)
{
    _one_to_n_data *data;


    data = PyArray_malloc(sizeof(_one_to_n_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }

    data->base.free = &_one_to_n_data_free;
    data->base.clone = &_one_to_n_data_clone;
    data->stransfer = stransfer_inner;
    data->data = data_inner;
    data->stransfer_finish_src = stransfer_finish_src;
    data->data_finish_src = data_finish_src;
    data->N = N;
    data->dst_itemsize = dst_itemsize;

    if (stransfer_finish_src == NULL) {
        *out_stransfer = &_strided_to_strided_one_to_n;
    }
    else {
        *out_stransfer = &_strided_to_strided_one_to_n_with_finish;
    }
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}

static int
get_one_to_n_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            npy_intp N,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyArray_StridedUnaryOp *stransfer, *stransfer_finish_src = NULL;
    NpyAuxData *data, *data_finish_src = NULL;

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
                    &stransfer, &data,
                    out_needs_api) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    /* If the src object will need a DECREF, set src_dtype */
    if (move_references && PyDataType_REFCHK(src_dtype)) {
        if (get_decsrcref_transfer_function(aligned,
                            src_stride,
                            src_dtype,
                            &stransfer_finish_src,
                            &data_finish_src,
                            out_needs_api) != NPY_SUCCEED) {
            NPY_AUXDATA_FREE(data);
            return NPY_FAIL;
        }
    }

    if (wrap_transfer_function_one_to_n(stransfer, data,
                            stransfer_finish_src, data_finish_src,
                            dst_dtype->elsize,
                            N,
                            out_stransfer, out_transferdata) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE(data);
        NPY_AUXDATA_FREE(data_finish_src);
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}

/**************************** COPY N TO N CONTIGUOUS ************************/

/* Copies N contiguous elements to N contiguous elements */
typedef struct {
    NpyAuxData base;
    PyArray_StridedUnaryOp *stransfer;
    NpyAuxData *data;
    npy_intp N, src_itemsize, dst_itemsize;
} _n_to_n_data;

/* transfer data free function */
static void _n_to_n_data_free(NpyAuxData *data)
{
    _n_to_n_data *d = (_n_to_n_data *)data;
    NPY_AUXDATA_FREE(d->data);
    PyArray_free(data);
}

/* transfer data copy function */
static NpyAuxData *_n_to_n_data_clone(NpyAuxData *data)
{
    _n_to_n_data *d = (_n_to_n_data *)data;
    _n_to_n_data *newdata;

    /* Allocate the data, and populate it */
    newdata = (_n_to_n_data *)PyArray_malloc(sizeof(_n_to_n_data));
    if (newdata == NULL) {
        return NULL;
    }
    memcpy(newdata, data, sizeof(_n_to_n_data));
    if (newdata->data != NULL) {
        newdata->data = NPY_AUXDATA_CLONE(d->data);
        if (newdata->data == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_n_to_n(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _n_to_n_data *d = (_n_to_n_data *)data;
    PyArray_StridedUnaryOp *subtransfer = d->stransfer;
    NpyAuxData *subdata = d->data;
    npy_intp subN = d->N, src_subitemsize = d->src_itemsize,
                dst_subitemsize = d->dst_itemsize;

    while (N > 0) {
        if (subtransfer(
                dst, dst_subitemsize, src, src_subitemsize,
                subN, src_subitemsize, subdata) < 0) {
            return -1;
        }
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

static int
_contig_to_contig_n_to_n(char *dst, npy_intp NPY_UNUSED(dst_stride),
                        char *src, npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data)
{
    _n_to_n_data *d = (_n_to_n_data *)data;
    PyArray_StridedUnaryOp *subtransfer = d->stransfer;
    NpyAuxData *subdata = d->data;
    npy_intp subN = d->N, src_subitemsize = d->src_itemsize,
                dst_subitemsize = d->dst_itemsize;

    if (subtransfer(
            dst, dst_subitemsize, src, src_subitemsize,
            subN*N, src_subitemsize, subdata) < 0) {
        return -1;
    }
    return 0;
}

/*
 * Wraps a transfer function to produce one that copies N contiguous elements
 * of src to N contiguous elements of dst.
 */
static int
wrap_transfer_function_n_to_n(
                            PyArray_StridedUnaryOp *stransfer_inner,
                            NpyAuxData *data_inner,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp src_itemsize, npy_intp dst_itemsize,
                            npy_intp N,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata)
{
    _n_to_n_data *data;

    data = PyArray_malloc(sizeof(_n_to_n_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }

    data->base.free = &_n_to_n_data_free;
    data->base.clone = &_n_to_n_data_clone;
    data->stransfer = stransfer_inner;
    data->data = data_inner;
    data->N = N;
    data->src_itemsize = src_itemsize;
    data->dst_itemsize = dst_itemsize;

    /*
     * If the N subarray elements exactly fit in the strides,
     * then can do a faster contiguous transfer.
     */
    if (src_stride == N * src_itemsize &&
                    dst_stride == N * dst_itemsize) {
        *out_stransfer = &_contig_to_contig_n_to_n;
    }
    else {
        *out_stransfer = &_strided_to_strided_n_to_n;
    }
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}

static int
get_n_to_n_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            npy_intp N,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyArray_StridedUnaryOp *stransfer;
    NpyAuxData *data;

    /*
     * src_stride and dst_stride are set to contiguous, because
     * subarrays are always contiguous.
     */
    if (PyArray_GetDTypeTransferFunction(aligned,
                    src_dtype->elsize, dst_dtype->elsize,
                    src_dtype, dst_dtype,
                    move_references,
                    &stransfer, &data,
                    out_needs_api) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    if (wrap_transfer_function_n_to_n(stransfer, data,
                            src_stride, dst_stride,
                            src_dtype->elsize, dst_dtype->elsize,
                            N,
                            out_stransfer,
                            out_transferdata) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE(data);
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}

/********************** COPY WITH SUBARRAY BROADCAST ************************/

typedef struct {
    npy_intp offset, count;
} _subarray_broadcast_offsetrun;

/* Copies element with subarray broadcasting */
typedef struct {
    NpyAuxData base;
    PyArray_StridedUnaryOp *stransfer;
    NpyAuxData *data;
    npy_intp src_N, dst_N, src_itemsize, dst_itemsize;
    PyArray_StridedUnaryOp *stransfer_decsrcref;
    NpyAuxData *data_decsrcref;
    PyArray_StridedUnaryOp *stransfer_decdstref;
    NpyAuxData *data_decdstref;
    /* This gets a run-length encoded representation of the transfer */
    npy_intp run_count;
    _subarray_broadcast_offsetrun offsetruns;
} _subarray_broadcast_data;


/* transfer data free function */
static void _subarray_broadcast_data_free(NpyAuxData *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;
    NPY_AUXDATA_FREE(d->data);
    NPY_AUXDATA_FREE(d->data_decsrcref);
    NPY_AUXDATA_FREE(d->data_decdstref);
    PyArray_free(data);
}

/* transfer data copy function */
static NpyAuxData *_subarray_broadcast_data_clone( NpyAuxData *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;
    _subarray_broadcast_data *newdata;
    npy_intp run_count = d->run_count, structsize;

    structsize = sizeof(_subarray_broadcast_data) +
                        run_count*sizeof(_subarray_broadcast_offsetrun);

    /* Allocate the data and populate it */
    newdata = (_subarray_broadcast_data *)PyArray_malloc(structsize);
    if (newdata == NULL) {
        return NULL;
    }
    memcpy(newdata, data, structsize);
    if (d->data != NULL) {
        newdata->data = NPY_AUXDATA_CLONE(d->data);
        if (newdata->data == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (d->data_decsrcref != NULL) {
        newdata->data_decsrcref = NPY_AUXDATA_CLONE(d->data_decsrcref);
        if (newdata->data_decsrcref == NULL) {
            NPY_AUXDATA_FREE(newdata->data);
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (d->data_decdstref != NULL) {
        newdata->data_decdstref = NPY_AUXDATA_CLONE(d->data_decdstref);
        if (newdata->data_decdstref == NULL) {
            NPY_AUXDATA_FREE(newdata->data);
            NPY_AUXDATA_FREE(newdata->data_decsrcref);
            PyArray_free(newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_subarray_broadcast(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;
    PyArray_StridedUnaryOp *subtransfer = d->stransfer;
    NpyAuxData *subdata = d->data;
    npy_intp run, run_count = d->run_count,
            src_subitemsize = d->src_itemsize,
            dst_subitemsize = d->dst_itemsize;
    npy_intp loop_index, offset, count;
    char *dst_ptr;
    _subarray_broadcast_offsetrun *offsetruns = &d->offsetruns;

    while (N > 0) {
        loop_index = 0;
        for (run = 0; run < run_count; ++run) {
            offset = offsetruns[run].offset;
            count = offsetruns[run].count;
            dst_ptr = dst + loop_index*dst_subitemsize;
            if (offset != -1) {
                if (subtransfer(
                        dst_ptr, dst_subitemsize, src + offset, src_subitemsize,
                        count, src_subitemsize, subdata) < 0) {
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
_strided_to_strided_subarray_broadcast_withrefs(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;
    PyArray_StridedUnaryOp *subtransfer = d->stransfer;
    NpyAuxData *subdata = d->data;
    PyArray_StridedUnaryOp *stransfer_decsrcref = d->stransfer_decsrcref;
    NpyAuxData *data_decsrcref = d->data_decsrcref;
    PyArray_StridedUnaryOp *stransfer_decdstref = d->stransfer_decdstref;
    NpyAuxData *data_decdstref = d->data_decdstref;
    npy_intp run, run_count = d->run_count,
            src_subitemsize = d->src_itemsize,
            dst_subitemsize = d->dst_itemsize,
            src_subN = d->src_N;
    npy_intp loop_index, offset, count;
    char *dst_ptr;
    _subarray_broadcast_offsetrun *offsetruns = &d->offsetruns;

    while (N > 0) {
        loop_index = 0;
        for (run = 0; run < run_count; ++run) {
            offset = offsetruns[run].offset;
            count = offsetruns[run].count;
            dst_ptr = dst + loop_index*dst_subitemsize;
            if (offset != -1) {
                if (subtransfer(
                        dst_ptr, dst_subitemsize, src + offset, src_subitemsize,
                        count, src_subitemsize, subdata) < 0) {
                    return -1;
                }
            }
            else {
                if (stransfer_decdstref != NULL) {
                    if (stransfer_decdstref(
                            NULL, 0, dst_ptr, dst_subitemsize,
                            count, dst_subitemsize, data_decdstref) < 0) {
                        return -1;
                    }
                }
                memset(dst_ptr, 0, count*dst_subitemsize);
            }
            loop_index += count;
        }

        if (stransfer_decsrcref != NULL) {
            if (stransfer_decsrcref(
                    NULL, 0, src, src_subitemsize,
                    src_subN, src_subitemsize, data_decsrcref) < 0) {
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
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    _subarray_broadcast_data *data;
    npy_intp structsize, loop_index, run, run_size,
             src_index, dst_index, i, ndim;
    _subarray_broadcast_offsetrun *offsetruns;

    structsize = sizeof(_subarray_broadcast_data) +
                        dst_size*sizeof(_subarray_broadcast_offsetrun);

    /* Allocate the data and populate it */
    data = (_subarray_broadcast_data *)PyArray_malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }

    /*
     * move_references is set to 0, handled in the wrapping transfer fn,
     * src_stride and dst_stride are set to contiguous, as N will always
     * be 1 when it's called.
     */
    if (PyArray_GetDTypeTransferFunction(aligned,
                    src_dtype->elsize, dst_dtype->elsize,
                    src_dtype, dst_dtype,
                    0,
                    &data->stransfer, &data->data,
                    out_needs_api) != NPY_SUCCEED) {
        PyArray_free(data);
        return NPY_FAIL;
    }
    data->base.free = &_subarray_broadcast_data_free;
    data->base.clone = &_subarray_broadcast_data_clone;
    data->src_N = src_size;
    data->dst_N = dst_size;
    data->src_itemsize = src_dtype->elsize;
    data->dst_itemsize = dst_dtype->elsize;

    /* If the src object will need a DECREF */
    if (move_references && PyDataType_REFCHK(src_dtype)) {
        if (PyArray_GetDTypeTransferFunction(aligned,
                        src_dtype->elsize, 0,
                        src_dtype, NULL,
                        1,
                        &data->stransfer_decsrcref,
                        &data->data_decsrcref,
                        out_needs_api) != NPY_SUCCEED) {
            NPY_AUXDATA_FREE(data->data);
            PyArray_free(data);
            return NPY_FAIL;
        }
    }
    else {
        data->stransfer_decsrcref = NULL;
        data->data_decsrcref = NULL;
    }

    /* If the dst object needs a DECREF to set it to NULL */
    if (PyDataType_REFCHK(dst_dtype)) {
        if (PyArray_GetDTypeTransferFunction(aligned,
                        dst_dtype->elsize, 0,
                        dst_dtype, NULL,
                        1,
                        &data->stransfer_decdstref,
                        &data->data_decdstref,
                        out_needs_api) != NPY_SUCCEED) {
            NPY_AUXDATA_FREE(data->data);
            NPY_AUXDATA_FREE(data->data_decsrcref);
            PyArray_free(data);
            return NPY_FAIL;
        }
    }
    else {
        data->stransfer_decdstref = NULL;
        data->data_decdstref = NULL;
    }

    /* Calculate the broadcasting and set the offsets */
    offsetruns = &data->offsetruns;
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

    if (data->stransfer_decsrcref == NULL &&
                                data->stransfer_decdstref == NULL) {
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
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyArray_Dims src_shape = {NULL, -1}, dst_shape = {NULL, -1};
    npy_intp src_size = 1, dst_size = 1;

    /* Get the subarray shapes and sizes */
    if (PyDataType_HASSUBARRAY(src_dtype)) {
        if (!(PyArray_IntpConverter(src_dtype->subarray->shape,
                                            &src_shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return NPY_FAIL;
        }
        src_size = PyArray_MultiplyList(src_shape.ptr, src_shape.len);
        src_dtype = src_dtype->subarray->base;
    }
    if (PyDataType_HASSUBARRAY(dst_dtype)) {
        if (!(PyArray_IntpConverter(dst_dtype->subarray->shape,
                                            &dst_shape))) {
            npy_free_cache_dim_obj(src_shape);
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return NPY_FAIL;
        }
        dst_size = PyArray_MultiplyList(dst_shape.ptr, dst_shape.len);
        dst_dtype = dst_dtype->subarray->base;
    }

    /*
     * Just a straight one-element copy.
     */
    if (dst_size == 1 && src_size == 1) {
        npy_free_cache_dim_obj(src_shape);
        npy_free_cache_dim_obj(dst_shape);

        return PyArray_GetDTypeTransferFunction(aligned,
                src_stride, dst_stride,
                src_dtype, dst_dtype,
                move_references,
                out_stransfer, out_transferdata,
                out_needs_api);
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
                        out_needs_api);
    }
    /* If the shapes match exactly, do an n to n copy */
    else if (src_shape.len == dst_shape.len &&
               PyArray_CompareLists(src_shape.ptr, dst_shape.ptr,
                                                    src_shape.len)) {
        npy_free_cache_dim_obj(src_shape);
        npy_free_cache_dim_obj(dst_shape);

        return get_n_to_n_transfer_function(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        src_size,
                        out_stransfer, out_transferdata,
                        out_needs_api);
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
                        out_needs_api);

        npy_free_cache_dim_obj(src_shape);
        npy_free_cache_dim_obj(dst_shape);
        return ret;
    }
}

/**************************** COPY FIELDS *******************************/
typedef struct {
    npy_intp src_offset, dst_offset, src_itemsize;
    PyArray_StridedUnaryOp *stransfer;
    NpyAuxData *data;
} _single_field_transfer;

typedef struct {
    NpyAuxData base;
    npy_intp field_count;

    _single_field_transfer fields;
} _field_transfer_data;

/* transfer data free function */
static void _field_transfer_data_free(NpyAuxData *data)
{
    _field_transfer_data *d = (_field_transfer_data *)data;
    npy_intp i, field_count;
    _single_field_transfer *fields;

    field_count = d->field_count;
    fields = &d->fields;

    for (i = 0; i < field_count; ++i) {
        NPY_AUXDATA_FREE(fields[i].data);
    }
    PyArray_free(d);
}

/* transfer data copy function */
static NpyAuxData *_field_transfer_data_clone(NpyAuxData *data)
{
    _field_transfer_data *d = (_field_transfer_data *)data;
    _field_transfer_data *newdata;
    npy_intp i, field_count = d->field_count, structsize;
    _single_field_transfer *fields, *newfields;

    structsize = sizeof(_field_transfer_data) +
                    field_count * sizeof(_single_field_transfer);

    /* Allocate the data and populate it */
    newdata = (_field_transfer_data *)PyArray_malloc(structsize);
    if (newdata == NULL) {
        return NULL;
    }
    memcpy(newdata, d, structsize);
    /* Copy all the fields transfer data */
    fields = &d->fields;
    newfields = &newdata->fields;
    for (i = 0; i < field_count; ++i) {
        if (fields[i].data != NULL) {
            newfields[i].data = NPY_AUXDATA_CLONE(fields[i].data);
            if (newfields[i].data == NULL) {
                for (i = i-1; i >= 0; --i) {
                    NPY_AUXDATA_FREE(newfields[i].data);
                }
                PyArray_free(newdata);
                return NULL;
            }
        }

    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_field_transfer(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data)
{
    _field_transfer_data *d = (_field_transfer_data *)data;
    npy_intp i, field_count = d->field_count;
    _single_field_transfer *field;

    /* Do the transfer a block at a time */
    for (;;) {
        field = &d->fields;
        if (N > NPY_LOWLEVEL_BUFFER_BLOCKSIZE) {
            for (i = 0; i < field_count; ++i, ++field) {
                if (field->stransfer(
                        dst + field->dst_offset, dst_stride,
                        src + field->src_offset, src_stride,
                        NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                        field->src_itemsize, field->data) < 0) {
                    return -1;
                }
            }
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            for (i = 0; i < field_count; ++i, ++field) {
                if (field->stransfer(
                        dst + field->dst_offset, dst_stride,
                        src + field->src_offset, src_stride,
                        N,
                        field->src_itemsize, field->data) < 0) {
                    return -1;
                }
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
get_fields_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyObject *key, *tup, *title;
    PyArray_Descr *src_fld_dtype, *dst_fld_dtype;
    npy_int i, field_count, structsize;
    int src_offset, dst_offset;
    _field_transfer_data *data;
    _single_field_transfer *fields;
    int failed = 0;

    /*
     * There are three cases to take care of: 1. src is non-structured,
     * 2. dst is non-structured, or 3. both are structured.
     */

    /* 1. src is non-structured. Copy the src value to all the fields of dst */
    if (!PyDataType_HASFIELDS(src_dtype)) {
        field_count = PyTuple_GET_SIZE(dst_dtype->names);

        /* Allocate the field-data structure and populate it */
        structsize = sizeof(_field_transfer_data) +
                        (field_count + 1) * sizeof(_single_field_transfer);
        data = (_field_transfer_data *)PyArray_malloc(structsize);
        if (data == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
        data->base.free = &_field_transfer_data_free;
        data->base.clone = &_field_transfer_data_clone;
        fields = &data->fields;

        for (i = 0; i < field_count; ++i) {
            key = PyTuple_GET_ITEM(dst_dtype->names, i);
            tup = PyDict_GetItem(dst_dtype->fields, key);
            if (!PyArg_ParseTuple(tup, "Oi|O", &dst_fld_dtype,
                                                    &dst_offset, &title)) {
                PyArray_free(data);
                return NPY_FAIL;
            }
            if (PyArray_GetDTypeTransferFunction(0,
                                    src_stride, dst_stride,
                                    src_dtype, dst_fld_dtype,
                                    0,
                                    &fields[i].stransfer,
                                    &fields[i].data,
                                    out_needs_api) != NPY_SUCCEED) {
                for (i = i-1; i >= 0; --i) {
                    NPY_AUXDATA_FREE(fields[i].data);
                }
                PyArray_free(data);
                return NPY_FAIL;
            }
            fields[i].src_offset = 0;
            fields[i].dst_offset = dst_offset;
            fields[i].src_itemsize = src_dtype->elsize;
        }

        /*
         * If references should be decrefd in src, add
         * another transfer function to do that.
         */
        if (move_references && PyDataType_REFCHK(src_dtype)) {
            if (get_decsrcref_transfer_function(0,
                                    src_stride,
                                    src_dtype,
                                    &fields[field_count].stransfer,
                                    &fields[field_count].data,
                                    out_needs_api) != NPY_SUCCEED) {
                for (i = 0; i < field_count; ++i) {
                    NPY_AUXDATA_FREE(fields[i].data);
                }
                PyArray_free(data);
                return NPY_FAIL;
            }
            fields[field_count].src_offset = 0;
            fields[field_count].dst_offset = 0;
            fields[field_count].src_itemsize = src_dtype->elsize;
            field_count++;
        }
        data->field_count = field_count;

        *out_stransfer = &_strided_to_strided_field_transfer;
        *out_transferdata = (NpyAuxData *)data;

        return NPY_SUCCEED;
    }

    /* 2. dst is non-structured. Allow transfer from single-field src to dst */
    if (!PyDataType_HASFIELDS(dst_dtype)) {
        if (PyTuple_GET_SIZE(src_dtype->names) != 1) {
            PyErr_SetString(PyExc_ValueError,
                    "Can't cast from structure to non-structure, except if the "
                    "structure only has a single field.");
            return NPY_FAIL;
        }

        /* Allocate the field-data structure and populate it */
        structsize = sizeof(_field_transfer_data) +
                        1 * sizeof(_single_field_transfer);
        data = (_field_transfer_data *)PyArray_malloc(structsize);
        if (data == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
        data->base.free = &_field_transfer_data_free;
        data->base.clone = &_field_transfer_data_clone;
        fields = &data->fields;

        key = PyTuple_GET_ITEM(src_dtype->names, 0);
        tup = PyDict_GetItem(src_dtype->fields, key);
        if (!PyArg_ParseTuple(tup, "Oi|O",
                              &src_fld_dtype, &src_offset, &title)) {
            return NPY_FAIL;
        }

        if (PyArray_GetDTypeTransferFunction(0,
                                             src_stride, dst_stride,
                                             src_fld_dtype, dst_dtype,
                                             move_references,
                                             &fields[0].stransfer,
                                             &fields[0].data,
                                             out_needs_api) != NPY_SUCCEED) {
            PyArray_free(data);
            return NPY_FAIL;
        }
        fields[0].src_offset = src_offset;
        fields[0].dst_offset = 0;
        fields[0].src_itemsize = src_fld_dtype->elsize;

        data->field_count = 1;

        *out_stransfer = &_strided_to_strided_field_transfer;
        *out_transferdata = (NpyAuxData *)data;

        return NPY_SUCCEED;
    }

    /* 3. Otherwise both src and dst are structured arrays */
    field_count = PyTuple_GET_SIZE(dst_dtype->names);

    /* Match up the fields to copy (field-by-field transfer) */
    if (PyTuple_GET_SIZE(src_dtype->names) != field_count) {
        PyErr_SetString(PyExc_ValueError, "structures must have the same size");
        return NPY_FAIL;
    }

    /* Allocate the field-data structure and populate it */
    structsize = sizeof(_field_transfer_data) +
                    field_count * sizeof(_single_field_transfer);
    data = (_field_transfer_data *)PyArray_malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->base.free = &_field_transfer_data_free;
    data->base.clone = &_field_transfer_data_clone;
    fields = &data->fields;

    /* set up the transfer function for each field */
    for (i = 0; i < field_count; ++i) {
        key = PyTuple_GET_ITEM(dst_dtype->names, i);
        tup = PyDict_GetItem(dst_dtype->fields, key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &dst_fld_dtype,
                                                &dst_offset, &title)) {
            failed = 1;
            break;
        }
        key = PyTuple_GET_ITEM(src_dtype->names, i);
        tup = PyDict_GetItem(src_dtype->fields, key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &src_fld_dtype,
                                                &src_offset, &title)) {
            failed = 1;
            break;
        }

        if (PyArray_GetDTypeTransferFunction(0,
                                             src_stride, dst_stride,
                                             src_fld_dtype, dst_fld_dtype,
                                             move_references,
                                             &fields[i].stransfer,
                                             &fields[i].data,
                                             out_needs_api) != NPY_SUCCEED) {
            failed = 1;
            break;
        }
        fields[i].src_offset = src_offset;
        fields[i].dst_offset = dst_offset;
        fields[i].src_itemsize = src_fld_dtype->elsize;
    }

    if (failed) {
        for (i = i-1; i >= 0; --i) {
            NPY_AUXDATA_FREE(fields[i].data);
        }
        PyArray_free(data);
        return NPY_FAIL;
    }

    data->field_count = field_count;

    *out_stransfer = &_strided_to_strided_field_transfer;
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}

static int
get_decsrcref_fields_transfer_function(int aligned,
                            npy_intp src_stride,
                            PyArray_Descr *src_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyObject *names, *key, *tup, *title;
    PyArray_Descr *src_fld_dtype;
    npy_int i, names_size, field_count, structsize;
    int src_offset;
    _field_transfer_data *data;
    _single_field_transfer *fields;

    names = src_dtype->names;
    names_size = PyTuple_GET_SIZE(src_dtype->names);

    field_count = names_size;
    structsize = sizeof(_field_transfer_data) +
                    field_count * sizeof(_single_field_transfer);
    /* Allocate the data and populate it */
    data = (_field_transfer_data *)PyArray_malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->base.free = &_field_transfer_data_free;
    data->base.clone = &_field_transfer_data_clone;
    fields = &data->fields;

    field_count = 0;
    for (i = 0; i < names_size; ++i) {
        key = PyTuple_GET_ITEM(names, i);
        tup = PyDict_GetItem(src_dtype->fields, key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &src_fld_dtype,
                                                &src_offset, &title)) {
            PyArray_free(data);
            return NPY_FAIL;
        }
        if (PyDataType_REFCHK(src_fld_dtype)) {
            if (out_needs_api) {
                *out_needs_api = 1;
            }
            if (get_decsrcref_transfer_function(0,
                                    src_stride,
                                    src_fld_dtype,
                                    &fields[field_count].stransfer,
                                    &fields[field_count].data,
                                    out_needs_api) != NPY_SUCCEED) {
                for (i = field_count-1; i >= 0; --i) {
                    NPY_AUXDATA_FREE(fields[i].data);
                }
                PyArray_free(data);
                return NPY_FAIL;
            }
            fields[field_count].src_offset = src_offset;
            fields[field_count].dst_offset = 0;
            fields[field_count].src_itemsize = src_dtype->elsize;
            field_count++;
        }
    }

    data->field_count = field_count;

    *out_stransfer = &_strided_to_strided_field_transfer;
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}


/************************* MASKED TRANSFER WRAPPER *************************/

typedef struct {
    NpyAuxData base;
    /* The transfer function being wrapped */
    PyArray_StridedUnaryOp *stransfer;
    NpyAuxData *transferdata;

    /* The src decref function if necessary */
    PyArray_StridedUnaryOp *decsrcref_stransfer;
    NpyAuxData *decsrcref_transferdata;
} _masked_wrapper_transfer_data;

/* transfer data free function */
static void _masked_wrapper_transfer_data_free(NpyAuxData *data)
{
    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)data;
    NPY_AUXDATA_FREE(d->transferdata);
    NPY_AUXDATA_FREE(d->decsrcref_transferdata);
    PyArray_free(data);
}

/* transfer data copy function */
static NpyAuxData *_masked_wrapper_transfer_data_clone(NpyAuxData *data)
{
    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)data;
    _masked_wrapper_transfer_data *newdata;

    /* Allocate the data and populate it */
    newdata = (_masked_wrapper_transfer_data *)PyArray_malloc(
                                    sizeof(_masked_wrapper_transfer_data));
    if (newdata == NULL) {
        return NULL;
    }
    memcpy(newdata, d, sizeof(_masked_wrapper_transfer_data));

    /* Clone all the owned auxdata as well */
    if (newdata->transferdata != NULL) {
        newdata->transferdata = NPY_AUXDATA_CLONE(newdata->transferdata);
        if (newdata->transferdata == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (newdata->decsrcref_transferdata != NULL) {
        newdata->decsrcref_transferdata =
                        NPY_AUXDATA_CLONE(newdata->decsrcref_transferdata);
        if (newdata->decsrcref_transferdata == NULL) {
            NPY_AUXDATA_FREE(newdata->transferdata);
            PyArray_free(newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_masked_wrapper_decsrcref_transfer_function(
                                    char *dst, npy_intp dst_stride,
                                    char *src, npy_intp src_stride,
                                    npy_bool *mask, npy_intp mask_stride,
                                    npy_intp N, npy_intp src_itemsize,
                                    NpyAuxData *transferdata)
{
    _masked_wrapper_transfer_data *d =
                        (_masked_wrapper_transfer_data *)transferdata;
    npy_intp subloopsize;
    PyArray_StridedUnaryOp *unmasked_stransfer, *decsrcref_stransfer;
    NpyAuxData *unmasked_transferdata, *decsrcref_transferdata;

    unmasked_stransfer = d->stransfer;
    unmasked_transferdata = d->transferdata;
    decsrcref_stransfer = d->decsrcref_stransfer;
    decsrcref_transferdata = d->decsrcref_transferdata;

    while (N > 0) {
        /* Skip masked values, still calling decsrcref for move_references */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 1);
        if (decsrcref_stransfer(
                NULL, 0, src, src_stride,
                subloopsize, src_itemsize, decsrcref_transferdata) < 0) {
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
        if (unmasked_stransfer(
                dst, dst_stride, src, src_stride,
                subloopsize, src_itemsize, unmasked_transferdata) < 0) {
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
                                    char *dst, npy_intp dst_stride,
                                    char *src, npy_intp src_stride,
                                    npy_bool *mask, npy_intp mask_stride,
                                    npy_intp N, npy_intp src_itemsize,
                                    NpyAuxData *transferdata)
{

    _masked_wrapper_transfer_data *d =
                            (_masked_wrapper_transfer_data *)transferdata;
    npy_intp subloopsize;
    PyArray_StridedUnaryOp *unmasked_stransfer;
    NpyAuxData *unmasked_transferdata;

    unmasked_stransfer = d->stransfer;
    unmasked_transferdata = d->transferdata;

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
        if (unmasked_stransfer(
                dst, dst_stride, src, src_stride,
                subloopsize, src_itemsize, unmasked_transferdata) < 0) {
            return -1;
        }
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
    }
    return 0;
}


/*************************** CLEAR SRC *******************************/

static int
_dec_src_ref_nop(char *NPY_UNUSED(dst),
                        npy_intp NPY_UNUSED(dst_stride),
                        char *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp NPY_UNUSED(N),
                        npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *NPY_UNUSED(data))
{
    /* NOP */
    return 0;
}

static int
_strided_to_null_dec_src_ref_reference(char *NPY_UNUSED(dst),
                        npy_intp NPY_UNUSED(dst_stride),
                        char *src, npy_intp src_stride,
                        npy_intp N,
                        npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *NPY_UNUSED(data))
{
    PyObject *src_ref = NULL;
    while (N > 0) {
        /* Release the reference in src and set it to NULL */
        NPY_DT_DBG_REFTRACE("dec src ref (null dst)", src_ref);
        memcpy(&src_ref, src, sizeof(src_ref));
        Py_XDECREF(src_ref);
        memset(src, 0, sizeof(PyObject *));

        src += src_stride;
        --N;
    }
    return 0;
}


NPY_NO_EXPORT int
get_decsrcref_transfer_function(int aligned,
                            npy_intp src_stride,
                            PyArray_Descr *src_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    /* If there are no references, it's a nop */
    if (!PyDataType_REFCHK(src_dtype)) {
        *out_stransfer = &_dec_src_ref_nop;
        *out_transferdata = NULL;

        return NPY_SUCCEED;
    }
    /* If it's a single reference, it's one decref */
    else if (src_dtype->type_num == NPY_OBJECT) {
        if (out_needs_api) {
            *out_needs_api = 1;
        }

        *out_stransfer = &_strided_to_null_dec_src_ref_reference;
        *out_transferdata = NULL;

        return NPY_SUCCEED;
    }
    /* If there are subarrays, need to wrap it */
    else if (PyDataType_HASSUBARRAY(src_dtype)) {
        PyArray_Dims src_shape = {NULL, -1};
        npy_intp src_size;
        PyArray_StridedUnaryOp *stransfer;
        NpyAuxData *data;

        if (out_needs_api) {
            *out_needs_api = 1;
        }

        if (!(PyArray_IntpConverter(src_dtype->subarray->shape,
                                            &src_shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return NPY_FAIL;
        }
        src_size = PyArray_MultiplyList(src_shape.ptr, src_shape.len);
        npy_free_cache_dim_obj(src_shape);

        /* Get a function for contiguous src of the subarray type */
        if (get_decsrcref_transfer_function(aligned,
                                src_dtype->subarray->base->elsize,
                                src_dtype->subarray->base,
                                &stransfer, &data,
                                out_needs_api) != NPY_SUCCEED) {
            return NPY_FAIL;
        }

        if (wrap_transfer_function_n_to_n(stransfer, data,
                                src_stride, 0,
                                src_dtype->subarray->base->elsize, 0,
                                src_size,
                                out_stransfer, out_transferdata) != NPY_SUCCEED) {
            NPY_AUXDATA_FREE(data);
            return NPY_FAIL;
        }

        return NPY_SUCCEED;
    }
    /* If there are fields, need to do each field */
    else {
        if (out_needs_api) {
            *out_needs_api = 1;
        }

        return get_decsrcref_fields_transfer_function(aligned,
                            src_stride, src_dtype,
                            out_stransfer,
                            out_transferdata,
                            out_needs_api);
    }
}

/********************* DTYPE COPY SWAP FUNCTION ***********************/

NPY_NO_EXPORT int
PyArray_GetDTypeCopySwapFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *dtype,
                            PyArray_StridedUnaryOp **outstransfer,
                            NpyAuxData **outtransferdata)
{
    npy_intp itemsize = dtype->elsize;

    /* If it's a custom data type, wrap its copy swap function */
    if (dtype->type_num >= NPY_NTYPES) {
        *outstransfer = NULL;
        wrap_copy_swap_function(aligned,
                            src_stride, dst_stride,
                            dtype,
                            !PyArray_ISNBO(dtype->byteorder),
                            outstransfer, outtransferdata);
    }
    /* A straight copy */
    else if (itemsize == 1 || PyArray_ISNBO(dtype->byteorder)) {
        *outstransfer = PyArray_GetStridedCopyFn(aligned,
                                    src_stride, dst_stride,
                                    itemsize);
        *outtransferdata = NULL;
    }
    else if (dtype->kind == 'U') {
        return wrap_copy_swap_function(aligned,
                                       src_stride, dst_stride, dtype, 1,
                                       outstransfer, outtransferdata);
    }
    /* If it's not complex, one swap */
    else if (dtype->kind != 'c') {
        *outstransfer = PyArray_GetStridedCopySwapFn(aligned,
                                    src_stride, dst_stride,
                                    itemsize);
        *outtransferdata = NULL;
    }
    /* If complex, a paired swap */
    else {
        *outstransfer = PyArray_GetStridedCopySwapPairFn(aligned,
                                    src_stride, dst_stride,
                                    itemsize);
        *outtransferdata = NULL;
    }

    return (*outstransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
}

/********************* MAIN DTYPE TRANSFER FUNCTION ***********************/

#if !NPY_USE_NEW_CASTINGIMPL
static int
PyArray_LegacyGetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    npy_intp src_itemsize, dst_itemsize;
    int src_type_num, dst_type_num;
    int is_builtin;

#if NPY_DT_DBG_TRACING
    printf("Calculating dtype transfer from ");
    if (PyObject_Print((PyObject *)src_dtype, stdout, 0) < 0) {
        return NPY_FAIL;
    }
    printf(" to ");
    if (PyObject_Print((PyObject *)dst_dtype, stdout, 0) < 0) {
        return NPY_FAIL;
    }
    printf("\n");
#endif

    /*
     * If one of the dtypes is NULL, we give back either a src decref
     * function or a dst setzero function
     */
    if (dst_dtype == NULL) {
        if (move_references) {
            return get_decsrcref_transfer_function(aligned,
                                src_dtype->elsize,
                                src_dtype,
                                out_stransfer, out_transferdata,
                                out_needs_api);
        }
        else {
            *out_stransfer = &_dec_src_ref_nop;
            *out_transferdata = NULL;
            return NPY_SUCCEED;
        }
    }

    src_itemsize = src_dtype->elsize;
    dst_itemsize = dst_dtype->elsize;
    src_type_num = src_dtype->type_num;
    dst_type_num = dst_dtype->type_num;
    is_builtin = src_type_num < NPY_NTYPES && dst_type_num < NPY_NTYPES;

    /* Common special case - number -> number NBO cast */
    if (PyTypeNum_ISNUMBER(src_type_num) &&
                    PyTypeNum_ISNUMBER(dst_type_num) &&
                    PyArray_ISNBO(src_dtype->byteorder) &&
                    PyArray_ISNBO(dst_dtype->byteorder)) {

        if (PyArray_EquivTypenums(src_type_num, dst_type_num)) {
            *out_stransfer = PyArray_GetStridedCopyFn(aligned,
                                        src_stride, dst_stride,
                                        src_itemsize);
            *out_transferdata = NULL;
            return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
        }
        else {
            return get_nbo_cast_numeric_transfer_function (aligned,
                                        src_stride, dst_stride,
                                        src_type_num, dst_type_num,
                                        out_stransfer, out_transferdata);
        }
    }

    /*
     * If there are no references and the data types are equivalent and builtin,
     * return a simple copy
     */
    if (PyArray_EquivTypes(src_dtype, dst_dtype) &&
            !PyDataType_REFCHK(src_dtype) && !PyDataType_REFCHK(dst_dtype) &&
            ( !PyDataType_HASFIELDS(dst_dtype) ||
              is_dtype_struct_simple_unaligned_layout(dst_dtype)) &&
            is_builtin) {
        /*
         * We can't pass through the aligned flag because it's not
         * appropriate. Consider a size-8 string, it will say it's
         * aligned because strings only need alignment 1, but the
         * copy function wants to know if it's alignment 8.
         *
         * TODO: Change align from a flag to a "best power of 2 alignment"
         *       which holds the strongest alignment value for all
         *       the data which will be used.
         */
        *out_stransfer = PyArray_GetStridedCopyFn(0,
                                        src_stride, dst_stride,
                                        src_dtype->elsize);
        *out_transferdata = NULL;
        return NPY_SUCCEED;
    }

    /* First look at the possibilities of just a copy or swap */
    if (src_itemsize == dst_itemsize && src_dtype->kind == dst_dtype->kind &&
                !PyDataType_HASFIELDS(src_dtype) &&
                !PyDataType_HASFIELDS(dst_dtype) &&
                !PyDataType_HASSUBARRAY(src_dtype) &&
                !PyDataType_HASSUBARRAY(dst_dtype) &&
                src_type_num != NPY_DATETIME && src_type_num != NPY_TIMEDELTA) {
        /* A custom data type requires that we use its copy/swap */
        if (!is_builtin) {
            /*
             * If the sizes and kinds are identical, but they're different
             * custom types, then get a cast function
             */
            if (src_type_num != dst_type_num) {
                return get_cast_transfer_function(aligned,
                                src_stride, dst_stride,
                                src_dtype, dst_dtype,
                                move_references,
                                out_stransfer, out_transferdata,
                                out_needs_api);
            }
            else {
                return wrap_copy_swap_function(aligned,
                                src_stride, dst_stride,
                                src_dtype,
                                PyArray_ISNBO(src_dtype->byteorder) !=
                                        PyArray_ISNBO(dst_dtype->byteorder),
                                out_stransfer, out_transferdata);
            }
        }

        /* The special types, which have no or subelement byte-order */
        switch (src_type_num) {
            case NPY_UNICODE:
                /* Wrap the copy swap function when swapping is necessary */
                if (PyArray_ISNBO(src_dtype->byteorder) !=
                        PyArray_ISNBO(dst_dtype->byteorder)) {
                    return wrap_copy_swap_function(aligned,
                                    src_stride, dst_stride,
                                    src_dtype, 1,
                                    out_stransfer, out_transferdata);
                }
            case NPY_VOID:
            case NPY_STRING:
                *out_stransfer = PyArray_GetStridedCopyFn(0,
                                    src_stride, dst_stride,
                                    src_itemsize);
                *out_transferdata = NULL;
                return NPY_SUCCEED;
            case NPY_OBJECT:
                if (out_needs_api) {
                    *out_needs_api = 1;
                }
                if (move_references) {
                    *out_stransfer = &_strided_to_strided_move_references;
                    *out_transferdata = NULL;
                }
                else {
                    *out_stransfer = &_strided_to_strided_copy_references;
                    *out_transferdata = NULL;
                }
                return NPY_SUCCEED;
        }

        /* This is a straight copy */
        if (src_itemsize == 1 || PyArray_ISNBO(src_dtype->byteorder) ==
                                 PyArray_ISNBO(dst_dtype->byteorder)) {
            *out_stransfer = PyArray_GetStridedCopyFn(aligned,
                                        src_stride, dst_stride,
                                        src_itemsize);
            *out_transferdata = NULL;
            return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
        }
        /* This is a straight copy + byte swap */
        else if (!PyTypeNum_ISCOMPLEX(src_type_num)) {
            *out_stransfer = PyArray_GetStridedCopySwapFn(aligned,
                                        src_stride, dst_stride,
                                        src_itemsize);
            *out_transferdata = NULL;
            return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
        }
        /* This is a straight copy + element pair byte swap */
        else {
            *out_stransfer = PyArray_GetStridedCopySwapPairFn(aligned,
                                        src_stride, dst_stride,
                                        src_itemsize);
            *out_transferdata = NULL;
            return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
        }
    }

    /* Handle subarrays */
    if (PyDataType_HASSUBARRAY(src_dtype) ||
                                PyDataType_HASSUBARRAY(dst_dtype)) {
        return get_subarray_transfer_function(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        out_stransfer, out_transferdata,
                        out_needs_api);
    }

    /* Handle fields */
    if ((PyDataType_HASFIELDS(src_dtype) || PyDataType_HASFIELDS(dst_dtype)) &&
            src_type_num != NPY_OBJECT && dst_type_num != NPY_OBJECT) {
        return get_fields_transfer_function(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        out_stransfer, out_transferdata,
                        out_needs_api);
    }

    /* Check for different-sized strings, unicodes, or voids */
    if (src_type_num == dst_type_num) {
        switch (src_type_num) {
        case NPY_UNICODE:
            if (PyArray_ISNBO(src_dtype->byteorder) !=
                                 PyArray_ISNBO(dst_dtype->byteorder)) {
                return PyArray_GetStridedZeroPadCopyFn(0, 1,
                                        src_stride, dst_stride,
                                        src_dtype->elsize, dst_dtype->elsize,
                                        out_stransfer, out_transferdata);
            }
        case NPY_STRING:
        case NPY_VOID:
            return PyArray_GetStridedZeroPadCopyFn(0, 0,
                                    src_stride, dst_stride,
                                    src_dtype->elsize, dst_dtype->elsize,
                                    out_stransfer, out_transferdata);
        }
    }

    /* Otherwise a cast is necessary */
    return get_cast_transfer_function(aligned,
                    src_stride, dst_stride,
                    src_dtype, dst_dtype,
                    move_references,
                    out_stransfer, out_transferdata,
                    out_needs_api);
}
#endif

/*
 * ********************* Generalized Multistep Cast ************************
 *
 * New general purpose multiple step cast function when resolve descriptors
 * implies that multiple cast steps are necessary.
 */
#if NPY_USE_NEW_CASTINGIMPL

/*
 * The full context passed in is never the correct context for each
 * individual cast, so we have to store each of these casts information.
 * Certain fields may be undefined (currently, the `caller`).
 */
typedef struct {
    PyArray_StridedUnaryOp *stransfer;
    NpyAuxData *auxdata;
    PyArrayMethod_Context context;
    PyArray_Descr *descriptors[2];
} _cast_info;

typedef struct {
    NpyAuxData base;
    /* Information for main cast */
    _cast_info main;
    /* Information for input preparation cast */
    _cast_info from;
    /* Information for output finalization cast */
    _cast_info to;
    char *from_buffer;
    char *to_buffer;
} _multistep_castdata;


static NPY_INLINE void
_cast_info_free(_cast_info *cast_info)
{
    NPY_AUXDATA_FREE(cast_info->auxdata);
    Py_DECREF(cast_info->descriptors[0]);
    Py_DECREF(cast_info->descriptors[1]);
    Py_DECREF(cast_info->context.method);
}


/* zero-padded data copy function */
static void
_multistep_cast_auxdata_free(NpyAuxData *auxdata)
{
    _multistep_castdata *data = (_multistep_castdata *)auxdata;
    _cast_info_free(&data->main);
    if (data->from.stransfer != NULL) {
        _cast_info_free(&data->from);
    }
    if (data->to.stransfer != NULL) {
        _cast_info_free(&data->to);
    }
    PyMem_Free(data);
}


static NpyAuxData *
_multistep_cast_auxdata_clone(NpyAuxData *auxdata_old);

static NpyAuxData *
_multistep_cast_auxdata_clone_int(NpyAuxData *auxdata_old, int move_auxdata)
{
    _multistep_castdata *castdata = (_multistep_castdata *)auxdata_old;

    /* Round up the structure size to 16-byte boundary for the buffers */
    ssize_t datasize = (sizeof(_multistep_castdata) + 15) & ~0xf;

    ssize_t from_buffer_offset = datasize;
    if (castdata->from.stransfer != NULL) {
        ssize_t src_itemsize = castdata->main.context.descriptors[0]->elsize;
        datasize += NPY_LOWLEVEL_BUFFER_BLOCKSIZE * src_itemsize;
        datasize = (datasize + 15) & ~0xf;
    }
    ssize_t to_buffer_offset = datasize;
    if (castdata->to.stransfer != NULL) {
        ssize_t dst_itemsize = castdata->main.context.descriptors[1]->elsize;
        datasize += NPY_LOWLEVEL_BUFFER_BLOCKSIZE * dst_itemsize;
    }

    char *char_data = PyMem_Malloc(datasize);
    if (char_data == NULL) {
        return NULL;
    }

    _multistep_castdata *auxdata = (_multistep_castdata *)char_data;

    /* Copy the prepared old and fix it up internal pointers */
    memcpy(char_data, castdata, sizeof(*castdata));

    auxdata->from_buffer = char_data + from_buffer_offset;
    auxdata->to_buffer = char_data + to_buffer_offset;

    auxdata->main.context.descriptors = auxdata->main.descriptors;
    auxdata->from.context.descriptors = auxdata->from.descriptors;
    auxdata->to.context.descriptors = auxdata->to.descriptors;

    auxdata->base.free = &_multistep_cast_auxdata_free;
    auxdata->base.clone = &_multistep_cast_auxdata_clone;

    /* Hold on to references and initialize buffers if necessary. */
    Py_INCREF(auxdata->main.descriptors[0]);
    Py_INCREF(auxdata->main.descriptors[1]);
    Py_INCREF(auxdata->main.context.method);

    if (!move_auxdata) {
        /* Ensure we don't free twice on error: */
        auxdata->from.auxdata = NULL;
        auxdata->to.auxdata = NULL;

        if (castdata->main.auxdata != NULL) {
            auxdata->main.auxdata = NPY_AUXDATA_CLONE(castdata->main.auxdata);
            if (auxdata->main.auxdata == NULL) {
                NPY_AUXDATA_FREE((NpyAuxData *)auxdata);
                return NULL;
            }
        }
    }
    else {
        /* Clear the original, to avoid double free. */
        castdata->main.auxdata = NULL;
        castdata->from.auxdata = NULL;
        castdata->to.auxdata = NULL;
    }

    if (castdata->from.stransfer != NULL) {
        Py_INCREF(auxdata->from.descriptors[0]);
        Py_INCREF(auxdata->from.descriptors[1]);
        Py_INCREF(auxdata->from.context.method);
        if (PyDataType_FLAGCHK(auxdata->main.descriptors[0], NPY_NEEDS_INIT)) {
            memset(auxdata->from_buffer, 0, to_buffer_offset - from_buffer_offset);
        }
        if (!move_auxdata && castdata->from.auxdata != NULL) {
            auxdata->from.auxdata = NPY_AUXDATA_CLONE(castdata->from.auxdata);
            if (auxdata->from.auxdata == NULL) {
                NPY_AUXDATA_FREE((NpyAuxData *)auxdata);
                return NULL;
            }
        }
    }
    if (castdata->to.stransfer != NULL) {
        Py_INCREF(auxdata->to.descriptors[0]);
        Py_INCREF(auxdata->to.descriptors[1]);
        Py_INCREF(auxdata->to.context.method);
        if (PyDataType_FLAGCHK(auxdata->main.descriptors[1], NPY_NEEDS_INIT)) {
            memset(auxdata->to_buffer, 0, datasize - to_buffer_offset);
        }
        if (!move_auxdata && castdata->to.auxdata != NULL) {
            auxdata->to.auxdata = NPY_AUXDATA_CLONE(castdata->to.auxdata);
            if (auxdata->to.auxdata == NULL) {
                NPY_AUXDATA_FREE((NpyAuxData *)auxdata);
                return NULL;
            }
        }
    }

    return (NpyAuxData *)auxdata;
}

static NpyAuxData *
_multistep_cast_auxdata_clone(NpyAuxData *auxdata_old)
{
    return _multistep_cast_auxdata_clone_int(auxdata_old, 0);
}


static int
_strided_to_strided_multistep_cast(
        char *dst, npy_intp dst_stride,
        char *src, npy_intp src_stride,
        npy_intp N, npy_intp src_itemsize,
        NpyAuxData *data)
{
    _multistep_castdata *castdata = (_multistep_castdata *)data;

    char *main_src, *main_dst;
    npy_intp main_src_stride, main_dst_stride, main_src_itemsize;

    npy_intp block_size = NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
    while (N > 0) {
        if (block_size > N) {
            block_size = N;
        }

        if (castdata->from.stransfer != NULL) {
            npy_intp out_stride = castdata->from.descriptors[1]->elsize;
            if (castdata->from.stransfer(
                    castdata->from_buffer, out_stride, src, src_stride,
                    block_size, src_itemsize, castdata->from.auxdata)) {
                /* TODO: Internal buffer may require cleanup on error. */
                return -1;
            }
            main_src = castdata->from_buffer;
            main_src_stride = out_stride;
            main_src_itemsize = out_stride;
        }
        else {
            main_src = src;
            main_src_stride = src_stride;
            main_src_itemsize = src_itemsize;
        }

        if (castdata->to.stransfer != NULL) {
            main_dst = castdata->to_buffer;
            main_dst_stride = castdata->main.descriptors[1]->elsize;
        }
        else {
            main_dst = dst;
            main_dst_stride = dst_stride;
        }

        if (castdata->main.stransfer(
                main_dst, main_dst_stride, main_src, main_src_stride,
                block_size, main_src_itemsize, castdata->main.auxdata)) {
            /* TODO: Internal buffer may require cleanup on error. */
            return -1;
        }

        if (castdata->to.stransfer != NULL) {
            if (castdata->to.stransfer(
                    dst, dst_stride, main_dst, main_dst_stride,
                    block_size, main_dst_stride, castdata->to.auxdata)) {
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
static NPY_INLINE int
init_cast_info(_cast_info *cast_info, NPY_CASTING *casting,
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
    cast_info->context.caller = NULL;
    cast_info->stransfer = NULL;
    cast_info->auxdata = NULL;

    cast_info->context.method = (PyArrayMethodObject *)meth;
    cast_info->context.descriptors = cast_info->descriptors;

    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(src_dtype), NPY_DTYPE(dst_dtype)};
    PyArray_Descr *in_descr[2] = {src_dtype, dst_dtype};

    *casting = cast_info->context.method->resolve_descriptors(
            cast_info->context.method, dtypes, in_descr, cast_info->descriptors);
    if (NPY_UNLIKELY(*casting < 0)) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError,
                    "Cannot cast data from %S to %S.", src_dtype, dst_dtype);
            Py_DECREF(meth);
            return -1;
        }
    }

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
        _cast_info_free(cast_info);
        return -1;
    }

    return 0;
}


/*
 * Helper for PyArray_GetDTypeTransferFunction, which fetches a single
 * transfer function from the each casting implementation (ArrayMethod).
 * May set the transfer function to NULL when the cast can be achieved using
 * a view.
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
get_transferfunction_for_descrs(
        int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArray_StridedUnaryOp **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api)
{
    *out_transferdata = NULL;  /* ensure NULL on error */
    /* Storage for all cast info in case multi-step casting is necessary */
    _multistep_castdata castdata;
    /* Initialize secondary `stransfer` to indicate whether they are used: */
    castdata.to.stransfer = NULL;
    castdata.from.stransfer = NULL;
    NPY_CASTING casting = -1;
    int res = -1;

    if (init_cast_info(&castdata.main, &casting, src_dtype, dst_dtype, 1) < 0) {
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
        (castdata.main.context.method->flags & NPY_METH_SUPPORTS_UNALIGNED) == 0);

    /*
     * Wrap the input with an additional cast if necessary.
     */
    if (NPY_UNLIKELY(src_dtype != castdata.main.descriptors[0] || must_wrap)) {
        NPY_CASTING from_casting = -1;
        /* Cast function may not support the input, wrap if necessary */
        if (init_cast_info(
                &castdata.from, &from_casting,
                src_dtype, castdata.main.descriptors[0], 0) < 0) {
            goto fail;
        }
        casting = PyArray_MinCastSafety(casting, from_casting);

        /* Prepare the actual cast (if necessary): */
        if (from_casting & _NPY_CAST_IS_VIEW && !must_wrap) {
            /* This step is not necessary and can be skipped. */
            _cast_info_free(&castdata.from);
        }
        else {
            /* Fetch the cast function and set up */
            PyArrayMethod_Context *context = &castdata.from.context;
            npy_intp strides[2] = {src_stride, castdata.main.descriptors[0]->elsize};
            NPY_ARRAYMETHOD_FLAGS flags;
            if (context->method->get_strided_loop(
                    context, aligned, move_references, strides,
                    &castdata.from.stransfer, &castdata.from.auxdata, &flags) < 0) {
                assert(castdata.from.auxdata != NULL);
                _cast_info_free(&castdata.from);
                castdata.from.stransfer = NULL;  /* ensure we cleanup once */
                goto fail;
            }
            assert(castdata.from.stransfer != NULL);

            *out_needs_api |= (flags & NPY_METH_REQUIRES_PYAPI) != 0;
            /* The main cast now uses a buffered input: */
            src_stride = strides[1];
            move_references = 1;  /* main cast has to clear the buffer */
        }
    }
    /*
     * Wrap the output with an additional cast if necessary.
     */
    if (NPY_UNLIKELY(dst_dtype != castdata.main.descriptors[1] || must_wrap)) {
        NPY_CASTING to_casting = -1;
        /* Cast function may not support the output, wrap if necessary */
        if (init_cast_info(
                &castdata.to, &to_casting,
                castdata.main.descriptors[1], dst_dtype,  0) < 0) {
            goto fail;
        }
        casting = PyArray_MinCastSafety(casting, to_casting);

        /* Prepare the actual cast (if necessary): */
        if (to_casting & _NPY_CAST_IS_VIEW && !must_wrap) {
            /* This step is not necessary and can be skipped. */
            _cast_info_free(&castdata.to);
        }
        else {
            /* Fetch the cast function and set up */
            PyArrayMethod_Context *context = &castdata.to.context;
            npy_intp strides[2] = {castdata.main.descriptors[1]->elsize, dst_stride};
            NPY_ARRAYMETHOD_FLAGS flags;
            if (context->method->get_strided_loop(
                    context, aligned, 1 /* clear buffer */, strides,
                    &castdata.to.stransfer, &castdata.to.auxdata, &flags) < 0) {
                assert(castdata.to.auxdata != NULL);
                _cast_info_free(&castdata.to);
                castdata.to.stransfer = NULL;  /* ensure we cleanup once */
                goto fail;
            }
            assert(castdata.to.stransfer != NULL);

            *out_needs_api |= (flags & NPY_METH_REQUIRES_PYAPI) != 0;
            /* The main cast now uses a buffered input: */
            dst_stride = strides[0];
            if (castdata.from.stransfer != NULL) {
                /* Both input and output are wrapped, now always aligned */
                aligned = 1;
            }
        }
    }

    /* Fetch the main cast function (with updated values) */
    PyArrayMethod_Context *context = &castdata.main.context;
    npy_intp strides[2] = {src_stride, dst_stride};
    NPY_ARRAYMETHOD_FLAGS flags;
    if (context->method->get_strided_loop(
            context, aligned, move_references, strides,
            &castdata.main.stransfer, &castdata.main.auxdata, &flags) < 0) {
        goto fail;
    }

    *out_needs_api |= (flags & NPY_METH_REQUIRES_PYAPI) != 0;

    if (castdata.from.stransfer == NULL && castdata.to.stransfer == NULL) {
        /* The main step is sufficient to do the cast */
        *out_stransfer = castdata.main.stransfer;
        *out_transferdata = castdata.main.auxdata;
        castdata.main.auxdata = NULL;  /* do not free the auxdata */
        _cast_info_free(&castdata.main);
        return 0;
    }

    /* Clone the castdata as it is currently not persistently stored. */
    *out_transferdata = _multistep_cast_auxdata_clone_int(
            (NpyAuxData *)&castdata, 1);
    if (*out_transferdata == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    *out_stransfer = &_strided_to_strided_multistep_cast;

    res = 0;  /* success */

  fail:
    _cast_info_free(&castdata.main);
    if (castdata.from.stransfer != NULL) {
        _cast_info_free(&castdata.from);
    }
    if (castdata.to.stransfer != NULL) {
        _cast_info_free(&castdata.to);
    }
    return res;
}
#endif


NPY_NO_EXPORT int
PyArray_GetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    assert(src_dtype != NULL);

#if NPY_USE_NEW_CASTINGIMPL
    /*
     * If one of the dtypes is NULL, we give back either a src decref
     * function or a dst setzero function
     *
     * TODO: Eventually, we may wish to support user dtype with references
     *       (including and beyond bare `PyObject *` this may require extending
     *       the ArrayMethod API and those paths should likely be split out
     *       from this function.)
     */
    if (dst_dtype == NULL) {
        if (move_references) {
            return get_decsrcref_transfer_function(aligned,
                                src_dtype->elsize,
                                src_dtype,
                                out_stransfer, out_transferdata,
                                out_needs_api);
        }
        else {
            *out_stransfer = &_dec_src_ref_nop;
            *out_transferdata = NULL;
            return NPY_SUCCEED;
        }
    }

    if (get_transferfunction_for_descrs(aligned,
            src_stride, dst_stride,
            src_dtype, dst_dtype, move_references,
            out_stransfer, out_transferdata, out_needs_api) < 0) {
        return NPY_FAIL;
    }

    return NPY_SUCCEED;

#else
    return PyArray_LegacyGetDTypeTransferFunction(
            aligned, src_stride, dst_stride, src_dtype, dst_dtype,
            move_references, out_stransfer, out_transferdata, out_needs_api);
#endif
}


/*
 * Basic version of PyArray_GetDTypeTransferFunction for legacy dtype
 * support.
 * It supports only wrapping the copyswapn functions and the legacy
 * cast functions registered with `PyArray_RegisterCastFunc`.
 * This function takes the easy way out: It does not wrap, so if wrapping
 * might be necessary due to unaligned data, the user has to ensure that
 * this is done and aligned is passed in as True (this is asserted only).
 */
NPY_NO_EXPORT int
PyArray_GetLegacyDTypeTransferFunction(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArray_StridedUnaryOp **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api, int wrap_if_unaligned)
{
    /* Note: We ignore `needs_wrap`; needs-wrap is handled by another cast */
    int needs_wrap = 0;

    if (src_dtype->type_num == dst_dtype->type_num) {
        /*
         * This is a cast within the same dtype. For legacy user-dtypes,
         * it is always valid to handle this using the copy swap function.
         */
        return wrap_copy_swap_function(aligned,
                src_stride, dst_stride,
                src_dtype,
                PyArray_ISNBO(src_dtype->byteorder) !=
                PyArray_ISNBO(dst_dtype->byteorder),
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
    if (NPY_UNLIKELY(!wrap_if_unaligned)) {
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
        return -1;
    }

    /*
     * If we are here, use the legacy code to wrap the above cast (which
     * does not support unaligned data) into copyswapn.
     */
    NpyAuxData *castdata = *out_transferdata;
    *out_transferdata = NULL;
    if (wrap_aligned_contig_transfer_function_with_copyswapn(
                aligned, src_stride, dst_stride, src_dtype, dst_dtype,
                out_stransfer, out_transferdata, out_needs_api,
                *out_stransfer, castdata) == NPY_FAIL) {
        NPY_AUXDATA_FREE(castdata);
        return -1;
    }
    return 0;
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
                            PyArray_MaskedStridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    _masked_wrapper_transfer_data *data;

    /* TODO: Add struct-based mask_dtype support later */
    if (mask_dtype->type_num != NPY_BOOL &&
                            mask_dtype->type_num != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError,
                "Only bool and uint8 masks are supported at the moment, "
                "structs of bool/uint8 is planned for the future");
        return NPY_FAIL;
    }

    /* TODO: Special case some important cases so they're fast */

    /* Fall back to wrapping a non-masked transfer function */
    assert(dst_dtype != NULL);
    if (PyArray_GetDTypeTransferFunction(aligned,
                                src_stride, dst_stride,
                                src_dtype, dst_dtype,
                                move_references,
                                &stransfer, &transferdata,
                                out_needs_api) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    /* Create the wrapper function's auxdata */
    data = (_masked_wrapper_transfer_data *)PyArray_malloc(
                            sizeof(_masked_wrapper_transfer_data));
    if (data == NULL) {
        PyErr_NoMemory();
        NPY_AUXDATA_FREE(transferdata);
        return NPY_FAIL;
    }

    /* Fill in the auxdata object */
    memset(data, 0, sizeof(_masked_wrapper_transfer_data));
    data->base.free = &_masked_wrapper_transfer_data_free;
    data->base.clone = &_masked_wrapper_transfer_data_clone;

    data->stransfer = stransfer;
    data->transferdata = transferdata;

    /* If the src object will need a DECREF, get a function to handle that */
    if (move_references && PyDataType_REFCHK(src_dtype)) {
        if (get_decsrcref_transfer_function(aligned,
                            src_stride,
                            src_dtype,
                            &data->decsrcref_stransfer,
                            &data->decsrcref_transferdata,
                            out_needs_api) != NPY_SUCCEED) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }

        *out_stransfer = &_strided_masked_wrapper_decsrcref_transfer_function;
    }
    else {
        *out_stransfer = &_strided_masked_wrapper_transfer_function;
    }

    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}

NPY_NO_EXPORT int
PyArray_CastRawArrays(npy_intp count,
                      char *src, char *dst,
                      npy_intp src_stride, npy_intp dst_stride,
                      PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                      int move_references)
{
    PyArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned = 1, needs_api = 0;

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
    if (PyArray_GetDTypeTransferFunction(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    /* Cast */
    stransfer(dst, dst_stride, src, src_stride, count,
                src_dtype->elsize, transferdata);

    /* Cleanup */
    NPY_AUXDATA_FREE(transferdata);

    /* If needs_api was set to 1, it may have raised a Python exception */
    return (needs_api && PyErr_Occurred()) ? NPY_FAIL : NPY_SUCCEED;
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
