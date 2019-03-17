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
#include <numpy/npy_cpu.h>

#include "npy_pycompat.h"

#include "convert_datatype.h"
#include "ctors.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "descriptor.h"
#include "array_assign.h"

#include "shape.h"
#include "lowlevel_strided_loops.h"
#include "alloc.h"

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

/*
 * Returns a transfer function which zeros out the dest values.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
static int
get_setdstzero_transfer_function(int aligned,
                            npy_intp dst_stride,
                            PyArray_Descr *dst_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api);

/*
 * Returns a transfer function which sets a boolean type to ones.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
get_bool_setdstone_transfer_function(npy_intp dst_stride,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *NPY_UNUSED(out_needs_api));

/*************************** COPY REFERENCES *******************************/

/* Moves references from src to dst */
static void
_strided_to_strided_move_references(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    PyObject *src_ref = NULL, *dst_ref = NULL;
    while (N > 0) {
        NPY_COPY_PYOBJECT_PTR(&src_ref, src);
        NPY_COPY_PYOBJECT_PTR(&dst_ref, dst);

        /* Release the reference in dst */
        NPY_DT_DBG_REFTRACE("dec dst ref", dst_ref);
        Py_XDECREF(dst_ref);
        /* Move the reference */
        NPY_DT_DBG_REFTRACE("move src ref", src_ref);
        NPY_COPY_PYOBJECT_PTR(dst, &src_ref);
        /* Set the source reference to NULL */
        src_ref = NULL;
        NPY_COPY_PYOBJECT_PTR(src, &src_ref);

        src += src_stride;
        dst += dst_stride;
        --N;
    }
}

/* Copies references from src to dst */
static void
_strided_to_strided_copy_references(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    PyObject *src_ref = NULL, *dst_ref = NULL;
    while (N > 0) {
        NPY_COPY_PYOBJECT_PTR(&src_ref, src);
        NPY_COPY_PYOBJECT_PTR(&dst_ref, dst);

        /* Copy the reference */
        NPY_DT_DBG_REFTRACE("copy src ref", src_ref);
        NPY_COPY_PYOBJECT_PTR(dst, &src_ref);
        /* Claim the reference */
        Py_XINCREF(src_ref);
        /* Release the reference in dst */
        NPY_DT_DBG_REFTRACE("dec dst ref", dst_ref);
        Py_XDECREF(dst_ref);

        src += src_stride;
        dst += dst_stride;
        --N;
    }
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
static void
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
}

/*
 * Does a strided to strided zero-padded copy for the case where
 * dst_itemsize < src_itemsize
 */
static void
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
}

/*
 * Does a strided to strided zero-padded or truncated copy for the case where
 * unicode swapping is needed.
 */
static void
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

    return (NpyAuxData *)newdata;
}

static void
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

    for(;;) {
        if (N > NPY_LOWLEVEL_BUFFER_BLOCKSIZE) {
            tobuffer(bufferin, inner_src_itemsize, src, src_stride,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    src_itemsize, todata);
            wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    inner_src_itemsize, wrappeddata);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    dst_itemsize, fromdata);
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            tobuffer(bufferin, inner_src_itemsize, src, src_stride, N,
                                            src_itemsize, todata);
            wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize, N,
                                            inner_src_itemsize, wrappeddata);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize, N,
                                            dst_itemsize, fromdata);
            return;
        }
    }
}

static void
_strided_to_strided_contig_align_wrap_init_dest(char *dst, npy_intp dst_stride,
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

    for(;;) {
        if (N > NPY_LOWLEVEL_BUFFER_BLOCKSIZE) {
            tobuffer(bufferin, inner_src_itemsize, src, src_stride,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    src_itemsize, todata);
            memset(bufferout, 0, dst_itemsize*NPY_LOWLEVEL_BUFFER_BLOCKSIZE);
            wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    inner_src_itemsize, wrappeddata);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    dst_itemsize, fromdata);
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            tobuffer(bufferin, inner_src_itemsize, src, src_stride, N,
                                            src_itemsize, todata);
            memset(bufferout, 0, dst_itemsize*N);
            wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize, N,
                                            inner_src_itemsize, wrappeddata);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize, N,
                                            dst_itemsize, fromdata);
            return;
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

    /* Set the function and data */
    if (init_dest) {
        *out_stransfer = &_strided_to_strided_contig_align_wrap_init_dest;
    }
    else {
        *out_stransfer = &_strided_to_strided_contig_align_wrap;
    }
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

static void
_strided_to_strided_wrap_copy_swap(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data)
{
    _wrap_copy_swap_data *d = (_wrap_copy_swap_data *)data;

    d->copyswapn(dst, dst_stride, src, src_stride, N, d->swap, d->arr);
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

static void
_aligned_strided_to_strided_cast(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_cast_data *d = (_strided_cast_data *)data;
    PyArray_VectorUnaryFunc *castfunc = d->castfunc;
    PyArrayObject *aip = d->aip, *aop = d->aop;

    while (N > 0) {
        castfunc(src, dst, 1, aip, aop);
        dst += dst_stride;
        src += src_stride;
        --N;
    }
}

/* This one requires src be of type NPY_OBJECT */
static void
_aligned_strided_to_strided_cast_decref_src(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data)
{
    _strided_cast_data *d = (_strided_cast_data *)data;
    PyArray_VectorUnaryFunc *castfunc = d->castfunc;
    PyArrayObject *aip = d->aip, *aop = d->aop;
    PyObject *src_ref;

    while (N > 0) {
        castfunc(src, dst, 1, aip, aop);

        /* After casting, decrement the source ref */
        NPY_COPY_PYOBJECT_PTR(&src_ref, src);
        NPY_DT_DBG_REFTRACE("dec src ref (cast object -> not object)", src_ref);
        Py_XDECREF(src_ref);

        dst += dst_stride;
        src += src_stride;
        --N;
    }
}

static void
_aligned_contig_to_contig_cast(char *dst, npy_intp NPY_UNUSED(dst_stride),
                        char *src, npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(itemsize),
                        NpyAuxData *data)
{
    _strided_cast_data *d = (_strided_cast_data *)data;

    d->castfunc(src, dst, N, d->aip, d->aop);
}

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

static void
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
            dt = NPY_DATETIME_NAT;
        }
        else {
            if (convert_datetimestruct_to_datetime(&d->dst_meta,
                                                   &dts, &dt) < 0) {
                dt = NPY_DATETIME_NAT;
            }
        }

        memcpy(dst, &dt, sizeof(dt));

        dst += dst_stride;
        src += src_stride;
        --N;
    }
}

static void
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
}

static void
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
}

static void
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
            /* For an error, produce a 'NaT' string */
            dts.year = NPY_DATETIME_NAT;
        }

        /* Initialize the destination to all zeros */
        memset(dst, 0, dst_itemsize);

        /*
         * This may also raise an error, but the caller needs
         * to use PyErr_Occurred().
         */
        make_iso_8601_datetime(&dts, dst, dst_itemsize,
                                0, 0, d->src_meta.base, -1,
                                NPY_UNSAFE_CASTING);

        dst += dst_stride;
        src += src_stride;
        --N;
    }
}

static void
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
                dt = NPY_DATETIME_NAT;
            }
        }
        /* Otherwise parse the data in place */
        else {
            if (parse_iso_8601_datetime(src, tmp - src,
                                    d->dst_meta.base, NPY_SAME_KIND_CASTING,
                                    &dts, NULL, NULL) < 0) {
                dt = NPY_DATETIME_NAT;
            }
        }

        /* Convert to the datetime */
        if (dt != NPY_DATETIME_NAT &&
                convert_datetimestruct_to_datetime(&d->dst_meta,
                                               &dts, &dt) < 0) {
            dt = NPY_DATETIME_NAT;
        }

        memcpy(dst, &dt, sizeof(dt));

        dst += dst_stride;
        src += src_stride;
        --N;
    }
}

/*
 * Assumes src_dtype and dst_dtype are both datetimes or both timedeltas
 */
static int
get_nbo_cast_datetime_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
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

static int
get_nbo_datetime_to_string_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata)
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

static int
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
    str_dtype = PyArray_DescrFromType(NPY_STRING);
    str_dtype = PyArray_AdaptFlexibleDType(NULL, dst_dtype, str_dtype);
    if (str_dtype == NULL) {
        return NPY_FAIL;
    }

    /* Get the copy/swap operation to dst */
    if (PyArray_GetDTypeCopySwapFn(aligned,
                            src_stride, src_dtype->elsize,
                            src_dtype,
                            &tobuffer, &todata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        return NPY_FAIL;
    }

    /* Get the NBO datetime to string aligned contig function */
    if (get_nbo_datetime_to_string_transfer_function(1,
                            src_dtype->elsize, str_dtype->elsize,
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
                        out_stransfer, out_transferdata) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE(castdata);
        NPY_AUXDATA_FREE(todata);
        NPY_AUXDATA_FREE(fromdata);
        return NPY_FAIL;
    }

    Py_DECREF(str_dtype);

    return NPY_SUCCEED;
}

static int
get_nbo_string_to_datetime_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata)
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

static int
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
    str_dtype = PyArray_DescrFromType(NPY_STRING);
    str_dtype = PyArray_AdaptFlexibleDType(NULL, src_dtype, str_dtype);
    if (str_dtype == NULL) {
        return NPY_FAIL;
    }

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
    if (get_nbo_string_to_datetime_transfer_function(1,
                            str_dtype->elsize, dst_dtype->elsize,
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
    _strided_cast_data *data;
    PyArray_VectorUnaryFunc *castfunc;
    PyArray_Descr *tmp_dtype;
    npy_intp shape = 1, src_itemsize = src_dtype->elsize,
            dst_itemsize = dst_dtype->elsize;

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
                                        src_stride, dst_stride,
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
                                        aligned,
                                        src_stride, dst_stride,
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
                                        aligned,
                                        src_stride, dst_stride,
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
    NpyAuxData *castdata, *todata = NULL, *fromdata = NULL;
    int needs_wrap = 0;
    npy_intp src_itemsize = src_dtype->elsize,
            dst_itemsize = dst_dtype->elsize;

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
        PyArray_StridedUnaryOp *tobuffer, *frombuffer;

        /* Get the copy/swap operation from src */
        PyArray_GetDTypeCopySwapFn(aligned,
                                src_stride, src_itemsize,
                                src_dtype,
                                &tobuffer, &todata);

        if (!PyDataType_REFCHK(dst_dtype)) {
            /* Copying from buffer is a simple copy/swap operation */
            PyArray_GetDTypeCopySwapFn(aligned,
                                    dst_itemsize, dst_stride,
                                    dst_dtype,
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
                            out_stransfer, out_transferdata) != NPY_SUCCEED) {
            NPY_AUXDATA_FREE(castdata);
            NPY_AUXDATA_FREE(todata);
            NPY_AUXDATA_FREE(fromdata);
            return NPY_FAIL;
        }

        return NPY_SUCCEED;
    }
}

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

static void
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
        subtransfer(dst, dst_itemsize,
                    src, 0,
                    subN, src_itemsize,
                    subdata);

        src += src_stride;
        dst += dst_stride;
        --N;
    }
}

static void
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
        subtransfer(dst, dst_itemsize,
                    src, 0,
                    subN, src_itemsize,
                    subdata);


        stransfer_finish_src(NULL, 0,
                            src, 0,
                            1, src_itemsize,
                            data_finish_src);

        src += src_stride;
        dst += dst_stride;
        --N;
    }
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

static void
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
        subtransfer(dst, dst_subitemsize,
                    src, src_subitemsize,
                    subN, src_subitemsize,
                    subdata);

        src += src_stride;
        dst += dst_stride;
        --N;
    }
}

static void
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

    subtransfer(dst, dst_subitemsize,
                src, src_subitemsize,
                subN*N, src_subitemsize,
                subdata);
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

static void
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
                subtransfer(dst_ptr, dst_subitemsize,
                            src + offset, src_subitemsize,
                            count, src_subitemsize,
                            subdata);
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
}


static void
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
                subtransfer(dst_ptr, dst_subitemsize,
                            src + offset, src_subitemsize,
                            count, src_subitemsize,
                            subdata);
            }
            else {
                if (stransfer_decdstref != NULL) {
                    stransfer_decdstref(NULL, 0, dst_ptr, dst_subitemsize,
                                        count, dst_subitemsize,
                                        data_decdstref);
                }
                memset(dst_ptr, 0, count*dst_subitemsize);
            }
            loop_index += count;
        }

        if (stransfer_decsrcref != NULL) {
            stransfer_decsrcref(NULL, 0, src, src_subitemsize,
                                    src_subN, src_subitemsize,
                                    data_decsrcref);
        }

        src += src_stride;
        dst += dst_stride;
        --N;
    }
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
static int
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

static void
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
                field->stransfer(dst + field->dst_offset, dst_stride,
                                 src + field->src_offset, src_stride,
                                 NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                 field->src_itemsize,
                                 field->data);
            }
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            for (i = 0; i < field_count; ++i, ++field) {
                field->stransfer(dst + field->dst_offset, dst_stride,
                                 src + field->src_offset, src_stride,
                                 N,
                                 field->src_itemsize,
                                 field->data);
            }
            return;
        }
    }
}

/*
 * Handles fields transfer.  To call this, at least one of the dtypes
 * must have fields. Does not take care of object<->structure conversion
 */
static int
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

static int
get_setdestzero_fields_transfer_function(int aligned,
                            npy_intp dst_stride,
                            PyArray_Descr *dst_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyObject *names, *key, *tup, *title;
    PyArray_Descr *dst_fld_dtype;
    npy_int i, names_size, field_count, structsize;
    int dst_offset;
    _field_transfer_data *data;
    _single_field_transfer *fields;

    names = dst_dtype->names;
    names_size = PyTuple_GET_SIZE(dst_dtype->names);

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

    for (i = 0; i < names_size; ++i) {
        key = PyTuple_GET_ITEM(names, i);
        tup = PyDict_GetItem(dst_dtype->fields, key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &dst_fld_dtype,
                                                &dst_offset, &title)) {
            PyArray_free(data);
            return NPY_FAIL;
        }
        if (get_setdstzero_transfer_function(0,
                                dst_stride,
                                dst_fld_dtype,
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
        fields[i].src_itemsize = 0;
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

static void _strided_masked_wrapper_decsrcref_transfer_function(
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
        decsrcref_stransfer(NULL, 0, src, src_stride,
                            subloopsize, src_itemsize, decsrcref_transferdata);
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
        if (N <= 0) {
            break;
        }

        /* Process unmasked values */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 0);
        unmasked_stransfer(dst, dst_stride, src, src_stride,
                            subloopsize, src_itemsize, unmasked_transferdata);
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
    }
}

static void _strided_masked_wrapper_transfer_function(
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
        unmasked_stransfer(dst, dst_stride, src, src_stride,
                            subloopsize, src_itemsize, unmasked_transferdata);
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
    }
}


/************************* DEST BOOL SETONE *******************************/

static void
_null_to_strided_set_bool_one(char *dst,
                        npy_intp dst_stride,
                        char *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *NPY_UNUSED(data))
{
    /* bool type is one byte, so can just use the char */

    while (N > 0) {
        *dst = 1;

        dst += dst_stride;
        --N;
    }
}

static void
_null_to_contig_set_bool_one(char *dst,
                        npy_intp NPY_UNUSED(dst_stride),
                        char *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *NPY_UNUSED(data))
{
    /* bool type is one byte, so can just use the char */

    memset(dst, 1, N);
}

/* Only for the bool type, sets the destination to 1 */
NPY_NO_EXPORT int
get_bool_setdstone_transfer_function(npy_intp dst_stride,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *NPY_UNUSED(out_needs_api))
{
    if (dst_stride == 1) {
        *out_stransfer = &_null_to_contig_set_bool_one;
    }
    else {
        *out_stransfer = &_null_to_strided_set_bool_one;
    }
    *out_transferdata = NULL;

    return NPY_SUCCEED;
}

/*************************** DEST SETZERO *******************************/

/* Sets dest to zero */
typedef struct {
    NpyAuxData base;
    npy_intp dst_itemsize;
} _dst_memset_zero_data;

/* zero-padded data copy function */
static NpyAuxData *_dst_memset_zero_data_clone(NpyAuxData *data)
{
    _dst_memset_zero_data *newdata =
            (_dst_memset_zero_data *)PyArray_malloc(
                                    sizeof(_dst_memset_zero_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_dst_memset_zero_data));

    return (NpyAuxData *)newdata;
}

static void
_null_to_strided_memset_zero(char *dst,
                        npy_intp dst_stride,
                        char *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data)
{
    _dst_memset_zero_data *d = (_dst_memset_zero_data *)data;
    npy_intp dst_itemsize = d->dst_itemsize;

    while (N > 0) {
        memset(dst, 0, dst_itemsize);
        dst += dst_stride;
        --N;
    }
}

static void
_null_to_contig_memset_zero(char *dst,
                        npy_intp dst_stride,
                        char *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data)
{
    _dst_memset_zero_data *d = (_dst_memset_zero_data *)data;
    npy_intp dst_itemsize = d->dst_itemsize;

    memset(dst, 0, N*dst_itemsize);
}

static void
_null_to_strided_reference_setzero(char *dst,
                        npy_intp dst_stride,
                        char *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *NPY_UNUSED(data))
{
    PyObject *dst_ref = NULL;

    while (N > 0) {
        NPY_COPY_PYOBJECT_PTR(&dst_ref, dst);

        /* Release the reference in dst */
        NPY_DT_DBG_REFTRACE("dec dest ref (to set zero)", dst_ref);
        Py_XDECREF(dst_ref);

        /* Set it to zero */
        dst_ref = NULL;
        NPY_COPY_PYOBJECT_PTR(dst, &dst_ref);

        dst += dst_stride;
        --N;
    }
}

NPY_NO_EXPORT int
get_setdstzero_transfer_function(int aligned,
                            npy_intp dst_stride,
                            PyArray_Descr *dst_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    _dst_memset_zero_data *data;

    /* If there are no references, just set the whole thing to zero */
    if (!PyDataType_REFCHK(dst_dtype)) {
        data = (_dst_memset_zero_data *)
                        PyArray_malloc(sizeof(_dst_memset_zero_data));
        if (data == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }

        data->base.free = (NpyAuxData_FreeFunc *)(&PyArray_free);
        data->base.clone = &_dst_memset_zero_data_clone;
        data->dst_itemsize = dst_dtype->elsize;

        if (dst_stride == data->dst_itemsize) {
            *out_stransfer = &_null_to_contig_memset_zero;
        }
        else {
            *out_stransfer = &_null_to_strided_memset_zero;
        }
        *out_transferdata = (NpyAuxData *)data;
    }
    /* If it's exactly one reference, use the decref function */
    else if (dst_dtype->type_num == NPY_OBJECT) {
        if (out_needs_api) {
            *out_needs_api = 1;
        }

        *out_stransfer = &_null_to_strided_reference_setzero;
        *out_transferdata = NULL;
    }
    /* If there are subarrays, need to wrap it */
    else if (PyDataType_HASSUBARRAY(dst_dtype)) {
        PyArray_Dims dst_shape = {NULL, -1};
        npy_intp dst_size = 1;
        PyArray_StridedUnaryOp *contig_stransfer;
        NpyAuxData *contig_data;

        if (out_needs_api) {
            *out_needs_api = 1;
        }

        if (!(PyArray_IntpConverter(dst_dtype->subarray->shape,
                                            &dst_shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return NPY_FAIL;
        }
        dst_size = PyArray_MultiplyList(dst_shape.ptr, dst_shape.len);
        npy_free_cache_dim_obj(dst_shape);

        /* Get a function for contiguous dst of the subarray type */
        if (get_setdstzero_transfer_function(aligned,
                                dst_dtype->subarray->base->elsize,
                                dst_dtype->subarray->base,
                                &contig_stransfer, &contig_data,
                                out_needs_api) != NPY_SUCCEED) {
            return NPY_FAIL;
        }

        if (wrap_transfer_function_n_to_n(contig_stransfer, contig_data,
                            0, dst_stride,
                            0, dst_dtype->subarray->base->elsize,
                            dst_size,
                            out_stransfer, out_transferdata) != NPY_SUCCEED) {
            NPY_AUXDATA_FREE(contig_data);
            return NPY_FAIL;
        }
    }
    /* If there are fields, need to do each field */
    else if (PyDataType_HASFIELDS(dst_dtype)) {
        if (out_needs_api) {
            *out_needs_api = 1;
        }

        return get_setdestzero_fields_transfer_function(aligned,
                            dst_stride, dst_dtype,
                            out_stransfer,
                            out_transferdata,
                            out_needs_api);
    }

    return NPY_SUCCEED;
}

static void
_dec_src_ref_nop(char *NPY_UNUSED(dst),
                        npy_intp NPY_UNUSED(dst_stride),
                        char *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp NPY_UNUSED(N),
                        npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *NPY_UNUSED(data))
{
    /* NOP */
}

static void
_strided_to_null_dec_src_ref_reference(char *NPY_UNUSED(dst),
                        npy_intp NPY_UNUSED(dst_stride),
                        char *src, npy_intp src_stride,
                        npy_intp N,
                        npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *NPY_UNUSED(data))
{
    PyObject *src_ref = NULL;
    while (N > 0) {
        NPY_COPY_PYOBJECT_PTR(&src_ref, src);

        /* Release the reference in src */
        NPY_DT_DBG_REFTRACE("dec src ref (null dst)", src_ref);
        Py_XDECREF(src_ref);

        src += src_stride;
        --N;
    }
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
        npy_intp src_size = 1;
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

NPY_NO_EXPORT int
PyArray_GetDTypeTransferFunction(int aligned,
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
    else if (src_dtype == NULL) {
        return get_setdstzero_transfer_function(aligned,
                                dst_dtype->elsize,
                                dst_dtype,
                                out_stransfer, out_transferdata,
                                out_needs_api);
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
PyArray_PrepareOneRawArrayIter(int ndim, npy_intp *shape,
                            char *data, npy_intp *strides,
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
PyArray_PrepareTwoRawArrayIter(int ndim, npy_intp *shape,
                            char *dataA, npy_intp *stridesA,
                            char *dataB, npy_intp *stridesB,
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
PyArray_PrepareThreeRawArrayIter(int ndim, npy_intp *shape,
                            char *dataA, npy_intp *stridesA,
                            char *dataB, npy_intp *stridesB,
                            char *dataC, npy_intp *stridesC,
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
