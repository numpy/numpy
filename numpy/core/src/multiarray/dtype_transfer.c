/*
 * This file contains low-level loops for data type transfers.
 * In particular the function PyArray_GetDTypeTransferFunction is
 * implemented here.
 *
 * Copyright (c) 2010 by Mark Wiebe (mwwiebe@gmail.com)
 * The Univerity of British Columbia
 *
 * See LICENSE.txt for the license.

 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define _MULTIARRAYMODULE
#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_cpu.h>

#include "lowlevel_strided_loops.h"

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

/*
 * Returns a transfer function which DECREFs any references in src_type.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
static int
get_decsrcref_transfer_function(int aligned,
                            npy_intp src_stride,
                            PyArray_Descr *src_dtype,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
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
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
                            int *out_needs_api);

/*
 * Returns a transfer function which sets a boolean type to ones.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
get_bool_setdstone_transfer_function(npy_intp dst_stride,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
                            int *NPY_UNUSED(out_needs_api));

/*************************** COPY REFERENCES *******************************/

/* Moves references from src to dst */
static void
_strided_to_strided_move_references(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        void *data)
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
                        void *data)
{
    PyObject *src_ref = NULL, *dst_ref = NULL;
    while (N > 0) {
        NPY_COPY_PYOBJECT_PTR(&src_ref, src);
        NPY_COPY_PYOBJECT_PTR(&dst_ref, dst);

        /* Release the reference in dst */
        NPY_DT_DBG_REFTRACE("dec dst ref", dst_ref);
        Py_XDECREF(dst_ref);
        /* Copy the reference */
        NPY_DT_DBG_REFTRACE("copy src ref", src_ref);
        NPY_COPY_PYOBJECT_PTR(dst, &src_ref);
        /* Claim the reference */
        Py_XINCREF(src_ref);

        src += src_stride;
        dst += dst_stride;
        --N;
    }
}

/************************** ZERO-PADDED COPY ******************************/

typedef void (*free_strided_transfer_data)(void *);
typedef void *(*copy_strided_transfer_data)(void *);

/* Does a zero-padded copy */
typedef struct {
    free_strided_transfer_data freefunc;
    copy_strided_transfer_data copyfunc;
    npy_intp dst_itemsize;
} _strided_zero_pad_data;

/* zero-padded data copy function */
void *_strided_zero_pad_data_copy(void *data)
{
    _strided_zero_pad_data *newdata =
            (_strided_zero_pad_data *)PyArray_malloc(
                                    sizeof(_strided_zero_pad_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_strided_zero_pad_data));

    return newdata;
}

/*
 * Does a strided to strided zero-padded copy for the case where
 * dst_itemsize > src_itemsize
 */
static void
_strided_to_strided_zero_pad_copy(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        void *data)
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
                        void *data)
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

NPY_NO_EXPORT int
PyArray_GetStridedZeroPadCopyFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp src_itemsize, npy_intp dst_itemsize,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata)
{
    if (src_itemsize == dst_itemsize) {
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
        d->freefunc = &PyArray_free;
        d->copyfunc = &_strided_zero_pad_data_copy;

        if (src_itemsize < dst_itemsize) {
            *out_stransfer = &_strided_to_strided_zero_pad_copy;
        }
        else {
            *out_stransfer = &_strided_to_strided_truncate_copy;
        }

        *out_transferdata = d;
        return NPY_SUCCEED;
    }
}

/***************** WRAP ALIGNED CONTIGUOUS TRANSFER FUNCTION **************/

/* Wraps a transfer function + data in alignment code */
typedef struct {
    free_strided_transfer_data freefunc;
    copy_strided_transfer_data copyfunc;
    PyArray_StridedTransferFn *wrapped,
                *tobuffer, *frombuffer;
    void *wrappeddata, *todata, *fromdata;
    npy_intp src_itemsize, dst_itemsize;
    char *bufferin, *bufferout;
} _align_wrap_data;

/* transfer data free function */
void _align_wrap_data_free(void *data)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    PyArray_FreeStridedTransferData(d->wrappeddata);
    PyArray_FreeStridedTransferData(d->todata);
    PyArray_FreeStridedTransferData(d->fromdata);
    PyArray_free(data);
}

/* transfer data copy function */
void *_align_wrap_data_copy(void *data)
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
        newdata->wrappeddata =
                        PyArray_CopyStridedTransferData(d->wrappeddata);
        if (newdata->wrappeddata == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (newdata->todata != NULL) {
        newdata->todata = PyArray_CopyStridedTransferData(d->todata);
        if (newdata->todata == NULL) {
            PyArray_FreeStridedTransferData(newdata->wrappeddata);
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (newdata->fromdata != NULL) {
        newdata->fromdata = PyArray_CopyStridedTransferData(d->fromdata);
        if (newdata->fromdata == NULL) {
            PyArray_FreeStridedTransferData(newdata->wrappeddata);
            PyArray_FreeStridedTransferData(newdata->todata);
            PyArray_free(newdata);
            return NULL;
        }
    }

    return (void *)newdata;
}

static void
_strided_to_strided_contig_align_wrap(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        void *data)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    PyArray_StridedTransferFn *wrapped = d->wrapped,
            *tobuffer = d->tobuffer,
            *frombuffer = d->frombuffer;
    npy_intp dst_itemsize = d->dst_itemsize;
    void *wrappeddata = d->wrappeddata,
            *todata = d->todata,
            *fromdata = d->fromdata;
    char *bufferin = d->bufferin, *bufferout = d->bufferout;

    for(;;) {
        if (N > NPY_LOWLEVEL_BUFFER_BLOCKSIZE) {
            tobuffer(bufferin, src_itemsize, src, src_stride,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    src_itemsize, todata);
            wrapped(bufferout, dst_itemsize, bufferin, src_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    src_itemsize, wrappeddata);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    dst_itemsize, fromdata);
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            tobuffer(bufferin, src_itemsize, src, src_stride, N,
                                            src_itemsize, todata);
            wrapped(bufferout, dst_itemsize, bufferin, src_itemsize, N,
                                            src_itemsize, wrappeddata);
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
                        void *data)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    PyArray_StridedTransferFn *wrapped = d->wrapped,
            *tobuffer = d->tobuffer,
            *frombuffer = d->frombuffer;
    npy_intp dst_itemsize = d->dst_itemsize;
    void *wrappeddata = d->wrappeddata,
            *todata = d->todata,
            *fromdata = d->fromdata;
    char *bufferin = d->bufferin, *bufferout = d->bufferout;

    for(;;) {
        if (N > NPY_LOWLEVEL_BUFFER_BLOCKSIZE) {
            tobuffer(bufferin, src_itemsize, src, src_stride,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    src_itemsize, todata);
            memset(bufferout, 0, dst_itemsize*NPY_LOWLEVEL_BUFFER_BLOCKSIZE);
            wrapped(bufferout, dst_itemsize, bufferin, src_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    src_itemsize, wrappeddata);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    dst_itemsize, fromdata);
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            tobuffer(bufferin, src_itemsize, src, src_stride, N,
                                            src_itemsize, todata);
            memset(bufferout, 0, dst_itemsize*N);
            wrapped(bufferout, dst_itemsize, bufferin, src_itemsize, N,
                                            src_itemsize, wrappeddata);
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
            PyArray_StridedTransferFn *tobuffer, void *todata,
            PyArray_StridedTransferFn *frombuffer, void *fromdata,
            PyArray_StridedTransferFn *wrapped, void *wrappeddata,
            int init_dest,
            PyArray_StridedTransferFn **out_stransfer,
            void **out_transferdata)
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
    data->freefunc = &_align_wrap_data_free;
    data->copyfunc = &_align_wrap_data_copy;
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
    *out_transferdata = data;

    return NPY_SUCCEED;
}

/*************************** WRAP DTYPE COPY/SWAP *************************/
/* Wraps the dtype copy swap function */
typedef struct {
    free_strided_transfer_data freefunc;
    copy_strided_transfer_data copyfunc;
    PyArray_CopySwapNFunc *copyswapn;
    int swap;
    PyArrayObject *arr;
} _wrap_copy_swap_data;

/* wrap copy swap data free function */
void _wrap_copy_swap_data_free(void *data)
{
    _wrap_copy_swap_data *d = (_wrap_copy_swap_data *)data;
    Py_DECREF(d->arr);
    PyArray_free(data);
}

/* wrap copy swap data copy function */
void *_wrap_copy_swap_data_copy(void *data)
{
    _wrap_copy_swap_data *newdata =
        (_wrap_copy_swap_data *)PyArray_malloc(sizeof(_wrap_copy_swap_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_wrap_copy_swap_data));
    Py_INCREF(newdata->arr);

    return (void *)newdata;
}

static void
_strided_to_strided_wrap_copy_swap(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        void *data)
{
    _wrap_copy_swap_data *d = (_wrap_copy_swap_data *)data;

    d->copyswapn(dst, dst_stride, src, src_stride, N, d->swap, d->arr);
}

/* This only gets used for custom data types */
static int
wrap_copy_swap_function(int aligned,
                npy_intp src_stride, npy_intp dst_stride,
                PyArray_Descr *dtype,
                int should_swap,
                PyArray_StridedTransferFn **out_stransfer,
                void **out_transferdata)
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

    data->freefunc = &_wrap_copy_swap_data_free;
    data->copyfunc = &_wrap_copy_swap_data_copy;
    data->copyswapn = dtype->f->copyswapn;
    data->swap = should_swap;

    /*
     * TODO: This is a hack so the copyswap functions have an array.
     *       The copyswap functions shouldn't need that.
     */
    Py_INCREF(dtype);
    data->arr = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                            1, &shape, NULL, NULL, 0, NULL);
    if (data->arr == NULL) {
        PyArray_free(data);
        return NPY_FAIL;
    }

    *out_stransfer = &_strided_to_strided_wrap_copy_swap;
    *out_transferdata = data;

    return NPY_SUCCEED;
}

/*************************** DTYPE CAST FUNCTIONS *************************/

/* Does a simple aligned cast */
typedef struct {
    free_strided_transfer_data freefunc;
    copy_strided_transfer_data copyfunc;
    PyArray_VectorUnaryFunc *castfunc;
    PyArrayObject *aip, *aop;
} _strided_cast_data;

/* strided cast data free function */
void _strided_cast_data_free(void *data)
{
    _strided_cast_data *d = (_strided_cast_data *)data;
    Py_DECREF(d->aip);
    Py_DECREF(d->aop);
    PyArray_free(data);
}

/* strided cast data copy function */
void *_strided_cast_data_copy(void *data)
{
    _strided_cast_data *newdata =
            (_strided_cast_data *)PyArray_malloc(sizeof(_strided_cast_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_strided_cast_data));
    Py_INCREF(newdata->aip);
    Py_INCREF(newdata->aop);

    return (void *)newdata;
}

static void
_aligned_strided_to_strided_cast(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        void *data)
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
                        void *data)
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
                        void *data)
{
    _strided_cast_data *d = (_strided_cast_data *)data;

    d->castfunc(src, dst, N, d->aip, d->aop);
}

static int
get_nbo_cast_numeric_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            int src_type_num, int dst_type_num,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata)
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
#if PY_VERSION_HEX >= 0x02050000
        ret = PyErr_WarnEx(cls,
                           "Casting complex values to real discards "
                           "the imaginary part", 1);
#else
        ret = PyErr_Warn(cls,
                         "Casting complex values to real discards "
                         "the imaginary part");
#endif
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

static int
get_nbo_cast_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
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
    data->freefunc = &_strided_cast_data_free;
    data->copyfunc = &_strided_cast_data_copy;
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
    data->aip = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, tmp_dtype,
                            1, &shape, NULL, NULL, 0, NULL);
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
    data->aop = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, tmp_dtype,
                            1, &shape, NULL, NULL, 0, NULL);
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
    *out_transferdata = data;

    return NPY_SUCCEED;
}

static int
get_cast_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
                            int *out_needs_api)
{
    PyArray_StridedTransferFn *caststransfer;
    void *castdata, *todata = NULL, *fromdata = NULL;
    int needs_wrap = 0;
    npy_intp src_itemsize = src_dtype->elsize,
            dst_itemsize = dst_dtype->elsize;

    if (src_dtype->type_num == dst_dtype->type_num) {
        PyErr_SetString(PyExc_ValueError,
                "low level cast function is for unequal type numbers");
        return NPY_FAIL;
    }

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
        PyArray_StridedTransferFn *tobuffer, *frombuffer;

        /* Get the copy/swap operation from src */

        /* If it's a custom data type, wrap its copy swap function */
        if (src_dtype->type_num >= NPY_NTYPES) {
            tobuffer = NULL;
            wrap_copy_swap_function(aligned,
                                src_stride, src_itemsize,
                                src_dtype,
                                !PyArray_ISNBO(src_dtype->byteorder),
                                &tobuffer, &todata);
        }
        /* A straight copy */
        else if (src_itemsize == 1 || PyArray_ISNBO(src_dtype->byteorder)) {
            tobuffer = PyArray_GetStridedCopyFn(aligned,
                                        src_stride, src_itemsize,
                                        src_itemsize);
        }
        /* If it's not complex, one swap */
        else if(src_dtype->kind != 'c') {
            tobuffer = PyArray_GetStridedCopySwapFn(aligned,
                                        src_stride, src_itemsize,
                                        src_itemsize);
        }
        /* If complex, a paired swap */
        else {
            tobuffer = PyArray_GetStridedCopySwapPairFn(aligned,
                                        src_stride, src_itemsize,
                                        src_itemsize);
        }

        /* Get the copy/swap operation to dst */

        /* If it's a custom data type, wrap its copy swap function */
        if (dst_dtype->type_num >= NPY_NTYPES) {
            frombuffer = NULL;
            wrap_copy_swap_function(aligned,
                                dst_itemsize, dst_stride,
                                dst_dtype,
                                !PyArray_ISNBO(dst_dtype->byteorder),
                                &frombuffer, &fromdata);
        }
        /* A straight copy */
        else if (dst_itemsize == 1 || PyArray_ISNBO(dst_dtype->byteorder)) {
            if (dst_dtype->type_num == NPY_OBJECT) {
                frombuffer = &_strided_to_strided_move_references;
            }
            else {
                frombuffer = PyArray_GetStridedCopyFn(aligned,
                                        dst_itemsize, dst_stride,
                                        dst_itemsize);
            }
        }
        /* If it's not complex, one swap */
        else if(dst_dtype->kind != 'c') {
            frombuffer = PyArray_GetStridedCopySwapFn(aligned,
                                        dst_itemsize, dst_stride,
                                        dst_itemsize);
        }
        /* If complex, a paired swap */
        else {
            frombuffer = PyArray_GetStridedCopySwapPairFn(aligned,
                                        dst_itemsize, dst_stride,
                                        dst_itemsize);
        }

        if (frombuffer == NULL || tobuffer == NULL) {
            PyArray_FreeStridedTransferData(castdata);
            PyArray_FreeStridedTransferData(todata);
            PyArray_FreeStridedTransferData(fromdata);
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
            PyArray_FreeStridedTransferData(castdata);
            PyArray_FreeStridedTransferData(todata);
            PyArray_FreeStridedTransferData(fromdata);
            return NPY_FAIL;
        }

        return NPY_SUCCEED;
    }
}

/**************************** COPY 1 TO N CONTIGUOUS ************************/

/* Copies 1 element to N contiguous elements */
typedef struct {
    free_strided_transfer_data freefunc;
    copy_strided_transfer_data copyfunc;
    PyArray_StridedTransferFn *stransfer;
    void *data;
    npy_intp N, dst_itemsize;
    /* If this is non-NULL the source type has references needing a decref */
    PyArray_StridedTransferFn *stransfer_finish_src;
    void *data_finish_src;
} _one_to_n_data;

/* transfer data free function */
void _one_to_n_data_free(void *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    PyArray_FreeStridedTransferData(d->data);
    PyArray_FreeStridedTransferData(d->data_finish_src);
    PyArray_free(data);
}

/* transfer data copy function */
void *_one_to_n_data_copy(void *data)
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
        newdata->data = PyArray_CopyStridedTransferData(d->data);
        if (newdata->data == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (d->data_finish_src != NULL) {
        newdata->data_finish_src =
                        PyArray_CopyStridedTransferData(d->data_finish_src);
        if (newdata->data_finish_src == NULL) {
            PyArray_FreeStridedTransferData(newdata->data);
            PyArray_free(newdata);
            return NULL;
        }
    }

    return (void *)newdata;
}

static void
_strided_to_strided_one_to_n(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        void *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    PyArray_StridedTransferFn *subtransfer = d->stransfer;
    void *subdata = d->data;
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
                        void *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    PyArray_StridedTransferFn *subtransfer = d->stransfer,
                *stransfer_finish_src = d->stransfer_finish_src;
    void *subdata = d->data, *data_finish_src = data_finish_src;
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
                            PyArray_StridedTransferFn *stransfer_inner,
                            void *data_inner,
                            PyArray_StridedTransferFn *stransfer_finish_src,
                            void *data_finish_src,
                            npy_intp dst_itemsize,
                            npy_intp N,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata)
{
    _one_to_n_data *data;


    data = PyArray_malloc(sizeof(_one_to_n_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }

    data->freefunc = &_one_to_n_data_free;
    data->copyfunc = &_one_to_n_data_copy;
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
    *out_transferdata = data;

    return NPY_SUCCEED;
}

static int
get_one_to_n_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            npy_intp N,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
                            int *out_needs_api)
{
    PyArray_StridedTransferFn *stransfer, *stransfer_finish_src = NULL;
    void *data, *data_finish_src = NULL;

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
            PyArray_FreeStridedTransferData(data);
            return NPY_FAIL;
        }
    }

    if (wrap_transfer_function_one_to_n(stransfer, data,
                            stransfer_finish_src, data_finish_src,
                            dst_dtype->elsize,
                            N,
                            out_stransfer, out_transferdata) != NPY_SUCCEED) {
        PyArray_FreeStridedTransferData(data);
        PyArray_FreeStridedTransferData(data_finish_src);
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}

/**************************** COPY N TO N CONTIGUOUS ************************/

/* Copies N contiguous elements to N contiguous elements */
typedef struct {
    free_strided_transfer_data freefunc;
    copy_strided_transfer_data copyfunc;
    PyArray_StridedTransferFn *stransfer;
    void *data;
    npy_intp N, src_itemsize, dst_itemsize;
} _n_to_n_data;

/* transfer data free function */
void _n_to_n_data_free(void *data)
{
    _n_to_n_data *d = (_n_to_n_data *)data;
    PyArray_FreeStridedTransferData(d->data);
    PyArray_free(data);
}

/* transfer data copy function */
void *_n_to_n_data_copy(void *data)
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
        newdata->data = PyArray_CopyStridedTransferData(d->data);
        if (newdata->data == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }

    return (void *)newdata;
}

static void
_strided_to_strided_n_to_n(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        void *data)
{
    _n_to_n_data *d = (_n_to_n_data *)data;
    PyArray_StridedTransferFn *subtransfer = d->stransfer;
    void *subdata = d->data;
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
                        void *data)
{
    _n_to_n_data *d = (_n_to_n_data *)data;
    PyArray_StridedTransferFn *subtransfer = d->stransfer;
    void *subdata = d->data;
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
                            PyArray_StridedTransferFn *stransfer_inner,
                            void *data_inner,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp src_itemsize, npy_intp dst_itemsize,
                            npy_intp N,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata)
{
    _n_to_n_data *data;

    data = PyArray_malloc(sizeof(_n_to_n_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }

    data->freefunc = &_n_to_n_data_free;
    data->copyfunc = &_n_to_n_data_copy;
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
    *out_transferdata = data;

    return NPY_SUCCEED;
}

static int
get_n_to_n_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            npy_intp N,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
                            int *out_needs_api)
{
    PyArray_StridedTransferFn *stransfer;
    void *data;

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
        PyArray_FreeStridedTransferData(data);
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
    free_strided_transfer_data freefunc;
    copy_strided_transfer_data copyfunc;
    PyArray_StridedTransferFn *stransfer;
    void *data;
    npy_intp src_N, dst_N, src_itemsize, dst_itemsize;
    PyArray_StridedTransferFn *stransfer_decsrcref;
    void *data_decsrcref;
    PyArray_StridedTransferFn *stransfer_decdstref;
    void *data_decdstref;
    /* This gets a run-length encoded representation of the transfer */
    npy_intp run_count;
    _subarray_broadcast_offsetrun offsetruns;
} _subarray_broadcast_data;

/* transfer data free function */
void _subarray_broadcast_data_free(void *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;
    PyArray_FreeStridedTransferData(d->data);
    PyArray_FreeStridedTransferData(d->data_decsrcref);
    PyArray_FreeStridedTransferData(d->data_decdstref);
    PyArray_free(data);
}

/* transfer data copy function */
void *_subarray_broadcast_data_copy( void *data)
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
        newdata->data = PyArray_CopyStridedTransferData(d->data);
        if (newdata->data == NULL) {
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (d->data_decsrcref != NULL) {
        newdata->data_decsrcref =
                        PyArray_CopyStridedTransferData(d->data_decsrcref);
        if (newdata->data_decsrcref == NULL) {
            PyArray_FreeStridedTransferData(newdata->data);
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (d->data_decdstref != NULL) {
        newdata->data_decdstref =
                        PyArray_CopyStridedTransferData(d->data_decdstref);
        if (newdata->data_decdstref == NULL) {
            PyArray_FreeStridedTransferData(newdata->data);
            PyArray_FreeStridedTransferData(newdata->data_decsrcref);
            PyArray_free(newdata);
            return NULL;
        }
    }

    return newdata;
}

static void
_strided_to_strided_subarray_broadcast(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        void *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;
    PyArray_StridedTransferFn *subtransfer = d->stransfer;
    void *subdata = d->data;
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
                        void *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;
    PyArray_StridedTransferFn *subtransfer = d->stransfer;
    void *subdata = d->data;
    PyArray_StridedTransferFn *stransfer_decsrcref = d->stransfer_decsrcref;
    void *data_decsrcref = d->data_decsrcref;
    PyArray_StridedTransferFn *stransfer_decdstref = d->stransfer_decdstref;
    void *data_decdstref = d->data_decdstref;
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
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
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
    data->freefunc = &_subarray_broadcast_data_free;
    data->copyfunc = &_subarray_broadcast_data_copy;
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
            PyArray_FreeStridedTransferData(data->data);
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
            PyArray_FreeStridedTransferData(data->data);
            PyArray_FreeStridedTransferData(data->data_decsrcref);
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
    *out_transferdata = data;

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
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
                            int *out_needs_api)
{
    PyArray_Dims src_shape = {NULL, -1}, dst_shape = {NULL, -1};
    npy_intp src_size = 1, dst_size = 1;

    /* Get the subarray shapes and sizes */
    if (src_dtype->subarray != NULL) {
       if (!(PyArray_IntpConverter(src_dtype->subarray->shape,
                                            &src_shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return NPY_FAIL;
        }
        src_size = PyArray_MultiplyList(src_shape.ptr, src_shape.len);
        src_dtype = src_dtype->subarray->base;
    }
    if (dst_dtype->subarray != NULL) {
       if (!(PyArray_IntpConverter(dst_dtype->subarray->shape,
                                            &dst_shape))) {
            if (src_shape.ptr != NULL) {
                PyDimMem_FREE(src_shape.ptr);
            }
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
        PyDimMem_FREE(src_shape.ptr);
        PyDimMem_FREE(dst_shape.ptr);

        return PyArray_GetDTypeTransferFunction(aligned,
                src_stride, dst_stride,
                src_dtype, dst_dtype,
                move_references,
                out_stransfer, out_transferdata,
                out_needs_api);
    }
    /* Copy the src value to all the dst values */
    else if (src_size == 1) {
        PyDimMem_FREE(src_shape.ptr);
        PyDimMem_FREE(dst_shape.ptr);

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
        PyDimMem_FREE(src_shape.ptr);
        PyDimMem_FREE(dst_shape.ptr);

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

        PyDimMem_FREE(src_shape.ptr);
        PyDimMem_FREE(dst_shape.ptr);
        return ret;
    }
}

/**************************** COPY FIELDS *******************************/
typedef struct {
    npy_intp src_offset, dst_offset, src_itemsize;
    PyArray_StridedTransferFn *stransfer;
    void *data;
} _single_field_transfer;

typedef struct {
    free_strided_transfer_data freefunc;
    copy_strided_transfer_data copyfunc;
    npy_intp field_count;

    _single_field_transfer fields;
} _field_transfer_data;

/* transfer data free function */
void _field_transfer_data_free(void *data)
{
    _field_transfer_data *d = (_field_transfer_data *)data;
    npy_intp i, field_count;
    _single_field_transfer *fields;

    field_count = d->field_count;
    fields = &d->fields;

    for (i = 0; i < field_count; ++i) {
        PyArray_FreeStridedTransferData(fields[i].data);
    }
    PyArray_free(d);
}

/* transfer data copy function */
void *_field_transfer_data_copy(void *data)
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
            newfields[i].data =
                        PyArray_CopyStridedTransferData(fields[i].data);
            if (newfields[i].data == NULL) {
                for (i = i-1; i >= 0; --i) {
                    PyArray_FreeStridedTransferData(newfields[i].data);
                }
                PyArray_free(newdata);
                return NULL;
            }
        }

    }

    return (void *)newdata;
}

static void
_strided_to_strided_field_transfer(char *dst, npy_intp dst_stride,
                        char *src, npy_intp src_stride,
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        void *data)
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
 * must have fields
 */
static int
get_fields_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
                            int *out_needs_api)
{
    PyObject *names, *key, *tup, *title;
    PyArray_Descr *src_fld_dtype, *dst_fld_dtype;
    npy_int i, names_size, field_count, structsize;
    int src_offset, dst_offset;
    _field_transfer_data *data;
    _single_field_transfer *fields;

    /* Copy the src value to all the fields of dst */
    if (!PyDescr_HASFIELDS(src_dtype)) {
        names = dst_dtype->names;
        names_size = PyTuple_GET_SIZE(dst_dtype->names);

        field_count = names_size;
        structsize = sizeof(_field_transfer_data) +
                        (field_count + 1) * sizeof(_single_field_transfer);
        /* Allocate the data and populate it */
        data = (_field_transfer_data *)PyArray_malloc(structsize);
        if (data == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
        data->freefunc = &_field_transfer_data_free;
        data->copyfunc = &_field_transfer_data_copy;
        fields = &data->fields;

        for (i = 0; i < names_size; ++i) {
            key = PyTuple_GET_ITEM(names, i);
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
                    PyArray_FreeStridedTransferData(fields[i].data);
                }
                PyArray_free(data);
                return NPY_FAIL;
            }
            fields[i].src_offset = 0;
            fields[i].dst_offset = dst_offset;
            fields[i].src_itemsize = src_dtype->elsize;
        }

        /*
         * If the references should be removed from src, add
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
                    PyArray_FreeStridedTransferData(fields[i].data);
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
        *out_transferdata = data;

        return NPY_SUCCEED;
    }
    /* Copy the value of the first field to dst */
    else if (!PyDescr_HASFIELDS(dst_dtype)) {
        names = src_dtype->names;
        names_size = PyTuple_GET_SIZE(src_dtype->names);

        /*
         * If DECREF is needed on source fields, may need
         * to process all the fields
         */
        if (move_references && PyDataType_REFCHK(src_dtype)) {
            field_count = names_size + 1;
        }
        else {
            field_count = 1;
        }
        structsize = sizeof(_field_transfer_data) +
                        field_count * sizeof(_single_field_transfer);
        /* Allocate the data and populate it */
        data = (_field_transfer_data *)PyArray_malloc(structsize);
        if (data == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }
        data->freefunc = &_field_transfer_data_free;
        data->copyfunc = &_field_transfer_data_copy;
        fields = &data->fields;

        key = PyTuple_GET_ITEM(names, 0);
        tup = PyDict_GetItem(src_dtype->fields, key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &src_fld_dtype,
                                                &src_offset, &title)) {
            PyArray_free(data);
            return NPY_FAIL;
        }
        field_count = 0;
        /*
         * Special case bool type, the existence of fields implies True
         *
         * TODO: Perhaps a better behavior would be to combine all the
         *       input fields with an OR?  The same would apply to subarrays.
         */
        if (dst_dtype->type_num == NPY_BOOL) {
            if (get_bool_setdstone_transfer_function(dst_stride,
                                    &fields[field_count].stransfer,
                                    &fields[field_count].data,
                                    out_needs_api) != NPY_SUCCEED) {
                PyArray_free(data);
                return NPY_FAIL;
            }
            fields[field_count].src_offset = 0;
            fields[field_count].dst_offset = 0;
            fields[field_count].src_itemsize = 0;
            field_count++;

            /* If the src field has references, may need to clear them */
            if (move_references && PyDataType_REFCHK(src_fld_dtype)) {
                if (get_decsrcref_transfer_function(0,
                            src_stride,
                            src_fld_dtype,
                            &fields[field_count].stransfer,
                            &fields[field_count].data,
                            out_needs_api) != NPY_SUCCEED) {
                    PyArray_FreeStridedTransferData(fields[0].data);
                    PyArray_free(data);
                    return NPY_FAIL;
                }
                fields[field_count].src_offset = src_offset;
                fields[field_count].dst_offset = 0;
                fields[field_count].src_itemsize = src_fld_dtype->elsize;
                field_count++;
            }
        }
        /* Transfer the first field to the output */
        else {
            if (PyArray_GetDTypeTransferFunction(0,
                                    src_stride, dst_stride,
                                    src_fld_dtype, dst_dtype,
                                    move_references,
                                    &fields[field_count].stransfer,
                                    &fields[field_count].data,
                                    out_needs_api) != NPY_SUCCEED) {
                PyArray_free(data);
                return NPY_FAIL;
            }
            fields[field_count].src_offset = src_offset;
            fields[field_count].dst_offset = 0;
            fields[field_count].src_itemsize = src_fld_dtype->elsize;
            field_count++;
        }

        /*
         * If the references should be removed from src, add
         * more transfer functions to decrement the references
         * for all the other fields.
         */
        if (move_references && PyDataType_REFCHK(src_dtype)) {
            for (i = 1; i < names_size; ++i) {
                key = PyTuple_GET_ITEM(names, i);
                tup = PyDict_GetItem(src_dtype->fields, key);
                if (!PyArg_ParseTuple(tup, "Oi|O", &src_fld_dtype,
                                                    &src_offset, &title)) {
                    return NPY_FAIL;
                }
                if (PyDataType_REFCHK(src_fld_dtype)) {
                    if (get_decsrcref_transfer_function(0,
                                src_stride,
                                src_fld_dtype,
                                &fields[field_count].stransfer,
                                &fields[field_count].data,
                                out_needs_api) != NPY_SUCCEED) {
                        for (i = field_count-1; i >= 0; --i) {
                            PyArray_FreeStridedTransferData(fields[i].data);
                        }
                        PyArray_free(data);
                        return NPY_FAIL;
                    }
                    fields[field_count].src_offset = src_offset;
                    fields[field_count].dst_offset = 0;
                    fields[field_count].src_itemsize = src_fld_dtype->elsize;
                    field_count++;
                }
            }
        }

        data->field_count = field_count;

        *out_stransfer = &_strided_to_strided_field_transfer;
        *out_transferdata = data;

        return NPY_SUCCEED;
    }
    /* Match up the fields to copy */
    else {
        /* Keeps track of the names we already used */
        PyObject *used_names_dict = NULL;

        names = dst_dtype->names;
        names_size = PyTuple_GET_SIZE(dst_dtype->names);

        /*
         * If DECREF is needed on source fields, will need
         * to also go through its fields.
         */
        if (move_references && PyDataType_REFCHK(src_dtype)) {
            field_count = names_size + PyTuple_GET_SIZE(src_dtype->names);
            used_names_dict = PyDict_New();
            if (used_names_dict == NULL) {
                return NPY_FAIL;
            }
        }
        else {
            field_count = names_size;
        }
        structsize = sizeof(_field_transfer_data) +
                        field_count * sizeof(_single_field_transfer);
        /* Allocate the data and populate it */
        data = (_field_transfer_data *)PyArray_malloc(structsize);
        if (data == NULL) {
            PyErr_NoMemory();
            Py_XDECREF(used_names_dict);
            return NPY_FAIL;
        }
        data->freefunc = &_field_transfer_data_free;
        data->copyfunc = &_field_transfer_data_copy;
        fields = &data->fields;

        for (i = 0; i < names_size; ++i) {
            key = PyTuple_GET_ITEM(names, i);
            tup = PyDict_GetItem(dst_dtype->fields, key);
            if (!PyArg_ParseTuple(tup, "Oi|O", &dst_fld_dtype,
                                                    &dst_offset, &title)) {
                for (i = i-1; i >= 0; --i) {
                    PyArray_FreeStridedTransferData(fields[i].data);
                }
                PyArray_free(data);
                Py_XDECREF(used_names_dict);
                return NPY_FAIL;
            }
            tup = PyDict_GetItem(src_dtype->fields, key);
            if (tup != NULL) {
                if (!PyArg_ParseTuple(tup, "Oi|O", &src_fld_dtype,
                                                        &src_offset, &title)) {
                    for (i = i-1; i >= 0; --i) {
                        PyArray_FreeStridedTransferData(fields[i].data);
                    }
                    PyArray_free(data);
                    Py_XDECREF(used_names_dict);
                    return NPY_FAIL;
                }
                if (PyArray_GetDTypeTransferFunction(0,
                                        src_stride, dst_stride,
                                        src_fld_dtype, dst_fld_dtype,
                                        move_references,
                                        &fields[i].stransfer,
                                        &fields[i].data,
                                        out_needs_api) != NPY_SUCCEED) {
                    for (i = i-1; i >= 0; --i) {
                        PyArray_FreeStridedTransferData(fields[i].data);
                    }
                    PyArray_free(data);
                    Py_XDECREF(used_names_dict);
                    return NPY_FAIL;
                }
                fields[i].src_offset = src_offset;
                fields[i].dst_offset = dst_offset;
                fields[i].src_itemsize = src_fld_dtype->elsize;

                if (used_names_dict != NULL) {
                    PyDict_SetItem(used_names_dict, key, Py_True);
                }
            }
            else {
                if (get_setdstzero_transfer_function(0,
                                            dst_stride,
                                            dst_fld_dtype,
                                            &fields[i].stransfer,
                                            &fields[i].data,
                                            out_needs_api) != NPY_SUCCEED) {
                    for (i = i-1; i >= 0; --i) {
                        PyArray_FreeStridedTransferData(fields[i].data);
                    }
                    PyArray_free(data);
                    Py_XDECREF(used_names_dict);
                    return NPY_FAIL;
                }
                fields[i].src_offset = 0;
                fields[i].dst_offset = dst_offset;
                fields[i].src_itemsize = 0;
            }
        }

        if (move_references && PyDataType_REFCHK(src_dtype)) {
            /* Use field_count to track additional functions added */
            field_count = names_size;

            names = src_dtype->names;
            names_size = PyTuple_GET_SIZE(src_dtype->names);
            for (i = 0; i < names_size; ++i) {
                key = PyTuple_GET_ITEM(names, i);
                if (PyDict_GetItem(used_names_dict, key) == NULL) {
                    tup = PyDict_GetItem(src_dtype->fields, key);
                    if (!PyArg_ParseTuple(tup, "Oi|O", &src_fld_dtype,
                                                    &src_offset, &title)) {
                        for (i = field_count-1; i >= 0; --i) {
                            PyArray_FreeStridedTransferData(fields[i].data);
                        }
                        PyArray_free(data);
                        Py_XDECREF(used_names_dict);
                        return NPY_FAIL;
                    }
                    if (PyDataType_REFCHK(src_fld_dtype)) {
                        if (get_decsrcref_transfer_function(0,
                                    src_stride,
                                    src_fld_dtype,
                                    &fields[field_count].stransfer,
                                    &fields[field_count].data,
                                    out_needs_api) != NPY_SUCCEED) {
                            for (i = field_count-1; i >= 0; --i) {
                                PyArray_FreeStridedTransferData(fields[i].data);
                            }
                            PyArray_free(data);
                            return NPY_FAIL;
                        }
                        fields[field_count].src_offset = src_offset;
                        fields[field_count].dst_offset = 0;
                        fields[field_count].src_itemsize =
                                                src_fld_dtype->elsize;
                        field_count++;
                    }
                }
            }
        }

        Py_XDECREF(used_names_dict);

        data->field_count = field_count;

        *out_stransfer = &_strided_to_strided_field_transfer;
        *out_transferdata = data;

        return NPY_SUCCEED;
    }
}

static int
get_decsrcref_fields_transfer_function(int aligned,
                            npy_intp src_stride,
                            PyArray_Descr *src_dtype,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
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
    data->freefunc = &_field_transfer_data_free;
    data->copyfunc = &_field_transfer_data_copy;
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
                    PyArray_FreeStridedTransferData(fields[i].data);
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
    *out_transferdata = data;

    return NPY_SUCCEED;
}

static int
get_setdestzero_fields_transfer_function(int aligned,
                            npy_intp dst_stride,
                            PyArray_Descr *dst_dtype,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
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
    data->freefunc = &_field_transfer_data_free;
    data->copyfunc = &_field_transfer_data_copy;
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
                PyArray_FreeStridedTransferData(fields[i].data);
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
    *out_transferdata = data;

    return NPY_SUCCEED;
}

/************************* DEST BOOL SETONE *******************************/

static void
_null_to_strided_set_bool_one(char *dst,
                        npy_intp dst_stride,
                        char *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        void *NPY_UNUSED(data))
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
                        void *NPY_UNUSED(data))
{
    /* bool type is one byte, so can just use the char */

    memset(dst, 1, N);
}

/* Only for the bool type, sets the destination to 1 */
NPY_NO_EXPORT int
get_bool_setdstone_transfer_function(npy_intp dst_stride,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
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
    free_strided_transfer_data freefunc;
    copy_strided_transfer_data copyfunc;
    npy_intp dst_itemsize;
} _dst_memset_zero_data;

/* zero-padded data copy function */
void *_dst_memset_zero_data_copy(void *data)
{
    _dst_memset_zero_data *newdata =
            (_dst_memset_zero_data *)PyArray_malloc(
                                    sizeof(_dst_memset_zero_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_dst_memset_zero_data));

    return newdata;
}

static void
_null_to_strided_memset_zero(char *dst,
                        npy_intp dst_stride,
                        char *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        void *data)
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
                        void *data)
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
                        void *NPY_UNUSED(data))
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
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
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

        data->freefunc = &PyArray_free;
        data->copyfunc = &_dst_memset_zero_data_copy;
        data->dst_itemsize = dst_dtype->elsize;

        if (dst_stride == data->dst_itemsize) {
            *out_stransfer = &_null_to_contig_memset_zero;
        }
        else {
            *out_stransfer = &_null_to_strided_memset_zero;
        }
        *out_transferdata = data;
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
    else if (dst_dtype->subarray != NULL) {
        PyArray_Dims dst_shape = {NULL, -1};
        npy_intp dst_size = 1;
        PyArray_StridedTransferFn *contig_stransfer;
        void *contig_data;

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
        PyDimMem_FREE(dst_shape.ptr);

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
            PyArray_FreeStridedTransferData(contig_data);
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
                        void *NPY_UNUSED(data))
{
    /* NOP */
}

static void
_strided_to_null_dec_src_ref_reference(char *NPY_UNUSED(dst),
                        npy_intp NPY_UNUSED(dst_stride),
                        char *src, npy_intp src_stride,
                        npy_intp N,
                        npy_intp NPY_UNUSED(src_itemsize),
                        void *NPY_UNUSED(data))
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
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
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
    else if (src_dtype->subarray != NULL) {
        PyArray_Dims src_shape = {NULL, -1};
        npy_intp src_size = 1;
        PyArray_StridedTransferFn *stransfer;
        void *data;

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
        PyDimMem_FREE(src_shape.ptr);

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
            PyArray_FreeStridedTransferData(data);
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

/********************* MAIN DTYPE TRANSFER FUNCTION ***********************/

NPY_NO_EXPORT int
PyArray_GetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
                            int *out_needs_api)
{
    npy_intp src_itemsize, dst_itemsize;
    int src_type_num, dst_type_num;

#if NPY_DT_DBG_TRACING
    printf("Calculating dtype transfer from ");
    PyObject_Print((PyObject *)src_dtype, stdout, 0);
    printf(" to ");
    PyObject_Print((PyObject *)dst_dtype, stdout, 0);
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
     * If there are no references and the data types are equivalent,
     * return a simple copy
     */
    if (!PyDataType_REFCHK(src_dtype) && !PyDataType_REFCHK(dst_dtype) &&
                            PyArray_EquivTypes(src_dtype, dst_dtype)) {
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
                src_dtype->subarray == NULL && dst_dtype->subarray == NULL) {
        /* A custom data type requires that we use its copy/swap */
        if (src_type_num >= NPY_NTYPES || dst_type_num >= NPY_NTYPES) {
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

        /* The special types, which have no byte-order */
        switch (src_type_num) {
            case NPY_VOID:
            case NPY_STRING:
            case NPY_UNICODE:
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
    if (src_dtype->subarray != NULL || dst_dtype->subarray != NULL) {
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
        case NPY_STRING:
        case NPY_UNICODE:
        case NPY_VOID:
            return PyArray_GetStridedZeroPadCopyFn(0,
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
