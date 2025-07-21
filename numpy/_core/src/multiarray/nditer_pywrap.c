/*
 * This file implements the CPython wrapper of NpyIter
 *
 * Copyright (c) 2010 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "npy_config.h"

#include "alloc.h"
#include "common.h"
#include "conversion_utils.h"
#include "ctors.h"
#include "npy_pycompat.h"

/* Functions not part of the public NumPy C API */
npy_bool npyiter_has_writeback(NpyIter *iter);


typedef struct NewNpyArrayIterObject_tag NewNpyArrayIterObject;

struct NewNpyArrayIterObject_tag {
    PyObject_HEAD
    /* The iterator */
    NpyIter *iter;
    /* Flag indicating iteration started/stopped */
    char started, finished;
    /* Child to update for nested iteration */
    NewNpyArrayIterObject *nested_child;
    /* Cached values from the iterator */
    NpyIter_IterNextFunc *iternext;
    NpyIter_GetMultiIndexFunc *get_multi_index;
    char **dataptrs;
    PyArray_Descr **dtypes;
    PyArrayObject **operands;
    npy_intp *innerstrides, *innerloopsizeptr;
    char *writeflags;  /* could inline allocation with variable sized object */
};

static int npyiter_cache_values(NewNpyArrayIterObject *self)
{
    NpyIter *iter = self->iter;

    /* iternext and get_multi_index functions */
    self->iternext = NpyIter_GetIterNext(iter, NULL);
    if (self->iternext == NULL) {
        return -1;
    }

    if (NpyIter_HasMultiIndex(iter) && !NpyIter_HasDelayedBufAlloc(iter)) {
        self->get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);
    }
    else {
        self->get_multi_index = NULL;
    }

    /* Internal data pointers */
    self->dataptrs = NpyIter_GetDataPtrArray(iter);
    self->dtypes = NpyIter_GetDescrArray(iter);
    self->operands = NpyIter_GetOperandArray(iter);

    if (NpyIter_HasExternalLoop(iter)) {
        self->innerstrides = NpyIter_GetInnerStrideArray(iter);
        self->innerloopsizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    }
    else {
        self->innerstrides = NULL;
        self->innerloopsizeptr = NULL;
    }

    if (self->writeflags == NULL) {
        self->writeflags = PyMem_Malloc(sizeof(char) * NpyIter_GetNOp(iter));
        if (self->writeflags == NULL) {
            PyErr_NoMemory();
            return -1;
        }
    }
    /* The write flags settings (not-readable cannot be signalled to Python) */
    NpyIter_GetWriteFlags(iter, self->writeflags);
    return 0;
}

static PyObject *
npyiter_new(PyTypeObject *subtype, PyObject *NPY_UNUSED(args),
            PyObject *NPY_UNUSED(kwds))
{
    NewNpyArrayIterObject *self;

    self = (NewNpyArrayIterObject *)subtype->tp_alloc(subtype, 0);
    if (self != NULL) {
        self->iter = NULL;
        self->nested_child = NULL;
        self->writeflags = NULL;
    }

    return (PyObject *)self;
}

static int
NpyIter_GlobalFlagsConverter(PyObject *flags_in, npy_uint32 *flags)
{
    npy_uint32 tmpflags = 0;
    int iflags, nflags;

    PyObject *f;
    char *str = NULL;
    Py_ssize_t length = 0;
    npy_uint32 flag;

    if (flags_in == NULL || flags_in == Py_None) {
        return 1;
    }

    if (!PyTuple_Check(flags_in) && !PyList_Check(flags_in)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator global flags must be a list or tuple of strings");
        return 0;
    }

    nflags = PySequence_Size(flags_in);

    for (iflags = 0; iflags < nflags; ++iflags) {
        f = PySequence_GetItem(flags_in, iflags);
        if (f == NULL) {
            return 0;
        }

        if (PyUnicode_Check(f)) {
            /* accept unicode input */
            PyObject *f_str;
            f_str = PyUnicode_AsASCIIString(f);
            if (f_str == NULL) {
                Py_DECREF(f);
                return 0;
            }
            Py_DECREF(f);
            f = f_str;
        }

        if (PyBytes_AsStringAndSize(f, &str, &length) < 0) {
            Py_DECREF(f);
            return 0;
        }
        /* Use switch statements to quickly isolate the right flag */
        flag = 0;
        switch (str[0]) {
            case 'b':
                if (strcmp(str, "buffered") == 0) {
                    flag = NPY_ITER_BUFFERED;
                }
                break;
            case 'c':
                if (length >= 6) switch (str[5]) {
                    case 'e':
                        if (strcmp(str, "c_index") == 0) {
                            flag = NPY_ITER_C_INDEX;
                        }
                        break;
                    case 'i':
                        if (strcmp(str, "copy_if_overlap") == 0) {
                            flag = NPY_ITER_COPY_IF_OVERLAP;
                        }
                        break;
                    case 'n':
                        if (strcmp(str, "common_dtype") == 0) {
                            flag = NPY_ITER_COMMON_DTYPE;
                        }
                        break;
                }
                break;
            case 'd':
                if (strcmp(str, "delay_bufalloc") == 0) {
                    flag = NPY_ITER_DELAY_BUFALLOC;
                }
                break;
            case 'e':
                if (strcmp(str, "external_loop") == 0) {
                    flag = NPY_ITER_EXTERNAL_LOOP;
                }
                break;
            case 'f':
                if (strcmp(str, "f_index") == 0) {
                    flag = NPY_ITER_F_INDEX;
                }
                break;
            case 'g':
                /*
                 * Documentation is grow_inner, but initial implementation
                 * was growinner, so allowing for either.
                 */
                if (strcmp(str, "grow_inner") == 0 ||
                            strcmp(str, "growinner") == 0) {
                    flag = NPY_ITER_GROWINNER;
                }
                break;
            case 'm':
                if (strcmp(str, "multi_index") == 0) {
                    flag = NPY_ITER_MULTI_INDEX;
                }
                break;
            case 'r':
                if (strcmp(str, "ranged") == 0) {
                    flag = NPY_ITER_RANGED;
                }
                else if (strcmp(str, "refs_ok") == 0) {
                    flag = NPY_ITER_REFS_OK;
                }
                else if (strcmp(str, "reduce_ok") == 0) {
                    flag = NPY_ITER_REDUCE_OK;
                }
                break;
            case 'z':
                if (strcmp(str, "zerosize_ok") == 0) {
                    flag = NPY_ITER_ZEROSIZE_OK;
                }
                break;
        }
        if (flag == 0) {
            PyErr_Format(PyExc_ValueError,
                    "Unexpected iterator global flag \"%s\"", str);
            Py_DECREF(f);
            return 0;
        }
        else {
            tmpflags |= flag;
        }
        Py_DECREF(f);
    }

    *flags |= tmpflags;
    return 1;
}

static int
NpyIter_OpFlagsConverter(PyObject *op_flags_in,
                         npy_uint32 *op_flags)
{
    int iflags, nflags;
    npy_uint32 flag;

    if (!PyTuple_Check(op_flags_in) && !PyList_Check(op_flags_in)) {
        PyErr_SetString(PyExc_ValueError,
                "op_flags must be a tuple or array of per-op flag-tuples");
        return 0;
    }

    nflags = PySequence_Size(op_flags_in);

    *op_flags = 0;
    for (iflags = 0; iflags < nflags; ++iflags) {
        PyObject *f;
        char *str = NULL;
        Py_ssize_t length = 0;

        f = PySequence_GetItem(op_flags_in, iflags);
        if (f == NULL) {
            return 0;
        }

        if (PyUnicode_Check(f)) {
            /* accept unicode input */
            PyObject *f_str;
            f_str = PyUnicode_AsASCIIString(f);
            if (f_str == NULL) {
                Py_DECREF(f);
                return 0;
            }
            Py_DECREF(f);
            f = f_str;
        }

        if (PyBytes_AsStringAndSize(f, &str, &length) < 0) {
            PyErr_Clear();
            Py_DECREF(f);
            PyErr_SetString(PyExc_ValueError,
                   "op_flags must be a tuple or array of per-op flag-tuples");
            return 0;
        }

        /* Use switch statements to quickly isolate the right flag */
        flag = 0;
        switch (str[0]) {
            case 'a':
                if (length > 2) switch(str[2]) {
                    case 'i':
                        if (strcmp(str, "aligned") == 0) {
                            flag = NPY_ITER_ALIGNED;
                        }
                        break;
                    case 'l':
                        if (strcmp(str, "allocate") == 0) {
                            flag = NPY_ITER_ALLOCATE;
                        }
                        break;
                    case 'r':
                        if (strcmp(str, "arraymask") == 0) {
                            flag = NPY_ITER_ARRAYMASK;
                        }
                        break;
                }
                break;
            case 'c':
                if (strcmp(str, "copy") == 0) {
                    flag = NPY_ITER_COPY;
                }
                if (strcmp(str, "contig") == 0) {
                    flag = NPY_ITER_CONTIG;
                }
                break;
            case 'n':
                switch (str[1]) {
                    case 'b':
                        if (strcmp(str, "nbo") == 0) {
                            flag = NPY_ITER_NBO;
                        }
                        break;
                    case 'o':
                        if (strcmp(str, "no_subtype") == 0) {
                            flag = NPY_ITER_NO_SUBTYPE;
                        }
                        else if (strcmp(str, "no_broadcast") == 0) {
                            flag = NPY_ITER_NO_BROADCAST;
                        }
                        break;
                }
                break;
            case 'o':
                if (strcmp(str, "overlap_assume_elementwise") == 0) {
                    flag = NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
                }
                break;
            case 'r':
                if (length > 4) switch (str[4]) {
                    case 'o':
                        if (strcmp(str, "readonly") == 0) {
                            flag = NPY_ITER_READONLY;
                        }
                        break;
                    case 'w':
                        if (strcmp(str, "readwrite") == 0) {
                            flag = NPY_ITER_READWRITE;
                        }
                        break;
                }
                break;
            case 'u':
                switch (str[1]) {
                    case 'p':
                        if (strcmp(str, "updateifcopy") == 0) {
                            flag = NPY_ITER_UPDATEIFCOPY;
                        }
                        break;
                }
                break;
            case 'v':
                if (strcmp(str, "virtual") == 0) {
                    flag = NPY_ITER_VIRTUAL;
                }
                break;
            case 'w':
                if (length > 5) switch (str[5]) {
                    case 'o':
                        if (strcmp(str, "writeonly") == 0) {
                            flag = NPY_ITER_WRITEONLY;
                        }
                        break;
                    case 'm':
                        if (strcmp(str, "writemasked") == 0) {
                            flag = NPY_ITER_WRITEMASKED;
                        }
                        break;
                }
                break;
        }
        if (flag == 0) {
            PyErr_Format(PyExc_ValueError,
                    "Unexpected per-op iterator flag \"%s\"", str);
            Py_DECREF(f);
            return 0;
        }
        else {
            *op_flags |= flag;
        }
        Py_DECREF(f);
    }

    return 1;
}

static int
npyiter_convert_op_flags_array(PyObject *op_flags_in,
                         npy_uint32 *op_flags_array, npy_intp nop)
{
    npy_intp iop;

    if (!PyTuple_Check(op_flags_in) && !PyList_Check(op_flags_in)) {
        PyErr_SetString(PyExc_ValueError,
                "op_flags must be a tuple or array of per-op flag-tuples");
        return 0;
    }

    if (PySequence_Size(op_flags_in) != nop) {
        goto try_single_flags;
    }

    for (iop = 0; iop < nop; ++iop) {
        PyObject *f = PySequence_GetItem(op_flags_in, iop);
        if (f == NULL) {
            return 0;
        }
        /* If the first item is a string, try as one set of flags */
        if (iop == 0 && (PyBytes_Check(f) || PyUnicode_Check(f))) {
            Py_DECREF(f);
            goto try_single_flags;
        }
        if (NpyIter_OpFlagsConverter(f,
                        &op_flags_array[iop]) != 1) {
            Py_DECREF(f);
            return 0;
        }

        Py_DECREF(f);
    }

    return 1;

try_single_flags:
    if (NpyIter_OpFlagsConverter(op_flags_in,
                        &op_flags_array[0]) != 1) {
        return 0;
    }

    for (iop = 1; iop < nop; ++iop) {
        op_flags_array[iop] = op_flags_array[0];
    }

    return 1;
}

static int
npyiter_convert_dtypes(PyObject *op_dtypes_in,
                        PyArray_Descr **op_dtypes,
                        npy_intp nop)
{
    npy_intp iop;

    /*
     * If the input isn't a tuple of dtypes, try converting it as-is
     * to a dtype, and replicating to all operands.
     */
    if ((!PyTuple_Check(op_dtypes_in) && !PyList_Check(op_dtypes_in)) ||
                                    PySequence_Size(op_dtypes_in) != nop) {
        goto try_single_dtype;
    }

    for (iop = 0; iop < nop; ++iop) {
        PyObject *dtype = PySequence_GetItem(op_dtypes_in, iop);
        if (dtype == NULL) {
            npy_intp i;
            for (i = 0; i < iop; ++i ) {
                Py_XDECREF(op_dtypes[i]);
            }
            return 0;
        }

        /* Try converting the object to a descr */
        if (PyArray_DescrConverter2(dtype, &op_dtypes[iop]) != 1) {
            npy_intp i;
            for (i = 0; i < iop; ++i ) {
                Py_XDECREF(op_dtypes[i]);
            }
            Py_DECREF(dtype);
            PyErr_Clear();
            goto try_single_dtype;
        }

        Py_DECREF(dtype);
    }

    return 1;

try_single_dtype:
    if (PyArray_DescrConverter2(op_dtypes_in, &op_dtypes[0]) == 1) {
        for (iop = 1; iop < nop; ++iop) {
            op_dtypes[iop] = op_dtypes[0];
            Py_XINCREF(op_dtypes[iop]);
        }
        return 1;
    }

    return 0;
}

static int
npyiter_convert_op_axes(PyObject *op_axes_in, int nop,
                        int **op_axes, int *oa_ndim)
{
    PyObject *a;
    int iop;

    if ((!PyTuple_Check(op_axes_in) && !PyList_Check(op_axes_in)) ||
                                PySequence_Size(op_axes_in) != nop) {
        PyErr_SetString(PyExc_ValueError,
                "op_axes must be a tuple/list matching the number of ops");
        return 0;
    }

    *oa_ndim = -1;

    /* Copy the tuples into op_axes */
    for (iop = 0; iop < nop; ++iop) {
        int idim;
        a = PySequence_GetItem(op_axes_in, iop);
        if (a == NULL) {
            return 0;
        }
        if (a == Py_None) {
            op_axes[iop] = NULL;
        } else {
            if (!PyTuple_Check(a) && !PyList_Check(a)) {
                PyErr_SetString(PyExc_ValueError,
                        "Each entry of op_axes must be None "
                        "or a tuple/list");
                Py_DECREF(a);
                return 0;
            }
            if (*oa_ndim == -1) {
                *oa_ndim = PySequence_Size(a);
                if (*oa_ndim > NPY_MAXDIMS) {
                    PyErr_SetString(PyExc_ValueError,
                            "Too many dimensions in op_axes");
                    Py_DECREF(a);
                    return 0;
                }
            }
            if (PySequence_Size(a) != *oa_ndim) {
                PyErr_SetString(PyExc_ValueError,
                        "Each entry of op_axes must have the same size");
                Py_DECREF(a);
                return 0;
            }
            for (idim = 0; idim < *oa_ndim; ++idim) {
                PyObject *v = PySequence_GetItem(a, idim);
                if (v == NULL) {
                    Py_DECREF(a);
                    return 0;
                }
                /* numpy.newaxis is None */
                if (v == Py_None) {
                    op_axes[iop][idim] = -1;
                }
                else {
                    op_axes[iop][idim] = PyArray_PyIntAsInt(v);
                    if (op_axes[iop][idim]==-1 &&
                                                PyErr_Occurred()) {
                        Py_DECREF(a);
                        Py_DECREF(v);
                        return 0;
                    }
                }
                Py_DECREF(v);
            }
        }
        Py_DECREF(a);
    }

    if (*oa_ndim == -1) {
        PyErr_SetString(PyExc_ValueError,
                "If op_axes is provided, at least one list of axes "
                "must be contained within it");
        return 0;
    }

    return 1;
}


static int
npyiter_prepare_ops(PyObject *op_in, PyObject **out_owner, PyObject ***out_objs)
{
    /* Take ownership of op_in (either a tuple/list or single element): */
    if (PyTuple_Check(op_in) || PyList_Check(op_in)) {
        PyObject *seq = PySequence_Fast(op_in, "failed accessing item list"); // noqa: borrowed-ref OK
        if (op_in == NULL) {
            Py_DECREF(op_in);
            return -1;
        }
        Py_ssize_t length = PySequence_Fast_GET_SIZE(op_in);
        if (length == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Must provide at least one operand");
            Py_DECREF(op_in);
            return -1;
        }
        if (length > NPY_MAX_INT) {
            /* NpyIter supports fewer args, but deal with it there. */
            PyErr_Format(PyExc_ValueError,
                "Too many operands to nditer, found %zd.", length);
            Py_DECREF(op_in);
            return -1;
        }
        *out_objs = PySequence_Fast_ITEMS(op_in);
        *out_owner = seq;
        return (int)length;
    }
    else {
        Py_INCREF(op_in);
        *out_objs = out_owner;  /* `out_owner` is in caller stack space */
        *out_owner = op_in;
        return 1;
    }
}

/*
 * Converts the operand array and op_flags array into the form
 * NpyIter_AdvancedNew needs.  On success, each op[i] owns a reference
 * to an array object.
 */
static int
npyiter_convert_ops(int nop, PyObject **op_objs, PyObject *op_flags_in,
                    PyArrayObject **op, npy_uint32 *op_flags)
{
    /* op_flags */
    if (op_flags_in == NULL || op_flags_in == Py_None) {
        for (int iop = 0; iop < nop; ++iop) {
            /*
             * By default, make NULL operands writeonly and flagged for
             * allocation, and everything else readonly.  To write
             * to a provided operand, you must specify the write flag manually.
             */
            if (op_objs[iop] == Py_None) {
                op_flags[iop] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
            }
            else {
                op_flags[iop] = NPY_ITER_READONLY;
            }
        }
    }
    else if (npyiter_convert_op_flags_array(op_flags_in,
                                      op_flags, nop) != 1) {
        return 0;
    }

    /* Now that we have the flags - convert all the ops to arrays */
    for (int iop = 0; iop < nop; ++iop) {
        if (op_objs[iop] != Py_None) {
            PyArrayObject *ao;
            int fromanyflags = 0;

            if (op_flags[iop]&(NPY_ITER_READWRITE|NPY_ITER_WRITEONLY)) {
                fromanyflags |= NPY_ARRAY_WRITEBACKIFCOPY;
            }
            ao = (PyArrayObject *)PyArray_FROM_OF((PyObject *)op_objs[iop],
                                                  fromanyflags);
            if (ao == NULL) {
                if (PyErr_Occurred() &&
                            PyErr_ExceptionMatches(PyExc_TypeError)) {
                    PyErr_SetString(PyExc_TypeError,
                            "Iterator operand is flagged as writeable, "
                            "but is an object which cannot be written "
                            "back to via WRITEBACKIFCOPY");
                }
                return 0;
            }
            op[iop] = ao;
        }
    }

    return 1;
}

static int
npyiter_init(NewNpyArrayIterObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"op", "flags", "op_flags", "op_dtypes",
                             "order", "casting", "op_axes", "itershape",
                             "buffersize",
                             NULL};

    PyObject *op_in = NULL, *op_flags_in = NULL,
                *op_dtypes_in = NULL, *op_axes_in = NULL;

    npy_uint32 flags = 0;
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    int oa_ndim = -1;
    PyArray_Dims itershape = {NULL, -1};
    int buffersize = 0;

    int res = -1;

    if (self->iter != NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was already initialized");
        return -1;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&OOO&O&OO&i:nditer", kwlist,
                    &op_in,
                    NpyIter_GlobalFlagsConverter, &flags,
                    &op_flags_in,
                    &op_dtypes_in,
                    PyArray_OrderConverter, &order,
                    PyArray_CastingConverter, &casting,
                    &op_axes_in,
                    PyArray_OptionalIntpConverter, &itershape,
                    &buffersize)) {
        npy_free_cache_dim_obj(itershape);
        return -1;
    }

    /* Need nop to set up workspaces */
    PyObject **op_objs = NULL;
    PyObject *op_in_owned = NULL; /* Sequence/object owning op_objs. */
    PyArray_Descr **op_request_dtypes = NULL;
    int pre_alloc_fail = 0;
    int post_alloc_fail = 0;
    int nop;

    NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(op_in, nditer_cs);

    nop = npyiter_prepare_ops(op_in, &op_in_owned, &op_objs);
    if (nop < 0) {
        pre_alloc_fail = 1;
        goto cleanup;
    }

    /* allocate workspace for Python objects (operands and dtypes) */
    NPY_ALLOC_WORKSPACE(op, PyArrayObject *, 2 * 8, 2 * nop);
    if (op == NULL) {
        pre_alloc_fail = 1;
        goto cleanup;
    }
    memset(op, 0, sizeof(PyObject *) * 2 * nop);
    op_request_dtypes = (PyArray_Descr **)(op + nop);

    /* And other workspaces (that do not need to clean up their content) */
    NPY_ALLOC_WORKSPACE(op_flags, npy_uint32, 8, nop);
    NPY_ALLOC_WORKSPACE(op_axes_storage, int, 8 * NPY_MAXDIMS, nop * NPY_MAXDIMS);
    NPY_ALLOC_WORKSPACE(op_axes, int *, 8, nop);
    /*
     * Trying to allocate should be OK if one failed, check for error now
     * that we can use `goto finish` to clean up everything.
     * (NPY_ALLOC_WORKSPACE has to be done before a goto fail currently.)
     */
    if (op_flags == NULL || op_axes_storage == NULL || op_axes == NULL) {
        post_alloc_fail = 1;
        goto cleanup;
    }

    /* op and op_flags */
    if (npyiter_convert_ops(nop, op_objs, op_flags_in, op, op_flags) != 1) {
        post_alloc_fail = 1;
        goto cleanup;
    }

cleanup:

    NPY_END_CRITICAL_SECTION_SEQUENCE_FAST_NO_BRACKETS(nditer_cs);

    if (pre_alloc_fail) {
        goto pre_alloc_fail;
    }

    if (post_alloc_fail) {
        goto finish;
    }

    /* op_request_dtypes */
    if (op_dtypes_in != NULL && op_dtypes_in != Py_None &&
            npyiter_convert_dtypes(op_dtypes_in,
                                   op_request_dtypes, nop) != 1) {
        goto finish;
    }

    /* op_axes */
    if (op_axes_in != NULL && op_axes_in != Py_None) {
        /* Initialize to point to the op_axes arrays */
        for (int iop = 0; iop < nop; ++iop) {
            op_axes[iop] = &op_axes_storage[iop * NPY_MAXDIMS];
        }

        if (npyiter_convert_op_axes(op_axes_in, nop,
                                    op_axes, &oa_ndim) != 1) {
            goto finish;
        }
    }

    if (itershape.len != -1) {
        if (oa_ndim == -1) {
            oa_ndim = itershape.len;
            memset(op_axes, 0, sizeof(op_axes[0]) * nop);
        }
        else if (oa_ndim != itershape.len) {
            PyErr_SetString(PyExc_ValueError,
                        "'op_axes' and 'itershape' must have the same number "
                        "of entries equal to the iterator ndim");
            goto finish;
        }
    }

    self->iter = NpyIter_AdvancedNew(nop, op, flags, order, casting, op_flags,
                                  op_request_dtypes,
                                  oa_ndim, oa_ndim >= 0 ? op_axes : NULL,
                                  itershape.ptr,
                                  buffersize);

    if (self->iter == NULL) {
        goto finish;
    }

    /* Cache some values for the member functions to use */
    if (npyiter_cache_values(self) < 0) {
        goto finish;
    }

    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    res = 0;

  finish:
    for (int iop = 0; iop < nop; ++iop) {
        Py_XDECREF(op[iop]);
        Py_XDECREF(op_request_dtypes[iop]);
    }
    npy_free_workspace(op);
    npy_free_workspace(op_flags);
    npy_free_workspace(op_axes_storage);
    npy_free_workspace(op_axes);

  pre_alloc_fail:
    Py_XDECREF(op_in_owned);
    npy_free_cache_dim_obj(itershape);
    return res;
}


NPY_NO_EXPORT PyObject *
NpyIter_NestedIters(PyObject *NPY_UNUSED(self),
                    PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"op", "axes", "flags", "op_flags",
                             "op_dtypes", "order",
                             "casting", "buffersize",
                             NULL};

    PyObject *op_in = NULL, *axes_in = NULL,
            *op_flags_in = NULL, *op_dtypes_in = NULL;

    int iop, inest, nnest = 0;
    npy_uint32 flags = 0, flags_inner;
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_SAFE_CASTING;

    int op_axes_data[NPY_MAXDIMS];
    int *nested_op_axes[NPY_MAXDIMS];
    int nested_naxes[NPY_MAXDIMS], iaxes, naxes;
    int negones[NPY_MAXDIMS];
    char used_axes[NPY_MAXDIMS];
    int buffersize = 0;

    PyObject *res = NULL;  /* returned */
    PyObject *ret = NULL;  /* intermediate object on failure */

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O&OOO&O&i", kwlist,
                    &op_in,
                    &axes_in,
                    NpyIter_GlobalFlagsConverter, &flags,
                    &op_flags_in,
                    &op_dtypes_in,
                    PyArray_OrderConverter, &order,
                    PyArray_CastingConverter, &casting,
                    &buffersize)) {
        return NULL;
    }

    /* axes */
    if (!PyTuple_Check(axes_in) && !PyList_Check(axes_in)) {
        PyErr_SetString(PyExc_ValueError,
                "axes must be a tuple of axis arrays");
        return NULL;
    }
    nnest = PySequence_Size(axes_in);
    if (nnest < 2) {
        PyErr_SetString(PyExc_ValueError,
                "axes must have at least 2 entries for nested iteration");
        return NULL;
    }
    naxes = 0;
    memset(used_axes, 0, NPY_MAXDIMS);
    for (inest = 0; inest < nnest; ++inest) {
        PyObject *item = PySequence_GetItem(axes_in, inest);
        npy_intp i;
        if (item == NULL) {
            return NULL;
        }
        if (!PyTuple_Check(item) && !PyList_Check(item)) {
            PyErr_SetString(PyExc_ValueError,
                    "Each item in axes must be a an integer tuple");
            Py_DECREF(item);
            return NULL;
        }
        nested_naxes[inest] = PySequence_Size(item);
        if (naxes + nested_naxes[inest] > NPY_MAXDIMS) {
            PyErr_SetString(PyExc_ValueError,
                    "Too many axes given");
            Py_DECREF(item);
            return NULL;
        }
        for (i = 0; i < nested_naxes[inest]; ++i) {
            PyObject *v = PySequence_GetItem(item, i);
            npy_intp axis;
            if (v == NULL) {
                Py_DECREF(item);
                return NULL;
            }
            axis = PyLong_AsLong(v);
            Py_DECREF(v);
            if (axis < 0 || axis >= NPY_MAXDIMS) {
                PyErr_SetString(PyExc_ValueError,
                        "An axis is out of bounds");
                Py_DECREF(item);
                return NULL;
            }
            /*
             * This check is very important, without it out of bounds
             * data accesses are possible.
             */
            if (used_axes[axis] != 0) {
                PyErr_SetString(PyExc_ValueError,
                        "An axis is used more than once");
                Py_DECREF(item);
                return NULL;
            }
            used_axes[axis] = 1;
            op_axes_data[naxes+i] = axis;
        }
        nested_op_axes[inest] = &op_axes_data[naxes];
        naxes += nested_naxes[inest];
        Py_DECREF(item);
    }

    /* Need nop to set up workspaces */
    PyObject **op_objs = NULL;
    PyObject *op_in_owned;  /* Sequence/object owning op_objs. */
    int nop = npyiter_prepare_ops(op_in, &op_in_owned, &op_objs);
    if (nop < 0) {
        return NULL;
    }

    /* allocate workspace for Python objects (operands and dtypes) */
    NPY_ALLOC_WORKSPACE(op, PyArrayObject *, 3 * 8, 3 * nop);
    if (op == NULL) {
        Py_DECREF(op_in_owned);
        return NULL;
    }
    memset(op, 0, sizeof(PyObject *) * 3 * nop);
    PyArray_Descr **op_request_dtypes = (PyArray_Descr **)(op + nop);
    PyArray_Descr **op_request_dtypes_inner = op_request_dtypes + nop;

    /* And other workspaces (that do not need to clean up their content) */
    NPY_ALLOC_WORKSPACE(op_flags, npy_uint32, 8, nop);
    NPY_ALLOC_WORKSPACE(op_flags_inner, npy_uint32, 8, nop);
    NPY_ALLOC_WORKSPACE(op_axes_storage, int, 8 * NPY_MAXDIMS, nop * NPY_MAXDIMS);
    NPY_ALLOC_WORKSPACE(op_axes, int *, 2 * 8, 2 * nop);
    /*
     * Trying to allocate should be OK if one failed, check for error now
     * that we can use `goto finish` to clean up everything.
     * (NPY_ALLOC_WORKSPACE has to be done before a goto fail currently.)
     */
    if (op_flags == NULL || op_axes_storage == NULL || op_axes == NULL) {
        goto finish;
    }
    /* Finalize shared workspace: */
    int **op_axes_nop = op_axes + nop;

    /* op and op_flags */
    if (npyiter_convert_ops(nop, op_objs, op_flags_in, op, op_flags) != 1) {
        goto finish;
    }

    /* op_request_dtypes */
    if (op_dtypes_in != NULL && op_dtypes_in != Py_None &&
            npyiter_convert_dtypes(op_dtypes_in,
                                   op_request_dtypes, nop) != 1) {
        goto finish;
    }

    ret = PyTuple_New(nnest);
    if (ret == NULL) {
        goto finish;
    }

    /* For broadcasting allocated arrays */
    for (iaxes = 0; iaxes < naxes; ++iaxes) {
        negones[iaxes] = -1;
    }

    /*
     * Clear any unnecessary ALLOCATE flags, so we can use them
     * to indicate exactly the allocated outputs.  Also, separate
     * the inner loop flags.
     */
    for (iop = 0; iop < nop; ++iop) {
        if ((op_flags[iop]&NPY_ITER_ALLOCATE) && op[iop] != NULL) {
            op_flags[iop] &= ~NPY_ITER_ALLOCATE;
        }

        /*
         * Clear any flags allowing copies or output allocation for
         * the inner loop.
         */
        op_flags_inner[iop] = op_flags[iop] & ~(NPY_ITER_COPY|
                             NPY_ITER_UPDATEIFCOPY|
                             NPY_ITER_ALLOCATE);
        /*
         * If buffering is enabled and copying is not,
         * clear the nbo_aligned flag and strip the data type
         * for the outer loops.
         */
        if ((flags&(NPY_ITER_BUFFERED)) &&
                !(op_flags[iop]&(NPY_ITER_COPY|
                                   NPY_ITER_UPDATEIFCOPY|
                                   NPY_ITER_ALLOCATE))) {
            op_flags[iop] &= ~(NPY_ITER_NBO|NPY_ITER_ALIGNED|NPY_ITER_CONTIG);
            op_request_dtypes_inner[iop] = op_request_dtypes[iop];
            op_request_dtypes[iop] = NULL;
        }
    }

    /* Only the inner loop gets the buffering and no inner flags */
    flags_inner = flags&~NPY_ITER_COMMON_DTYPE;
    flags &= ~(NPY_ITER_EXTERNAL_LOOP|
                    NPY_ITER_BUFFERED);

    for (inest = 0; inest < nnest; ++inest) {
        NewNpyArrayIterObject *iter;
        /*
         * All the operands' op_axes are the same, except for
         * allocated outputs.
         */
        for (iop = 0; iop < nop; ++iop) {
            if (op_flags[iop]&NPY_ITER_ALLOCATE) {
                if (inest == 0) {
                    op_axes_nop[iop] = NULL;
                }
                else {
                    op_axes_nop[iop] = negones;
                }
            }
            else {
                op_axes_nop[iop] = nested_op_axes[inest];
            }
        }

        /*
        printf("\n");
        for (iop = 0; iop < nop; ++iop) {
            npy_intp i;

            for (i = 0; i < nested_naxes[inest]; ++i) {
                printf("%d ", (int)op_axes_nop[iop][i]);
            }
            printf("\n");
        }
        */

        /* Allocate the iterator */
        iter = (NewNpyArrayIterObject *)npyiter_new(&NpyIter_Type, NULL, NULL);
        if (iter == NULL) {
            goto finish;
        }

        /* Store iter into return tuple (owns the reference). */
        PyTuple_SET_ITEM(ret, inest, (PyObject *)iter);

        if (inest < nnest-1) {
            iter->iter = NpyIter_AdvancedNew(nop, op, flags, order,
                                casting, op_flags, op_request_dtypes,
                                nested_naxes[inest], op_axes_nop,
                                NULL,
                                0);
        }
        else {
            iter->iter = NpyIter_AdvancedNew(nop, op, flags_inner, order,
                                casting, op_flags_inner,
                                op_request_dtypes_inner,
                                nested_naxes[inest], op_axes_nop,
                                NULL,
                                buffersize);
        }

        if (iter->iter == NULL) {
            goto finish;
        }

        /* Cache some values for the member functions to use */
        if (npyiter_cache_values(iter) < 0) {
            goto finish;
        }

        if (NpyIter_GetIterSize(iter->iter) == 0) {
            iter->started = 1;
            iter->finished = 1;
        }
        else {
            iter->started = 0;
            iter->finished = 0;
        }

        /*
         * If there are any allocated outputs or any copies were made,
         * adjust op so that the other iterators use the same ones.
         */
        if (inest == 0) {
            PyArrayObject **operands = NpyIter_GetOperandArray(iter->iter);
            for (iop = 0; iop < nop; ++iop) {
                if (op[iop] != operands[iop]) {
                    Py_XDECREF(op[iop]);
                    op[iop] = operands[iop];
                    Py_INCREF(op[iop]);
                }

                /*
                 * Clear any flags allowing copies for
                 * the rest of the iterators
                 */
                op_flags[iop] &= ~(NPY_ITER_COPY|
                                 NPY_ITER_UPDATEIFCOPY);
            }
            /* Clear the common dtype flag for the rest of the iterators */
            flags &= ~NPY_ITER_COMMON_DTYPE;
        }
    }

    /* Set up the nested child references */
    for (inest = 0; inest < nnest-1; ++inest) {
        NewNpyArrayIterObject *iter;
        iter = (NewNpyArrayIterObject *)PyTuple_GET_ITEM(ret, inest);
        /*
         * Indicates which iterator to reset with new base pointers
         * each iteration step.
         */
        iter->nested_child =
                (NewNpyArrayIterObject *)PyTuple_GET_ITEM(ret, inest+1);
        Py_INCREF(iter->nested_child);
        /*
         * Need to do a nested reset so all the iterators point
         * at the right data
         */
        if (NpyIter_ResetBasePointers(iter->nested_child->iter,
                                iter->dataptrs, NULL) != NPY_SUCCEED) {
            goto finish;
        }
    }

    res = Py_NewRef(ret);

finish:
    Py_DECREF(op_in_owned);
    Py_XDECREF(ret);

    for (iop = 0; iop < nop; ++iop) {
        Py_XDECREF(op[iop]);
        Py_XDECREF(op_request_dtypes[iop]);
        Py_XDECREF(op_request_dtypes_inner[iop]);
    }

    npy_free_workspace(op);
    npy_free_workspace(op_flags);
    npy_free_workspace(op_flags_inner);
    npy_free_workspace(op_axes_storage);
    npy_free_workspace(op_axes);

    return res;
}


static void
npyiter_dealloc(NewNpyArrayIterObject *self)
{
    if (self->iter) {
        /* Store error, so that WriteUnraisable cannot clear an existing one */
        PyObject *exc, *val, *tb;
        PyErr_Fetch(&exc, &val, &tb);
        if (npyiter_has_writeback(self->iter)) {
            if (PyErr_WarnEx(PyExc_RuntimeWarning,
                    "Temporary data has not been written back to one of the "
                    "operands. Typically nditer is used as a context manager "
                    "otherwise 'close' must be called before reading iteration "
                    "results.", 1) < 0) {
                PyObject *s;

                s = PyUnicode_FromString("npyiter_dealloc");
                if (s) {
                    PyErr_WriteUnraisable(s);
                    Py_DECREF(s);
                }
                else {
                    PyErr_WriteUnraisable(Py_None);
                }
            }
        }
        if (!NpyIter_Deallocate(self->iter)) {
            PyErr_WriteUnraisable(Py_None);
        }
        self->iter = NULL;
        Py_XDECREF(self->nested_child);
        self->nested_child = NULL;
        PyErr_Restore(exc, val, tb);
    }
    PyMem_Free(self->writeflags);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
npyiter_resetbasepointers(NewNpyArrayIterObject *self)
{
    while (self->nested_child) {
        if (NpyIter_ResetBasePointers(self->nested_child->iter,
                                        self->dataptrs, NULL) != NPY_SUCCEED) {
            return NPY_FAIL;
        }
        self = self->nested_child;
        if (NpyIter_GetIterSize(self->iter) == 0) {
            self->started = 1;
            self->finished = 1;
        }
        else {
            self->started = 0;
            self->finished = 0;
        }
    }

    return NPY_SUCCEED;
}

static PyObject *
npyiter_reset(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    if (NpyIter_Reset(self->iter, NULL) != NPY_SUCCEED) {
        return NULL;
    }
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    if (self->get_multi_index == NULL && NpyIter_HasMultiIndex(self->iter)) {
        self->get_multi_index = NpyIter_GetGetMultiIndex(self->iter, NULL);
    }

    /* If there is nesting, the nested iterators should be reset */
    if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
        return NULL;
    }

    Py_RETURN_NONE;
}

/*
 * Makes a copy of the iterator.  Note that the nesting is not
 * copied.
 */
static PyObject *
npyiter_copy(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    NewNpyArrayIterObject *iter;

    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    /* Allocate the iterator */
    iter = (NewNpyArrayIterObject *)npyiter_new(&NpyIter_Type, NULL, NULL);
    if (iter == NULL) {
        return NULL;
    }

    /* Copy the C iterator */
    iter->iter = NpyIter_Copy(self->iter);
    if (iter->iter == NULL) {
        Py_DECREF(iter);
        return NULL;
    }

    /* Cache some values for the member functions to use */
    if (npyiter_cache_values(iter) < 0) {
        Py_DECREF(iter);
        return NULL;
    }

    iter->started = self->started;
    iter->finished = self->finished;

    return (PyObject *)iter;
}

static PyObject *
npyiter_iternext(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    if (self->iter != NULL && self->iternext != NULL &&
                        !self->finished && self->iternext(self->iter)) {
        /* If there is nesting, the nested iterators should be reset */
        if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
            return NULL;
        }

        Py_RETURN_TRUE;
    }
    else {
        if (PyErr_Occurred()) {
            /* casting error, buffer cleanup will occur at reset or dealloc */
            return NULL;
        }
        self->finished = 1;
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_remove_axis(NewNpyArrayIterObject *self, PyObject *args)
{
    int axis = 0;

    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "i:remove_axis", &axis)) {
        return NULL;
    }

    if (NpyIter_RemoveAxis(self->iter, axis) != NPY_SUCCEED) {
        return NULL;
    }
    /* RemoveAxis invalidates cached values */
    if (npyiter_cache_values(self) < 0) {
        return NULL;
    }
    /* RemoveAxis also resets the iterator */
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    Py_RETURN_NONE;
}

static PyObject *
npyiter_remove_multi_index(
    NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    NpyIter_RemoveMultiIndex(self->iter);
    /* RemoveMultiIndex invalidates cached values */
    if (npyiter_cache_values(self) < 0) {
        return NULL;
    }
    /* RemoveMultiIndex also resets the iterator */
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    Py_RETURN_NONE;
}

static PyObject *
npyiter_enable_external_loop(
    NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    NpyIter_EnableExternalLoop(self->iter);
    /* EnableExternalLoop invalidates cached values */
    if (npyiter_cache_values(self) < 0) {
        return NULL;
    }
    /* EnableExternalLoop also resets the iterator */
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    Py_RETURN_NONE;
}

static PyObject *
npyiter_debug_print(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    if (self->iter != NULL) {
        NpyIter_DebugPrint(self->iter);
    }
    else {
        printf("Iterator: (nil)\n");
    }

    Py_RETURN_NONE;
}

NPY_NO_EXPORT PyObject *
npyiter_seq_item(NewNpyArrayIterObject *self, Py_ssize_t i);

static PyObject *
npyiter_value_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;

    npy_intp iop, nop;

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    nop = NpyIter_GetNOp(self->iter);

    /* Return an array  or tuple of arrays with the values */
    if (nop == 1) {
        ret = npyiter_seq_item(self, 0);
    }
    else {
        ret = PyTuple_New(nop);
        if (ret == NULL) {
            return NULL;
        }
        for (iop = 0; iop < nop; ++iop) {
            PyObject *a = npyiter_seq_item(self, iop);
            if (a == NULL) {
                Py_DECREF(ret);
                return NULL;
            }
            PyTuple_SET_ITEM(ret, iop, a);
        }
    }

    return ret;
}

static PyObject *
npyiter_operands_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;

    npy_intp iop, nop;
    PyArrayObject **operands;

    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }
    nop = NpyIter_GetNOp(self->iter);
    operands = self->operands;

    ret = PyTuple_New(nop);
    if (ret == NULL) {
        return NULL;
    }
    for (iop = 0; iop < nop; ++iop) {
        PyObject *operand = (PyObject *)operands[iop];

        Py_INCREF(operand);
        PyTuple_SET_ITEM(ret, iop, operand);
    }

    return ret;
}

static PyObject *
npyiter_itviews_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;

    npy_intp iop, nop;

    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }
    nop = NpyIter_GetNOp(self->iter);

    ret = PyTuple_New(nop);
    if (ret == NULL) {
        return NULL;
    }
    for (iop = 0; iop < nop; ++iop) {
        PyArrayObject *view = NpyIter_GetIterView(self->iter, iop);

        if (view == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
        PyTuple_SET_ITEM(ret, iop, (PyObject *)view);
    }

    return ret;
}

static PyObject *
npyiter_next(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL || self->iternext == NULL ||
                self->finished) {
        return NULL;
    }

    /*
     * Use the started flag for the Python iteration protocol to work
     * when buffering is enabled.
     */
    if (self->started) {
        if (!self->iternext(self->iter)) {
            /*
             * A casting error may be set here (or no error causing a
             * StopIteration). Buffers may only be cleaned up later.
             */
            self->finished = 1;
            return NULL;
        }

        /* If there is nesting, the nested iterators should be reset */
        if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
            return NULL;
        }
    }
    self->started = 1;

    return npyiter_value_get(self, NULL);
};

static PyObject *
npyiter_shape_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    npy_intp ndim, shape[NPY_MAXDIMS];

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    if (NpyIter_GetShape(self->iter, shape) == NPY_SUCCEED) {
        ndim = NpyIter_GetNDim(self->iter);
        return PyArray_IntTupleFromIntp(ndim, shape);
    }

    return NULL;
}

static PyObject *
npyiter_multi_index_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    npy_intp ndim, multi_index[NPY_MAXDIMS];

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    if (self->get_multi_index != NULL) {
        ndim = NpyIter_GetNDim(self->iter);
        self->get_multi_index(self->iter, multi_index);
        return PyArray_IntTupleFromIntp(ndim, multi_index);
    }
    else {
        if (!NpyIter_HasMultiIndex(self->iter)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator is not tracking a multi-index");
            return NULL;
        }
        else if (NpyIter_HasDelayedBufAlloc(self->iter)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator construction used delayed buffer allocation, "
                    "and no reset has been done yet");
            return NULL;
        }
        else {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator is in an invalid state");
            return NULL;
        }
    }
}

static int
npyiter_multi_index_set(
        NewNpyArrayIterObject *self, PyObject *value, void *NPY_UNUSED(ignored))
{
    npy_intp idim, ndim, multi_index[NPY_MAXDIMS];

    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete nditer multi_index");
        return -1;
    }
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return -1;
    }

    if (NpyIter_HasMultiIndex(self->iter)) {
        ndim = NpyIter_GetNDim(self->iter);
        if (!PySequence_Check(value)) {
            PyErr_SetString(PyExc_ValueError,
                    "multi_index must be set with a sequence");
            return -1;
        }
        if (PySequence_Size(value) != ndim) {
            PyErr_SetString(PyExc_ValueError,
                    "Wrong number of indices");
            return -1;
        }
        for (idim = 0; idim < ndim; ++idim) {
            PyObject *v = PySequence_GetItem(value, idim);
            multi_index[idim] = PyLong_AsLong(v);
            Py_DECREF(v);
            if (error_converting(multi_index[idim])) {
                return -1;
            }
        }
        if (NpyIter_GotoMultiIndex(self->iter, multi_index) != NPY_SUCCEED) {
            return -1;
        }
        self->started = 0;
        self->finished = 0;

        /* If there is nesting, the nested iterators should be reset */
        if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
            return -1;
        }

        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is not tracking a multi-index");
        return -1;
    }
}

static PyObject *
npyiter_index_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    if (NpyIter_HasIndex(self->iter)) {
        npy_intp ind = *NpyIter_GetIndexPtr(self->iter);
        return PyLong_FromLong(ind);
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator does not have an index");
        return NULL;
    }
}

static int
npyiter_index_set(
        NewNpyArrayIterObject *self, PyObject *value, void *NPY_UNUSED(ignored))
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete nditer index");
        return -1;
    }
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return -1;
    }

    if (NpyIter_HasIndex(self->iter)) {
        npy_intp ind;
        ind = PyLong_AsLong(value);
        if (error_converting(ind)) {
            return -1;
        }
        if (NpyIter_GotoIndex(self->iter, ind) != NPY_SUCCEED) {
            return -1;
        }
        self->started = 0;
        self->finished = 0;

        /* If there is nesting, the nested iterators should be reset */
        if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
            return -1;
        }

        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator does not have an index");
        return -1;
    }
}

static PyObject *
npyiter_iterindex_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    return PyLong_FromLong(NpyIter_GetIterIndex(self->iter));
}

static int
npyiter_iterindex_set(
        NewNpyArrayIterObject *self, PyObject *value, void *NPY_UNUSED(ignored))
{
    npy_intp iterindex;

    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete nditer iterindex");
        return -1;
    }
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return -1;
    }

    iterindex = PyLong_AsLong(value);
    if (error_converting(iterindex)) {
        return -1;
    }
    if (NpyIter_GotoIterIndex(self->iter, iterindex) != NPY_SUCCEED) {
        return -1;
    }
    self->started = 0;
    self->finished = 0;

    /* If there is nesting, the nested iterators should be reset */
    if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
        return -1;
    }

    return 0;
}

static PyObject *
npyiter_iterrange_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    npy_intp istart = 0, iend = 0;
    PyObject *ret;

    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    NpyIter_GetIterIndexRange(self->iter, &istart, &iend);

    ret = PyTuple_New(2);
    if (ret == NULL) {
        return NULL;
    }

    PyTuple_SET_ITEM(ret, 0, PyLong_FromLong(istart));
    PyTuple_SET_ITEM(ret, 1, PyLong_FromLong(iend));

    return ret;
}

static int
npyiter_iterrange_set(
        NewNpyArrayIterObject *self, PyObject *value, void *NPY_UNUSED(ignored))
{
    npy_intp istart = 0, iend = 0;

    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete nditer iterrange");
        return -1;
    }
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return -1;
    }

    if (!PyArg_ParseTuple(value, "nn", &istart, &iend)) {
        return -1;
    }

    if (NpyIter_ResetToIterIndexRange(self->iter, istart, iend, NULL)
                                                    != NPY_SUCCEED) {
        return -1;
    }
    if (istart < iend) {
        self->started = self->finished = 0;
    }
    else {
        self->started = self->finished = 1;
    }

    if (self->get_multi_index == NULL && NpyIter_HasMultiIndex(self->iter)) {
        self->get_multi_index = NpyIter_GetGetMultiIndex(self->iter, NULL);
    }

    /* If there is nesting, the nested iterators should be reset */
    if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
        return -1;
    }

    return 0;
}

static PyObject *
npyiter_has_delayed_bufalloc_get(
        NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_iterationneedsapi_get(
        NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    if (NpyIter_IterationNeedsAPI(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_has_multi_index_get(
        NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    if (NpyIter_HasMultiIndex(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_has_index_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    if (NpyIter_HasIndex(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_dtypes_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;

    npy_intp iop, nop;
    PyArray_Descr **dtypes;

    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }
    nop = NpyIter_GetNOp(self->iter);

    ret = PyTuple_New(nop);
    if (ret == NULL) {
        return NULL;
    }
    dtypes = self->dtypes;
    for (iop = 0; iop < nop; ++iop) {
        PyArray_Descr *dtype = dtypes[iop];

        Py_INCREF(dtype);
        PyTuple_SET_ITEM(ret, iop, (PyObject *)dtype);
    }

    return ret;
}

static PyObject *
npyiter_ndim_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    return PyLong_FromLong(NpyIter_GetNDim(self->iter));
}

static PyObject *
npyiter_nop_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    return PyLong_FromLong(NpyIter_GetNOp(self->iter));
}

static PyObject *
npyiter_itersize_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    return PyLong_FromLong(NpyIter_GetIterSize(self->iter));
}

static PyObject *
npyiter_finished_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL || !self->finished) {
        Py_RETURN_FALSE;
    }
    else {
        Py_RETURN_TRUE;
    }
}

NPY_NO_EXPORT Py_ssize_t
npyiter_seq_length(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL) {
        return 0;
    }
    else {
        return NpyIter_GetNOp(self->iter);
    }
}

NPY_NO_EXPORT PyObject *
npyiter_seq_item(NewNpyArrayIterObject *self, Py_ssize_t i)
{
    npy_intp ret_ndim;
    npy_intp nop, innerloopsize, innerstride;
    char *dataptr;
    PyArray_Descr *dtype;
    int has_external_loop;
    Py_ssize_t i_orig = i;

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return NULL;
    }
    nop = NpyIter_GetNOp(self->iter);

    /* Negative indexing */
    if (i < 0) {
        i += nop;
    }

    if (i < 0 || i >= nop) {
        PyErr_Format(PyExc_IndexError,
                "Iterator operand index %zd is out of bounds", i_orig);
        return NULL;
    }

    dataptr = self->dataptrs[i];
    dtype = self->dtypes[i];
    has_external_loop = NpyIter_HasExternalLoop(self->iter);

    if (has_external_loop) {
        innerloopsize = *self->innerloopsizeptr;
        innerstride = self->innerstrides[i];
        ret_ndim = 1;
    }
    else {
        innerloopsize = 1;
        innerstride = 0;
        /* If the iterator is going over every element, return array scalars */
        ret_ndim = 0;
    }

    Py_INCREF(dtype);
    return PyArray_NewFromDescrAndBase(
            &PyArray_Type, dtype,
            ret_ndim, &innerloopsize, &innerstride, dataptr,
            self->writeflags[i] ? NPY_ARRAY_WRITEABLE : 0,
            NULL, (PyObject *)self);
}

NPY_NO_EXPORT PyObject *
npyiter_seq_slice(NewNpyArrayIterObject *self,
                    Py_ssize_t ilow, Py_ssize_t ihigh)
{
    PyObject *ret;
    npy_intp nop;
    Py_ssize_t i;

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return NULL;
    }
    nop = NpyIter_GetNOp(self->iter);
    if (ilow < 0) {
        ilow = 0;
    }
    else if (ilow >= nop) {
        ilow = nop-1;
    }
    if (ihigh < ilow) {
        ihigh = ilow;
    }
    else if (ihigh > nop) {
        ihigh = nop;
    }

    ret = PyTuple_New(ihigh-ilow);
    if (ret == NULL) {
        return NULL;
    }
    for (i = ilow; i < ihigh ; ++i) {
        PyObject *item = npyiter_seq_item(self, i);
        if (item == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
        PyTuple_SET_ITEM(ret, i-ilow, item);
    }
    return ret;
}

NPY_NO_EXPORT int
npyiter_seq_ass_item(NewNpyArrayIterObject *self, Py_ssize_t i, PyObject *v)
{

    npy_intp nop, innerloopsize, innerstride;
    char *dataptr;
    PyArray_Descr *dtype;
    PyArrayObject *tmp;
    int ret, has_external_loop;
    Py_ssize_t i_orig = i;


    if (v == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Cannot delete iterator elements");
        return -1;
    }

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return -1;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return -1;
    }
    nop = NpyIter_GetNOp(self->iter);

    /* Negative indexing */
    if (i < 0) {
        i += nop;
    }

    if (i < 0 || i >= nop) {
        PyErr_Format(PyExc_IndexError,
                "Iterator operand index %zd is out of bounds", i_orig);
        return -1;
    }
    if (!self->writeflags[i]) {
        PyErr_Format(PyExc_RuntimeError,
                "Iterator operand %zd is not writeable", i_orig);
        return -1;
    }

    dataptr = self->dataptrs[i];
    dtype = self->dtypes[i];
    has_external_loop = NpyIter_HasExternalLoop(self->iter);

    if (has_external_loop) {
        innerloopsize = *self->innerloopsizeptr;
        innerstride = self->innerstrides[i];
    }
    else {
        innerloopsize = 1;
        innerstride = 0;
    }

    /* TODO - there should be a better way than this... */
    Py_INCREF(dtype);
    tmp = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                                1, &innerloopsize,
                                &innerstride, dataptr,
                                NPY_ARRAY_WRITEABLE, NULL);
    if (tmp == NULL) {
        return -1;
    }

    ret = PyArray_CopyObject(tmp, v);
    Py_DECREF(tmp);
    return ret;
}

static int
npyiter_seq_ass_slice(NewNpyArrayIterObject *self, Py_ssize_t ilow,
                Py_ssize_t ihigh, PyObject *v)
{
    npy_intp nop;
    Py_ssize_t i;

    if (v == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Cannot delete iterator elements");
        return -1;
    }

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return -1;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return -1;
    }
    nop = NpyIter_GetNOp(self->iter);
    if (ilow < 0) {
        ilow = 0;
    }
    else if (ilow >= nop) {
        ilow = nop-1;
    }
    if (ihigh < ilow) {
        ihigh = ilow;
    }
    else if (ihigh > nop) {
        ihigh = nop;
    }

    if (!PySequence_Check(v) || PySequence_Size(v) != ihigh-ilow) {
        PyErr_SetString(PyExc_ValueError,
                "Wrong size to assign to iterator slice");
        return -1;
    }

    for (i = ilow; i < ihigh ; ++i) {
        PyObject *item = PySequence_GetItem(v, i-ilow);
        if (item == NULL) {
            return -1;
        }
        if (npyiter_seq_ass_item(self, i, item) < 0) {
            Py_DECREF(item);
            return -1;
        }
        Py_DECREF(item);
    }

    return 0;
}

static PyObject *
npyiter_subscript(NewNpyArrayIterObject *self, PyObject *op)
{
    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return NULL;
    }

    if (PyLong_Check(op) ||
                    (PyIndex_Check(op) && !PySequence_Check(op))) {
        npy_intp i = PyArray_PyIntAsIntp(op);
        if (error_converting(i)) {
            return NULL;
        }
        return npyiter_seq_item(self, i);
    }
    else if (PySlice_Check(op)) {
        Py_ssize_t istart = 0, iend = 0, istep = 0, islicelength;
        if (PySlice_GetIndicesEx(op, NpyIter_GetNOp(self->iter),
                                 &istart, &iend, &istep, &islicelength) < 0) {
            return NULL;
        }
        if (istep != 1) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator slicing only supports a step of 1");
            return NULL;
        }
        return npyiter_seq_slice(self, istart, iend);
    }

    PyErr_SetString(PyExc_TypeError,
            "invalid index type for iterator indexing");
    return NULL;
}

static int
npyiter_ass_subscript(NewNpyArrayIterObject *self, PyObject *op,
                        PyObject *value)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Cannot delete iterator elements");
        return -1;
    }
    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return -1;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return -1;
    }

    if (PyLong_Check(op) ||
                    (PyIndex_Check(op) && !PySequence_Check(op))) {
        npy_intp i = PyArray_PyIntAsIntp(op);
        if (error_converting(i)) {
            return -1;
        }
        return npyiter_seq_ass_item(self, i, value);
    }
    else if (PySlice_Check(op)) {
        Py_ssize_t istart = 0, iend = 0, istep = 0, islicelength = 0;
        if (PySlice_GetIndicesEx(op, NpyIter_GetNOp(self->iter),
                                 &istart, &iend, &istep, &islicelength) < 0) {
            return -1;
        }
        if (istep != 1) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator slice assignment only supports a step of 1");
            return -1;
        }
        return npyiter_seq_ass_slice(self, istart, iend, value);
    }

    PyErr_SetString(PyExc_TypeError,
            "invalid index type for iterator indexing");
    return -1;
}

static PyObject *
npyiter_enter(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "operation on non-initialized iterator");
        return NULL;
    }
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
npyiter_close(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    NpyIter *iter = self->iter;
    int ret;
    if (self->iter == NULL) {
        Py_RETURN_NONE;
    }
    ret = NpyIter_Deallocate(iter);
    self->iter = NULL;
    Py_XDECREF(self->nested_child);
    self->nested_child = NULL;
    if (ret != NPY_SUCCEED) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
npyiter_exit(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    /* even if called via exception handling, writeback any data */
    return npyiter_close(self, NULL);
}

static PyMethodDef npyiter_methods[] = {
    {"reset",
        (PyCFunction)npyiter_reset,
        METH_NOARGS, NULL},
    {"copy",
        (PyCFunction)npyiter_copy,
        METH_NOARGS, NULL},
    {"__copy__",
        (PyCFunction)npyiter_copy,
        METH_NOARGS, NULL},
    {"iternext",
        (PyCFunction)npyiter_iternext,
        METH_NOARGS, NULL},
    {"remove_axis",
        (PyCFunction)npyiter_remove_axis,
        METH_VARARGS, NULL},
    {"remove_multi_index",
        (PyCFunction)npyiter_remove_multi_index,
        METH_NOARGS, NULL},
    {"enable_external_loop",
        (PyCFunction)npyiter_enable_external_loop,
        METH_NOARGS, NULL},
    {"debug_print",
        (PyCFunction)npyiter_debug_print,
        METH_NOARGS, NULL},
    {"__enter__", (PyCFunction)npyiter_enter,
         METH_NOARGS,  NULL},
    {"__exit__",  (PyCFunction)npyiter_exit,
         METH_VARARGS, NULL},
    {"close",  (PyCFunction)npyiter_close,
         METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static PyMemberDef npyiter_members[] = {
    {NULL, 0, 0, 0, NULL},
};

static PyGetSetDef npyiter_getsets[] = {
    {"value",
        (getter)npyiter_value_get,
        NULL, NULL, NULL},
    {"shape",
        (getter)npyiter_shape_get,
        NULL, NULL, NULL},
    {"multi_index",
        (getter)npyiter_multi_index_get,
        (setter)npyiter_multi_index_set,
        NULL, NULL},
    {"index",
        (getter)npyiter_index_get,
        (setter)npyiter_index_set,
        NULL, NULL},
    {"iterindex",
        (getter)npyiter_iterindex_get,
        (setter)npyiter_iterindex_set,
        NULL, NULL},
    {"iterrange",
        (getter)npyiter_iterrange_get,
        (setter)npyiter_iterrange_set,
        NULL, NULL},
    {"operands",
        (getter)npyiter_operands_get,
        NULL, NULL, NULL},
    {"itviews",
        (getter)npyiter_itviews_get,
        NULL, NULL, NULL},
    {"has_delayed_bufalloc",
        (getter)npyiter_has_delayed_bufalloc_get,
        NULL, NULL, NULL},
    {"iterationneedsapi",
        (getter)npyiter_iterationneedsapi_get,
        NULL, NULL, NULL},
    {"has_multi_index",
        (getter)npyiter_has_multi_index_get,
        NULL, NULL, NULL},
    {"has_index",
        (getter)npyiter_has_index_get,
        NULL, NULL, NULL},
    {"dtypes",
        (getter)npyiter_dtypes_get,
        NULL, NULL, NULL},
    {"ndim",
        (getter)npyiter_ndim_get,
        NULL, NULL, NULL},
    {"nop",
        (getter)npyiter_nop_get,
        NULL, NULL, NULL},
    {"itersize",
        (getter)npyiter_itersize_get,
        NULL, NULL, NULL},
    {"finished",
        (getter)npyiter_finished_get,
        NULL, NULL, NULL},

    {NULL, NULL, NULL, NULL, NULL}
};

NPY_NO_EXPORT PySequenceMethods npyiter_as_sequence = {
    (lenfunc)npyiter_seq_length,            /*sq_length*/
    (binaryfunc)NULL,                       /*sq_concat*/
    (ssizeargfunc)NULL,                     /*sq_repeat*/
    (ssizeargfunc)npyiter_seq_item,         /*sq_item*/
    (ssizessizeargfunc)NULL,                /*sq_slice*/
    (ssizeobjargproc)npyiter_seq_ass_item,  /*sq_ass_item*/
    (ssizessizeobjargproc)NULL,             /*sq_ass_slice*/
    (objobjproc)NULL,                       /*sq_contains */
    (binaryfunc)NULL,                       /*sq_inplace_concat */
    (ssizeargfunc)NULL,                     /*sq_inplace_repeat */
};

NPY_NO_EXPORT PyMappingMethods npyiter_as_mapping = {
    (lenfunc)npyiter_seq_length,          /*mp_length*/
    (binaryfunc)npyiter_subscript,        /*mp_subscript*/
    (objobjargproc)npyiter_ass_subscript, /*mp_ass_subscript*/
};

NPY_NO_EXPORT PyTypeObject NpyIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.nditer",
    .tp_basicsize = sizeof(NewNpyArrayIterObject),
    .tp_dealloc = (destructor)npyiter_dealloc,
    .tp_as_sequence = &npyiter_as_sequence,
    .tp_as_mapping = &npyiter_as_mapping,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iternext = (iternextfunc)npyiter_next,
    .tp_methods = npyiter_methods,
    .tp_members = npyiter_members,
    .tp_getset = npyiter_getsets,
    .tp_init = (initproc)npyiter_init,
    .tp_new = npyiter_new,
};
