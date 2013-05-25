/*
 * This file implements the construction, copying, and destruction
 * aspects of NumPy's nditer.
 *
 * Copyright (c) 2010-2011 by Mark Wiebe (mwwiebe@gmail.com)
 * The Univerity of British Columbia
 *
 * Copyright (c) 2011 Enthought, Inc
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* Indicate that this .c file is allowed to include the header */
#define NPY_ITERATOR_IMPLEMENTATION_CODE
#include "nditer_impl.h"

#include "arrayobject.h"

/* Internal helper functions private to this file */
static int
npyiter_check_global_flags(npy_uint32 flags, npy_uint32* itflags);
static int
npyiter_check_op_axes(int nop, int oa_ndim, int **op_axes,
                        npy_intp *itershape);
static int
npyiter_calculate_ndim(int nop, PyArrayObject **op_in,
                       int oa_ndim);
static int
npyiter_check_per_op_flags(npy_uint32 flags, npyiter_opitflags *op_itflags);
static int
npyiter_prepare_one_operand(PyArrayObject **op,
                        char **op_dataptr,
                        PyArray_Descr *op_request_dtype,
                        PyArray_Descr** op_dtype,
                        npy_uint32 flags,
                        npy_uint32 op_flags, npyiter_opitflags *op_itflags);
static int
npyiter_prepare_operands(int nop,
                    PyArrayObject **op_in,
                    PyArrayObject **op,
                    char **op_dataptr,
                    PyArray_Descr **op_request_dtypes,
                    PyArray_Descr **op_dtype,
                    npy_uint32 flags,
                    npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                    npy_int8 *out_maskop);
static int
npyiter_check_casting(int nop, PyArrayObject **op,
                    PyArray_Descr **op_dtype,
                    NPY_CASTING casting,
                    npyiter_opitflags *op_itflags);
static int
npyiter_fill_axisdata(NpyIter *iter, npy_uint32 flags, npyiter_opitflags *op_itflags,
                    char **op_dataptr,
                    npy_uint32 *op_flags, int **op_axes,
                    npy_intp *itershape);
static void
npyiter_replace_axisdata(NpyIter *iter, int iop,
                      PyArrayObject *op,
                      int op_ndim, char *op_dataptr,
                      int *op_axes);
static void
npyiter_compute_index_strides(NpyIter *iter, npy_uint32 flags);
static void
npyiter_apply_forced_iteration_order(NpyIter *iter, NPY_ORDER order);
static void
npyiter_flip_negative_strides(NpyIter *iter);
static void
npyiter_reverse_axis_ordering(NpyIter *iter);
static void
npyiter_find_best_axis_ordering(NpyIter *iter);
static PyArray_Descr *
npyiter_get_common_dtype(int nop, PyArrayObject **op,
                        npyiter_opitflags *op_itflags, PyArray_Descr **op_dtype,
                        PyArray_Descr **op_request_dtypes,
                        int only_inputs);
static PyArrayObject *
npyiter_new_temp_array(NpyIter *iter, PyTypeObject *subtype,
                npy_uint32 flags, npyiter_opitflags *op_itflags,
                int op_ndim, npy_intp *shape,
                PyArray_Descr *op_dtype, int *op_axes);
static int
npyiter_allocate_arrays(NpyIter *iter,
                        npy_uint32 flags,
                        PyArray_Descr **op_dtype, PyTypeObject *subtype,
                        npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                        int **op_axes);
static void
npyiter_get_priority_subtype(int nop, PyArrayObject **op,
                            npyiter_opitflags *op_itflags,
                            double *subtype_priority, PyTypeObject **subtype);
static int
npyiter_allocate_transfer_functions(NpyIter *iter);


/*NUMPY_API
 * Allocate a new iterator for multiple array objects, and advanced
 * options for controlling the broadcasting, shape, and buffer size.
 */
NPY_NO_EXPORT NpyIter *
NpyIter_AdvancedNew(int nop, PyArrayObject **op_in, npy_uint32 flags,
                 NPY_ORDER order, NPY_CASTING casting,
                 npy_uint32 *op_flags,
                 PyArray_Descr **op_request_dtypes,
                 int oa_ndim, int **op_axes, npy_intp *itershape,
                 npy_intp buffersize)
{
    npy_uint32 itflags = NPY_ITFLAG_IDENTPERM;
    int idim, ndim;
    int iop;

    /* The iterator being constructed */
    NpyIter *iter;

    /* Per-operand values */
    PyArrayObject **op;
    PyArray_Descr **op_dtype;
    npyiter_opitflags *op_itflags;
    char **op_dataptr;

    npy_int8 *perm;
    NpyIter_BufferData *bufferdata = NULL;
    int any_allocate = 0, any_missing_dtypes = 0, need_subtype = 0;

    /* The subtype for automatically allocated outputs */
    double subtype_priority = NPY_PRIORITY;
    PyTypeObject *subtype = &PyArray_Type;

#if NPY_IT_CONSTRUCTION_TIMING
    npy_intp c_temp,
            c_start,
            c_check_op_axes,
            c_check_global_flags,
            c_calculate_ndim,
            c_malloc,
            c_prepare_operands,
            c_fill_axisdata,
            c_compute_index_strides,
            c_apply_forced_iteration_order,
            c_find_best_axis_ordering,
            c_get_priority_subtype,
            c_find_output_common_dtype,
            c_check_casting,
            c_allocate_arrays,
            c_coalesce_axes,
            c_prepare_buffers;
#endif

    NPY_IT_TIME_POINT(c_start);

    if (nop > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
            "Cannot construct an iterator with more than %d operands "
            "(%d were requested)", (int)NPY_MAXARGS, (int)nop);
        return NULL;
    }

    /*
     * Before 1.8, if `oa_ndim == 0`, this meant `op_axes != NULL` was an error.
     * With 1.8, `oa_ndim == -1` takes this role, while op_axes in that case
     * enforces a 0-d iterator. Using `oa_ndim == 0` with `op_axes == NULL`
     * is thus deprecated with version 1.8.
     */
    if ((oa_ndim == 0) && (op_axes == NULL)) {
        char* mesg = "using `oa_ndim == 0` when `op_axes` is NULL is "
                     "deprecated. Use `oa_ndim == -1` or the MultiNew "
                     "iterator for NumPy <1.8 compatibility";
        if (DEPRECATE(mesg) < 0) {
            return NULL;
        }
        oa_ndim = -1;
    }

    /* Error check 'oa_ndim' and 'op_axes', which must be used together */
    if (!npyiter_check_op_axes(nop, oa_ndim, op_axes, itershape)) {
        return NULL;
    }

    NPY_IT_TIME_POINT(c_check_op_axes);

    /* Check the global iterator flags */
    if (!npyiter_check_global_flags(flags, &itflags)) {
        return NULL;
    }

    NPY_IT_TIME_POINT(c_check_global_flags);

    /* Calculate how many dimensions the iterator should have */
    ndim = npyiter_calculate_ndim(nop, op_in, oa_ndim);

    NPY_IT_TIME_POINT(c_calculate_ndim);

    /* Allocate memory for the iterator */
    iter = (NpyIter*)
                PyArray_malloc(NIT_SIZEOF_ITERATOR(itflags, ndim, nop));

    NPY_IT_TIME_POINT(c_malloc);

    /* Fill in the basic data */
    NIT_ITFLAGS(iter) = itflags;
    NIT_NDIM(iter) = ndim;
    NIT_NOP(iter) = nop;
    NIT_MASKOP(iter) = -1;
    NIT_ITERINDEX(iter) = 0;
    memset(NIT_BASEOFFSETS(iter), 0, (nop+1)*NPY_SIZEOF_INTP);

    op = NIT_OPERANDS(iter);
    op_dtype = NIT_DTYPES(iter);
    op_itflags = NIT_OPITFLAGS(iter);
    op_dataptr = NIT_RESETDATAPTR(iter);

    /* Prepare all the operands */
    if (!npyiter_prepare_operands(nop, op_in, op, op_dataptr,
                        op_request_dtypes, op_dtype,
                        flags,
                        op_flags, op_itflags,
                        &NIT_MASKOP(iter))) {
        PyArray_free(iter);
        return NULL;
    }
    /* Set resetindex to zero as well (it's just after the resetdataptr) */
    op_dataptr[nop] = 0;

    NPY_IT_TIME_POINT(c_prepare_operands);

    /*
     * Initialize buffer data (must set the buffers and transferdata
     * to NULL before we might deallocate the iterator).
     */
    if (itflags & NPY_ITFLAG_BUFFER) {
        bufferdata = NIT_BUFFERDATA(iter);
        NBF_SIZE(bufferdata) = 0;
        memset(NBF_BUFFERS(bufferdata), 0, nop*NPY_SIZEOF_INTP);
        memset(NBF_PTRS(bufferdata), 0, nop*NPY_SIZEOF_INTP);
        memset(NBF_READTRANSFERDATA(bufferdata), 0, nop*NPY_SIZEOF_INTP);
        memset(NBF_WRITETRANSFERDATA(bufferdata), 0, nop*NPY_SIZEOF_INTP);
    }

    /* Fill in the AXISDATA arrays and set the ITERSIZE field */
    if (!npyiter_fill_axisdata(iter, flags, op_itflags, op_dataptr,
                                        op_flags, op_axes, itershape)) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    NPY_IT_TIME_POINT(c_fill_axisdata);

    if (itflags & NPY_ITFLAG_BUFFER) {
        /*
         * If buffering is enabled and no buffersize was given, use a default
         * chosen to be big enough to get some amortization benefits, but
         * small enough to be cache-friendly.
         */
        if (buffersize <= 0) {
            buffersize = NPY_BUFSIZE;
        }
        /* No point in a buffer bigger than the iteration size */
        if (buffersize > NIT_ITERSIZE(iter)) {
            buffersize = NIT_ITERSIZE(iter);
        }
        NBF_BUFFERSIZE(bufferdata) = buffersize;
    }

    /*
     * If an index was requested, compute the strides for it.
     * Note that we must do this before changing the order of the
     * axes
     */
    npyiter_compute_index_strides(iter, flags);

    NPY_IT_TIME_POINT(c_compute_index_strides);

    /* Initialize the perm to the identity */
    perm = NIT_PERM(iter);
    for(idim = 0; idim < ndim; ++idim) {
        perm[idim] = (npy_int8)idim;
    }

    /*
     * If an iteration order is being forced, apply it.
     */
    npyiter_apply_forced_iteration_order(iter, order);
    itflags = NIT_ITFLAGS(iter);

    NPY_IT_TIME_POINT(c_apply_forced_iteration_order);

    /* Set some flags for allocated outputs */
    for (iop = 0; iop < nop; ++iop) {
        if (op[iop] == NULL) {
            /* Flag this so later we can avoid flipping axes */
            any_allocate = 1;
            /* If a subtype may be used, indicate so */
            if (!(op_flags[iop] & NPY_ITER_NO_SUBTYPE)) {
                need_subtype = 1;
            }
            /*
             * If the data type wasn't provided, will need to
             * calculate it.
             */
            if (op_dtype[iop] == NULL) {
                any_missing_dtypes = 1;
            }
        }
    }

    /*
     * If the ordering was not forced, reorder the axes
     * and flip negative strides to find the best one.
     */
    if (!(itflags & NPY_ITFLAG_FORCEDORDER)) {
        if (ndim > 1) {
            npyiter_find_best_axis_ordering(iter);
        }
        /*
         * If there's an output being allocated, we must not negate
         * any strides.
         */
        if (!any_allocate && !(flags & NPY_ITER_DONT_NEGATE_STRIDES)) {
            npyiter_flip_negative_strides(iter);
        }
        itflags = NIT_ITFLAGS(iter);
    }

    NPY_IT_TIME_POINT(c_find_best_axis_ordering);

    if (need_subtype) {
        npyiter_get_priority_subtype(nop, op, op_itflags,
                                     &subtype_priority, &subtype);
    }

    NPY_IT_TIME_POINT(c_get_priority_subtype);

    /*
     * If an automatically allocated output didn't have a specified
     * dtype, we need to figure it out now, before allocating the outputs.
     */
    if (any_missing_dtypes || (flags & NPY_ITER_COMMON_DTYPE)) {
        PyArray_Descr *dtype;
        int only_inputs = !(flags & NPY_ITER_COMMON_DTYPE);

        op = NIT_OPERANDS(iter);
        op_dtype = NIT_DTYPES(iter);

        dtype = npyiter_get_common_dtype(nop, op,
                                    op_itflags, op_dtype,
                                    op_request_dtypes,
                                    only_inputs);
        if (dtype == NULL) {
            NpyIter_Deallocate(iter);
            return NULL;
        }
        if (flags & NPY_ITER_COMMON_DTYPE) {
            NPY_IT_DBG_PRINT("Iterator: Replacing all data types\n");
            /* Replace all the data types */
            for (iop = 0; iop < nop; ++iop) {
                if (op_dtype[iop] != dtype) {
                    Py_XDECREF(op_dtype[iop]);
                    Py_INCREF(dtype);
                    op_dtype[iop] = dtype;
                }
            }
        }
        else {
            NPY_IT_DBG_PRINT("Iterator: Setting unset output data types\n");
            /* Replace the NULL data types */
            for (iop = 0; iop < nop; ++iop) {
                if (op_dtype[iop] == NULL) {
                    Py_INCREF(dtype);
                    op_dtype[iop] = dtype;
                }
            }
        }
        Py_DECREF(dtype);
    }

    NPY_IT_TIME_POINT(c_find_output_common_dtype);

    /*
     * All of the data types have been settled, so it's time
     * to check that data type conversions are following the
     * casting rules.
     */
    if (!npyiter_check_casting(nop, op, op_dtype, casting, op_itflags)) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    NPY_IT_TIME_POINT(c_check_casting);

    /*
     * At this point, the iteration order has been finalized. so
     * any allocation of ops that were NULL, or any temporary
     * copying due to casting/byte order/alignment can be
     * done now using a memory layout matching the iterator.
     */
    if (!npyiter_allocate_arrays(iter, flags, op_dtype, subtype, op_flags,
                            op_itflags, op_axes)) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    NPY_IT_TIME_POINT(c_allocate_arrays);

    /*
     * Finally, if a multi-index wasn't requested,
     * it may be possible to coalesce some axes together.
     */
    if (ndim > 1 && !(itflags & NPY_ITFLAG_HASMULTIINDEX)) {
        npyiter_coalesce_axes(iter);
        /*
         * The operation may have changed the layout, so we have to
         * get the internal pointers again.
         */
        itflags = NIT_ITFLAGS(iter);
        ndim = NIT_NDIM(iter);
        op = NIT_OPERANDS(iter);
        op_dtype = NIT_DTYPES(iter);
        op_itflags = NIT_OPITFLAGS(iter);
        op_dataptr = NIT_RESETDATAPTR(iter);
    }

    NPY_IT_TIME_POINT(c_coalesce_axes);

    /*
     * Now that the axes are finished, check whether we can apply
     * the single iteration optimization to the iternext function.
     */
    if (!(itflags & NPY_ITFLAG_BUFFER)) {
        NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
        if (itflags & NPY_ITFLAG_EXLOOP) {
            if (NIT_ITERSIZE(iter) == NAD_SHAPE(axisdata)) {
                NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
            }
        }
        else if (NIT_ITERSIZE(iter) == 1) {
            NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
        }
    }

    /*
     * If REFS_OK was specified, check whether there are any
     * reference arrays and flag it if so.
     */
    if (flags & NPY_ITER_REFS_OK) {
        for (iop = 0; iop < nop; ++iop) {
            PyArray_Descr *rdt = op_dtype[iop];
            if ((rdt->flags & (NPY_ITEM_REFCOUNT |
                                     NPY_ITEM_IS_POINTER |
                                     NPY_NEEDS_PYAPI)) != 0) {
                /* Iteration needs API access */
                NIT_ITFLAGS(iter) |= NPY_ITFLAG_NEEDSAPI;
            }
        }
    }

    /* If buffering is set without delayed allocation */
    if (itflags & NPY_ITFLAG_BUFFER) {
        if (!npyiter_allocate_transfer_functions(iter)) {
            NpyIter_Deallocate(iter);
            return NULL;
        }
        if (!(itflags & NPY_ITFLAG_DELAYBUF)) {
            /* Allocate the buffers */
            if (!npyiter_allocate_buffers(iter, NULL)) {
                NpyIter_Deallocate(iter);
                return NULL;
            }

            /* Prepare the next buffers and set iterend/size */
            npyiter_copy_to_buffers(iter, NULL);
        }
    }

    NPY_IT_TIME_POINT(c_prepare_buffers);

#if NPY_IT_CONSTRUCTION_TIMING
    printf("\nIterator construction timing:\n");
    NPY_IT_PRINT_TIME_START(c_start);
    NPY_IT_PRINT_TIME_VAR(c_check_op_axes);
    NPY_IT_PRINT_TIME_VAR(c_check_global_flags);
    NPY_IT_PRINT_TIME_VAR(c_calculate_ndim);
    NPY_IT_PRINT_TIME_VAR(c_malloc);
    NPY_IT_PRINT_TIME_VAR(c_prepare_operands);
    NPY_IT_PRINT_TIME_VAR(c_fill_axisdata);
    NPY_IT_PRINT_TIME_VAR(c_compute_index_strides);
    NPY_IT_PRINT_TIME_VAR(c_apply_forced_iteration_order);
    NPY_IT_PRINT_TIME_VAR(c_find_best_axis_ordering);
    NPY_IT_PRINT_TIME_VAR(c_get_priority_subtype);
    NPY_IT_PRINT_TIME_VAR(c_find_output_common_dtype);
    NPY_IT_PRINT_TIME_VAR(c_check_casting);
    NPY_IT_PRINT_TIME_VAR(c_allocate_arrays);
    NPY_IT_PRINT_TIME_VAR(c_coalesce_axes);
    NPY_IT_PRINT_TIME_VAR(c_prepare_buffers);
    printf("\n");
#endif

    return iter;
}

/*NUMPY_API
 * Allocate a new iterator for more than one array object, using
 * standard NumPy broadcasting rules and the default buffer size.
 */
NPY_NO_EXPORT NpyIter *
NpyIter_MultiNew(int nop, PyArrayObject **op_in, npy_uint32 flags,
                 NPY_ORDER order, NPY_CASTING casting,
                 npy_uint32 *op_flags,
                 PyArray_Descr **op_request_dtypes)
{
    return NpyIter_AdvancedNew(nop, op_in, flags, order, casting,
                            op_flags, op_request_dtypes,
                            -1, NULL, NULL, 0);
}

/*NUMPY_API
 * Allocate a new iterator for one array object.
 */
NPY_NO_EXPORT NpyIter *
NpyIter_New(PyArrayObject *op, npy_uint32 flags,
                  NPY_ORDER order, NPY_CASTING casting,
                  PyArray_Descr* dtype)
{
    /* Split the flags into separate global and op flags */
    npy_uint32 op_flags = flags & NPY_ITER_PER_OP_FLAGS;
    flags &= NPY_ITER_GLOBAL_FLAGS;

    return NpyIter_AdvancedNew(1, &op, flags, order, casting,
                            &op_flags, &dtype,
                            -1, NULL, NULL, 0);
}

/*NUMPY_API
 * Makes a copy of the iterator
 */
NPY_NO_EXPORT NpyIter *
NpyIter_Copy(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);
    int out_of_memory = 0;

    npy_intp size;
    NpyIter *newiter;
    PyArrayObject **objects;
    PyArray_Descr **dtypes;

    /* Allocate memory for the new iterator */
    size = NIT_SIZEOF_ITERATOR(itflags, ndim, nop);
    newiter = (NpyIter*)PyArray_malloc(size);

    /* Copy the raw values to the new iterator */
    memcpy(newiter, iter, size);

    /* Take ownership of references to the operands and dtypes */
    objects = NIT_OPERANDS(newiter);
    dtypes = NIT_DTYPES(newiter);
    for (iop = 0; iop < nop; ++iop) {
        Py_INCREF(objects[iop]);
        Py_INCREF(dtypes[iop]);
    }

    /* Allocate buffers and make copies of the transfer data if necessary */
    if (itflags & NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata;
        npy_intp buffersize, itemsize;
        char **buffers;
        NpyAuxData **readtransferdata, **writetransferdata;

        bufferdata = NIT_BUFFERDATA(newiter);
        buffers = NBF_BUFFERS(bufferdata);
        readtransferdata = NBF_READTRANSFERDATA(bufferdata);
        writetransferdata = NBF_WRITETRANSFERDATA(bufferdata);
        buffersize = NBF_BUFFERSIZE(bufferdata);

        for (iop = 0; iop < nop; ++iop) {
            if (buffers[iop] != NULL) {
                if (out_of_memory) {
                    buffers[iop] = NULL;
                }
                else {
                    itemsize = dtypes[iop]->elsize;
                    buffers[iop] = PyArray_malloc(itemsize*buffersize);
                    if (buffers[iop] == NULL) {
                        out_of_memory = 1;
                    }
                }
            }

            if (readtransferdata[iop] != NULL) {
                if (out_of_memory) {
                    readtransferdata[iop] = NULL;
                }
                else {
                    readtransferdata[iop] =
                          NPY_AUXDATA_CLONE(readtransferdata[iop]);
                    if (readtransferdata[iop] == NULL) {
                        out_of_memory = 1;
                    }
                }
            }

            if (writetransferdata[iop] != NULL) {
                if (out_of_memory) {
                    writetransferdata[iop] = NULL;
                }
                else {
                    writetransferdata[iop] =
                          NPY_AUXDATA_CLONE(writetransferdata[iop]);
                    if (writetransferdata[iop] == NULL) {
                        out_of_memory = 1;
                    }
                }
            }
        }

        /* Initialize the buffers to the current iterindex */
        if (!out_of_memory && NBF_SIZE(bufferdata) > 0) {
            npyiter_goto_iterindex(newiter, NIT_ITERINDEX(newiter));

            /* Prepare the next buffers and set iterend/size */
            npyiter_copy_to_buffers(newiter, NULL);
        }
    }

    if (out_of_memory) {
        NpyIter_Deallocate(newiter);
        PyErr_NoMemory();
        return NULL;
    }

    return newiter;
}

/*NUMPY_API
 * Deallocate an iterator
 */
NPY_NO_EXPORT int
NpyIter_Deallocate(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int iop, nop = NIT_NOP(iter);

    PyArray_Descr **dtype = NIT_DTYPES(iter);
    PyArrayObject **object = NIT_OPERANDS(iter);

    /* Deallocate any buffers and buffering data */
    if (itflags & NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        char **buffers;
        NpyAuxData **transferdata;

        /* buffers */
        buffers = NBF_BUFFERS(bufferdata);
        for(iop = 0; iop < nop; ++iop, ++buffers) {
            if (*buffers) {
                PyArray_free(*buffers);
            }
        }
        /* read bufferdata */
        transferdata = NBF_READTRANSFERDATA(bufferdata);
        for(iop = 0; iop < nop; ++iop, ++transferdata) {
            if (*transferdata) {
                NPY_AUXDATA_FREE(*transferdata);
            }
        }
        /* write bufferdata */
        transferdata = NBF_WRITETRANSFERDATA(bufferdata);
        for(iop = 0; iop < nop; ++iop, ++transferdata) {
            if (*transferdata) {
                NPY_AUXDATA_FREE(*transferdata);
            }
        }
    }

    /* Deallocate all the dtypes and objects that were iterated */
    for(iop = 0; iop < nop; ++iop, ++dtype, ++object) {
        Py_XDECREF(*dtype);
        Py_XDECREF(*object);
    }

    /* Deallocate the iterator memory */
    PyArray_free(iter);

    return NPY_SUCCEED;
}

/* Checks 'flags' for (C|F)_ORDER_INDEX, MULTI_INDEX, and EXTERNAL_LOOP,
 * setting the appropriate internal flags in 'itflags'.
 *
 * Returns 1 on success, 0 on error.
 */
static int
npyiter_check_global_flags(npy_uint32 flags, npy_uint32* itflags)
{
    if ((flags & NPY_ITER_PER_OP_FLAGS) != 0) {
        PyErr_SetString(PyExc_ValueError,
                    "A per-operand flag was passed as a global flag "
                    "to the iterator constructor");
        return 0;
    }

    /* Check for an index */
    if (flags & (NPY_ITER_C_INDEX | NPY_ITER_F_INDEX)) {
        if ((flags & (NPY_ITER_C_INDEX | NPY_ITER_F_INDEX)) ==
                    (NPY_ITER_C_INDEX | NPY_ITER_F_INDEX)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator flags C_INDEX and "
                    "F_INDEX cannot both be specified");
            return 0;
        }
        (*itflags) |= NPY_ITFLAG_HASINDEX;
    }
    /* Check if a multi-index was requested */
    if (flags & NPY_ITER_MULTI_INDEX) {
        /*
         * This flag primarily disables dimension manipulations that
         * would produce an incorrect multi-index.
         */
        (*itflags) |= NPY_ITFLAG_HASMULTIINDEX;
    }
    /* Check if the caller wants to handle inner iteration */
    if (flags & NPY_ITER_EXTERNAL_LOOP) {
        if ((*itflags) & (NPY_ITFLAG_HASINDEX | NPY_ITFLAG_HASMULTIINDEX)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator flag EXTERNAL_LOOP cannot be used "
                    "if an index or multi-index is being tracked");
            return 0;
        }
        (*itflags) |= NPY_ITFLAG_EXLOOP;
    }
    /* Ranged */
    if (flags & NPY_ITER_RANGED) {
        (*itflags) |= NPY_ITFLAG_RANGE;
        if ((flags & NPY_ITER_EXTERNAL_LOOP) &&
                                    !(flags & NPY_ITER_BUFFERED)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator flag RANGED cannot be used with "
                    "the flag EXTERNAL_LOOP unless "
                    "BUFFERED is also enabled");
            return 0;
        }
    }
    /* Buffering */
    if (flags & NPY_ITER_BUFFERED) {
        (*itflags) |= NPY_ITFLAG_BUFFER;
        if (flags & NPY_ITER_GROWINNER) {
            (*itflags) |= NPY_ITFLAG_GROWINNER;
        }
        if (flags & NPY_ITER_DELAY_BUFALLOC) {
            (*itflags) |= NPY_ITFLAG_DELAYBUF;
        }
    }

    return 1;
}

static int
npyiter_check_op_axes(int nop, int oa_ndim, int **op_axes,
                        npy_intp *itershape)
{
    char axes_dupcheck[NPY_MAXDIMS];
    int iop, idim;

    if (oa_ndim < 0) {
        /*
         * If `oa_ndim < 0`, `op_axes` and `itershape` are signalled to
         * be unused and should be NULL. (Before NumPy 1.8 this was
         * signalled by `oa_ndim == 0`.)
         */
        if (op_axes != NULL || itershape != NULL) {
            PyErr_Format(PyExc_ValueError,
                    "If 'op_axes' or 'itershape' is not NULL in the iterator "
                    "constructor, 'oa_ndim' must be zero or greater");
            return 0;
        }
        return 1;
    }
    if (oa_ndim > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                "Cannot construct an iterator with more than %d dimensions "
                "(%d were requested for op_axes)",
                (int)NPY_MAXDIMS, oa_ndim);
        return 0;
    }
    if (op_axes == NULL) {
        PyErr_Format(PyExc_ValueError,
                "If 'oa_ndim' is zero or greater in the iterator "
                "constructor, then op_axes cannot be NULL");
        return 0;
    }

    /* Check that there are no duplicates in op_axes */
    for (iop = 0; iop < nop; ++iop) {
        int *axes = op_axes[iop];
        if (axes != NULL) {
            memset(axes_dupcheck, 0, NPY_MAXDIMS);
            for (idim = 0; idim < oa_ndim; ++idim) {
                npy_intp i = axes[idim];
                if (i >= 0) {
                    if (i >= NPY_MAXDIMS) {
                        PyErr_Format(PyExc_ValueError,
                                "The 'op_axes' provided to the iterator "
                                "constructor for operand %d "
                                "contained invalid "
                                "values %d", (int)iop, (int)i);
                        return 0;
                    }
                    else if (axes_dupcheck[i] == 1) {
                        PyErr_Format(PyExc_ValueError,
                                "The 'op_axes' provided to the iterator "
                                "constructor for operand %d "
                                "contained duplicate "
                                "value %d", (int)iop, (int)i);
                        return 0;
                    }
                    else {
                        axes_dupcheck[i] = 1;
                    }
                }
            }
        }
    }

    return 1;
}

static int
npyiter_calculate_ndim(int nop, PyArrayObject **op_in,
                       int oa_ndim)
{
    /* If 'op_axes' is being used, force 'ndim' */
    if (oa_ndim >= 0 ) {
        return oa_ndim;
    }
    /* Otherwise it's the maximum 'ndim' from the operands */
    else {
        int ndim = 0, iop;

        for (iop = 0; iop < nop; ++iop) {
            if (op_in[iop] != NULL) {
                int ondim = PyArray_NDIM(op_in[iop]);
                if (ondim > ndim) {
                    ndim = ondim;
                }
            }

        }

        return ndim;
    }
}

/*
 * Checks the per-operand input flags, and fills in op_itflags.
 *
 * Returns 1 on success, 0 on failure.
 */
static int
npyiter_check_per_op_flags(npy_uint32 op_flags, npyiter_opitflags *op_itflags)
{
    if ((op_flags & NPY_ITER_GLOBAL_FLAGS) != 0) {
        PyErr_SetString(PyExc_ValueError,
                    "A global iterator flag was passed as a per-operand flag "
                    "to the iterator constructor");
        return 0;
    }

    /* Check the read/write flags */
    if (op_flags & NPY_ITER_READONLY) {
        /* The read/write flags are mutually exclusive */
        if (op_flags & (NPY_ITER_READWRITE|NPY_ITER_WRITEONLY)) {
            PyErr_SetString(PyExc_ValueError,
                    "Only one of the iterator flags READWRITE, "
                    "READONLY, and WRITEONLY may be "
                    "specified for an operand");
            return 0;
        }

        *op_itflags = NPY_OP_ITFLAG_READ;
    }
    else if (op_flags & NPY_ITER_READWRITE) {
        /* The read/write flags are mutually exclusive */
        if (op_flags & NPY_ITER_WRITEONLY) {
            PyErr_SetString(PyExc_ValueError,
                    "Only one of the iterator flags READWRITE, "
                    "READONLY, and WRITEONLY may be "
                    "specified for an operand");
            return 0;
        }

        *op_itflags = NPY_OP_ITFLAG_READ|NPY_OP_ITFLAG_WRITE;
    }
    else if(op_flags & NPY_ITER_WRITEONLY) {
        *op_itflags = NPY_OP_ITFLAG_WRITE;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "None of the iterator flags READWRITE, "
                "READONLY, or WRITEONLY were "
                "specified for an operand");
        return 0;
    }

    /* Check the flags for temporary copies */
    if (((*op_itflags) & NPY_OP_ITFLAG_WRITE) &&
                (op_flags & (NPY_ITER_COPY |
                           NPY_ITER_UPDATEIFCOPY)) == NPY_ITER_COPY) {
        PyErr_SetString(PyExc_ValueError,
                "If an iterator operand is writeable, must use "
                "the flag UPDATEIFCOPY instead of "
                "COPY");
        return 0;
    }

    /* Check the flag for a write masked operands */
    if (op_flags & NPY_ITER_WRITEMASKED) {
        if (!((*op_itflags) & NPY_OP_ITFLAG_WRITE)) {
            PyErr_SetString(PyExc_ValueError,
                "The iterator flag WRITEMASKED may only "
                "be used with READWRITE or WRITEONLY");
            return 0;
        }
        if ((op_flags & NPY_ITER_ARRAYMASK) != 0) {
            PyErr_SetString(PyExc_ValueError,
                "The iterator flag WRITEMASKED may not "
                "be used together with ARRAYMASK");
            return 0;
        }
        *op_itflags |= NPY_OP_ITFLAG_WRITEMASKED;
    }

    if ((op_flags & NPY_ITER_VIRTUAL) != 0) {
        if ((op_flags & NPY_ITER_READWRITE) == 0) {
            PyErr_SetString(PyExc_ValueError,
                "The iterator flag VIRTUAL should be "
                "be used together with READWRITE");
            return 0;
        }
        *op_itflags |= NPY_OP_ITFLAG_VIRTUAL;
    }

    return 1;
}

/*
 * Prepares a a constructor operand.  Assumes a reference to 'op'
 * is owned, and that 'op' may be replaced.  Fills in 'op_dataptr',
 * 'op_dtype', and may modify 'op_itflags'.
 *
 * Returns 1 on success, 0 on failure.
 */
static int
npyiter_prepare_one_operand(PyArrayObject **op,
                        char **op_dataptr,
                        PyArray_Descr *op_request_dtype,
                        PyArray_Descr **op_dtype,
                        npy_uint32 flags,
                        npy_uint32 op_flags, npyiter_opitflags *op_itflags)
{
    /* NULL operands must be automatically allocated outputs */
    if (*op == NULL) {
        /* ALLOCATE or VIRTUAL should be enabled */
        if ((op_flags & (NPY_ITER_ALLOCATE|NPY_ITER_VIRTUAL)) == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator operand was NULL, but neither the "
                    "ALLOCATE nor the VIRTUAL flag was specified");
            return 0;
        }

        if (op_flags & NPY_ITER_ALLOCATE) {
            /* Writing should be enabled */
            if (!((*op_itflags) & NPY_OP_ITFLAG_WRITE)) {
                PyErr_SetString(PyExc_ValueError,
                        "Automatic allocation was requested for an iterator "
                        "operand, but it wasn't flagged for writing");
                return 0;
            }
            /*
             * Reading should be disabled if buffering is enabled without
             * also enabling NPY_ITER_DELAY_BUFALLOC.  In all other cases,
             * the caller may initialize the allocated operand to a value
             * before beginning iteration.
             */
            if (((flags & (NPY_ITER_BUFFERED |
                            NPY_ITER_DELAY_BUFALLOC)) == NPY_ITER_BUFFERED) &&
                    ((*op_itflags) & NPY_OP_ITFLAG_READ)) {
                PyErr_SetString(PyExc_ValueError,
                        "Automatic allocation was requested for an iterator "
                        "operand, and it was flagged as readable, but "
                        "buffering  without delayed allocation was enabled");
                return 0;
            }

            /* If a requested dtype was provided, use it, otherwise NULL */
            Py_XINCREF(op_request_dtype);
            *op_dtype = op_request_dtype;
        }
        else {
            *op_dtype = NULL;
        }

        /* Specify bool if no dtype was requested for the mask */
        if (op_flags & NPY_ITER_ARRAYMASK) {
            if (*op_dtype == NULL) {
                *op_dtype = PyArray_DescrFromType(NPY_BOOL);
                if (*op_dtype == NULL) {
                    return 0;
                }
            }
        }

        *op_dataptr = NULL;

        return 1;
    }

    /* VIRTUAL operands must be NULL */
    if (op_flags & NPY_ITER_VIRTUAL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator operand flag VIRTUAL was specified, "
                "but the operand was not NULL");
        return 0;
    }


    if (PyArray_Check(*op)) {

        if ((*op_itflags) & NPY_OP_ITFLAG_WRITE
            && PyArray_FailUnlessWriteable(*op, "operand array with iterator "
                                           "write flag set") < 0) {
            return 0;
        }
        if (!(flags & NPY_ITER_ZEROSIZE_OK) && PyArray_SIZE(*op) == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Iteration of zero-sized operands is not enabled");
            return 0;
        }
        *op_dataptr = PyArray_BYTES(*op);
        /* PyArray_DESCR does not give us a reference */
        *op_dtype = PyArray_DESCR(*op);
        if (*op_dtype == NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator input operand has no dtype descr");
            return 0;
        }
        Py_INCREF(*op_dtype);
        /*
         * If references weren't specifically allowed, make sure there
         * are no references in the inputs or requested dtypes.
         */
        if (!(flags & NPY_ITER_REFS_OK)) {
            PyArray_Descr *dt = PyArray_DESCR(*op);
            if (((dt->flags & (NPY_ITEM_REFCOUNT |
                           NPY_ITEM_IS_POINTER)) != 0) ||
                    (dt != *op_dtype &&
                        (((*op_dtype)->flags & (NPY_ITEM_REFCOUNT |
                                             NPY_ITEM_IS_POINTER))) != 0)) {
                PyErr_SetString(PyExc_TypeError,
                        "Iterator operand or requested dtype holds "
                        "references, but the REFS_OK flag was not enabled");
                return 0;
            }
        }
        /*
         * Checking whether casts are valid is done later, once the
         * final data types have been selected.  For now, just store the
         * requested type.
         */
        if (op_request_dtype != NULL) {
            /* We just have a borrowed reference to op_request_dtype */
            Py_INCREF(op_request_dtype);
            /* If the requested dtype is flexible, adapt it */
            PyArray_AdaptFlexibleDType((PyObject *)(*op), PyArray_DESCR(*op),
                                        &op_request_dtype);
            if (op_request_dtype == NULL) {
                return 0;
            }

            /* Store the requested dtype */
            Py_DECREF(*op_dtype);
            *op_dtype = op_request_dtype;
        }

        /* Check if the operand is in the byte order requested */
        if (op_flags & NPY_ITER_NBO) {
            /* Check byte order */
            if (!PyArray_ISNBO((*op_dtype)->byteorder)) {
                PyArray_Descr *nbo_dtype;

                /* Replace with a new descr which is in native byte order */
                nbo_dtype = PyArray_DescrNewByteorder(*op_dtype, NPY_NATIVE);
                Py_DECREF(*op_dtype);
                *op_dtype = nbo_dtype;

                NPY_IT_DBG_PRINT("Iterator: Setting NPY_OP_ITFLAG_CAST "
                                    "because of NPY_ITER_NBO\n");
                /* Indicate that byte order or alignment needs fixing */
                *op_itflags |= NPY_OP_ITFLAG_CAST;
            }
        }
        /* Check if the operand is aligned */
        if (op_flags & NPY_ITER_ALIGNED) {
            /* Check alignment */
            if (!PyArray_ISALIGNED(*op)) {
                NPY_IT_DBG_PRINT("Iterator: Setting NPY_OP_ITFLAG_CAST "
                                    "because of NPY_ITER_ALIGNED\n");
                *op_itflags |= NPY_OP_ITFLAG_CAST;
            }
        }
        /*
         * The check for NPY_ITER_CONTIG can only be done later,
         * once the final iteration order is settled.
         */
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator inputs must be ndarrays");
        return 0;
    }

    return 1;
}

/*
 * Process all the operands, copying new references so further processing
 * can replace the arrays if copying is necessary.
 */
static int
npyiter_prepare_operands(int nop, PyArrayObject **op_in,
                    PyArrayObject **op,
                    char **op_dataptr,
                    PyArray_Descr **op_request_dtypes,
                    PyArray_Descr **op_dtype,
                    npy_uint32 flags,
                    npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                    npy_int8 *out_maskop)
{
    int iop, i;
    npy_int8 maskop = -1;
    int any_writemasked_ops = 0;

    /*
     * Here we just prepare the provided operands.
     */
    for (iop = 0; iop < nop; ++iop) {
        op[iop] = op_in[iop];
        Py_XINCREF(op[iop]);
        op_dtype[iop] = NULL;

        /* Check the readonly/writeonly flags, and fill in op_itflags */
        if (!npyiter_check_per_op_flags(op_flags[iop], &op_itflags[iop])) {
            for (i = 0; i <= iop; ++i) {
                Py_XDECREF(op[i]);
                Py_XDECREF(op_dtype[i]);
            }
            return 0;
        }

        /* Extract the operand which is for masked iteration */
        if ((op_flags[iop] & NPY_ITER_ARRAYMASK) != 0) {
            if (maskop != -1) {
                PyErr_SetString(PyExc_ValueError,
                        "Only one iterator operand may receive an "
                        "ARRAYMASK flag");
                for (i = 0; i <= iop; ++i) {
                    Py_XDECREF(op[i]);
                    Py_XDECREF(op_dtype[i]);
                }
                return 0;
            }

            maskop = iop;
            *out_maskop = iop;
        }

        if (op_flags[iop] & NPY_ITER_WRITEMASKED) {
            any_writemasked_ops = 1;
        }

        /*
         * Prepare the operand.  This produces an op_dtype[iop] reference
         * on success.
         */
        if (!npyiter_prepare_one_operand(&op[iop],
                        &op_dataptr[iop],
                        op_request_dtypes ? op_request_dtypes[iop] : NULL,
                        &op_dtype[iop],
                        flags,
                        op_flags[iop], &op_itflags[iop])) {
            for (i = 0; i <= iop; ++i) {
                Py_XDECREF(op[i]);
                Py_XDECREF(op_dtype[i]);
            }
            return 0;
        }
    }

    /* If all the operands were NULL, it's an error */
    if (op[0] == NULL) {
        int all_null = 1;
        for (iop = 1; iop < nop; ++iop) {
            if (op[iop] != NULL) {
                all_null = 0;
                break;
            }
        }
        if (all_null) {
            for (i = 0; i < nop; ++i) {
                Py_XDECREF(op[i]);
                Py_XDECREF(op_dtype[i]);
            }
            PyErr_SetString(PyExc_ValueError,
                    "At least one iterator operand must be non-NULL");
            return 0;
        }
    }

    if (any_writemasked_ops && maskop < 0) {
        PyErr_SetString(PyExc_ValueError,
                "An iterator operand was flagged as WRITEMASKED, "
                "but no ARRAYMASK operand was given to supply "
                "the mask");
        return 0;
    }
    else if (!any_writemasked_ops && maskop >= 0) {
        PyErr_SetString(PyExc_ValueError,
                "An iterator operand was flagged as the ARRAYMASK, "
                "but no WRITEMASKED operands were given to use "
                "the mask");
        return 0;
    }

    return 1;
}

static const char *
npyiter_casting_to_string(NPY_CASTING casting)
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

static PyObject *
npyiter_shape_string(npy_intp n, npy_intp *vals, char *ending)
{
    npy_intp i;
    PyObject *ret, *tmp;

    /*
     * Negative dimension indicates "newaxis", which can
     * be discarded for printing if it's a leading dimension.
     * Find the first non-"newaxis" dimension.
     */
    i = 0;
    while (i < n && vals[i] < 0) {
        ++i;
    }

    if (i == n) {
        return PyUString_FromFormat("()%s", ending);
    }
    else {
        ret = PyUString_FromFormat("(%" NPY_INTP_FMT, vals[i++]);
        if (ret == NULL) {
            return NULL;
        }
    }

    for (; i < n; ++i) {
        if (vals[i] < 0) {
            tmp = PyUString_FromString(",newaxis");
        }
        else {
            tmp = PyUString_FromFormat(",%" NPY_INTP_FMT, vals[i]);
        }
        if (tmp == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        PyUString_ConcatAndDel(&ret, tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    tmp = PyUString_FromFormat(")%s", ending);
    PyUString_ConcatAndDel(&ret, tmp);
    return ret;
}

static int
npyiter_check_casting(int nop, PyArrayObject **op,
                    PyArray_Descr **op_dtype,
                    NPY_CASTING casting,
                    npyiter_opitflags *op_itflags)
{
    int iop;

    for(iop = 0; iop < nop; ++iop) {
        NPY_IT_DBG_PRINT1("Iterator: Checking casting for operand %d\n",
                            (int)iop);
#if NPY_IT_DBG_TRACING
        printf("op: ");
        if (op[iop] != NULL) {
            PyObject_Print((PyObject *)PyArray_DESCR(op[iop]), stdout, 0);
        }
        else {
            printf("<null>");
        }
        printf(", iter: ");
        PyObject_Print((PyObject *)op_dtype[iop], stdout, 0);
        printf("\n");
#endif
        /* If the types aren't equivalent, a cast is necessary */
        if (op[iop] != NULL && !PyArray_EquivTypes(PyArray_DESCR(op[iop]),
                                                     op_dtype[iop])) {
            /* Check read (op -> temp) casting */
            if ((op_itflags[iop] & NPY_OP_ITFLAG_READ) &&
                        !PyArray_CanCastArrayTo(op[iop],
                                          op_dtype[iop],
                                          casting)) {
                PyObject *errmsg;
                errmsg = PyUString_FromFormat(
                        "Iterator operand %d dtype could not be cast from ",
                        (int)iop);
                PyUString_ConcatAndDel(&errmsg,
                        PyObject_Repr((PyObject *)PyArray_DESCR(op[iop])));
                PyUString_ConcatAndDel(&errmsg,
                        PyUString_FromString(" to "));
                PyUString_ConcatAndDel(&errmsg,
                        PyObject_Repr((PyObject *)op_dtype[iop]));
                PyUString_ConcatAndDel(&errmsg,
                        PyUString_FromFormat(" according to the rule %s",
                                npyiter_casting_to_string(casting)));
                PyErr_SetObject(PyExc_TypeError, errmsg);
                Py_DECREF(errmsg);
                return 0;
            }
            /* Check write (temp -> op) casting */
            if ((op_itflags[iop] & NPY_OP_ITFLAG_WRITE) &&
                        !PyArray_CanCastTypeTo(op_dtype[iop],
                                          PyArray_DESCR(op[iop]),
                                          casting)) {
                PyObject *errmsg;
                errmsg = PyUString_FromString(
                        "Iterator requested dtype could not be cast from ");
                PyUString_ConcatAndDel(&errmsg,
                        PyObject_Repr((PyObject *)op_dtype[iop]));
                PyUString_ConcatAndDel(&errmsg,
                        PyUString_FromString(" to "));
                PyUString_ConcatAndDel(&errmsg,
                        PyObject_Repr((PyObject *)PyArray_DESCR(op[iop])));
                PyUString_ConcatAndDel(&errmsg,
                        PyUString_FromFormat(", the operand %d dtype, "
                                "according to the rule %s",
                                (int)iop,
                                npyiter_casting_to_string(casting)));
                PyErr_SetObject(PyExc_TypeError, errmsg);
                Py_DECREF(errmsg);
                return 0;
            }

            NPY_IT_DBG_PRINT("Iterator: Setting NPY_OP_ITFLAG_CAST "
                                "because the types aren't equivalent\n");
            /* Indicate that this operand needs casting */
            op_itflags[iop] |= NPY_OP_ITFLAG_CAST;
        }
    }

    return 1;
}

/*
 * Checks that the mask broadcasts to the WRITEMASK REDUCE
 * operand 'iop', but 'iop' never broadcasts to the mask.
 * If 'iop' broadcasts to the mask, the result would be more
 * than one mask value per reduction element, something which
 * is invalid.
 *
 * This check should only be called after all the operands
 * have been filled in.
 *
 * Returns 1 on success, 0 on error.
 */
static int
check_mask_for_writemasked_reduction(NpyIter *iter, int iop)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);
    int maskop = NIT_MASKOP(iter);

    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    for(idim = 0; idim < ndim; ++idim) {
        npy_intp maskstride, istride;

        istride = NAD_STRIDES(axisdata)[iop];
        maskstride = NAD_STRIDES(axisdata)[maskop];

        /*
         * If 'iop' is being broadcast to 'maskop', we have
         * the invalid situation described above.
         */
        if (maskstride != 0 && istride == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator reduction operand is WRITEMASKED, "
                    "but also broadcasts to multiple mask values. "
                    "There can be only one mask value per WRITEMASKED "
                    "element.");
            return 0;
        }

        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    return 1;
}

/*
 * Fills in the AXISDATA for the 'nop' operands, broadcasting
 * the dimensionas as necessary.  Also fills
 * in the ITERSIZE data member.
 *
 * If op_axes is not NULL, it should point to an array of ndim-sized
 * arrays, one for each op.
 *
 * Returns 1 on success, 0 on failure.
 */
static int
npyiter_fill_axisdata(NpyIter *iter, npy_uint32 flags, npyiter_opitflags *op_itflags,
                    char **op_dataptr,
                    npy_uint32 *op_flags, int **op_axes,
                    npy_intp *itershape)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);
    int maskop = NIT_MASKOP(iter);

    int ondim;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    PyArrayObject **op = NIT_OPERANDS(iter), *op_cur;
    npy_intp broadcast_shape[NPY_MAXDIMS];

    /* First broadcast the shapes together */
    if (itershape == NULL) {
        for (idim = 0; idim < ndim; ++idim) {
            broadcast_shape[idim] = 1;
        }
    }
    else {
        for (idim = 0; idim < ndim; ++idim) {
            broadcast_shape[idim] = itershape[idim];
            /* Negative shape entries are deduced from the operands */
            if (broadcast_shape[idim] < 0) {
                broadcast_shape[idim] = 1;
            }
        }
    }
    for (iop = 0; iop < nop; ++iop) {
        op_cur = op[iop];
        if (op_cur != NULL) {
            npy_intp *shape = PyArray_DIMS(op_cur);
            ondim = PyArray_NDIM(op_cur);

            if (op_axes == NULL || op_axes[iop] == NULL) {
                /*
                 * Possible if op_axes are being used, but
                 * op_axes[iop] is NULL
                 */
                if (ondim > ndim) {
                    PyErr_SetString(PyExc_ValueError,
                            "input operand has more dimensions than allowed "
                            "by the axis remapping");
                    return 0;
                }
                for (idim = 0; idim < ondim; ++idim) {
                    npy_intp bshape = broadcast_shape[idim+ndim-ondim],
                                      op_shape = shape[idim];
                    if (bshape == 1) {
                        broadcast_shape[idim+ndim-ondim] = op_shape;
                    }
                    else if (bshape != op_shape && op_shape != 1) {
                        goto broadcast_error;
                    }
                }
            }
            else {
                int *axes = op_axes[iop];
                for (idim = 0; idim < ndim; ++idim) {
                    int i = axes[idim];
                    if (i >= 0) {
                        if (i < ondim) {
                            npy_intp bshape = broadcast_shape[idim],
                                              op_shape = shape[i];
                            if (bshape == 1) {
                                broadcast_shape[idim] = op_shape;
                            }
                            else if (bshape != op_shape && op_shape != 1) {
                                goto broadcast_error;
                            }
                        }
                        else {
                            PyErr_Format(PyExc_ValueError,
                                    "Iterator input op_axes[%d][%d] (==%d) "
                                    "is not a valid axis of op[%d], which "
                                    "has %d dimensions ",
                                    (int)iop, (int)(ndim-idim-1), (int)i,
                                    (int)iop, (int)ondim);
                            return 0;
                        }
                    }
                }
            }
        }
    }
    /*
     * If a shape was provided with a 1 entry, make sure that entry didn't
     * get expanded by broadcasting.
     */
    if (itershape != NULL) {
        for (idim = 0; idim < ndim; ++idim) {
            if (itershape[idim] == 1 && broadcast_shape[idim] != 1) {
                goto broadcast_error;
            }
        }
    }

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    if (ndim == 0) {
        /* Need to fill the first axisdata, even if the iterator is 0-d */
        NAD_SHAPE(axisdata) = 1;
        NAD_INDEX(axisdata) = 0;
        memcpy(NAD_PTRS(axisdata), op_dataptr, NPY_SIZEOF_INTP*nop);
    }

    /* Now process the operands, filling in the axisdata */
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp bshape = broadcast_shape[ndim-idim-1];
        npy_intp *strides = NAD_STRIDES(axisdata);

        NAD_SHAPE(axisdata) = bshape;
        NAD_INDEX(axisdata) = 0;
        memcpy(NAD_PTRS(axisdata), op_dataptr, NPY_SIZEOF_INTP*nop);

        for (iop = 0; iop < nop; ++iop) {
            op_cur = op[iop];

            if (op_axes == NULL || op_axes[iop] == NULL) {
                if (op_cur == NULL) {
                    strides[iop] = 0;
                }
                else {
                    ondim = PyArray_NDIM(op_cur);
                    if (bshape == 1) {
                        strides[iop] = 0;
                        if (idim >= ondim &&
                                    (op_flags[iop] & NPY_ITER_NO_BROADCAST)) {
                            goto operand_different_than_broadcast;
                        }
                    }
                    else if (idim >= ondim ||
                                    PyArray_DIM(op_cur, ondim-idim-1) == 1) {
                        strides[iop] = 0;
                        if (op_flags[iop] & NPY_ITER_NO_BROADCAST) {
                            goto operand_different_than_broadcast;
                        }
                        /* If it's writeable, this means a reduction */
                        if (op_itflags[iop] & NPY_OP_ITFLAG_WRITE) {
                            if (!(flags & NPY_ITER_REDUCE_OK)) {
                                PyErr_SetString(PyExc_ValueError,
                                        "output operand requires a "
                                        "reduction, but reduction is "
                                        "not enabled");
                                return 0;
                            }
                            if (!(op_itflags[iop] & NPY_OP_ITFLAG_READ)) {
                                PyErr_SetString(PyExc_ValueError,
                                        "output operand requires a "
                                        "reduction, but is flagged as "
                                        "write-only, not read-write");
                                return 0;
                            }
                            /*
                             * The ARRAYMASK can't be a reduction, because
                             * it would be possible to write back to the
                             * array once when the ARRAYMASK says 'True',
                             * then have the reduction on the ARRAYMASK
                             * later flip to 'False', indicating that the
                             * write back should never have been done,
                             * and violating the strict masking semantics
                             */
                            if (iop == maskop) {
                                PyErr_SetString(PyExc_ValueError,
                                        "output operand requires a "
                                        "reduction, but is flagged as "
                                        "the ARRAYMASK operand which "
                                        "is not permitted to be the "
                                        "result of a reduction");
                                return 0;
                            }

                            NIT_ITFLAGS(iter) |= NPY_ITFLAG_REDUCE;
                            op_itflags[iop] |= NPY_OP_ITFLAG_REDUCE;
                        }
                    }
                    else {
                        strides[iop] = PyArray_STRIDE(op_cur, ondim-idim-1);
                    }
                }
            }
            else {
                int *axes = op_axes[iop];
                int i = axes[ndim-idim-1];
                if (i >= 0) {
                    if (bshape == 1 || op_cur == NULL) {
                        strides[iop] = 0;
                    }
                    else if (PyArray_DIM(op_cur, i) == 1) {
                        strides[iop] = 0;
                        if (op_flags[iop] & NPY_ITER_NO_BROADCAST) {
                            goto operand_different_than_broadcast;
                        }
                        /* If it's writeable, this means a reduction */
                        if (op_itflags[iop] & NPY_OP_ITFLAG_WRITE) {
                            if (!(flags & NPY_ITER_REDUCE_OK)) {
                                PyErr_SetString(PyExc_ValueError,
                                        "output operand requires a reduction, but "
                                        "reduction is not enabled");
                                return 0;
                            }
                            if (!(op_itflags[iop] & NPY_OP_ITFLAG_READ)) {
                                PyErr_SetString(PyExc_ValueError,
                                        "output operand requires a reduction, but "
                                        "is flagged as write-only, not "
                                        "read-write");
                                return 0;
                            }
                            NIT_ITFLAGS(iter) |= NPY_ITFLAG_REDUCE;
                            op_itflags[iop] |= NPY_OP_ITFLAG_REDUCE;
                        }
                    }
                    else {
                        strides[iop] = PyArray_STRIDE(op_cur, i);
                    }
                }
                else if (bshape == 1) {
                    strides[iop] = 0;
                }
                else {
                    strides[iop] = 0;
                    /* If it's writeable, this means a reduction */
                    if (op_itflags[iop] & NPY_OP_ITFLAG_WRITE) {
                        if (!(flags & NPY_ITER_REDUCE_OK)) {
                            PyErr_SetString(PyExc_ValueError,
                                    "output operand requires a reduction, but "
                                    "reduction is not enabled");
                            return 0;
                        }
                        if (!(op_itflags[iop] & NPY_OP_ITFLAG_READ)) {
                            PyErr_SetString(PyExc_ValueError,
                                    "output operand requires a reduction, but "
                                    "is flagged as write-only, not "
                                    "read-write");
                            return 0;
                        }
                        NIT_ITFLAGS(iter) |= NPY_ITFLAG_REDUCE;
                        op_itflags[iop] |= NPY_OP_ITFLAG_REDUCE;
                    }
                }
            }
        }

        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    /* Now fill in the ITERSIZE member */
    NIT_ITERSIZE(iter) = 1;
    for (idim = 0; idim < ndim; ++idim) {
        NIT_ITERSIZE(iter) *= broadcast_shape[idim];
    }
    /* The range defaults to everything */
    NIT_ITERSTART(iter) = 0;
    NIT_ITEREND(iter) = NIT_ITERSIZE(iter);

    return 1;

broadcast_error: {
        PyObject *errmsg, *tmp;
        npy_intp remdims[NPY_MAXDIMS];
        char *tmpstr;

        if (op_axes == NULL) {
            errmsg = PyUString_FromString("operands could not be broadcast "
                                          "together with shapes ");
            if (errmsg == NULL) {
                return 0;
            }
            for (iop = 0; iop < nop; ++iop) {
                if (op[iop] != NULL) {
                    tmp = npyiter_shape_string(PyArray_NDIM(op[iop]),
                                                    PyArray_DIMS(op[iop]),
                                                    " ");
                    if (tmp == NULL) {
                        Py_DECREF(errmsg);
                        return 0;
                    }
                    PyUString_ConcatAndDel(&errmsg, tmp);
                    if (errmsg == NULL) {
                        return 0;
                    }
                }
            }
            if (itershape != NULL) {
                tmp = PyUString_FromString("and requested shape ");
                if (tmp == NULL) {
                    Py_DECREF(errmsg);
                    return 0;
                }
                PyUString_ConcatAndDel(&errmsg, tmp);
                if (errmsg == NULL) {
                    return 0;
                }

                tmp = npyiter_shape_string(ndim, itershape, "");
                if (tmp == NULL) {
                    Py_DECREF(errmsg);
                    return 0;
                }
                PyUString_ConcatAndDel(&errmsg, tmp);
                if (errmsg == NULL) {
                    return 0;
                }

            }
            PyErr_SetObject(PyExc_ValueError, errmsg);
            Py_DECREF(errmsg);
        }
        else {
            errmsg = PyUString_FromString("operands could not be broadcast "
                                          "together with remapped shapes "
                                          "[original->remapped]: ");
            for (iop = 0; iop < nop; ++iop) {
                if (op[iop] != NULL) {
                    int *axes = op_axes[iop];

                    tmpstr = (axes == NULL) ? " " : "->";
                    tmp = npyiter_shape_string(PyArray_NDIM(op[iop]),
                                                    PyArray_DIMS(op[iop]),
                                                    tmpstr);
                    if (tmp == NULL) {
                        return 0;
                    }
                    PyUString_ConcatAndDel(&errmsg, tmp);
                    if (errmsg == NULL) {
                        return 0;
                    }

                    if (axes != NULL) {
                        for (idim = 0; idim < ndim; ++idim) {
                            npy_intp i = axes[idim];

                            if (i >= 0 && i < PyArray_NDIM(op[iop])) {
                                remdims[idim] = PyArray_DIM(op[iop], i);
                            }
                            else {
                                remdims[idim] = -1;
                            }
                        }
                        tmp = npyiter_shape_string(ndim, remdims, " ");
                        if (tmp == NULL) {
                            return 0;
                        }
                        PyUString_ConcatAndDel(&errmsg, tmp);
                        if (errmsg == NULL) {
                            return 0;
                        }
                    }
                }
            }
            if (itershape != NULL) {
                tmp = PyUString_FromString("and requested shape ");
                if (tmp == NULL) {
                    Py_DECREF(errmsg);
                    return 0;
                }
                PyUString_ConcatAndDel(&errmsg, tmp);
                if (errmsg == NULL) {
                    return 0;
                }

                tmp = npyiter_shape_string(ndim, itershape, "");
                if (tmp == NULL) {
                    Py_DECREF(errmsg);
                    return 0;
                }
                PyUString_ConcatAndDel(&errmsg, tmp);
                if (errmsg == NULL) {
                    return 0;
                }

            }
            PyErr_SetObject(PyExc_ValueError, errmsg);
            Py_DECREF(errmsg);
        }

        return 0;
    }

operand_different_than_broadcast: {
        npy_intp remdims[NPY_MAXDIMS];
        PyObject *errmsg, *tmp;

        /* Start of error message */
        if (op_flags[iop] & NPY_ITER_READONLY) {
            errmsg = PyUString_FromString("non-broadcastable operand "
                                          "with shape ");
        }
        else {
            errmsg = PyUString_FromString("non-broadcastable output "
                                          "operand with shape ");
        }
        if (errmsg == NULL) {
            return 0;
        }

        /* Operand shape */
        tmp = npyiter_shape_string(PyArray_NDIM(op[iop]),
                                        PyArray_DIMS(op[iop]), "");
        if (tmp == NULL) {
            return 0;
        }
        PyUString_ConcatAndDel(&errmsg, tmp);
        if (errmsg == NULL) {
            return 0;
        }
        /* Remapped operand shape */
        if (op_axes != NULL && op_axes[iop] != NULL) {
            int *axes = op_axes[iop];

            for (idim = 0; idim < ndim; ++idim) {
                npy_intp i = axes[ndim-idim-1];

                if (i >= 0 && i < PyArray_NDIM(op[iop])) {
                    remdims[idim] = PyArray_DIM(op[iop], i);
                }
                else {
                    remdims[idim] = -1;
                }
            }

            tmp = PyUString_FromString(" [remapped to ");
            if (tmp == NULL) {
                return 0;
            }
            PyUString_ConcatAndDel(&errmsg, tmp);
            if (errmsg == NULL) {
                return 0;
            }

            tmp = npyiter_shape_string(ndim, remdims, "]");
            if (tmp == NULL) {
                return 0;
            }
            PyUString_ConcatAndDel(&errmsg, tmp);
            if (errmsg == NULL) {
                return 0;
            }
        }

        tmp = PyUString_FromString(" doesn't match the broadcast shape ");
        if (tmp == NULL) {
            return 0;
        }
        PyUString_ConcatAndDel(&errmsg, tmp);
        if (errmsg == NULL) {
            return 0;
        }

        /* Broadcast shape */
        tmp = npyiter_shape_string(ndim, broadcast_shape, "");
        if (tmp == NULL) {
            return 0;
        }
        PyUString_ConcatAndDel(&errmsg, tmp);
        if (errmsg == NULL) {
            return 0;
        }

        PyErr_SetObject(PyExc_ValueError, errmsg);
        Py_DECREF(errmsg);

        return 0;
    }
}

/*
 * Replaces the AXISDATA for the iop'th operand, broadcasting
 * the dimensions as necessary.  Assumes the replacement array is
 * exactly the same shape as the original array used when
 * npy_fill_axisdata was called.
 *
 * If op_axes is not NULL, it should point to an ndim-sized
 * array.
 */
static void
npyiter_replace_axisdata(NpyIter *iter, int iop,
                      PyArrayObject *op,
                      int op_ndim, char *op_dataptr,
                      int *op_axes)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    NpyIter_AxisData *axisdata0, *axisdata;
    npy_intp sizeof_axisdata;
    npy_int8 *perm;
    npy_intp baseoffset = 0;

    perm = NIT_PERM(iter);
    axisdata0 = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /*
     * Replace just the strides which were non-zero, and compute
     * the base data address.
     */
    axisdata = axisdata0;

    if (op_axes != NULL) {
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            npy_int8 p;
            int i;
            npy_intp shape;

            /* Apply the perm to get the original axis */
            p = perm[idim];
            if (p < 0) {
                i = op_axes[ndim+p];
            }
            else {
                i = op_axes[ndim-p-1];
            }

            if (0 <= i && i < op_ndim) {
                shape = PyArray_DIM(op, i);
                if (shape != 1) {
                    npy_intp stride = PyArray_STRIDE(op, i);
                    if (p < 0) {
                        /* If the perm entry is negative, flip the axis */
                        NAD_STRIDES(axisdata)[iop] = -stride;
                        baseoffset += stride*(shape-1);
                    }
                    else {
                        NAD_STRIDES(axisdata)[iop] = stride;
                    }
                }
            }
        }
    }
    else {
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            npy_int8 p;
            int i;
            npy_intp shape;

            /* Apply the perm to get the original axis */
            p = perm[idim];
            if (p < 0) {
                i = op_ndim+p;
            }
            else {
                i = op_ndim-p-1;
            }

            if (i >= 0) {
                shape = PyArray_DIM(op, i);
                if (shape != 1) {
                    npy_intp stride = PyArray_STRIDE(op, i);
                    if (p < 0) {
                        /* If the perm entry is negative, flip the axis */
                        NAD_STRIDES(axisdata)[iop] = -stride;
                        baseoffset += stride*(shape-1);
                    }
                    else {
                        NAD_STRIDES(axisdata)[iop] = stride;
                    }
                }
            }
        }
    }

    op_dataptr += baseoffset;

    /* Now the base data pointer is calculated, set it everywhere it's needed */
    NIT_RESETDATAPTR(iter)[iop] = op_dataptr;
    NIT_BASEOFFSETS(iter)[iop] = baseoffset;
    axisdata = axisdata0;
    /* Fill at least one axisdata, for the 0-d case */
    NAD_PTRS(axisdata)[iop] = op_dataptr;
    NIT_ADVANCE_AXISDATA(axisdata, 1);
    for (idim = 1; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        NAD_PTRS(axisdata)[iop] = op_dataptr;
    }
}

/*
 * Computes the iterator's index strides and initializes the index values
 * to zero.
 *
 * This must be called before the axes (i.e. the AXISDATA array) may
 * be reordered.
 */
static void
npyiter_compute_index_strides(NpyIter *iter, npy_uint32 flags)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp indexstride;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    /*
     * If there is only one element being iterated, we just have
     * to touch the first AXISDATA because nothing will ever be
     * incremented. This also initializes the data for the 0-d case.
     */
    if (NIT_ITERSIZE(iter) == 1) {
        if (itflags & NPY_ITFLAG_HASINDEX) {
            axisdata = NIT_AXISDATA(iter);
            NAD_PTRS(axisdata)[nop] = 0;
        }
        return;
    }

    if (flags & NPY_ITER_C_INDEX) {
        sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
        axisdata = NIT_AXISDATA(iter);
        indexstride = 1;
        for(idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            npy_intp shape = NAD_SHAPE(axisdata);

            if (shape == 1) {
                NAD_STRIDES(axisdata)[nop] = 0;
            }
            else {
                NAD_STRIDES(axisdata)[nop] = indexstride;
            }
            NAD_PTRS(axisdata)[nop] = 0;
            indexstride *= shape;
        }
    }
    else if (flags & NPY_ITER_F_INDEX) {
        sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
        axisdata = NIT_INDEX_AXISDATA(NIT_AXISDATA(iter), ndim-1);
        indexstride = 1;
        for(idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, -1)) {
            npy_intp shape = NAD_SHAPE(axisdata);

            if (shape == 1) {
                NAD_STRIDES(axisdata)[nop] = 0;
            }
            else {
                NAD_STRIDES(axisdata)[nop] = indexstride;
            }
            NAD_PTRS(axisdata)[nop] = 0;
            indexstride *= shape;
        }
    }
}

/*
 * If the order is NPY_KEEPORDER, lets the iterator find the best
 * iteration order, otherwise forces it.  Indicates in the itflags that
 * whether the iteration order was forced.
 */
static void
npyiter_apply_forced_iteration_order(NpyIter *iter, NPY_ORDER order)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    switch (order) {
    case NPY_CORDER:
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_FORCEDORDER;
        break;
    case NPY_FORTRANORDER:
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_FORCEDORDER;
        /* Only need to actually do something if there is more than 1 dim */
        if (ndim > 1) {
            npyiter_reverse_axis_ordering(iter);
        }
        break;
    case NPY_ANYORDER:
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_FORCEDORDER;
        /* Only need to actually do something if there is more than 1 dim */
        if (ndim > 1) {
            PyArrayObject **op = NIT_OPERANDS(iter);
            int forder = 1;

            /* Check that all the array inputs are fortran order */
            for (iop = 0; iop < nop; ++iop, ++op) {
                if (*op && !PyArray_CHKFLAGS(*op, NPY_ARRAY_F_CONTIGUOUS)) {
                   forder = 0;
                   break;
                }
            }

            if (forder) {
                npyiter_reverse_axis_ordering(iter);
            }
        }
        break;
    case NPY_KEEPORDER:
        /* Don't set the forced order flag here... */
        break;
    }
}

/*
 * This function negates any strides in the iterator
 * which are negative.  When iterating more than one
 * object, it only flips strides when they are all
 * negative or zero.
 */
static void
npyiter_flip_negative_strides(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    npy_intp istrides, nstrides = NAD_NSTRIDES();
    NpyIter_AxisData *axisdata, *axisdata0;
    npy_intp *baseoffsets;
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    int any_flipped = 0;

    axisdata0 = axisdata = NIT_AXISDATA(iter);
    baseoffsets = NIT_BASEOFFSETS(iter);
    for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        npy_intp *strides = NAD_STRIDES(axisdata);
        int any_negative = 0;

        /*
         * Check the signs of all the operand strides.
         */
        for (iop = 0; iop < nop; ++iop) {
            if (strides[iop] < 0) {
                any_negative = 1;
            }
            else if (strides[iop] != 0) {
                break;
            }
        }
        /*
         * If at least one stride is negative and none are positive,
         * flip all the strides for this dimension.
         */
        if (any_negative && iop == nop) {
            npy_intp shapem1 = NAD_SHAPE(axisdata) - 1;

            for (istrides = 0; istrides < nstrides; ++istrides) {
                npy_intp stride = strides[istrides];

                /* Adjust the base pointers to start at the end */
                baseoffsets[istrides] += shapem1 * stride;
                /* Flip the stride */
                strides[istrides] = -stride;
            }
            /*
             * Make the perm entry negative so get_multi_index
             * knows it's flipped
             */
            NIT_PERM(iter)[idim] = -1-NIT_PERM(iter)[idim];

            any_flipped = 1;
        }
    }

    /*
     * If any strides were flipped, the base pointers were adjusted
     * in the first AXISDATA, and need to be copied to all the rest
     */
    if (any_flipped) {
        char **resetdataptr = NIT_RESETDATAPTR(iter);

        for (istrides = 0; istrides < nstrides; ++istrides) {
            resetdataptr[istrides] += baseoffsets[istrides];
        }
        axisdata = axisdata0;
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            char **ptrs = NAD_PTRS(axisdata);
            for (istrides = 0; istrides < nstrides; ++istrides) {
                ptrs[istrides] = resetdataptr[istrides];
            }
        }
        /*
         * Indicate that some of the perm entries are negative,
         * and that it's not (strictly speaking) the identity perm.
         */
        NIT_ITFLAGS(iter) = (NIT_ITFLAGS(iter)|NPY_ITFLAG_NEGPERM) &
                            ~NPY_ITFLAG_IDENTPERM;
    }
}

static void
npyiter_reverse_axis_ordering(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp i, temp, size;
    npy_intp *first, *last;
    npy_int8 *perm;

    size = NIT_AXISDATA_SIZEOF(itflags, ndim, nop)/NPY_SIZEOF_INTP;
    first = (npy_intp*)NIT_AXISDATA(iter);
    last = first + (ndim-1)*size;

    /* This loop reverses the order of the AXISDATA array */
    while (first < last) {
        for (i = 0; i < size; ++i) {
            temp = first[i];
            first[i] = last[i];
            last[i] = temp;
        }
        first += size;
        last -= size;
    }

    /* Store the perm we applied */
    perm = NIT_PERM(iter);
    for(i = ndim-1; i >= 0; --i, ++perm) {
        *perm = (npy_int8)i;
    }

    NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_IDENTPERM;
}

static NPY_INLINE npy_intp
intp_abs(npy_intp x)
{
    return (x < 0) ? -x : x;
}

static void
npyiter_find_best_axis_ordering(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    npy_intp ax_i0, ax_i1, ax_ipos;
    npy_int8 ax_j0, ax_j1;
    npy_int8 *perm;
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    int permuted = 0;

    perm = NIT_PERM(iter);

    /*
     * Do a custom stable insertion sort.  Note that because
     * the AXISDATA has been reversed from C order, this
     * is sorting from smallest stride to biggest stride.
     */
    for (ax_i0 = 1; ax_i0 < ndim; ++ax_i0) {
        npy_intp *strides0;

        /* 'ax_ipos' is where perm[ax_i0] will get inserted */
        ax_ipos = ax_i0;
        ax_j0 = perm[ax_i0];

        strides0 = NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, ax_j0));
        for (ax_i1 = ax_i0-1; ax_i1 >= 0; --ax_i1) {
            int ambig = 1, shouldswap = 0;
            npy_intp *strides1;

            ax_j1 = perm[ax_i1];

            strides1 = NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, ax_j1));

            for (iop = 0; iop < nop; ++iop) {
                if (strides0[iop] != 0 && strides1[iop] != 0) {
                    if (intp_abs(strides1[iop]) <=
                                            intp_abs(strides0[iop])) {
                        /*
                         * Set swap even if it's not ambiguous already,
                         * because in the case of conflicts between
                         * different operands, C-order wins.
                         */
                        shouldswap = 0;
                    }
                    else {
                        /* Only set swap if it's still ambiguous */
                        if (ambig) {
                            shouldswap = 1;
                        }
                    }

                    /*
                     * A comparison has been done, so it's
                     * no longer ambiguous
                     */
                    ambig = 0;
                }
            }
            /*
             * If the comparison was unambiguous, either shift
             * 'ax_ipos' to 'ax_i1' or stop looking for an insertion
             * point
             */
            if (!ambig) {
                if (shouldswap) {
                    ax_ipos = ax_i1;
                }
                else {
                    break;
                }
            }
        }

        /* Insert perm[ax_i0] into the right place */
        if (ax_ipos != ax_i0) {
            for (ax_i1 = ax_i0; ax_i1 > ax_ipos; --ax_i1) {
                perm[ax_i1] = perm[ax_i1-1];
            }
            perm[ax_ipos] = ax_j0;
            permuted = 1;
        }
    }

    /* Apply the computed permutation to the AXISDATA array */
    if (permuted == 1) {
        npy_intp i, size = sizeof_axisdata/NPY_SIZEOF_INTP;
        NpyIter_AxisData *ad_i;

        /* Use the index as a flag, set each to 1 */
        ad_i = axisdata;
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(ad_i, 1)) {
            NAD_INDEX(ad_i) = 1;
        }
        /* Apply the permutation by following the cycles */
        for (idim = 0; idim < ndim; ++idim) {
            ad_i = NIT_INDEX_AXISDATA(axisdata, idim);

            /* If this axis hasn't been touched yet, process it */
            if (NAD_INDEX(ad_i) == 1) {
                npy_int8 pidim = perm[idim];
                npy_intp tmp;
                NpyIter_AxisData *ad_p, *ad_q;

                if (pidim != idim) {
                    /* Follow the cycle, copying the data */
                    for (i = 0; i < size; ++i) {
                        pidim = perm[idim];
                        ad_q = ad_i;
                        tmp = *((npy_intp*)ad_q + i);
                        while (pidim != idim) {
                            ad_p = NIT_INDEX_AXISDATA(axisdata, pidim);
                            *((npy_intp*)ad_q + i) = *((npy_intp*)ad_p + i);

                            ad_q = ad_p;
                            pidim = perm[(int)pidim];
                        }
                        *((npy_intp*)ad_q + i) = tmp;
                    }
                    /* Follow the cycle again, marking it as done */
                    pidim = perm[idim];

                    while (pidim != idim) {
                        NAD_INDEX(NIT_INDEX_AXISDATA(axisdata, pidim)) = 0;
                        pidim = perm[(int)pidim];
                    }
                }
                NAD_INDEX(ad_i) = 0;
            }
        }
        /* Clear the identity perm flag */
        NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_IDENTPERM;
    }
}

/*
 * Calculates a dtype that all the types can be promoted to, using the
 * ufunc rules.  If only_inputs is 1, it leaves any operands that
 * are not read from out of the calculation.
 */
static PyArray_Descr *
npyiter_get_common_dtype(int nop, PyArrayObject **op,
                        npyiter_opitflags *op_itflags, PyArray_Descr **op_dtype,
                        PyArray_Descr **op_request_dtypes,
                        int only_inputs)
{
    int iop;
    npy_intp narrs = 0, ndtypes = 0;
    PyArrayObject *arrs[NPY_MAXARGS];
    PyArray_Descr *dtypes[NPY_MAXARGS];
    PyArray_Descr *ret;

    NPY_IT_DBG_PRINT("Iterator: Getting a common data type from operands\n");

    for (iop = 0; iop < nop; ++iop) {
        if (op_dtype[iop] != NULL &&
                    (!only_inputs || (op_itflags[iop] & NPY_OP_ITFLAG_READ))) {
            /* If no dtype was requested and the op is a scalar, pass the op */
            if ((op_request_dtypes == NULL ||
                            op_request_dtypes[iop] == NULL) &&
                                            PyArray_NDIM(op[iop]) == 0) {
                arrs[narrs++] = op[iop];
            }
            /* Otherwise just pass in the dtype */
            else {
                dtypes[ndtypes++] = op_dtype[iop];
            }
        }
    }

    if (narrs == 0) {
        npy_intp i;
        ret = dtypes[0];
        for (i = 1; i < ndtypes; ++i) {
            if (ret != dtypes[i])
                break;
        }
        if (i == ndtypes) {
            if (ndtypes == 1 || PyArray_ISNBO(ret->byteorder)) {
                Py_INCREF(ret);
            }
            else {
                ret = PyArray_DescrNewByteorder(ret, NPY_NATIVE);
            }
        }
        else {
            ret = PyArray_ResultType(narrs, arrs, ndtypes, dtypes);
        }
    }
    else {
        ret = PyArray_ResultType(narrs, arrs, ndtypes, dtypes);
    }

    return ret;
}

/*
 * Allocates a temporary array which can be used to replace op
 * in the iteration.  Its dtype will be op_dtype.
 *
 * The result array has a memory ordering which matches the iterator,
 * which may or may not match that of op.  The parameter 'shape' may be
 * NULL, in which case it is filled in from the iterator's shape.
 *
 * This function must be called before any axes are coalesced.
 */
static PyArrayObject *
npyiter_new_temp_array(NpyIter *iter, PyTypeObject *subtype,
                npy_uint32 flags, npyiter_opitflags *op_itflags,
                int op_ndim, npy_intp *shape,
                PyArray_Descr *op_dtype, int *op_axes)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_int8 *perm = NIT_PERM(iter);
    npy_intp new_shape[NPY_MAXDIMS], strides[NPY_MAXDIMS],
             stride = op_dtype->elsize;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    npy_intp i;

    PyArrayObject *ret;

    /*
     * There is an interaction with array-dtypes here, which
     * generally works. Let's say you make an nditer with an
     * output dtype of a 2-double array. All-scalar inputs
     * will result in a 1-dimensional output with shape (2).
     * Everything still works out in the nditer, because the
     * new dimension is always added on the end, and it cares
     * about what happens at the beginning.
     */

    /* If it's a scalar, don't need to check the axes */
    if (op_ndim == 0) {
        Py_INCREF(op_dtype);
        ret = (PyArrayObject *)PyArray_NewFromDescr(subtype, op_dtype, 0,
                               NULL, NULL, NULL, 0, NULL);

        return ret;
    }

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /* Initialize the strides to invalid values */
    for (i = 0; i < NPY_MAXDIMS; ++i) {
        strides[i] = NPY_MAX_INTP;
    }

    if (op_axes != NULL) {
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            npy_int8 p;

            /* Apply the perm to get the original axis */
            p = perm[idim];
            if (p < 0) {
                i = op_axes[ndim+p];
            }
            else {
                i = op_axes[ndim-p-1];
            }

            if (i >= 0) {
                NPY_IT_DBG_PRINT3("Iterator: Setting allocated stride %d "
                                    "for iterator dimension %d to %d\n", (int)i,
                                    (int)idim, (int)stride);
                strides[i] = stride;
                if (shape == NULL) {
                    new_shape[i] = NAD_SHAPE(axisdata);
                    stride *= new_shape[i];
                    if (i >= ndim) {
                        PyErr_SetString(PyExc_ValueError,
                                "automatically allocated output array "
                                "specified with an inconsistent axis mapping");
                        return NULL;
                    }
                }
                else {
                    stride *= shape[i];
                }
            }
            else {
                if (shape == NULL) {
                    /*
                     * If deleting this axis produces a reduction, but
                     * reduction wasn't enabled, throw an error
                     */
                    if (NAD_SHAPE(axisdata) != 1) {
                        if (!(flags & NPY_ITER_REDUCE_OK)) {
                            PyErr_SetString(PyExc_ValueError,
                                    "output requires a reduction, but "
                                    "reduction is not enabled");
                            return NULL;
                        }
                        if (!((*op_itflags) & NPY_OP_ITFLAG_READ)) {
                            PyErr_SetString(PyExc_ValueError,
                                    "output requires a reduction, but "
                                    "is flagged as write-only, not read-write");
                            return NULL;
                        }

                        NPY_IT_DBG_PRINT("Iterator: Indicating that a "
                                          "reduction is occurring\n");
                        /* Indicate that a reduction is occurring */
                        NIT_ITFLAGS(iter) |= NPY_ITFLAG_REDUCE;
                        (*op_itflags) |= NPY_OP_ITFLAG_REDUCE;
                    }
                }
            }
        }
    }
    else {
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            npy_int8 p;

            /* Apply the perm to get the original axis */
            p = perm[idim];
            if (p < 0) {
                i = op_ndim + p;
            }
            else {
                i = op_ndim - p - 1;
            }

            if (i >= 0) {
                NPY_IT_DBG_PRINT3("Iterator: Setting allocated stride %d "
                                    "for iterator dimension %d to %d\n", (int)i,
                                    (int)idim, (int)stride);
                strides[i] = stride;
                if (shape == NULL) {
                    new_shape[i] = NAD_SHAPE(axisdata);
                    stride *= new_shape[i];
                }
                else {
                    stride *= shape[i];
                }
            }
        }
    }

    /*
     * If custom axes were specified, some dimensions may not have been used.
     * Add the REDUCE itflag if this creates a reduction situation.
     */
    if (shape == NULL) {
        /* Ensure there are no dimension gaps in op_axes, and find op_ndim */
        op_ndim = ndim;
        if (op_axes != NULL) {
            for (i = 0; i < ndim; ++i) {
                if (strides[i] == NPY_MAX_INTP) {
                    if (op_ndim == ndim) {
                        op_ndim = i;
                    }
                }
                /*
                 * If there's a gap in the array's dimensions, it's an error.
                 * For example, op_axes of [0,2] for the automatically
                 * allocated output.
                 */
                else if (op_ndim != ndim) {
                    PyErr_SetString(PyExc_ValueError,
                            "automatically allocated output array "
                            "specified with an inconsistent axis mapping");
                    return NULL;
                }
            }
        }
    }
    else {
        for (i = 0; i < op_ndim; ++i) {
            if (strides[i] == NPY_MAX_INTP) {
                npy_intp factor, new_strides[NPY_MAXDIMS],
                         itemsize;

                /* Fill in the missing strides in C order */
                factor = 1;
                itemsize = op_dtype->elsize;
                for (i = op_ndim-1; i >= 0; --i) {
                    if (strides[i] == NPY_MAX_INTP) {
                        new_strides[i] = factor * itemsize;
                        factor *= shape[i];
                    }
                }

                /*
                 * Copy the missing strides, and multiply the existing strides
                 * by the calculated factor.  This way, the missing strides
                 * are tighter together in memory, which is good for nested
                 * loops.
                 */
                for (i = 0; i < op_ndim; ++i) {
                    if (strides[i] == NPY_MAX_INTP) {
                        strides[i] = new_strides[i];
                    }
                    else {
                        strides[i] *= factor;
                    }
                }

                break;
            }
        }
    }

    /* If shape was NULL, set it to the shape we calculated */
    if (shape == NULL) {
        shape = new_shape;
    }

    /* Allocate the temporary array */
    Py_INCREF(op_dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(subtype, op_dtype, op_ndim,
                               shape, strides, NULL, 0, NULL);
    if (ret == NULL) {
        return NULL;
    }

    /* Make sure all the flags are good */
    PyArray_UpdateFlags(ret, NPY_ARRAY_UPDATE_ALL);

    /* Double-check that the subtype didn't mess with the dimensions */
    if (subtype != &PyArray_Type) {
        if (PyArray_NDIM(ret) != op_ndim ||
                    !PyArray_CompareLists(shape, PyArray_DIMS(ret), op_ndim)) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Iterator automatic output has an array subtype "
                    "which changed the dimensions of the output");
            Py_DECREF(ret);
            return NULL;
        }
    }

    return ret;
}

static int
npyiter_allocate_arrays(NpyIter *iter,
                        npy_uint32 flags,
                        PyArray_Descr **op_dtype, PyTypeObject *subtype,
                        npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                        int **op_axes)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    int check_writemasked_reductions = 0;

    NpyIter_BufferData *bufferdata = NULL;
    PyArrayObject **op = NIT_OPERANDS(iter);

    if (itflags & NPY_ITFLAG_BUFFER) {
        bufferdata = NIT_BUFFERDATA(iter);
    }

    for (iop = 0; iop < nop; ++iop) {
        /*
         * Check whether there are any WRITEMASKED REDUCE operands
         * which should be validated after all the strides are filled
         * in.
         */
        if ((op_itflags[iop] &
                (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) ==
                        (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) {
            check_writemasked_reductions = 1;
        }

        /* NULL means an output the iterator should allocate */
        if (op[iop] == NULL) {
            PyArrayObject *out;
            PyTypeObject *op_subtype;
            int ondim = ndim;

            /* Check whether the subtype was disabled */
            op_subtype = (op_flags[iop] & NPY_ITER_NO_SUBTYPE) ?
                                                &PyArray_Type : subtype;

            /* Allocate the output array */
            out = npyiter_new_temp_array(iter, op_subtype,
                                        flags, &op_itflags[iop],
                                        ondim,
                                        NULL,
                                        op_dtype[iop],
                                        op_axes ? op_axes[iop] : NULL);
            if (out == NULL) {
                return 0;
            }

            op[iop] = out;

            /*
             * Now we need to replace the pointers and strides with values
             * from the new array.
             */
            npyiter_replace_axisdata(iter, iop, op[iop], ondim,
                    PyArray_DATA(op[iop]), op_axes ? op_axes[iop] : NULL);

            /* New arrays are aligned and need no cast */
            op_itflags[iop] |= NPY_OP_ITFLAG_ALIGNED;
            op_itflags[iop] &= ~NPY_OP_ITFLAG_CAST;
        }
        /*
         * If casting is required, the operand is read-only, and
         * it's an array scalar, make a copy whether or not the
         * copy flag is enabled.
         */
        else if ((op_itflags[iop] & (NPY_OP_ITFLAG_CAST |
                         NPY_OP_ITFLAG_READ |
                         NPY_OP_ITFLAG_WRITE)) == (NPY_OP_ITFLAG_CAST |
                                                   NPY_OP_ITFLAG_READ) &&
                          PyArray_NDIM(op[iop]) == 0) {
            PyArrayObject *temp;
            Py_INCREF(op_dtype[iop]);
            temp = (PyArrayObject *)PyArray_NewFromDescr(
                                        &PyArray_Type, op_dtype[iop],
                                        0, NULL, NULL, NULL, 0, NULL);
            if (temp == NULL) {
                return 0;
            }
            if (PyArray_CopyInto(temp, op[iop]) != 0) {
                Py_DECREF(temp);
                return 0;
            }
            Py_DECREF(op[iop]);
            op[iop] = temp;

            /*
             * Now we need to replace the pointers and strides with values
             * from the temporary array.
             */
            npyiter_replace_axisdata(iter, iop, op[iop], 0,
                    PyArray_DATA(op[iop]), NULL);

            /*
             * New arrays are aligned need no cast, and in the case
             * of scalars, always have stride 0 so never need buffering
             */
            op_itflags[iop] |= (NPY_OP_ITFLAG_ALIGNED |
                                  NPY_OP_ITFLAG_BUFNEVER);
            op_itflags[iop] &= ~NPY_OP_ITFLAG_CAST;
            if (itflags & NPY_ITFLAG_BUFFER) {
                NBF_STRIDES(bufferdata)[iop] = 0;
            }
        }
        /* If casting is required and permitted */
        else if ((op_itflags[iop] & NPY_OP_ITFLAG_CAST) &&
                   (op_flags[iop] & (NPY_ITER_COPY|NPY_ITER_UPDATEIFCOPY))) {
            PyArrayObject *temp;
            int ondim = PyArray_NDIM(op[iop]);

            /* Allocate the temporary array, if possible */
            temp = npyiter_new_temp_array(iter, &PyArray_Type,
                                        flags, &op_itflags[iop],
                                        ondim,
                                        PyArray_DIMS(op[iop]),
                                        op_dtype[iop],
                                        op_axes ? op_axes[iop] : NULL);
            if (temp == NULL) {
                return 0;
            }

            /*
             * If the data will be read, copy it into temp.
             * TODO: It might be possible to do a view into
             *       op[iop]'s mask instead here.
             */
            if (op_itflags[iop] & NPY_OP_ITFLAG_READ) {
                if (PyArray_CopyInto(temp, op[iop]) != 0) {
                    Py_DECREF(temp);
                    return 0;
                }
            }
            /* If the data will be written to, set UPDATEIFCOPY */
            if (op_itflags[iop] & NPY_OP_ITFLAG_WRITE) {
                Py_INCREF(op[iop]);
                if (PyArray_SetUpdateIfCopyBase(temp, op[iop]) < 0) {
                    Py_DECREF(temp);
                    return 0;
                }
            }

            Py_DECREF(op[iop]);
            op[iop] = temp;

            /*
             * Now we need to replace the pointers and strides with values
             * from the temporary array.
             */
            npyiter_replace_axisdata(iter, iop, op[iop], ondim,
                    PyArray_DATA(op[iop]), op_axes ? op_axes[iop] : NULL);

            /* The temporary copy is aligned and needs no cast */
            op_itflags[iop] |= NPY_OP_ITFLAG_ALIGNED;
            op_itflags[iop] &= ~NPY_OP_ITFLAG_CAST;
        }
        else {
            /*
             * Buffering must be enabled for casting/conversion if copy
             * wasn't specified.
             */
            if ((op_itflags[iop] & NPY_OP_ITFLAG_CAST) &&
                                  !(itflags & NPY_ITFLAG_BUFFER)) {
                PyErr_SetString(PyExc_TypeError,
                        "Iterator operand required copying or buffering, "
                        "but neither copying nor buffering was enabled");
                return 0;
            }

            /*
             * If the operand is aligned, any buffering can use aligned
             * optimizations.
             */
            if (PyArray_ISALIGNED(op[iop])) {
                op_itflags[iop] |= NPY_OP_ITFLAG_ALIGNED;
            }
        }

        /* Here we can finally check for contiguous iteration */
        if (op_flags[iop] & NPY_ITER_CONTIG) {
            NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
            npy_intp stride = NAD_STRIDES(axisdata)[iop];

            if (stride != op_dtype[iop]->elsize) {
                NPY_IT_DBG_PRINT("Iterator: Setting NPY_OP_ITFLAG_CAST "
                                    "because of NPY_ITER_CONTIG\n");
                op_itflags[iop] |= NPY_OP_ITFLAG_CAST;
                if (!(itflags & NPY_ITFLAG_BUFFER)) {
                    PyErr_SetString(PyExc_TypeError,
                            "Iterator operand required buffering, "
                            "to be contiguous as requested, but "
                            "buffering is not enabled");
                    return 0;
                }
            }
        }

        /*
         * If no alignment, byte swap, or casting is needed,
         * the inner stride of this operand works for the whole
         * array, we can set NPY_OP_ITFLAG_BUFNEVER.
         */
        if ((itflags & NPY_ITFLAG_BUFFER) &&
                                !(op_itflags[iop] & NPY_OP_ITFLAG_CAST)) {
            NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
            if (ndim <= 1) {
                op_itflags[iop] |= NPY_OP_ITFLAG_BUFNEVER;
                NBF_STRIDES(bufferdata)[iop] = NAD_STRIDES(axisdata)[iop];
            }
            else if (PyArray_NDIM(op[iop]) > 0) {
                npy_intp stride, shape, innerstride = 0, innershape;
                npy_intp sizeof_axisdata =
                                    NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
                /* Find stride of the first non-empty shape */
                for (idim = 0; idim < ndim; ++idim) {
                    innershape = NAD_SHAPE(axisdata);
                    if (innershape != 1) {
                        innerstride = NAD_STRIDES(axisdata)[iop];
                        break;
                    }
                    NIT_ADVANCE_AXISDATA(axisdata, 1);
                }
                ++idim;
                NIT_ADVANCE_AXISDATA(axisdata, 1);
                /* Check that everything could have coalesced together */
                for (; idim < ndim; ++idim) {
                    stride = NAD_STRIDES(axisdata)[iop];
                    shape = NAD_SHAPE(axisdata);
                    if (shape != 1) {
                        /*
                         * If N times the inner stride doesn't equal this
                         * stride, the multi-dimensionality is needed.
                         */
                        if (innerstride*innershape != stride) {
                            break;
                        }
                        else {
                            innershape *= shape;
                        }
                    }
                    NIT_ADVANCE_AXISDATA(axisdata, 1);
                }
                /*
                 * If we looped all the way to the end, one stride works.
                 * Set that stride, because it may not belong to the first
                 * dimension.
                 */
                if (idim == ndim) {
                    op_itflags[iop] |= NPY_OP_ITFLAG_BUFNEVER;
                    NBF_STRIDES(bufferdata)[iop] = innerstride;
                }
            }
        }
    }

    if (check_writemasked_reductions) {
        for (iop = 0; iop < nop; ++iop) {
            /*
             * Check whether there are any WRITEMASKED REDUCE operands
             * which should be validated now that all the strides are filled
             * in.
             */
            if ((op_itflags[iop] &
                    (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) ==
                        (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) {
                /*
                 * If the ARRAYMASK has 'bigger' dimensions
                 * than this REDUCE WRITEMASKED operand,
                 * the result would be more than one mask
                 * value per reduction element, something which
                 * is invalid. This function provides validation
                 * for that.
                 */
                if (!check_mask_for_writemasked_reduction(iter, iop)) {
                    return 0;
                }
            }
        }
    }

    return 1;
}

/*
 * The __array_priority__ attribute of the inputs determines
 * the subtype of any output arrays.  This function finds the
 * subtype of the input array with highest priority.
 */
static void
npyiter_get_priority_subtype(int nop, PyArrayObject **op,
                            npyiter_opitflags *op_itflags,
                            double *subtype_priority,
                            PyTypeObject **subtype)
{
    int iop;

    for (iop = 0; iop < nop; ++iop) {
        if (op[iop] != NULL && op_itflags[iop] & NPY_OP_ITFLAG_READ) {
            double priority = PyArray_GetPriority((PyObject *)op[iop], 0.0);
            if (priority > *subtype_priority) {
                *subtype_priority = priority;
                *subtype = Py_TYPE(op[iop]);
            }
        }
    }
}

static int
npyiter_allocate_transfer_functions(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int iop = 0, nop = NIT_NOP(iter);

    npy_intp i;
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    PyArrayObject **op = NIT_OPERANDS(iter);
    PyArray_Descr **op_dtype = NIT_DTYPES(iter);
    npy_intp *strides = NAD_STRIDES(axisdata), op_stride;
    PyArray_StridedUnaryOp **readtransferfn = NBF_READTRANSFERFN(bufferdata),
                        **writetransferfn = NBF_WRITETRANSFERFN(bufferdata);
    NpyAuxData **readtransferdata = NBF_READTRANSFERDATA(bufferdata),
               **writetransferdata = NBF_WRITETRANSFERDATA(bufferdata);

    PyArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int needs_api = 0;

    for (iop = 0; iop < nop; ++iop) {
        npyiter_opitflags flags = op_itflags[iop];
        /*
         * Reduction operands may be buffered with a different stride,
         * so we must pass NPY_MAX_INTP to the transfer function factory.
         */
        op_stride = (flags & NPY_OP_ITFLAG_REDUCE) ? NPY_MAX_INTP :
                                                   strides[iop];

        /*
         * If we have determined that a buffer may be needed,
         * allocate the appropriate transfer functions
         */
        if (!(flags & NPY_OP_ITFLAG_BUFNEVER)) {
            if (flags & NPY_OP_ITFLAG_READ) {
                int move_references = 0;
                if (PyArray_GetDTypeTransferFunction(
                                        (flags & NPY_OP_ITFLAG_ALIGNED) != 0,
                                        op_stride,
                                        op_dtype[iop]->elsize,
                                        PyArray_DESCR(op[iop]),
                                        op_dtype[iop],
                                        move_references,
                                        &stransfer,
                                        &transferdata,
                                        &needs_api) != NPY_SUCCEED) {
                    goto fail;
                }
                readtransferfn[iop] = stransfer;
                readtransferdata[iop] = transferdata;
            }
            else {
                readtransferfn[iop] = NULL;
            }
            if (flags & NPY_OP_ITFLAG_WRITE) {
                int move_references = 1;

                /* If the operand is WRITEMASKED, use a masked transfer fn */
                if (flags & NPY_OP_ITFLAG_WRITEMASKED) {
                    int maskop = NIT_MASKOP(iter);
                    PyArray_Descr *mask_dtype = PyArray_DESCR(op[maskop]);

                    /*
                     * If the mask's stride is contiguous, use it, otherwise
                     * the mask may or may not be buffered, so the stride
                     * could be inconsistent.
                     */
                    if (PyArray_GetMaskedDTypeTransferFunction(
                                (flags & NPY_OP_ITFLAG_ALIGNED) != 0,
                                op_dtype[iop]->elsize,
                                op_stride,
                                (strides[maskop] == mask_dtype->elsize) ?
                                                mask_dtype->elsize :
                                                NPY_MAX_INTP,
                                op_dtype[iop],
                                PyArray_DESCR(op[iop]),
                                mask_dtype,
                                move_references,
                                (PyArray_MaskedStridedUnaryOp **)&stransfer,
                                &transferdata,
                                &needs_api) != NPY_SUCCEED) {
                        goto fail;
                    }
                }
                else {
                    if (PyArray_GetDTypeTransferFunction(
                                        (flags & NPY_OP_ITFLAG_ALIGNED) != 0,
                                        op_dtype[iop]->elsize,
                                        op_stride,
                                        op_dtype[iop],
                                        PyArray_DESCR(op[iop]),
                                        move_references,
                                        &stransfer,
                                        &transferdata,
                                        &needs_api) != NPY_SUCCEED) {
                        goto fail;
                    }
                }
                writetransferfn[iop] = stransfer;
                writetransferdata[iop] = transferdata;
            }
            /* If no write back but there are references make a decref fn */
            else if (PyDataType_REFCHK(op_dtype[iop])) {
                /*
                 * By passing NULL to dst_type and setting move_references
                 * to 1, we get back a function that just decrements the
                 * src references.
                 */
                if (PyArray_GetDTypeTransferFunction(
                                        (flags & NPY_OP_ITFLAG_ALIGNED) != 0,
                                        op_dtype[iop]->elsize, 0,
                                        op_dtype[iop], NULL,
                                        1,
                                        &stransfer,
                                        &transferdata,
                                        &needs_api) != NPY_SUCCEED) {
                    goto fail;
                }
                writetransferfn[iop] = stransfer;
                writetransferdata[iop] = transferdata;
            }
            else {
                writetransferfn[iop] = NULL;
            }
        }
        else {
            readtransferfn[iop] = NULL;
            writetransferfn[iop] = NULL;
        }
    }

    /* If any of the dtype transfer functions needed the API, flag it */
    if (needs_api) {
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_NEEDSAPI;
    }

    return 1;

fail:
    for (i = 0; i < iop; ++i) {
        if (readtransferdata[iop] != NULL) {
            NPY_AUXDATA_FREE(readtransferdata[iop]);
            readtransferdata[iop] = NULL;
        }
        if (writetransferdata[iop] != NULL) {
            NPY_AUXDATA_FREE(writetransferdata[iop]);
            writetransferdata[iop] = NULL;
        }
    }
    return 0;
}

#undef NPY_ITERATOR_IMPLEMENTATION_CODE
