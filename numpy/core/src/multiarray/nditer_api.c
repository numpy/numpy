/*
 * This file implements most of the main API functions of NumPy's nditer.
 * This excludes functions specialized using the templating system.
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

/* Internal helper functions private to this file */
static npy_intp
npyiter_checkreducesize(NpyIter *iter, npy_intp count,
                                npy_intp *reduce_innersize,
                                npy_intp *reduce_outerdim);

/*NUMPY_API
 * Removes an axis from iteration. This requires that NPY_ITER_MULTI_INDEX
 * was set for iterator creation, and does not work if buffering is
 * enabled. This function also resets the iterator to its initial state.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
NpyIter_RemoveAxis(NpyIter *iter, int axis)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    int xdim = 0;
    npy_int8 *perm = NIT_PERM(iter);
    NpyIter_AxisData *axisdata_del = NIT_AXISDATA(iter), *axisdata;
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    npy_intp *baseoffsets = NIT_BASEOFFSETS(iter);
    char **resetdataptr = NIT_RESETDATAPTR(iter);

    if (!(itflags&NPY_ITFLAG_HASMULTIINDEX)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Iterator RemoveAxis may only be called "
                "if a multi-index is being tracked");
        return NPY_FAIL;
    }
    else if (itflags&NPY_ITFLAG_HASINDEX) {
        PyErr_SetString(PyExc_RuntimeError,
                "Iterator RemoveAxis may not be called on "
                "an index is being tracked");
        return NPY_FAIL;
    }
    else if (itflags&NPY_ITFLAG_BUFFER) {
        PyErr_SetString(PyExc_RuntimeError,
                "Iterator RemoveAxis may not be called on "
                "a buffered iterator");
        return NPY_FAIL;
    }
    else if (axis < 0 || axis >= ndim) {
        PyErr_SetString(PyExc_ValueError,
                "axis out of bounds in iterator RemoveAxis");
        return NPY_FAIL;
    }

    /* Reverse axis, since the iterator treats them that way */
    axis = ndim - 1 - axis;

    /* First find the axis in question */
    for (idim = 0; idim < ndim; ++idim) {
        /* If this is it, and it's iterated forward, done */
        if (perm[idim] == axis) {
            xdim = idim;
            break;
        }
        /* If this is it, but it's iterated backward, must reverse the axis */
        else if (-1 - perm[idim] == axis) {
            npy_intp *strides = NAD_STRIDES(axisdata_del);
            npy_intp shape = NAD_SHAPE(axisdata_del), offset;

            xdim = idim;

            /*
             * Adjust baseoffsets and resetbaseptr back to the start of
             * this axis.
             */
            for (iop = 0; iop < nop; ++iop) {
                offset = (shape-1)*strides[iop];
                baseoffsets[iop] += offset;
                resetdataptr[iop] += offset;
            }
            break;
        }

        NIT_ADVANCE_AXISDATA(axisdata_del, 1);
    }

    if (idim == ndim) {
        PyErr_SetString(PyExc_RuntimeError,
                "internal error in iterator perm");
        return NPY_FAIL;
    }

    if (NAD_SHAPE(axisdata_del) == 0) {
        PyErr_SetString(PyExc_ValueError,
                "cannot remove a zero-sized axis from an iterator");
        return NPY_FAIL;
    }

    /* Adjust the permutation */
    for (idim = 0; idim < ndim-1; ++idim) {
        npy_int8 p = (idim < xdim) ? perm[idim] : perm[idim+1];
        if (p >= 0) {
            if (p > axis) {
                --p;
            }
        }
        else if (p <= 0) {
            if (p < -1-axis) {
                ++p;
            }
        }
        perm[idim] = p;
    }

    /* Adjust the iteration size */
    NIT_ITERSIZE(iter) /= NAD_SHAPE(axisdata_del);

    /* Shift all the axisdata structures by one */
    axisdata = NIT_INDEX_AXISDATA(axisdata_del, 1);
    memmove(axisdata_del, axisdata, (ndim-1-xdim)*sizeof_axisdata);

    /* Shrink the iterator */
    NIT_NDIM(iter) = ndim - 1;
    /* If it is now 0-d fill the singleton dimension */
    if (ndim == 1) {
        npy_intp *strides = NAD_STRIDES(axisdata_del);
        NAD_SHAPE(axisdata_del) = 1;
        for (iop = 0; iop < nop; ++iop) {
            strides[iop] = 0;
        }
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
    }

    return NpyIter_Reset(iter, NULL);
}

/*NUMPY_API
 * Removes multi-index support from an iterator.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
NpyIter_RemoveMultiIndex(NpyIter *iter)
{
    npy_uint32 itflags;

    /* Make sure the iterator is reset */
    if (NpyIter_Reset(iter, NULL) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    itflags = NIT_ITFLAGS(iter);
    if (itflags&NPY_ITFLAG_HASMULTIINDEX) {
        NIT_ITFLAGS(iter) = itflags & ~NPY_ITFLAG_HASMULTIINDEX;
        npyiter_coalesce_axes(iter);
    }

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Removes the inner loop handling (so HasExternalLoop returns true)
 */
NPY_NO_EXPORT int
NpyIter_EnableExternalLoop(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    /* Check conditions under which this can be done */
    if (itflags&(NPY_ITFLAG_HASINDEX|NPY_ITFLAG_HASMULTIINDEX)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator flag EXTERNAL_LOOP cannot be used "
                "if an index or multi-index is being tracked");
        return NPY_FAIL;
    }
    if ((itflags&(NPY_ITFLAG_BUFFER|NPY_ITFLAG_RANGE|NPY_ITFLAG_EXLOOP))
                        == (NPY_ITFLAG_RANGE|NPY_ITFLAG_EXLOOP)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator flag EXTERNAL_LOOP cannot be used "
                "with ranged iteration unless buffering is also enabled");
        return NPY_FAIL;
    }
    /* Set the flag */
    if (!(itflags&NPY_ITFLAG_EXLOOP)) {
        itflags |= NPY_ITFLAG_EXLOOP;
        NIT_ITFLAGS(iter) = itflags;

        /*
         * Check whether we can apply the single iteration
         * optimization to the iternext function.
         */
        if (!(itflags&NPY_ITFLAG_BUFFER)) {
            NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
            if (NIT_ITERSIZE(iter) == NAD_SHAPE(axisdata)) {
                NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
            }
        }
    }

    /* Reset the iterator */
    return NpyIter_Reset(iter, NULL);
}

/*NUMPY_API
 * Resets the iterator to its initial state
 *
 * If errmsg is non-NULL, it should point to a variable which will
 * receive the error message, and no Python exception will be set.
 * This is so that the function can be called from code not holding
 * the GIL.
 */
NPY_NO_EXPORT int
NpyIter_Reset(NpyIter *iter, char **errmsg)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata;

        /* If buffer allocation was delayed, do it now */
        if (itflags&NPY_ITFLAG_DELAYBUF) {
            if (!npyiter_allocate_buffers(iter, errmsg)) {
                return NPY_FAIL;
            }
            NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_DELAYBUF;
        }
        else {
            /*
             * If the iterindex is already right, no need to
             * do anything
             */
            bufferdata = NIT_BUFFERDATA(iter);
            if (NIT_ITERINDEX(iter) == NIT_ITERSTART(iter) &&
                    NBF_BUFITEREND(bufferdata) <= NIT_ITEREND(iter) &&
                    NBF_SIZE(bufferdata) > 0) {
                return NPY_SUCCEED;
            }

            /* Copy any data from the buffers back to the arrays */
            npyiter_copy_from_buffers(iter);
        }
    }

    npyiter_goto_iterindex(iter, NIT_ITERSTART(iter));

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* Prepare the next buffers and set iterend/size */
        npyiter_copy_to_buffers(iter, NULL);
    }

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Resets the iterator to its initial state, with new base data pointers.
 * This function requires great caution.
 *
 * If errmsg is non-NULL, it should point to a variable which will
 * receive the error message, and no Python exception will be set.
 * This is so that the function can be called from code not holding
 * the GIL.
 */
NPY_NO_EXPORT int
NpyIter_ResetBasePointers(NpyIter *iter, char **baseptrs, char **errmsg)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int iop, nop = NIT_NOP(iter);

    char **resetdataptr = NIT_RESETDATAPTR(iter);
    npy_intp *baseoffsets = NIT_BASEOFFSETS(iter);

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* If buffer allocation was delayed, do it now */
        if (itflags&NPY_ITFLAG_DELAYBUF) {
            if (!npyiter_allocate_buffers(iter, errmsg)) {
                return NPY_FAIL;
            }
            NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_DELAYBUF;
        }
        else {
            /* Copy any data from the buffers back to the arrays */
            npyiter_copy_from_buffers(iter);
        }
    }

    /* The new data pointers for resetting */
    for (iop = 0; iop < nop; ++iop) {
        resetdataptr[iop] = baseptrs[iop] + baseoffsets[iop];
    }

    npyiter_goto_iterindex(iter, NIT_ITERSTART(iter));

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* Prepare the next buffers and set iterend/size */
        npyiter_copy_to_buffers(iter, NULL);
    }

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Resets the iterator to a new iterator index range
 *
 * If errmsg is non-NULL, it should point to a variable which will
 * receive the error message, and no Python exception will be set.
 * This is so that the function can be called from code not holding
 * the GIL.
 */
NPY_NO_EXPORT int
NpyIter_ResetToIterIndexRange(NpyIter *iter,
                              npy_intp istart, npy_intp iend, char **errmsg)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    /*int nop = NIT_NOP(iter);*/

    if (!(itflags&NPY_ITFLAG_RANGE)) {
        if (errmsg == NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Cannot call ResetToIterIndexRange on an iterator without "
                    "requesting ranged iteration support in the constructor");
        }
        else {
            *errmsg = "Cannot call ResetToIterIndexRange on an iterator "
                      "without requesting ranged iteration support in the "
                    "constructor";
        }
        return NPY_FAIL;
    }

    if (istart < 0 || iend > NIT_ITERSIZE(iter)) {
        if (errmsg == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "Out-of-bounds range [%d, %d) passed to "
                    "ResetToIterIndexRange", (int)istart, (int)iend);
        }
        else {
            *errmsg = "Out-of-bounds range passed to ResetToIterIndexRange";
        }
        return NPY_FAIL;
    }
    else if (iend < istart) {
        if (errmsg == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "Invalid range [%d, %d) passed to ResetToIterIndexRange",
                    (int)istart, (int)iend);
        }
        else {
            *errmsg = "Invalid range passed to ResetToIterIndexRange";
        }
        return NPY_FAIL;
    }

    NIT_ITERSTART(iter) = istart;
    NIT_ITEREND(iter) = iend;

    return NpyIter_Reset(iter, errmsg);
}

/*NUMPY_API
 * Sets the iterator to the specified multi-index, which must have the
 * correct number of entries for 'ndim'.  It is only valid
 * when NPY_ITER_MULTI_INDEX was passed to the constructor.  This operation
 * fails if the multi-index is out of bounds.
 *
 * Returns NPY_SUCCEED on success, NPY_FAIL on failure.
 */
NPY_NO_EXPORT int
NpyIter_GotoMultiIndex(NpyIter *iter, npy_intp *multi_index)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp iterindex, factor;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    npy_int8 *perm;

    if (!(itflags&NPY_ITFLAG_HASMULTIINDEX)) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoMultiIndex on an iterator without "
                "requesting a multi-index in the constructor");
        return NPY_FAIL;
    }

    if (itflags&NPY_ITFLAG_BUFFER) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoMultiIndex on an iterator which "
                "is buffered");
        return NPY_FAIL;
    }

    if (itflags&NPY_ITFLAG_EXLOOP) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoMultiIndex on an iterator which "
                "has the flag EXTERNAL_LOOP");
        return NPY_FAIL;
    }

    perm = NIT_PERM(iter);
    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /* Compute the iterindex corresponding to the multi-index */
    iterindex = 0;
    factor = 1;
    for (idim = 0; idim < ndim; ++idim) {
        npy_int8 p = perm[idim];
        npy_intp i, shape;

        shape = NAD_SHAPE(axisdata);
        if (p < 0) {
            /* If the perm entry is negative, reverse the index */
            i = shape - multi_index[ndim+p] - 1;
        }
        else {
            i = multi_index[ndim-p-1];
        }

        /* Bounds-check this index */
        if (i >= 0 && i < shape) {
            iterindex += factor * i;
            factor *= shape;
        }
        else {
            PyErr_SetString(PyExc_IndexError,
                    "Iterator GotoMultiIndex called with an out-of-bounds "
                    "multi-index");
            return NPY_FAIL;
        }

        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    if (iterindex < NIT_ITERSTART(iter) || iterindex >= NIT_ITEREND(iter)) {
        PyErr_SetString(PyExc_IndexError,
                "Iterator GotoMultiIndex called with a multi-index outside the "
                "restricted iteration range");
        return NPY_FAIL;
    }

    npyiter_goto_iterindex(iter, iterindex);

    return NPY_SUCCEED;
}

/*NUMPY_API
 * If the iterator is tracking an index, sets the iterator
 * to the specified index.
 *
 * Returns NPY_SUCCEED on success, NPY_FAIL on failure.
 */
NPY_NO_EXPORT int
NpyIter_GotoIndex(NpyIter *iter, npy_intp flat_index)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp iterindex, factor;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    if (!(itflags&NPY_ITFLAG_HASINDEX)) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoIndex on an iterator without "
                "requesting a C or Fortran index in the constructor");
        return NPY_FAIL;
    }

    if (itflags&NPY_ITFLAG_BUFFER) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoIndex on an iterator which "
                "is buffered");
        return NPY_FAIL;
    }

    if (itflags&NPY_ITFLAG_EXLOOP) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoIndex on an iterator which "
                "has the flag EXTERNAL_LOOP");
        return NPY_FAIL;
    }

    if (flat_index < 0 || flat_index >= NIT_ITERSIZE(iter)) {
        PyErr_SetString(PyExc_IndexError,
                "Iterator GotoIndex called with an out-of-bounds "
                "index");
        return NPY_FAIL;
    }

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /* Compute the iterindex corresponding to the flat_index */
    iterindex = 0;
    factor = 1;
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp i, shape, iterstride;

        iterstride = NAD_STRIDES(axisdata)[nop];
        shape = NAD_SHAPE(axisdata);

        /* Extract the index from the flat_index */
        if (iterstride == 0) {
            i = 0;
        }
        else if (iterstride < 0) {
            i = shape - (flat_index/(-iterstride))%shape - 1;
        }
        else {
            i = (flat_index/iterstride)%shape;
        }

        /* Add its contribution to iterindex */
        iterindex += factor * i;
        factor *= shape;

        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }


    if (iterindex < NIT_ITERSTART(iter) || iterindex >= NIT_ITEREND(iter)) {
        PyErr_SetString(PyExc_IndexError,
                "Iterator GotoIndex called with an index outside the "
                "restricted iteration range.");
        return NPY_FAIL;
    }

    npyiter_goto_iterindex(iter, iterindex);

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Sets the iterator position to the specified iterindex,
 * which matches the iteration order of the iterator.
 *
 * Returns NPY_SUCCEED on success, NPY_FAIL on failure.
 */
NPY_NO_EXPORT int
NpyIter_GotoIterIndex(NpyIter *iter, npy_intp iterindex)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int iop, nop = NIT_NOP(iter);

    if (itflags&NPY_ITFLAG_EXLOOP) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoIterIndex on an iterator which "
                "has the flag EXTERNAL_LOOP");
        return NPY_FAIL;
    }

    if (iterindex < NIT_ITERSTART(iter) || iterindex >= NIT_ITEREND(iter)) {
        PyErr_SetString(PyExc_IndexError,
                "Iterator GotoIterIndex called with an iterindex outside the "
                "iteration range.");
        return NPY_FAIL;
    }

    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        npy_intp bufiterend, size;

        size = NBF_SIZE(bufferdata);
        bufiterend = NBF_BUFITEREND(bufferdata);
        /* Check if the new iterindex is already within the buffer */
        if (!(itflags&NPY_ITFLAG_REDUCE) && iterindex < bufiterend &&
                                        iterindex >= bufiterend - size) {
            npy_intp *strides, delta;
            char **ptrs;

            strides = NBF_STRIDES(bufferdata);
            ptrs = NBF_PTRS(bufferdata);
            delta = iterindex - NIT_ITERINDEX(iter);

            for (iop = 0; iop < nop; ++iop) {
                ptrs[iop] += delta * strides[iop];
            }

            NIT_ITERINDEX(iter) = iterindex;
        }
        /* Start the buffer at the provided iterindex */
        else {
            /* Write back to the arrays */
            npyiter_copy_from_buffers(iter);

            npyiter_goto_iterindex(iter, iterindex);

            /* Prepare the next buffers and set iterend/size */
            npyiter_copy_to_buffers(iter, NULL);
        }
    }
    else {
        npyiter_goto_iterindex(iter, iterindex);
    }

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Gets the current iteration index
 */
NPY_NO_EXPORT npy_intp
NpyIter_GetIterIndex(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    /* iterindex is only used if NPY_ITER_RANGED or NPY_ITER_BUFFERED was set */
    if (itflags&(NPY_ITFLAG_RANGE|NPY_ITFLAG_BUFFER)) {
        return NIT_ITERINDEX(iter);
    }
    else {
        npy_intp iterindex;
        NpyIter_AxisData *axisdata;
        npy_intp sizeof_axisdata;

        iterindex = 0;
        if (ndim == 0) {
            return 0;
        }
        sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
        axisdata = NIT_INDEX_AXISDATA(NIT_AXISDATA(iter), ndim-1);

        for (idim = ndim-2; idim >= 0; --idim) {
            iterindex += NAD_INDEX(axisdata);
            NIT_ADVANCE_AXISDATA(axisdata, -1);
            iterindex *= NAD_SHAPE(axisdata);
        }
        iterindex += NAD_INDEX(axisdata);

        return iterindex;
    }
}

/*NUMPY_API
 * Whether the buffer allocation is being delayed
 */
NPY_NO_EXPORT npy_bool
NpyIter_HasDelayedBufAlloc(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_DELAYBUF) != 0;
}

/*NUMPY_API
 * Whether the iterator handles the inner loop
 */
NPY_NO_EXPORT npy_bool
NpyIter_HasExternalLoop(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_EXLOOP) != 0;
}

/*NUMPY_API
 * Whether the iterator is tracking a multi-index
 */
NPY_NO_EXPORT npy_bool
NpyIter_HasMultiIndex(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_HASMULTIINDEX) != 0;
}

/*NUMPY_API
 * Whether the iterator is tracking an index
 */
NPY_NO_EXPORT npy_bool
NpyIter_HasIndex(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_HASINDEX) != 0;
}

/*NUMPY_API
 * Checks to see whether this is the first time the elements
 * of the specified reduction operand which the iterator points at are
 * being seen for the first time. The function returns
 * a reasonable answer for reduction operands and when buffering is
 * disabled. The answer may be incorrect for buffered non-reduction
 * operands.
 *
 * This function is intended to be used in EXTERNAL_LOOP mode only,
 * and will produce some wrong answers when that mode is not enabled.
 *
 * If this function returns true, the caller should also
 * check the inner loop stride of the operand, because if
 * that stride is 0, then only the first element of the innermost
 * external loop is being visited for the first time.
 *
 * WARNING: For performance reasons, 'iop' is not bounds-checked,
 *          it is not confirmed that 'iop' is actually a reduction
 *          operand, and it is not confirmed that EXTERNAL_LOOP
 *          mode is enabled. These checks are the responsibility of
 *          the caller, and should be done outside of any inner loops.
 */
NPY_NO_EXPORT npy_bool
NpyIter_IsFirstVisit(NpyIter *iter, int iop)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    axisdata = NIT_AXISDATA(iter);

    for (idim = 0; idim < ndim; ++idim) {
        npy_intp coord = NAD_INDEX(axisdata);
        npy_intp stride = NAD_STRIDES(axisdata)[iop];

        /*
         * If this is a reduction dimension and the coordinate
         * is not at the start, it's definitely not the first visit
         */
        if (stride == 0 && coord != 0) {
            return 0;
        }

        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    /*
     * In reduction buffering mode, there's a double loop being
     * tracked in the buffer part of the iterator data structure.
     * We only need to check the outer level of this two-level loop,
     * because of the requirement that EXTERNAL_LOOP be enabled.
     */
    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        /* The outer reduce loop */
        if (NBF_REDUCE_OUTERSTRIDES(bufferdata)[iop] == 0 &&
                NBF_REDUCE_POS(bufferdata) != 0) {
            return 0;
        }
    }

    return 1;
}

/*NUMPY_API
 * Whether the iteration could be done with no buffering.
 */
NPY_NO_EXPORT npy_bool
NpyIter_RequiresBuffering(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int iop, nop = NIT_NOP(iter);

    npyiter_opitflags *op_itflags;

    if (!(itflags&NPY_ITFLAG_BUFFER)) {
        return 0;
    }

    op_itflags = NIT_OPITFLAGS(iter);

    /* If any operand requires a cast, buffering is mandatory */
    for (iop = 0; iop < nop; ++iop) {
        if (op_itflags[iop]&NPY_OP_ITFLAG_CAST) {
            return 1;
        }
    }

    return 0;
}

/*NUMPY_API
 * Whether the iteration loop, and in particular the iternext()
 * function, needs API access.  If this is true, the GIL must
 * be retained while iterating.
 */
NPY_NO_EXPORT npy_bool
NpyIter_IterationNeedsAPI(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_NEEDSAPI) != 0;
}

/*NUMPY_API
 * Gets the number of dimensions being iterated
 */
NPY_NO_EXPORT int
NpyIter_GetNDim(NpyIter *iter)
{
    return NIT_NDIM(iter);
}

/*NUMPY_API
 * Gets the number of operands being iterated
 */
NPY_NO_EXPORT int
NpyIter_GetNOp(NpyIter *iter)
{
    return NIT_NOP(iter);
}

/*NUMPY_API
 * Gets the number of elements being iterated
 */
NPY_NO_EXPORT npy_intp
NpyIter_GetIterSize(NpyIter *iter)
{
    return NIT_ITERSIZE(iter);
}

/*NUMPY_API
 * Whether the iterator is buffered
 */
NPY_NO_EXPORT npy_bool
NpyIter_IsBuffered(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_BUFFER) != 0;
}

/*NUMPY_API
 * Whether the inner loop can grow if buffering is unneeded
 */
NPY_NO_EXPORT npy_bool
NpyIter_IsGrowInner(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_GROWINNER) != 0;
}

/*NUMPY_API
 * Gets the size of the buffer, or 0 if buffering is not enabled
 */
NPY_NO_EXPORT npy_intp
NpyIter_GetBufferSize(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        return NBF_BUFFERSIZE(bufferdata);
    }
    else {
        return 0;
    }

}

/*NUMPY_API
 * Gets the range of iteration indices being iterated
 */
NPY_NO_EXPORT void
NpyIter_GetIterIndexRange(NpyIter *iter,
                          npy_intp *istart, npy_intp *iend)
{
    *istart = NIT_ITERSTART(iter);
    *iend = NIT_ITEREND(iter);
}

/*NUMPY_API
 * Gets the broadcast shape if a multi-index is being tracked by the iterator,
 * otherwise gets the shape of the iteration as Fortran-order
 * (fastest-changing index first).
 *
 * The reason Fortran-order is returned when a multi-index
 * is not enabled is that this is providing a direct view into how
 * the iterator traverses the n-dimensional space. The iterator organizes
 * its memory from fastest index to slowest index, and when
 * a multi-index is enabled, it uses a permutation to recover the original
 * order.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
NpyIter_GetShape(NpyIter *iter, npy_intp *outshape)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    int idim, sizeof_axisdata;
    NpyIter_AxisData *axisdata;
    npy_int8 *perm;

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    if (itflags&NPY_ITFLAG_HASMULTIINDEX) {
        perm = NIT_PERM(iter);
        for(idim = 0; idim < ndim; ++idim) {
            npy_int8 p = perm[idim];
            if (p < 0) {
                outshape[ndim+p] = NAD_SHAPE(axisdata);
            }
            else {
                outshape[ndim-p-1] = NAD_SHAPE(axisdata);
            }

            NIT_ADVANCE_AXISDATA(axisdata, 1);
        }
    }
    else {
        for(idim = 0; idim < ndim; ++idim) {
            outshape[idim] = NAD_SHAPE(axisdata);
            NIT_ADVANCE_AXISDATA(axisdata, 1);
        }
    }

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Builds a set of strides which are the same as the strides of an
 * output array created using the NPY_ITER_ALLOCATE flag, where NULL
 * was passed for op_axes.  This is for data packed contiguously,
 * but not necessarily in C or Fortran order. This should be used
 * together with NpyIter_GetShape and NpyIter_GetNDim.
 *
 * A use case for this function is to match the shape and layout of
 * the iterator and tack on one or more dimensions.  For example,
 * in order to generate a vector per input value for a numerical gradient,
 * you pass in ndim*itemsize for itemsize, then add another dimension to
 * the end with size ndim and stride itemsize.  To do the Hessian matrix,
 * you do the same thing but add two dimensions, or take advantage of
 * the symmetry and pack it into 1 dimension with a particular encoding.
 *
 * This function may only be called if the iterator is tracking a multi-index
 * and if NPY_ITER_DONT_NEGATE_STRIDES was used to prevent an axis from
 * being iterated in reverse order.
 *
 * If an array is created with this method, simply adding 'itemsize'
 * for each iteration will traverse the new array matching the
 * iterator.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
NpyIter_CreateCompatibleStrides(NpyIter *iter,
                            npy_intp itemsize, npy_intp *outstrides)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp sizeof_axisdata;
    NpyIter_AxisData *axisdata;
    npy_int8 *perm;

    if (!(itflags&NPY_ITFLAG_HASMULTIINDEX)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Iterator CreateCompatibleStrides may only be called "
                "if a multi-index is being tracked");
        return NPY_FAIL;
    }

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    perm = NIT_PERM(iter);
    for(idim = 0; idim < ndim; ++idim) {
        npy_int8 p = perm[idim];
        if (p < 0) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Iterator CreateCompatibleStrides may only be called "
                    "if DONT_NEGATE_STRIDES was used to prevent reverse "
                    "iteration of an axis");
            return NPY_FAIL;
        }
        else {
            outstrides[ndim-p-1] = itemsize;
        }

        itemsize *= NAD_SHAPE(axisdata);
        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Get the array of data pointers (1 per object being iterated)
 *
 * This function may be safely called without holding the Python GIL.
 */
NPY_NO_EXPORT char **
NpyIter_GetDataPtrArray(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        return NBF_PTRS(bufferdata);
    }
    else {
        NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
        return NAD_PTRS(axisdata);
    }
}

/*NUMPY_API
 * Get the array of data pointers (1 per object being iterated),
 * directly into the arrays (never pointing to a buffer), for starting
 * unbuffered iteration. This always returns the addresses for the
 * iterator position as reset to iterator index 0.
 *
 * These pointers are different from the pointers accepted by
 * NpyIter_ResetBasePointers, because the direction along some
 * axes may have been reversed, requiring base offsets.
 *
 * This function may be safely called without holding the Python GIL.
 */
NPY_NO_EXPORT char **
NpyIter_GetInitialDataPtrArray(NpyIter *iter)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    return NIT_RESETDATAPTR(iter);
}

/*NUMPY_API
 * Get the array of data type pointers (1 per object being iterated)
 */
NPY_NO_EXPORT PyArray_Descr **
NpyIter_GetDescrArray(NpyIter *iter)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /*int ndim = NIT_NDIM(iter);*/
    /*int nop = NIT_NOP(iter);*/

    return NIT_DTYPES(iter);
}

/*NUMPY_API
 * Get the array of objects being iterated
 */
NPY_NO_EXPORT PyArrayObject **
NpyIter_GetOperandArray(NpyIter *iter)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    return NIT_OPERANDS(iter);
}

/*NUMPY_API
 * Returns a view to the i-th object with the iterator's internal axes
 */
NPY_NO_EXPORT PyArrayObject *
NpyIter_GetIterView(NpyIter *iter, npy_intp i)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    PyArrayObject *obj, *view;
    PyArray_Descr *dtype;
    char *dataptr;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    int writeable;

    if (i < 0) {
        PyErr_SetString(PyExc_IndexError,
                "index provided for an iterator view was out of bounds");
        return NULL;
    }

    /* Don't provide views if buffering is enabled */
    if (itflags&NPY_ITFLAG_BUFFER) {
        PyErr_SetString(PyExc_ValueError,
                "cannot provide an iterator view when buffering is enabled");
        return NULL;
    }

    obj = NIT_OPERANDS(iter)[i];
    dtype = PyArray_DESCR(obj);
    writeable = NIT_OPITFLAGS(iter)[i]&NPY_OP_ITFLAG_WRITE;
    dataptr = NIT_RESETDATAPTR(iter)[i];
    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /* Retrieve the shape and strides from the axisdata */
    for (idim = 0; idim < ndim; ++idim) {
        shape[ndim-idim-1] = NAD_SHAPE(axisdata);
        strides[ndim-idim-1] = NAD_STRIDES(axisdata)[i];

        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    Py_INCREF(dtype);
    view = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, ndim,
                                shape, strides, dataptr,
                                writeable ? NPY_ARRAY_WRITEABLE : 0,
                                NULL);
    if (view == NULL) {
        return NULL;
    }
    /* Tell the view who owns the data */
    Py_INCREF(obj);
    if (PyArray_SetBaseObject(view, (PyObject *)obj) < 0) {
        Py_DECREF(view);
        return NULL;
    }
    /* Make sure all the flags are good */
    PyArray_UpdateFlags(view, NPY_ARRAY_UPDATE_ALL);

    return view;
}

/*NUMPY_API
 * Get a pointer to the index, if it is being tracked
 */
NPY_NO_EXPORT npy_intp *
NpyIter_GetIndexPtr(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);

    if (itflags&NPY_ITFLAG_HASINDEX) {
        /* The index is just after the data pointers */
        return (npy_intp*)NAD_PTRS(axisdata) + nop;
    }
    else {
        return NULL;
    }
}

/*NUMPY_API
 * Gets an array of read flags (1 per object being iterated)
 */
NPY_NO_EXPORT void
NpyIter_GetReadFlags(NpyIter *iter, char *outreadflags)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /*int ndim = NIT_NDIM(iter);*/
    int iop, nop = NIT_NOP(iter);

    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);

    for (iop = 0; iop < nop; ++iop) {
        outreadflags[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_READ) != 0;
    }
}

/*NUMPY_API
 * Gets an array of write flags (1 per object being iterated)
 */
NPY_NO_EXPORT void
NpyIter_GetWriteFlags(NpyIter *iter, char *outwriteflags)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /*int ndim = NIT_NDIM(iter);*/
    int iop, nop = NIT_NOP(iter);

    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);

    for (iop = 0; iop < nop; ++iop) {
        outwriteflags[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_WRITE) != 0;
    }
}


/*NUMPY_API
 * Get the array of strides for the inner loop (when HasExternalLoop is true)
 *
 * This function may be safely called without holding the Python GIL.
 */
NPY_NO_EXPORT npy_intp *
NpyIter_GetInnerStrideArray(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *data = NIT_BUFFERDATA(iter);
        return NBF_STRIDES(data);
    }
    else {
        NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
        return NAD_STRIDES(axisdata);
    }
}

/*NUMPY_API
 * Gets the array of strides for the specified axis.
 * If the iterator is tracking a multi-index, gets the strides
 * for the axis specified, otherwise gets the strides for
 * the iteration axis as Fortran order (fastest-changing axis first).
 *
 * Returns NULL if an error occurs.
 */
NPY_NO_EXPORT npy_intp *
NpyIter_GetAxisStrideArray(NpyIter *iter, int axis)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_int8 *perm = NIT_PERM(iter);
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    if (axis < 0 || axis >= ndim) {
        PyErr_SetString(PyExc_ValueError,
                "axis out of bounds in iterator GetStrideAxisArray");
        return NULL;
    }

    if (itflags&NPY_ITFLAG_HASMULTIINDEX) {
        /* Reverse axis, since the iterator treats them that way */
        axis = ndim-1-axis;

        /* First find the axis in question */
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            if (perm[idim] == axis || -1 - perm[idim] == axis) {
                return NAD_STRIDES(axisdata);
            }
        }
    }
    else {
        return NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, axis));
    }

    PyErr_SetString(PyExc_RuntimeError,
            "internal error in iterator perm");
    return  NULL;
}

/*NUMPY_API
 * Get an array of strides which are fixed.  Any strides which may
 * change during iteration receive the value NPY_MAX_INTP.  Once
 * the iterator is ready to iterate, call this to get the strides
 * which will always be fixed in the inner loop, then choose optimized
 * inner loop functions which take advantage of those fixed strides.
 *
 * This function may be safely called without holding the Python GIL.
 */
NPY_NO_EXPORT void
NpyIter_GetInnerFixedStrideArray(NpyIter *iter, npy_intp *out_strides)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    NpyIter_AxisData *axisdata0 = NIT_AXISDATA(iter);
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *data = NIT_BUFFERDATA(iter);
        npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
        npy_intp stride, *strides = NBF_STRIDES(data),
                *ad_strides = NAD_STRIDES(axisdata0);
        PyArray_Descr **dtypes = NIT_DTYPES(iter);

        for (iop = 0; iop < nop; ++iop) {
            stride = strides[iop];
            /*
             * Operands which are always/never buffered have fixed strides,
             * and everything has fixed strides when ndim is 0 or 1
             */
            if (ndim <= 1 || (op_itflags[iop]&
                            (NPY_OP_ITFLAG_CAST|NPY_OP_ITFLAG_BUFNEVER))) {
                out_strides[iop] = stride;
            }
            /* If it's a reduction, 0-stride inner loop may have fixed stride */
            else if (stride == 0 && (itflags&NPY_ITFLAG_REDUCE)) {
                /* If it's a reduction operand, definitely fixed stride */
                if (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE) {
                    out_strides[iop] = stride;
                }
                /*
                 * Otherwise it's a fixed stride if the stride is 0
                 * for all inner dimensions of the reduction double loop
                 */
                else {
                    NpyIter_AxisData *axisdata = axisdata0;
                    int idim,
                            reduce_outerdim = NBF_REDUCE_OUTERDIM(data);
                    for (idim = 0; idim < reduce_outerdim; ++idim) {
                        if (NAD_STRIDES(axisdata)[iop] != 0) {
                            break;
                        }
                        NIT_ADVANCE_AXISDATA(axisdata, 1);
                    }
                    /* If all the strides were 0, the stride won't change */
                    if (idim == reduce_outerdim) {
                        out_strides[iop] = stride;
                    }
                    else {
                        out_strides[iop] = NPY_MAX_INTP;
                    }
                }
            }
            /*
             * Inner loop contiguous array means its stride won't change when
             * switching between buffering and not buffering
             */
            else if (ad_strides[iop] == dtypes[iop]->elsize) {
                out_strides[iop] = ad_strides[iop];
            }
            /*
             * Otherwise the strides can change if the operand is sometimes
             * buffered, sometimes not.
             */
            else {
                out_strides[iop] = NPY_MAX_INTP;
            }
        }
    }
    else {
        /* If there's no buffering, the strides are always fixed */
        memcpy(out_strides, NAD_STRIDES(axisdata0), nop*NPY_SIZEOF_INTP);
    }
}

/*NUMPY_API
 * Get a pointer to the size of the inner loop  (when HasExternalLoop is true)
 *
 * This function may be safely called without holding the Python GIL.
 */
NPY_NO_EXPORT npy_intp *
NpyIter_GetInnerLoopSizePtr(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *data = NIT_BUFFERDATA(iter);
        return &NBF_SIZE(data);
    }
    else {
        NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
        return &NAD_SHAPE(axisdata);
    }
}

/*NUMPY_API
 * For debugging
 */
NPY_NO_EXPORT void
NpyIter_DebugPrint(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    PyGILState_STATE gilstate = PyGILState_Ensure();

    printf("\n------ BEGIN ITERATOR DUMP ------\n");
    printf("| Iterator Address: %p\n", (void *)iter);
    printf("| ItFlags: ");
    if (itflags&NPY_ITFLAG_IDENTPERM)
        printf("IDENTPERM ");
    if (itflags&NPY_ITFLAG_NEGPERM)
        printf("NEGPERM ");
    if (itflags&NPY_ITFLAG_HASINDEX)
        printf("HASINDEX ");
    if (itflags&NPY_ITFLAG_HASMULTIINDEX)
        printf("HASMULTIINDEX ");
    if (itflags&NPY_ITFLAG_FORCEDORDER)
        printf("FORCEDORDER ");
    if (itflags&NPY_ITFLAG_EXLOOP)
        printf("EXLOOP ");
    if (itflags&NPY_ITFLAG_RANGE)
        printf("RANGE ");
    if (itflags&NPY_ITFLAG_BUFFER)
        printf("BUFFER ");
    if (itflags&NPY_ITFLAG_GROWINNER)
        printf("GROWINNER ");
    if (itflags&NPY_ITFLAG_ONEITERATION)
        printf("ONEITERATION ");
    if (itflags&NPY_ITFLAG_DELAYBUF)
        printf("DELAYBUF ");
    if (itflags&NPY_ITFLAG_NEEDSAPI)
        printf("NEEDSAPI ");
    if (itflags&NPY_ITFLAG_REDUCE)
        printf("REDUCE ");
    if (itflags&NPY_ITFLAG_REUSE_REDUCE_LOOPS)
        printf("REUSE_REDUCE_LOOPS ");

    printf("\n");
    printf("| NDim: %d\n", (int)ndim);
    printf("| NOp: %d\n", (int)nop);
    if (NIT_MASKOP(iter) >= 0) {
        printf("| MaskOp: %d\n", (int)NIT_MASKOP(iter));
    }
    printf("| IterSize: %d\n", (int)NIT_ITERSIZE(iter));
    printf("| IterStart: %d\n", (int)NIT_ITERSTART(iter));
    printf("| IterEnd: %d\n", (int)NIT_ITEREND(iter));
    printf("| IterIndex: %d\n", (int)NIT_ITERINDEX(iter));
    printf("| Iterator SizeOf: %d\n",
                            (int)NIT_SIZEOF_ITERATOR(itflags, ndim, nop));
    printf("| BufferData SizeOf: %d\n",
                            (int)NIT_BUFFERDATA_SIZEOF(itflags, ndim, nop));
    printf("| AxisData SizeOf: %d\n",
                            (int)NIT_AXISDATA_SIZEOF(itflags, ndim, nop));
    printf("|\n");

    printf("| Perm: ");
    for (idim = 0; idim < ndim; ++idim) {
        printf("%d ", (int)NIT_PERM(iter)[idim]);
    }
    printf("\n");
    printf("| DTypes: ");
    for (iop = 0; iop < nop; ++iop) {
        printf("%p ", (void *)NIT_DTYPES(iter)[iop]);
    }
    printf("\n");
    printf("| DTypes: ");
    for (iop = 0; iop < nop; ++iop) {
        if (NIT_DTYPES(iter)[iop] != NULL)
            PyObject_Print((PyObject*)NIT_DTYPES(iter)[iop], stdout, 0);
        else
            printf("(nil) ");
        printf(" ");
    }
    printf("\n");
    printf("| InitDataPtrs: ");
    for (iop = 0; iop < nop; ++iop) {
        printf("%p ", (void *)NIT_RESETDATAPTR(iter)[iop]);
    }
    printf("\n");
    printf("| BaseOffsets: ");
    for (iop = 0; iop < nop; ++iop) {
        printf("%i ", (int)NIT_BASEOFFSETS(iter)[iop]);
    }
    printf("\n");
    if (itflags&NPY_ITFLAG_HASINDEX) {
        printf("| InitIndex: %d\n",
                        (int)(npy_intp)NIT_RESETDATAPTR(iter)[nop]);
    }
    printf("| Operands: ");
    for (iop = 0; iop < nop; ++iop) {
        printf("%p ", (void *)NIT_OPERANDS(iter)[iop]);
    }
    printf("\n");
    printf("| Operand DTypes: ");
    for (iop = 0; iop < nop; ++iop) {
        PyArray_Descr *dtype;
        if (NIT_OPERANDS(iter)[iop] != NULL) {
            dtype = PyArray_DESCR(NIT_OPERANDS(iter)[iop]);
            if (dtype != NULL)
                PyObject_Print((PyObject *)dtype, stdout, 0);
            else
                printf("(nil) ");
        }
        else {
            printf("(op nil) ");
        }
        printf(" ");
    }
    printf("\n");
    printf("| OpItFlags:\n");
    for (iop = 0; iop < nop; ++iop) {
        printf("|   Flags[%d]: ", (int)iop);
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_READ)
            printf("READ ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_WRITE)
            printf("WRITE ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_CAST)
            printf("CAST ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_BUFNEVER)
            printf("BUFNEVER ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_ALIGNED)
            printf("ALIGNED ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_REDUCE)
            printf("REDUCE ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_VIRTUAL)
            printf("VIRTUAL ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_WRITEMASKED)
            printf("WRITEMASKED ");
        printf("\n");
    }
    printf("|\n");

    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        printf("| BufferData:\n");
        printf("|   BufferSize: %d\n", (int)NBF_BUFFERSIZE(bufferdata));
        printf("|   Size: %d\n", (int)NBF_SIZE(bufferdata));
        printf("|   BufIterEnd: %d\n", (int)NBF_BUFITEREND(bufferdata));
        if (itflags&NPY_ITFLAG_REDUCE) {
            printf("|   REDUCE Pos: %d\n",
                        (int)NBF_REDUCE_POS(bufferdata));
            printf("|   REDUCE OuterSize: %d\n",
                        (int)NBF_REDUCE_OUTERSIZE(bufferdata));
            printf("|   REDUCE OuterDim: %d\n",
                        (int)NBF_REDUCE_OUTERDIM(bufferdata));
        }
        printf("|   Strides: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%d ", (int)NBF_STRIDES(bufferdata)[iop]);
        printf("\n");
        /* Print the fixed strides when there's no inner loop */
        if (itflags&NPY_ITFLAG_EXLOOP) {
            npy_intp fixedstrides[NPY_MAXDIMS];
            printf("|   Fixed Strides: ");
            NpyIter_GetInnerFixedStrideArray(iter, fixedstrides);
            for (iop = 0; iop < nop; ++iop)
                printf("%d ", (int)fixedstrides[iop]);
            printf("\n");
        }
        printf("|   Ptrs: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)NBF_PTRS(bufferdata)[iop]);
        printf("\n");
        if (itflags&NPY_ITFLAG_REDUCE) {
            printf("|   REDUCE Outer Strides: ");
            for (iop = 0; iop < nop; ++iop)
                printf("%d ", (int)NBF_REDUCE_OUTERSTRIDES(bufferdata)[iop]);
            printf("\n");
            printf("|   REDUCE Outer Ptrs: ");
            for (iop = 0; iop < nop; ++iop)
                printf("%p ", (void *)NBF_REDUCE_OUTERPTRS(bufferdata)[iop]);
            printf("\n");
        }
        printf("|   ReadTransferFn: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)NBF_READTRANSFERFN(bufferdata)[iop]);
        printf("\n");
        printf("|   ReadTransferData: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)NBF_READTRANSFERDATA(bufferdata)[iop]);
        printf("\n");
        printf("|   WriteTransferFn: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)NBF_WRITETRANSFERFN(bufferdata)[iop]);
        printf("\n");
        printf("|   WriteTransferData: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)NBF_WRITETRANSFERDATA(bufferdata)[iop]);
        printf("\n");
        printf("|   Buffers: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)NBF_BUFFERS(bufferdata)[iop]);
        printf("\n");
        printf("|\n");
    }

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        printf("| AxisData[%d]:\n", (int)idim);
        printf("|   Shape: %d\n", (int)NAD_SHAPE(axisdata));
        printf("|   Index: %d\n", (int)NAD_INDEX(axisdata));
        printf("|   Strides: ");
        for (iop = 0; iop < nop; ++iop) {
            printf("%d ", (int)NAD_STRIDES(axisdata)[iop]);
        }
        printf("\n");
        if (itflags&NPY_ITFLAG_HASINDEX) {
            printf("|   Index Stride: %d\n", (int)NAD_STRIDES(axisdata)[nop]);
        }
        printf("|   Ptrs: ");
        for (iop = 0; iop < nop; ++iop) {
            printf("%p ", (void *)NAD_PTRS(axisdata)[iop]);
        }
        printf("\n");
        if (itflags&NPY_ITFLAG_HASINDEX) {
            printf("|   Index Value: %d\n",
                               (int)((npy_intp*)NAD_PTRS(axisdata))[nop]);
        }
    }

    printf("------- END ITERATOR DUMP -------\n");
    fflush(stdout);

    PyGILState_Release(gilstate);
}

NPY_NO_EXPORT void
npyiter_coalesce_axes(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp istrides, nstrides = NAD_NSTRIDES();
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    NpyIter_AxisData *ad_compress;
    npy_intp new_ndim = 1;

    /* The HASMULTIINDEX or IDENTPERM flags do not apply after coalescing */
    NIT_ITFLAGS(iter) &= ~(NPY_ITFLAG_IDENTPERM|NPY_ITFLAG_HASMULTIINDEX);

    axisdata = NIT_AXISDATA(iter);
    ad_compress = axisdata;

    for (idim = 0; idim < ndim-1; ++idim) {
        int can_coalesce = 1;
        npy_intp shape0 = NAD_SHAPE(ad_compress);
        npy_intp shape1 = NAD_SHAPE(NIT_INDEX_AXISDATA(axisdata, 1));
        npy_intp *strides0 = NAD_STRIDES(ad_compress);
        npy_intp *strides1 = NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, 1));

        /* Check that all the axes can be coalesced */
        for (istrides = 0; istrides < nstrides; ++istrides) {
            if (!((shape0 == 1 && strides0[istrides] == 0) ||
                  (shape1 == 1 && strides1[istrides] == 0)) &&
                     (strides0[istrides]*shape0 != strides1[istrides])) {
                can_coalesce = 0;
                break;
            }
        }

        if (can_coalesce) {
            npy_intp *strides = NAD_STRIDES(ad_compress);

            NIT_ADVANCE_AXISDATA(axisdata, 1);
            NAD_SHAPE(ad_compress) *= NAD_SHAPE(axisdata);
            for (istrides = 0; istrides < nstrides; ++istrides) {
                if (strides[istrides] == 0) {
                    strides[istrides] = NAD_STRIDES(axisdata)[istrides];
                }
            }
        }
        else {
            NIT_ADVANCE_AXISDATA(axisdata, 1);
            NIT_ADVANCE_AXISDATA(ad_compress, 1);
            if (ad_compress != axisdata) {
                memcpy(ad_compress, axisdata, sizeof_axisdata);
            }
            ++new_ndim;
        }
    }

    /*
     * If the number of axes shrunk, reset the perm and
     * compress the data into the new layout.
     */
    if (new_ndim < ndim) {
        npy_int8 *perm = NIT_PERM(iter);

        /* Reset to an identity perm */
        for (idim = 0; idim < new_ndim; ++idim) {
            perm[idim] = (npy_int8)idim;
        }
        NIT_NDIM(iter) = new_ndim;
    }
}

/*
 * If errmsg is non-NULL, it should point to a variable which will
 * receive the error message, and no Python exception will be set.
 * This is so that the function can be called from code not holding
 * the GIL.
 */
NPY_NO_EXPORT int
npyiter_allocate_buffers(NpyIter *iter, char **errmsg)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /*int ndim = NIT_NDIM(iter);*/
    int iop = 0, nop = NIT_NOP(iter);

    npy_intp i;
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    PyArray_Descr **op_dtype = NIT_DTYPES(iter);
    npy_intp buffersize = NBF_BUFFERSIZE(bufferdata);
    char *buffer, **buffers = NBF_BUFFERS(bufferdata);

    for (iop = 0; iop < nop; ++iop) {
        npyiter_opitflags flags = op_itflags[iop];

        /*
         * If we have determined that a buffer may be needed,
         * allocate one.
         */
        if (!(flags&NPY_OP_ITFLAG_BUFNEVER)) {
            npy_intp itemsize = op_dtype[iop]->elsize;
            buffer = PyArray_malloc(itemsize*buffersize);
            if (buffer == NULL) {
                if (errmsg == NULL) {
                    PyErr_NoMemory();
                }
                else {
                    *errmsg = "out of memory";
                }
                goto fail;
            }
            buffers[iop] = buffer;
        }
    }

    return 1;

fail:
    for (i = 0; i < iop; ++i) {
        if (buffers[i] != NULL) {
            PyArray_free(buffers[i]);
            buffers[i] = NULL;
        }
    }
    return 0;
}

/*
 * This sets the AXISDATA portion of the iterator to the specified
 * iterindex, updating the pointers as well.  This function does
 * no error checking.
 */
NPY_NO_EXPORT void
npyiter_goto_iterindex(NpyIter *iter, npy_intp iterindex)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    char **dataptr;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    npy_intp istrides, nstrides, i, shape;

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    nstrides = NAD_NSTRIDES();

    NIT_ITERINDEX(iter) = iterindex;

    ndim = ndim ? ndim : 1;

    if (iterindex == 0) {
        dataptr = NIT_RESETDATAPTR(iter);

        for (idim = 0; idim < ndim; ++idim) {
            char **ptrs;
            NAD_INDEX(axisdata) = 0;
            ptrs = NAD_PTRS(axisdata);
            for (istrides = 0; istrides < nstrides; ++istrides) {
                ptrs[istrides] = dataptr[istrides];
            }

            NIT_ADVANCE_AXISDATA(axisdata, 1);
        }
    }
    else {
        /*
         * Set the multi-index, from the fastest-changing to the
         * slowest-changing.
         */
        axisdata = NIT_AXISDATA(iter);
        shape = NAD_SHAPE(axisdata);
        i = iterindex;
        iterindex /= shape;
        NAD_INDEX(axisdata) = i - iterindex * shape;
        for (idim = 0; idim < ndim-1; ++idim) {
            NIT_ADVANCE_AXISDATA(axisdata, 1);

            shape = NAD_SHAPE(axisdata);
            i = iterindex;
            iterindex /= shape;
            NAD_INDEX(axisdata) = i - iterindex * shape;
        }

        dataptr = NIT_RESETDATAPTR(iter);

        /*
         * Accumulate the successive pointers with their
         * offsets in the opposite order, starting from the
         * original data pointers.
         */
        for (idim = 0; idim < ndim; ++idim) {
            npy_intp *strides;
            char **ptrs;

            strides = NAD_STRIDES(axisdata);
            ptrs = NAD_PTRS(axisdata);

            i = NAD_INDEX(axisdata);

            for (istrides = 0; istrides < nstrides; ++istrides) {
                ptrs[istrides] = dataptr[istrides] + i*strides[istrides];
            }

            dataptr = ptrs;

            NIT_ADVANCE_AXISDATA(axisdata, -1);
        }
    }
}

/*
 * This gets called after the the buffers have been exhausted, and
 * their data needs to be written back to the arrays.  The multi-index
 * must be positioned for the beginning of the buffer.
 */
NPY_NO_EXPORT void
npyiter_copy_from_buffers(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);
    int maskop = NIT_MASKOP(iter);

    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter),
                    *reduce_outeraxisdata = NULL;

    PyArray_Descr **dtypes = NIT_DTYPES(iter);
    npy_intp transfersize = NBF_SIZE(bufferdata),
                buffersize = NBF_BUFFERSIZE(bufferdata);
    npy_intp *strides = NBF_STRIDES(bufferdata),
             *ad_strides = NAD_STRIDES(axisdata);
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    char **ptrs = NBF_PTRS(bufferdata), **ad_ptrs = NAD_PTRS(axisdata);
    char **buffers = NBF_BUFFERS(bufferdata);
    char *buffer;

    npy_intp reduce_outerdim = 0;
    npy_intp *reduce_outerstrides = NULL;

    PyArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;

    npy_intp axisdata_incr = NIT_AXISDATA_SIZEOF(itflags, ndim, nop) /
                                NPY_SIZEOF_INTP;

    /* If we're past the end, nothing to copy */
    if (NBF_SIZE(bufferdata) == 0) {
        return;
    }

    NPY_IT_DBG_PRINT("Iterator: Copying buffers to outputs\n");

    if (itflags&NPY_ITFLAG_REDUCE) {
        reduce_outerdim = NBF_REDUCE_OUTERDIM(bufferdata);
        reduce_outerstrides = NBF_REDUCE_OUTERSTRIDES(bufferdata);
        reduce_outeraxisdata = NIT_INDEX_AXISDATA(axisdata, reduce_outerdim);
        transfersize *= NBF_REDUCE_OUTERSIZE(bufferdata);
    }

    for (iop = 0; iop < nop; ++iop) {
        stransfer = NBF_WRITETRANSFERFN(bufferdata)[iop];
        transferdata = NBF_WRITETRANSFERDATA(bufferdata)[iop];
        buffer = buffers[iop];
        /*
         * Copy the data back to the arrays.  If the type has refs,
         * this function moves them so the buffer's refs are released.
         *
         * The flag USINGBUFFER is set when the buffer was used, so
         * only copy back when this flag is on.
         */
        if ((stransfer != NULL) &&
               (op_itflags[iop]&(NPY_OP_ITFLAG_WRITE|NPY_OP_ITFLAG_USINGBUFFER))
                        == (NPY_OP_ITFLAG_WRITE|NPY_OP_ITFLAG_USINGBUFFER)) {
            npy_intp op_transfersize;

            npy_intp src_stride, *dst_strides, *dst_coords, *dst_shape;
            int ndim_transfer;

            NPY_IT_DBG_PRINT1("Iterator: Operand %d was buffered\n",
                                        (int)iop);

            /*
             * If this operand is being reduced in the inner loop,
             * its buffering stride was set to zero, and just
             * one element was copied.
             */
            if (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE) {
                if (strides[iop] == 0) {
                    if (reduce_outerstrides[iop] == 0) {
                        op_transfersize = 1;
                        src_stride = 0;
                        dst_strides = &src_stride;
                        dst_coords = &NAD_INDEX(reduce_outeraxisdata);
                        dst_shape = &NAD_SHAPE(reduce_outeraxisdata);
                        ndim_transfer = 1;
                    }
                    else {
                        op_transfersize = NBF_REDUCE_OUTERSIZE(bufferdata);
                        src_stride = reduce_outerstrides[iop];
                        dst_strides =
                                &NAD_STRIDES(reduce_outeraxisdata)[iop];
                        dst_coords = &NAD_INDEX(reduce_outeraxisdata);
                        dst_shape = &NAD_SHAPE(reduce_outeraxisdata);
                        ndim_transfer = ndim - reduce_outerdim;
                    }
                }
                else {
                    if (reduce_outerstrides[iop] == 0) {
                        op_transfersize = NBF_SIZE(bufferdata);
                        src_stride = strides[iop];
                        dst_strides = &ad_strides[iop];
                        dst_coords = &NAD_INDEX(axisdata);
                        dst_shape = &NAD_SHAPE(axisdata);
                        ndim_transfer = reduce_outerdim ?
                                        reduce_outerdim : 1;
                    }
                    else {
                        op_transfersize = transfersize;
                        src_stride = strides[iop];
                        dst_strides = &ad_strides[iop];
                        dst_coords = &NAD_INDEX(axisdata);
                        dst_shape = &NAD_SHAPE(axisdata);
                        ndim_transfer = ndim;
                    }
                }
            }
            else {
                op_transfersize = transfersize;
                src_stride = strides[iop];
                dst_strides = &ad_strides[iop];
                dst_coords = &NAD_INDEX(axisdata);
                dst_shape = &NAD_SHAPE(axisdata);
                ndim_transfer = ndim;
            }

            NPY_IT_DBG_PRINT2("Iterator: Copying buffer to "
                                "operand %d (%d items)\n",
                                (int)iop, (int)op_transfersize);

            /* WRITEMASKED operand */
            if (op_itflags[iop] & NPY_OP_ITFLAG_WRITEMASKED) {
                npy_bool *maskptr;

                /*
                 * The mask pointer may be in the buffer or in
                 * the array, detect which one.
                 */
                if ((op_itflags[maskop]&NPY_OP_ITFLAG_USINGBUFFER) != 0) {
                    maskptr = (npy_bool *)buffers[maskop];
                }
                else {
                    maskptr = (npy_bool *)ad_ptrs[maskop];
                }

                PyArray_TransferMaskedStridedToNDim(ndim_transfer,
                        ad_ptrs[iop], dst_strides, axisdata_incr,
                        buffer, src_stride,
                        maskptr, strides[maskop],
                        dst_coords, axisdata_incr,
                        dst_shape, axisdata_incr,
                        op_transfersize, dtypes[iop]->elsize,
                        (PyArray_MaskedStridedUnaryOp *)stransfer,
                        transferdata);
            }
            /* Regular operand */
            else {
                PyArray_TransferStridedToNDim(ndim_transfer,
                        ad_ptrs[iop], dst_strides, axisdata_incr,
                        buffer, src_stride,
                        dst_coords, axisdata_incr,
                        dst_shape, axisdata_incr,
                        op_transfersize, dtypes[iop]->elsize,
                        stransfer,
                        transferdata);
            }
        }
        /* If there's no copy back, we may have to decrement refs.  In
         * this case, the transfer function has a 'decsrcref' transfer
         * function, so we can use it to do the decrement.
         *
         * The flag USINGBUFFER is set when the buffer was used, so
         * only decrement refs when this flag is on.
         */
        else if (stransfer != NULL &&
                       (op_itflags[iop]&NPY_OP_ITFLAG_USINGBUFFER) != 0) {
            NPY_IT_DBG_PRINT1("Iterator: Freeing refs and zeroing buffer "
                                "of operand %d\n", (int)iop);
            /* Decrement refs */
            stransfer(NULL, 0, buffer, dtypes[iop]->elsize,
                        transfersize, dtypes[iop]->elsize,
                        transferdata);
            /*
             * Zero out the memory for safety.  For instance,
             * if during iteration some Python code copied an
             * array pointing into the buffer, it will get None
             * values for its references after this.
             */
            memset(buffer, 0, dtypes[iop]->elsize*transfersize);
        }
    }

    NPY_IT_DBG_PRINT("Iterator: Finished copying buffers to outputs\n");
}

/*
 * This gets called after the iterator has been positioned to a multi-index
 * for the start of a buffer.  It decides which operands need a buffer,
 * and copies the data into the buffers.
 */
NPY_NO_EXPORT void
npyiter_copy_to_buffers(NpyIter *iter, char **prev_dataptrs)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter),
                    *reduce_outeraxisdata = NULL;

    PyArray_Descr **dtypes = NIT_DTYPES(iter);
    PyArrayObject **operands = NIT_OPERANDS(iter);
    npy_intp *strides = NBF_STRIDES(bufferdata),
             *ad_strides = NAD_STRIDES(axisdata);
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    char **ptrs = NBF_PTRS(bufferdata), **ad_ptrs = NAD_PTRS(axisdata);
    char **buffers = NBF_BUFFERS(bufferdata);
    npy_intp iterindex, iterend, transfersize,
            singlestridesize, reduce_innersize = 0, reduce_outerdim = 0;
    int is_onestride = 0, any_buffered = 0;

    npy_intp *reduce_outerstrides = NULL;
    char **reduce_outerptrs = NULL;

    PyArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;

    /*
     * Have to get this flag before npyiter_checkreducesize sets
     * it for the next iteration.
     */
    npy_bool reuse_reduce_loops = (prev_dataptrs != NULL) &&
                    ((itflags&NPY_ITFLAG_REUSE_REDUCE_LOOPS) != 0);

    npy_intp axisdata_incr = NIT_AXISDATA_SIZEOF(itflags, ndim, nop) /
                                NPY_SIZEOF_INTP;

    NPY_IT_DBG_PRINT("Iterator: Copying inputs to buffers\n");

    /* Calculate the size if using any buffers */
    iterindex = NIT_ITERINDEX(iter);
    iterend = NIT_ITEREND(iter);
    transfersize = NBF_BUFFERSIZE(bufferdata);
    if (transfersize > iterend - iterindex) {
        transfersize = iterend - iterindex;
    }

    /* If last time around, the reduce loop structure was full, we reuse it */
    if (reuse_reduce_loops) {
        npy_intp full_transfersize, prev_reduce_outersize;

        prev_reduce_outersize = NBF_REDUCE_OUTERSIZE(bufferdata);
        reduce_outerstrides = NBF_REDUCE_OUTERSTRIDES(bufferdata);
        reduce_outerptrs = NBF_REDUCE_OUTERPTRS(bufferdata);
        reduce_outerdim = NBF_REDUCE_OUTERDIM(bufferdata);
        reduce_outeraxisdata = NIT_INDEX_AXISDATA(axisdata, reduce_outerdim);
        reduce_innersize = NBF_SIZE(bufferdata);
        NBF_REDUCE_POS(bufferdata) = 0;
        /*
         * Try to do make the outersize as big as possible. This allows
         * it to shrink when processing the last bit of the outer reduce loop,
         * then grow again at the beginnning of the next outer reduce loop.
         */
        NBF_REDUCE_OUTERSIZE(bufferdata) = (NAD_SHAPE(reduce_outeraxisdata)-
                                            NAD_INDEX(reduce_outeraxisdata));
        full_transfersize = NBF_REDUCE_OUTERSIZE(bufferdata)*reduce_innersize;
        /* If the full transfer size doesn't fit in the buffer, truncate it */
        if (full_transfersize > NBF_BUFFERSIZE(bufferdata)) {
            NBF_REDUCE_OUTERSIZE(bufferdata) = transfersize/reduce_innersize;
            transfersize = NBF_REDUCE_OUTERSIZE(bufferdata)*reduce_innersize;
        }
        else {
            transfersize = full_transfersize;
        }
        if (prev_reduce_outersize < NBF_REDUCE_OUTERSIZE(bufferdata)) {
            /*
             * If the previous time around less data was copied it may not
             * be safe to reuse the buffers even if the pointers match.
             */
            reuse_reduce_loops = 0;
        }
        NBF_BUFITEREND(bufferdata) = iterindex + reduce_innersize;

        NPY_IT_DBG_PRINT3("Reused reduce transfersize: %d innersize: %d "
                        "itersize: %d\n",
                            (int)transfersize,
                            (int)reduce_innersize,
                            (int)NpyIter_GetIterSize(iter));
        NPY_IT_DBG_PRINT1("Reduced reduce outersize: %d",
                            (int)NBF_REDUCE_OUTERSIZE(bufferdata));
    }
    /*
     * If there are any reduction operands, we may have to make
     * the size smaller so we don't copy the same value into
     * a buffer twice, as the buffering does not have a mechanism
     * to combine values itself.
     */
    else if (itflags&NPY_ITFLAG_REDUCE) {
        NPY_IT_DBG_PRINT("Iterator: Calculating reduce loops\n");
        transfersize = npyiter_checkreducesize(iter, transfersize,
                                                &reduce_innersize,
                                                &reduce_outerdim);
        NPY_IT_DBG_PRINT3("Reduce transfersize: %d innersize: %d "
                        "itersize: %d\n",
                            (int)transfersize,
                            (int)reduce_innersize,
                            (int)NpyIter_GetIterSize(iter));

        reduce_outerstrides = NBF_REDUCE_OUTERSTRIDES(bufferdata);
        reduce_outerptrs = NBF_REDUCE_OUTERPTRS(bufferdata);
        reduce_outeraxisdata = NIT_INDEX_AXISDATA(axisdata, reduce_outerdim);
        NBF_SIZE(bufferdata) = reduce_innersize;
        NBF_REDUCE_POS(bufferdata) = 0;
        NBF_REDUCE_OUTERDIM(bufferdata) = reduce_outerdim;
        NBF_BUFITEREND(bufferdata) = iterindex + reduce_innersize;
        if (reduce_innersize == 0) {
            NBF_REDUCE_OUTERSIZE(bufferdata) = 0;
            return;
        }
        else {
            NBF_REDUCE_OUTERSIZE(bufferdata) = transfersize/reduce_innersize;
        }
    }
    else {
        NBF_SIZE(bufferdata) = transfersize;
        NBF_BUFITEREND(bufferdata) = iterindex + transfersize;
    }

    /* Calculate the maximum size if using a single stride and no buffers */
    singlestridesize = NAD_SHAPE(axisdata)-NAD_INDEX(axisdata);
    if (singlestridesize > iterend - iterindex) {
        singlestridesize = iterend - iterindex;
    }
    if (singlestridesize >= transfersize) {
        is_onestride = 1;
    }

    for (iop = 0; iop < nop; ++iop) {
        /*
         * If the buffer is write-only, these two are NULL, and the buffer
         * pointers will be set up but the read copy won't be done
         */
        stransfer = NBF_READTRANSFERFN(bufferdata)[iop];
        transferdata = NBF_READTRANSFERDATA(bufferdata)[iop];
        switch (op_itflags[iop]&
                        (NPY_OP_ITFLAG_BUFNEVER|
                         NPY_OP_ITFLAG_CAST|
                         NPY_OP_ITFLAG_REDUCE)) {
            /* Never need to buffer this operand */
            case NPY_OP_ITFLAG_BUFNEVER:
                ptrs[iop] = ad_ptrs[iop];
                if (itflags&NPY_ITFLAG_REDUCE) {
                    reduce_outerstrides[iop] = reduce_innersize *
                                                 strides[iop];
                    reduce_outerptrs[iop] = ptrs[iop];
                }
                /*
                 * Should not adjust the stride - ad_strides[iop]
                 * could be zero, but strides[iop] was initialized
                 * to the first non-trivial stride.
                 */
                stransfer = NULL;
                /* The flag NPY_OP_ITFLAG_USINGBUFFER can be ignored here */
                break;
            /* Never need to buffer this operand */
            case NPY_OP_ITFLAG_BUFNEVER|NPY_OP_ITFLAG_REDUCE:
                ptrs[iop] = ad_ptrs[iop];
                reduce_outerptrs[iop] = ptrs[iop];
                reduce_outerstrides[iop] = 0;
                /*
                 * Should not adjust the stride - ad_strides[iop]
                 * could be zero, but strides[iop] was initialized
                 * to the first non-trivial stride.
                 */
                stransfer = NULL;
                /* The flag NPY_OP_ITFLAG_USINGBUFFER can be ignored here */
                break;
            /* Just a copy */
            case 0:
                /* Do not reuse buffer if it did not exist */
                if (!(op_itflags[iop] & NPY_OP_ITFLAG_USINGBUFFER) &&
                                                (prev_dataptrs != NULL)) {
                    prev_dataptrs[iop] = NULL;
                }
                /*
                 * No copyswap or cast was requested, so all we're
                 * doing is copying the data to fill the buffer and
                 * produce a single stride.  If the underlying data
                 * already does that, no need to copy it.
                 */
                if (is_onestride) {
                    ptrs[iop] = ad_ptrs[iop];
                    strides[iop] = ad_strides[iop];
                    stransfer = NULL;
                    /* Signal that the buffer is not being used */
                    op_itflags[iop] &= (~NPY_OP_ITFLAG_USINGBUFFER);
                }
                /* If some other op is reduced, we have a double reduce loop */
                else if ((itflags&NPY_ITFLAG_REDUCE) &&
                                (reduce_outerdim == 1) &&
                                (transfersize/reduce_innersize <=
                                            NAD_SHAPE(reduce_outeraxisdata) -
                                            NAD_INDEX(reduce_outeraxisdata))) {
                    ptrs[iop] = ad_ptrs[iop];
                    reduce_outerptrs[iop] = ptrs[iop];
                    strides[iop] = ad_strides[iop];
                    reduce_outerstrides[iop] =
                                    NAD_STRIDES(reduce_outeraxisdata)[iop];
                    stransfer = NULL;
                    /* Signal that the buffer is not being used */
                    op_itflags[iop] &= (~NPY_OP_ITFLAG_USINGBUFFER);
                }
                else {
                    /* In this case, the buffer is being used */
                    ptrs[iop] = buffers[iop];
                    strides[iop] = dtypes[iop]->elsize;
                    if (itflags&NPY_ITFLAG_REDUCE) {
                        reduce_outerstrides[iop] = reduce_innersize *
                                                     strides[iop];
                        reduce_outerptrs[iop] = ptrs[iop];
                    }
                    /* Signal that the buffer is being used */
                    op_itflags[iop] |= NPY_OP_ITFLAG_USINGBUFFER;
                }
                break;
            /* Just a copy, but with a reduction */
            case NPY_OP_ITFLAG_REDUCE:
                /* Do not reuse buffer if it did not exist */
                if (!(op_itflags[iop] & NPY_OP_ITFLAG_USINGBUFFER) &&
                                                (prev_dataptrs != NULL)) {
                    prev_dataptrs[iop] = NULL;
                }
                if (ad_strides[iop] == 0) {
                    strides[iop] = 0;
                    /* It's all in one stride in the inner loop dimension */
                    if (is_onestride) {
                        NPY_IT_DBG_PRINT1("reduce op %d all one stride\n", (int)iop);
                        ptrs[iop] = ad_ptrs[iop];
                        reduce_outerstrides[iop] = 0;
                        stransfer = NULL;
                        /* Signal that the buffer is not being used */
                        op_itflags[iop] &= (~NPY_OP_ITFLAG_USINGBUFFER);
                    }
                    /* It's all in one stride in the reduce outer loop */
                    else if ((reduce_outerdim > 0) &&
                                    (transfersize/reduce_innersize <=
                                            NAD_SHAPE(reduce_outeraxisdata) -
                                            NAD_INDEX(reduce_outeraxisdata))) {
                        NPY_IT_DBG_PRINT1("reduce op %d all one outer stride\n",
                                                            (int)iop);
                        ptrs[iop] = ad_ptrs[iop];
                        /* Outer reduce loop advances by one item */
                        reduce_outerstrides[iop] =
                                NAD_STRIDES(reduce_outeraxisdata)[iop];
                        stransfer = NULL;
                        /* Signal that the buffer is not being used */
                        op_itflags[iop] &= (~NPY_OP_ITFLAG_USINGBUFFER);
                    }
                    /* In this case, the buffer is being used */
                    else {
                        NPY_IT_DBG_PRINT1("reduce op %d must buffer\n", (int)iop);
                        ptrs[iop] = buffers[iop];
                        /* Both outer and inner reduce loops have stride 0 */
                        if (NAD_STRIDES(reduce_outeraxisdata)[iop] == 0) {
                            reduce_outerstrides[iop] = 0;
                        }
                        /* Outer reduce loop advances by one item */
                        else {
                            reduce_outerstrides[iop] = dtypes[iop]->elsize;
                        }
                        /* Signal that the buffer is being used */
                        op_itflags[iop] |= NPY_OP_ITFLAG_USINGBUFFER;
                    }

                }
                else if (is_onestride) {
                    NPY_IT_DBG_PRINT1("reduce op %d all one stride in dim 0\n", (int)iop);
                    ptrs[iop] = ad_ptrs[iop];
                    strides[iop] = ad_strides[iop];
                    reduce_outerstrides[iop] = 0;
                    stransfer = NULL;
                    /* Signal that the buffer is not being used */
                    op_itflags[iop] &= (~NPY_OP_ITFLAG_USINGBUFFER);
                }
                else {
                    /* It's all in one stride in the reduce outer loop */
                    if ((reduce_outerdim > 0) &&
                                    (transfersize/reduce_innersize <=
                                            NAD_SHAPE(reduce_outeraxisdata) -
                                            NAD_INDEX(reduce_outeraxisdata))) {
                        ptrs[iop] = ad_ptrs[iop];
                        strides[iop] = ad_strides[iop];
                        /* Outer reduce loop advances by one item */
                        reduce_outerstrides[iop] =
                                NAD_STRIDES(reduce_outeraxisdata)[iop];
                        stransfer = NULL;
                        /* Signal that the buffer is not being used */
                        op_itflags[iop] &= (~NPY_OP_ITFLAG_USINGBUFFER);
                    }
                    /* In this case, the buffer is being used */
                    else {
                        ptrs[iop] = buffers[iop];
                        strides[iop] = dtypes[iop]->elsize;

                        if (NAD_STRIDES(reduce_outeraxisdata)[iop] == 0) {
                            /* Reduction in outer reduce loop */
                            reduce_outerstrides[iop] = 0;
                        }
                        else {
                            /* Advance to next items in outer reduce loop */
                            reduce_outerstrides[iop] = reduce_innersize *
                                                         dtypes[iop]->elsize;
                        }
                        /* Signal that the buffer is being used */
                        op_itflags[iop] |= NPY_OP_ITFLAG_USINGBUFFER;
                    }
                }
                reduce_outerptrs[iop] = ptrs[iop];
                break;
            default:
                /* In this case, the buffer is always being used */
                any_buffered = 1;

                /* Signal that the buffer is being used */
                op_itflags[iop] |= NPY_OP_ITFLAG_USINGBUFFER;

                if (!(op_itflags[iop]&NPY_OP_ITFLAG_REDUCE)) {
                    ptrs[iop] = buffers[iop];
                    strides[iop] = dtypes[iop]->elsize;
                    if (itflags&NPY_ITFLAG_REDUCE) {
                        reduce_outerstrides[iop] = reduce_innersize *
                                                     strides[iop];
                        reduce_outerptrs[iop] = ptrs[iop];
                    }
                }
                /* The buffer is being used with reduction */
                else {
                    ptrs[iop] = buffers[iop];
                    if (ad_strides[iop] == 0) {
                        NPY_IT_DBG_PRINT1("cast op %d has innermost stride 0\n", (int)iop);
                        strides[iop] = 0;
                        /* Both outer and inner reduce loops have stride 0 */
                        if (NAD_STRIDES(reduce_outeraxisdata)[iop] == 0) {
                            NPY_IT_DBG_PRINT1("cast op %d has outermost stride 0\n", (int)iop);
                            reduce_outerstrides[iop] = 0;
                        }
                        /* Outer reduce loop advances by one item */
                        else {
                            NPY_IT_DBG_PRINT1("cast op %d has outermost stride !=0\n", (int)iop);
                            reduce_outerstrides[iop] = dtypes[iop]->elsize;
                        }
                    }
                    else {
                        NPY_IT_DBG_PRINT1("cast op %d has innermost stride !=0\n", (int)iop);
                        strides[iop] = dtypes[iop]->elsize;

                        if (NAD_STRIDES(reduce_outeraxisdata)[iop] == 0) {
                            NPY_IT_DBG_PRINT1("cast op %d has outermost stride 0\n", (int)iop);
                            /* Reduction in outer reduce loop */
                            reduce_outerstrides[iop] = 0;
                        }
                        else {
                            NPY_IT_DBG_PRINT1("cast op %d has outermost stride !=0\n", (int)iop);
                            /* Advance to next items in outer reduce loop */
                            reduce_outerstrides[iop] = reduce_innersize *
                                                         dtypes[iop]->elsize;
                        }
                    }
                    reduce_outerptrs[iop] = ptrs[iop];
                }
                break;
        }

        if (stransfer != NULL) {
            npy_intp src_itemsize;
            npy_intp op_transfersize;

            npy_intp dst_stride, *src_strides, *src_coords, *src_shape;
            int ndim_transfer;

            npy_bool skip_transfer = 0;

            src_itemsize = PyArray_DTYPE(operands[iop])->elsize;

            /* If stransfer wasn't set to NULL, buffering is required */
            any_buffered = 1;

            /*
             * If this operand is being reduced in the inner loop,
             * set its buffering stride to zero, and just copy
             * one element.
             */
            if (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE) {
                if (ad_strides[iop] == 0) {
                    strides[iop] = 0;
                    if (reduce_outerstrides[iop] == 0) {
                        op_transfersize = 1;
                        dst_stride = 0;
                        src_strides = &dst_stride;
                        src_coords = &NAD_INDEX(reduce_outeraxisdata);
                        src_shape = &NAD_SHAPE(reduce_outeraxisdata);
                        ndim_transfer = 1;

                        /*
                         * When we're reducing a single element, and
                         * it's still the same element, don't overwrite
                         * it even when reuse reduce loops is unset.
                         * This preserves the precision of the
                         * intermediate calculation.
                         */
                        if (prev_dataptrs &&
                                    prev_dataptrs[iop] == ad_ptrs[iop]) {
                            NPY_IT_DBG_PRINT1("Iterator: skipping operand %d"
                                    " copy because it's a 1-element reduce\n",
                                    (int)iop);

                            skip_transfer = 1;
                        }
                    }
                    else {
                        op_transfersize = NBF_REDUCE_OUTERSIZE(bufferdata);
                        dst_stride = reduce_outerstrides[iop];
                        src_strides = &NAD_STRIDES(reduce_outeraxisdata)[iop];
                        src_coords = &NAD_INDEX(reduce_outeraxisdata);
                        src_shape = &NAD_SHAPE(reduce_outeraxisdata);
                        ndim_transfer = ndim - reduce_outerdim;
                    }
                }
                else {
                    if (reduce_outerstrides[iop] == 0) {
                        op_transfersize = NBF_SIZE(bufferdata);
                        dst_stride = strides[iop];
                        src_strides = &ad_strides[iop];
                        src_coords = &NAD_INDEX(axisdata);
                        src_shape = &NAD_SHAPE(axisdata);
                        ndim_transfer = reduce_outerdim ? reduce_outerdim : 1;
                    }
                    else {
                        op_transfersize = transfersize;
                        dst_stride = strides[iop];
                        src_strides = &ad_strides[iop];
                        src_coords = &NAD_INDEX(axisdata);
                        src_shape = &NAD_SHAPE(axisdata);
                        ndim_transfer = ndim;
                    }
                }
            }
            else {
                op_transfersize = transfersize;
                dst_stride = strides[iop];
                src_strides = &ad_strides[iop];
                src_coords = &NAD_INDEX(axisdata);
                src_shape = &NAD_SHAPE(axisdata);
                ndim_transfer = ndim;
            }

            /*
             * If the whole buffered loop structure remains the same,
             * and the source pointer for this data didn't change,
             * we don't have to copy the data again.
             */
            if (reuse_reduce_loops && prev_dataptrs[iop] == ad_ptrs[iop]) {
                NPY_IT_DBG_PRINT2("Iterator: skipping operands %d "
                        "copy (%d items) because loops are reused and the data "
                        "pointer didn't change\n",
                        (int)iop, (int)op_transfersize);
                skip_transfer = 1;
            }

            /* If the data type requires zero-inititialization */
            if (PyDataType_FLAGCHK(dtypes[iop], NPY_NEEDS_INIT)) {
                NPY_IT_DBG_PRINT("Iterator: Buffer requires init, "
                                    "memsetting to 0\n");
                memset(ptrs[iop], 0, dtypes[iop]->elsize*op_transfersize);
                /* Can't skip the transfer in this case */
                skip_transfer = 0;
            }

            if (!skip_transfer) {
                NPY_IT_DBG_PRINT2("Iterator: Copying operand %d to "
                                "buffer (%d items)\n",
                                (int)iop, (int)op_transfersize);

                PyArray_TransferNDimToStrided(ndim_transfer,
                        ptrs[iop], dst_stride,
                        ad_ptrs[iop], src_strides, axisdata_incr,
                        src_coords, axisdata_incr,
                        src_shape, axisdata_incr,
                        op_transfersize, src_itemsize,
                        stransfer,
                        transferdata);
            }
        }
        else if (ptrs[iop] == buffers[iop]) {
            /* If the data type requires zero-inititialization */
            if (PyDataType_FLAGCHK(dtypes[iop], NPY_NEEDS_INIT)) {
                NPY_IT_DBG_PRINT1("Iterator: Write-only buffer for "
                                    "operand %d requires init, "
                                    "memsetting to 0\n", (int)iop);
                memset(ptrs[iop], 0, dtypes[iop]->elsize*transfersize);
            }
        }

    }

    /*
     * If buffering wasn't needed, we can grow the inner
     * loop to as large as possible.
     *
     * TODO: Could grow REDUCE loop too with some more logic above.
     */
    if (!any_buffered && (itflags&NPY_ITFLAG_GROWINNER) &&
                        !(itflags&NPY_ITFLAG_REDUCE)) {
        if (singlestridesize > transfersize) {
            NPY_IT_DBG_PRINT2("Iterator: Expanding inner loop size "
                    "from %d to %d since buffering wasn't needed\n",
                    (int)NBF_SIZE(bufferdata), (int)singlestridesize);
            NBF_SIZE(bufferdata) = singlestridesize;
            NBF_BUFITEREND(bufferdata) = iterindex + singlestridesize;
        }
    }

    NPY_IT_DBG_PRINT1("Any buffering needed: %d\n", any_buffered);

    NPY_IT_DBG_PRINT1("Iterator: Finished copying inputs to buffers "
                        "(buffered size is %d)\n", (int)NBF_SIZE(bufferdata));
}

/*
 * This checks how much space can be buffered without encountering the
 * same value twice, or for operands whose innermost stride is zero,
 * without encountering a different value.  By reducing the buffered
 * amount to this size, reductions can be safely buffered.
 *
 * Reductions are buffered with two levels of looping, to avoid
 * frequent copying to the buffers.  The return value is the over-all
 * buffer size, and when the flag NPY_ITFLAG_REDUCE is set, reduce_innersize
 * receives the size of the inner of the two levels of looping.
 *
 * The value placed in reduce_outerdim is the index into the AXISDATA
 * for where the second level of the double loop begins.
 *
 * The return value is always a multiple of the value placed in
 * reduce_innersize.
 */
static npy_intp
npyiter_checkreducesize(NpyIter *iter, npy_intp count,
                                npy_intp *reduce_innersize,
                                npy_intp *reduce_outerdim)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    npy_intp coord, shape, *strides;
    npy_intp reducespace = 1, factor;
    npy_bool nonzerocoord;

    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    char stride0op[NPY_MAXARGS];

    /* Default to no outer axis */
    *reduce_outerdim = 0;

    /* If there's only one dimension, no need to calculate anything */
    if (ndim == 1 || count == 0) {
        *reduce_innersize = count;
        return count;
    }

    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    axisdata = NIT_AXISDATA(iter);

    /* Indicate which REDUCE operands have stride 0 in the inner loop */
    strides = NAD_STRIDES(axisdata);
    for (iop = 0; iop < nop; ++iop) {
        stride0op[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE) &&
                           (strides[iop] == 0);
        NPY_IT_DBG_PRINT2("Iterator: Operand %d has stride 0 in "
                        "the inner loop? %d\n", iop, (int)stride0op[iop]);
    }
    shape = NAD_SHAPE(axisdata);
    coord = NAD_INDEX(axisdata);
    reducespace += (shape-coord-1);
    factor = shape;
    NIT_ADVANCE_AXISDATA(axisdata, 1);

    /* Initialize nonzerocoord based on the first coordinate */
    nonzerocoord = (coord != 0);

    /* Go forward through axisdata, calculating the space available */
    for (idim = 1; idim < ndim && reducespace < count;
                                ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        NPY_IT_DBG_PRINT2("Iterator: inner loop reducespace %d, count %d\n",
                                (int)reducespace, (int)count);

        strides = NAD_STRIDES(axisdata);
        for (iop = 0; iop < nop; ++iop) {
            /*
             * If a reduce stride switched from zero to non-zero, or
             * vice versa, that's the point where the data will stop
             * being the same element or will repeat, and if the
             * buffer starts with an all zero multi-index up to this
             * point, gives us the reduce_innersize.
             */
            if((stride0op[iop] && (strides[iop] != 0)) ||
                        (!stride0op[iop] &&
                         (strides[iop] == 0) &&
                         (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE))) {
                NPY_IT_DBG_PRINT1("Iterator: Reduce operation limits "
                                    "buffer to %d\n", (int)reducespace);
                /*
                 * If we already found more elements than count, or
                 * the starting coordinate wasn't zero, the two-level
                 * looping is unnecessary/can't be done, so return.
                 */
                if (count <= reducespace) {
                    *reduce_innersize = count;
                    NIT_ITFLAGS(iter) |= NPY_ITFLAG_REUSE_REDUCE_LOOPS;
                    return count;
                }
                else if (nonzerocoord) {
                    if (reducespace < count) {
                        count = reducespace;
                    }
                    *reduce_innersize = count;
                    /* NOTE: This is similar to the (coord != 0) case below. */
                    NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_REUSE_REDUCE_LOOPS;
                    return count;
                }
                else {
                    *reduce_innersize = reducespace;
                    break;
                }
            }
        }
        /* If we broke out of the loop early, we found reduce_innersize */
        if (iop != nop) {
            NPY_IT_DBG_PRINT2("Iterator: Found first dim not "
                            "reduce (%d of %d)\n", iop, nop);
            break;
        }

        shape = NAD_SHAPE(axisdata);
        coord = NAD_INDEX(axisdata);
        if (coord != 0) {
            nonzerocoord = 1;
        }
        reducespace += (shape-coord-1) * factor;
        factor *= shape;
    }

    /*
     * If there was any non-zero coordinate, the reduction inner
     * loop doesn't fit in the buffersize, or the reduction inner loop
     * covered the entire iteration size, can't do the double loop.
     */
    if (nonzerocoord || count < reducespace || idim == ndim) {
        if (reducespace < count) {
            count = reducespace;
        }
        *reduce_innersize = count;
        /* In this case, we can't reuse the reduce loops */
        NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_REUSE_REDUCE_LOOPS;
        return count;
    }

    coord = NAD_INDEX(axisdata);
    if (coord != 0) {
        /*
         * In this case, it is only safe to reuse the buffer if the amount
         * of data copied is not more then the current axes, as is the
         * case when reuse_reduce_loops was active already.
         * It should be in principle OK when the idim loop returns immidiatly.
         */
        NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_REUSE_REDUCE_LOOPS;
    }
    else {
        /* In this case, we can reuse the reduce loops */
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_REUSE_REDUCE_LOOPS;
    }

    *reduce_innersize = reducespace;
    count /= reducespace;

    NPY_IT_DBG_PRINT2("Iterator: reduce_innersize %d count /ed %d\n",
                    (int)reducespace, (int)count);

    /*
     * Continue through the rest of the dimensions.  If there are
     * two separated reduction axes, we may have to cut the buffer
     * short again.
     */
    *reduce_outerdim = idim;
    reducespace = 1;
    factor = 1;
    /* Indicate which REDUCE operands have stride 0 at the current level */
    strides = NAD_STRIDES(axisdata);
    for (iop = 0; iop < nop; ++iop) {
        stride0op[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE) &&
                           (strides[iop] == 0);
        NPY_IT_DBG_PRINT2("Iterator: Operand %d has stride 0 in "
                        "the outer loop? %d\n", iop, (int)stride0op[iop]);
    }
    shape = NAD_SHAPE(axisdata);
    reducespace += (shape-coord-1) * factor;
    factor *= shape;
    NIT_ADVANCE_AXISDATA(axisdata, 1);
    ++idim;

    for (; idim < ndim && reducespace < count;
                                ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        NPY_IT_DBG_PRINT2("Iterator: outer loop reducespace %d, count %d\n",
                                (int)reducespace, (int)count);
        strides = NAD_STRIDES(axisdata);
        for (iop = 0; iop < nop; ++iop) {
            /*
             * If a reduce stride switched from zero to non-zero, or
             * vice versa, that's the point where the data will stop
             * being the same element or will repeat, and if the
             * buffer starts with an all zero multi-index up to this
             * point, gives us the reduce_innersize.
             */
            if((stride0op[iop] && (strides[iop] != 0)) ||
                        (!stride0op[iop] &&
                         (strides[iop] == 0) &&
                         (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE))) {
                NPY_IT_DBG_PRINT1("Iterator: Reduce operation limits "
                                    "buffer to %d\n", (int)reducespace);
                /*
                 * This terminates the outer level of our double loop.
                 */
                if (count <= reducespace) {
                    return count * (*reduce_innersize);
                }
                else {
                    return reducespace * (*reduce_innersize);
                }
            }
        }

        shape = NAD_SHAPE(axisdata);
        coord = NAD_INDEX(axisdata);
        if (coord != 0) {
            nonzerocoord = 1;
        }
        reducespace += (shape-coord-1) * factor;
        factor *= shape;
    }

    if (reducespace < count) {
        count = reducespace;
    }
    return count * (*reduce_innersize);
}

#undef NPY_ITERATOR_IMPLEMENTATION_CODE
