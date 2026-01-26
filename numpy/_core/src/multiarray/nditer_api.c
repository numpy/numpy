/*
 * This file implements most of the main API functions of NumPy's nditer.
 * This excludes functions specialized using the templating system.
 *
 * Copyright (c) 2010-2011 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * Copyright (c) 2011 Enthought, Inc
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* Allow this .c file to include nditer_impl.h */
#define NPY_ITERATOR_IMPLEMENTATION_CODE

#include "nditer_impl.h"
#include "templ_common.h"
#include "ctors.h"


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

    /* Adjust the permutation */
    for (idim = 0; idim < ndim-1; ++idim) {
        npy_int8 p = (idim < xdim) ? perm[idim] : perm[idim+1];
        if (p >= 0) {
            if (p > axis) {
                --p;
            }
        }
        else {
            if (p < -1-axis) {
                ++p;
            }
        }
        perm[idim] = p;
    }

    /* Shift all the axisdata structures by one */
    axisdata = NIT_INDEX_AXISDATA(axisdata_del, 1);
    memmove(axisdata_del, axisdata, (ndim-1-xdim)*sizeof_axisdata);

    /* Adjust the iteration size and reset iterend */
    NIT_ITERSIZE(iter) = 1;
    axisdata = NIT_AXISDATA(iter);
    for (idim = 0; idim < ndim-1; ++idim) {
        if (npy_mul_sizes_with_overflow(&NIT_ITERSIZE(iter),
                    NIT_ITERSIZE(iter), NAD_SHAPE(axisdata))) {
            NIT_ITERSIZE(iter) = -1;
            break;
        }
        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }
    NIT_ITEREND(iter) = NIT_ITERSIZE(iter);

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
        if (NIT_ITERSIZE(iter) < 0) {
            PyErr_SetString(PyExc_ValueError, "iterator is too large");
            return NPY_FAIL;
        }

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


static char *_reset_cast_error = (
        "Iterator reset failed due to a casting failure. "
        "This error is set as a Python error.");

/*NUMPY_API
 * Resets the iterator to its initial state
 *
 * The use of errmsg is discouraged, it cannot be guaranteed that the GIL
 * will not be grabbed on casting errors even when this is passed.
 *
 * If errmsg is non-NULL, it should point to a variable which will
 * receive the error message, and no Python exception will be set.
 * This is so that the function can be called from code not holding
 * the GIL. Note that cast errors may still lead to the GIL being
 * grabbed temporarily.
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
                if (errmsg != NULL) {
                    *errmsg = _reset_cast_error;
                }
                return NPY_FAIL;
            }
            NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_DELAYBUF;
        }
        else {
            /*
             * If the iterindex is already right, no need to
             * do anything (and no cast error has previously occurred).
             */
            bufferdata = NIT_BUFFERDATA(iter);
            if (NIT_ITERINDEX(iter) == NIT_ITERSTART(iter) &&
                    NBF_BUFITEREND(bufferdata) <= NIT_ITEREND(iter) &&
                    NBF_SIZE(bufferdata) > 0) {
                return NPY_SUCCEED;
            }
            if (npyiter_copy_from_buffers(iter) < 0) {
                if (errmsg != NULL) {
                    *errmsg = _reset_cast_error;
                }
                return NPY_FAIL;
            }
        }
    }

    npyiter_goto_iterindex(iter, NIT_ITERSTART(iter));

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* Prepare the next buffers and set iterend/size */
        if (npyiter_copy_to_buffers(iter, NULL) < 0) {
            if (errmsg != NULL) {
                *errmsg = _reset_cast_error;
            }
            return NPY_FAIL;
        }
    }
    else if (itflags&NPY_ITFLAG_EXLOOP)  {
        /* make sure to update the user pointers (buffer copy does it above). */
        memcpy(NIT_USERPTRS(iter), NIT_DATAPTRS(iter), NPY_SIZEOF_INTP*nop);
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
 * the GIL. Note that cast errors may still lead to the GIL being
 * grabbed temporarily.
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
            if (npyiter_copy_from_buffers(iter) < 0) {
                if (errmsg != NULL) {
                    *errmsg = _reset_cast_error;
                }
                return NPY_FAIL;
            }
        }
    }

    /* The new data pointers for resetting */
    for (iop = 0; iop < nop; ++iop) {
        resetdataptr[iop] = baseptrs[iop] + baseoffsets[iop];
    }

    npyiter_goto_iterindex(iter, NIT_ITERSTART(iter));

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* Prepare the next buffers and set iterend/size */
        if (npyiter_copy_to_buffers(iter, NULL) < 0) {
            if (errmsg != NULL) {
                *errmsg = _reset_cast_error;
            }
            return NPY_FAIL;
        }
    }

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Resets the iterator to a new iterator index range
 *
 * If errmsg is non-NULL, it should point to a variable which will
 * receive the error message, and no Python exception will be set.
 * This is so that the function can be called from code not holding
 * the GIL. Note that cast errors may still lead to the GIL being
 * grabbed temporarily.
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
        if (NIT_ITERSIZE(iter) < 0) {
            if (errmsg == NULL) {
                PyErr_SetString(PyExc_ValueError, "iterator is too large");
            }
            else {
                *errmsg = "iterator is too large";
            }
            return NPY_FAIL;
        }
        if (errmsg == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "Out-of-bounds range [%" NPY_INTP_FMT ", %" NPY_INTP_FMT ") passed to "
                    "ResetToIterIndexRange", istart, iend);
        }
        else {
            *errmsg = "Out-of-bounds range passed to ResetToIterIndexRange";
        }
        return NPY_FAIL;
    }
    else if (iend < istart) {
        if (errmsg == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "Invalid range [%" NPY_INTP_FMT ", %" NPY_INTP_FMT ") passed to ResetToIterIndexRange",
                    istart, iend);
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
NpyIter_GotoMultiIndex(NpyIter *iter, npy_intp const *multi_index)
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
        if (NIT_ITERSIZE(iter) < 0) {
            PyErr_SetString(PyExc_ValueError, "iterator is too large");
            return NPY_FAIL;
        }
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
        if (NIT_ITERSIZE(iter) < 0) {
            PyErr_SetString(PyExc_ValueError, "iterator is too large");
            return NPY_FAIL;
        }
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
            ptrs = NIT_USERPTRS(iter);
            delta = iterindex - NIT_ITERINDEX(iter);

            for (iop = 0; iop < nop; ++iop) {
                ptrs[iop] += delta * strides[iop];
            }

            NIT_ITERINDEX(iter) = iterindex;
        }
        /* Start the buffer at the provided iterindex */
        else {
            /* Write back to the arrays */
            if (npyiter_copy_from_buffers(iter) < 0) {
                return NPY_FAIL;
            }

            npyiter_goto_iterindex(iter, iterindex);

            /* Prepare the next buffers and set iterend/size */
            if (npyiter_copy_to_buffers(iter, NULL) < 0) {
                return NPY_FAIL;
            }
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
        if (NBF_REDUCE_POS(bufferdata) != 0 &&
                NBF_REDUCE_OUTERSTRIDES(bufferdata)[iop] == 0) {
            return 0;
        }
    }

    return 1;
}

/*NUMPY_API
 * Whether the iteration could be done with no buffering.
 *
 * Note that the iterator may use buffering to increase the inner loop size
 * even when buffering is not required.
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
 *
 * NOTE: Internally (currently), `NpyIter_GetTransferFlags` will
 *       additionally provide information on whether floating point errors
 *       may be given during casts.  The flags only require the API use
 *       necessary for buffering though.  So an iterate which does not require
 *       buffering may indicate `NpyIter_IterationNeedsAPI`, but not include
 *       the flag in `NpyIter_GetTransferFlags`.
 */
NPY_NO_EXPORT npy_bool
NpyIter_IterationNeedsAPI(NpyIter *iter)
{
    int nop = NIT_NOP(iter);
    /* If any of the buffer filling need the API, flag it as well. */
    if (NpyIter_GetTransferFlags(iter) & NPY_METH_REQUIRES_PYAPI) {
        return NPY_TRUE;
    }

    for (int iop = 0; iop < nop; ++iop) {
        PyArray_Descr *rdt = NIT_DTYPES(iter)[iop];
        if ((rdt->flags & (NPY_ITEM_REFCOUNT |
                                 NPY_ITEM_IS_POINTER |
                                 NPY_NEEDS_PYAPI)) != 0) {
            /* Iteration needs API access */
            return NPY_TRUE;
        }
    }

    return NPY_FALSE;
}


/*NUMPY_API
 * Fetch the NPY_ARRAYMETHOD_FLAGS (runtime) flags for all "transfer functions'
 * (i.e. copy to buffer/casts).
 *
 * It is the preferred way to check whether the iteration requires to hold the
 * GIL or may set floating point errors during buffer copies.
 *
 * I.e. use `NpyIter_GetTransferFlags(iter) & NPY_METH_REQUIRES_PYAPI` to check
 * if you cannot release the GIL.
 */
NPY_NO_EXPORT NPY_ARRAYMETHOD_FLAGS
NpyIter_GetTransferFlags(NpyIter *iter)
{
    return NIT_ITFLAGS(iter) >> NPY_ITFLAG_TRANSFERFLAGS_SHIFT;
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
            int axis = npyiter_undo_iter_axis_perm(idim, ndim, perm, NULL);
            outshape[axis] = NAD_SHAPE(axisdata);

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
        npy_bool flipped;
        npy_int8 axis = npyiter_undo_iter_axis_perm(idim, ndim, perm, &flipped);
        if (flipped) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Iterator CreateCompatibleStrides may only be called "
                    "if DONT_NEGATE_STRIDES was used to prevent reverse "
                    "iteration of an axis");
            return NPY_FAIL;
        }
        else {
            outstrides[axis] = itemsize;
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

    if (itflags&(NPY_ITFLAG_BUFFER|NPY_ITFLAG_EXLOOP)) {
        return NIT_USERPTRS(iter);
    }
    else {
        return NIT_DATAPTRS(iter);
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
    view = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            &PyArray_Type, dtype,
            ndim, shape, strides, dataptr,
            writeable ? NPY_ARRAY_WRITEABLE : 0, NULL, (PyObject *)obj);

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

    if (itflags&NPY_ITFLAG_HASINDEX) {
        /* The index is just after the data pointers */
        return (npy_intp*)(NpyIter_GetDataPtrArray(iter) + nop);
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
 * change during iteration receive the value NPY_MAX_INTP
 * (as of NumPy 2.3, `NPY_MAX_INTP` will never happen but must be supported;
 * we could guarantee this, but not sure if we should).
 * Once the iterator is ready to iterate, call this to get the strides
 * which will always be fixed in the inner loop, then choose optimized
 * inner loop functions which take advantage of those fixed strides.
 *
 * This function may be safely called without holding the Python GIL.
 */
NPY_NO_EXPORT void
NpyIter_GetInnerFixedStrideArray(NpyIter *iter, npy_intp *out_strides)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int nop = NIT_NOP(iter);

    NpyIter_AxisData *axisdata0 = NIT_AXISDATA(iter);

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* If there is buffering we wrote the strides into the bufferdata. */
        memcpy(out_strides, NBF_STRIDES(NIT_BUFFERDATA(iter)), nop*NPY_SIZEOF_INTP);
    }
    else {
        /* If there's no buffering, the strides come from the operands. */
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

    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API

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
    if (itflags&NPY_ITFLAG_REDUCE)
        printf("REDUCE ");
    if (itflags&NPY_ITFLAG_REUSE_REDUCE_LOOPS)
        printf("REUSE_REDUCE_LOOPS ");

    printf("\n");
    printf("| NDim: %d\n", ndim);
    printf("| NOp: %d\n", nop);
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
    printf("| Ptrs: ");
    for (iop = 0; iop < nop; ++iop) {
        printf("%p ", (void *)NIT_DATAPTRS(iter)[iop]);
    }
    printf("\n");
    if (itflags&(NPY_ITFLAG_EXLOOP|NPY_ITFLAG_BUFFER)) {
        printf("| User/buffer ptrs: ");
        for (iop = 0; iop < nop; ++iop) {
            printf("%p ", (void *)NIT_USERPTRS(iter)[iop]);
        }
        printf("\n");
    }
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
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_REDUCE)
            printf("REDUCE ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_VIRTUAL)
            printf("VIRTUAL ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_WRITEMASKED)
            printf("WRITEMASKED ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_BUF_SINGLESTRIDE)
            printf("BUF_SINGLESTRIDE ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_CONTIG)
            printf("CONTIG ");
        printf("\n");
    }
    printf("|\n");

    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);

        printf("| BufferData:\n");
        printf("|   BufferSize: %d\n", (int)NBF_BUFFERSIZE(bufferdata));
        printf("|   Size: %d\n", (int)NBF_SIZE(bufferdata));
        printf("|   BufIterEnd: %d\n", (int)NBF_BUFITEREND(bufferdata));
        printf("|   BUFFER CoreSize: %d\n",
                    (int)NBF_CORESIZE(bufferdata));
        if (itflags&NPY_ITFLAG_REDUCE) {
            printf("|   REDUCE Pos: %d\n",
                        (int)NBF_REDUCE_POS(bufferdata));
            printf("|   REDUCE OuterSize: %d\n",
                        (int)NBF_REDUCE_OUTERSIZE(bufferdata));
            printf("|   REDUCE OuterDim: %d\n",
                        (int)NBF_OUTERDIM(bufferdata));
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
            printf("%p ", (void *)transferinfo[iop].read.func);
        printf("\n");
        printf("|   ReadTransferData: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)transferinfo[iop].read.auxdata);
        printf("\n");
        printf("|   WriteTransferFn: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)transferinfo[iop].write.func);
        printf("\n");
        printf("|   WriteTransferData: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)transferinfo[iop].write.auxdata);
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
        if (itflags&NPY_ITFLAG_HASINDEX) {
            printf("|   Index Value: %d\n",
                               (int)((npy_intp*)NIT_DATAPTRS(iter))[nop]);
        }
    }

    printf("------- END ITERATOR DUMP -------\n");
    fflush(stdout);

    NPY_DISABLE_C_API
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
    NpyIter_AxisData *ad_compress = axisdata;
    npy_intp new_ndim = 1;

    /* The HASMULTIINDEX or IDENTPERM flags do not apply after coalescing */
    NIT_ITFLAGS(iter) &= ~(NPY_ITFLAG_IDENTPERM|NPY_ITFLAG_HASMULTIINDEX);

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
            if (PyDataType_FLAGCHK(op_dtype[iop], NPY_NEEDS_INIT)) {
                memset(buffer, '\0', itemsize*buffersize);
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

    char **dataptrs = NIT_DATAPTRS(iter);

    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    npy_intp istrides, nstrides, i, shape;

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    nstrides = NAD_NSTRIDES();

    NIT_ITERINDEX(iter) = iterindex;

    ndim = ndim ? ndim : 1;

    for (istrides = 0; istrides < nstrides; ++istrides) {
        dataptrs[istrides] = NIT_RESETDATAPTR(iter)[istrides];
    }

    if (iterindex == 0) {
        for (idim = 0; idim < ndim; ++idim) {
            NAD_INDEX(axisdata) = 0;
            NIT_ADVANCE_AXISDATA(axisdata, 1);
        }
    }
    else {
        /*
         * Set the multi-index, from the fastest-changing to the
         * slowest-changing.
         */
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            shape = NAD_SHAPE(axisdata);
            i = iterindex;
            iterindex /= shape;
            NAD_INDEX(axisdata) = i - iterindex * shape;

            npy_intp *strides = NAD_STRIDES(axisdata);
            for (istrides = 0; istrides < nstrides; ++istrides) {
                dataptrs[istrides] += NAD_INDEX(axisdata) * strides[istrides];
            }
        }
    }

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* Find the remainder if chunking to the buffers coresize */
        npy_intp fact = NIT_ITERINDEX(iter) / NIT_BUFFERDATA(iter)->coresize;
        npy_intp offset = NIT_ITERINDEX(iter) - fact * NIT_BUFFERDATA(iter)->coresize;
        NIT_BUFFERDATA(iter)->coreoffset = offset;
    }
    else if (itflags&NPY_ITFLAG_EXLOOP) {
        /* If buffered, user pointers are updated during buffer copy. */
        memcpy(NIT_USERPTRS(iter), dataptrs, nstrides * sizeof(void *));
    }
}


/*
 * This helper fills the bufferdata copy information for an operand.  It
 * is very specific to copy from and to buffers.
 */
static inline void
npyiter_fill_buffercopy_params(
        int nop, int iop, int ndim, npy_uint32 opitflags, npy_intp transfersize,
        NpyIter_BufferData *bufferdata,
        NpyIter_AxisData *axisdata,
        NpyIter_AxisData *outer_axisdata,
        int *ndim_transfer,
        npy_intp *op_transfersize,
        npy_intp *buf_stride,
        npy_intp *op_strides[], npy_intp *op_shape[], npy_intp *op_coords[])
{
    /*
     * Set up if we had to do the full generic copy.
     * NOTE: Except the transfersize itself everything here is fixed
     *       and we could create it once early on.
     */
    *ndim_transfer = ndim;
    *op_transfersize = transfersize;

    if ((opitflags & NPY_OP_ITFLAG_REDUCE) && (NAD_STRIDES(outer_axisdata)[iop] != 0)) {
        /*
         * Reduce with all inner strides ==0 (outer !=0).  We buffer the outer
         * stride which also means buffering only outersize items.
         * (If the outer stride is 0, some inner ones are guaranteed nonzero.)
         */
        assert(NAD_STRIDES(axisdata)[iop] == 0);
        *ndim_transfer = 1;
        *op_transfersize = NBF_REDUCE_OUTERSIZE(bufferdata);
        *buf_stride = NBF_REDUCE_OUTERSTRIDES(bufferdata)[iop];

        *op_shape = op_transfersize;
        assert(**op_coords == 0);  /* initialized by caller currently */
        *op_strides = &NAD_STRIDES(outer_axisdata)[iop];
        return;
    }

    /*
     * The copy is now a typical copy into a contiguous buffer.
     * If it is a reduce, we only copy the inner part (i.e. less).
     * The buffer strides are now always contiguous.
     */
    *buf_stride = NBF_STRIDES(bufferdata)[iop];

    if (opitflags & NPY_OP_ITFLAG_REDUCE) {
        /* Outer dim is reduced, so omit it from copying */
        *ndim_transfer -= 1;
        if (*op_transfersize > bufferdata->coresize) {
            *op_transfersize = bufferdata->coresize;
        }
        /* copy setup is identical to non-reduced now. */
    }

    if (opitflags & NPY_OP_ITFLAG_BUF_SINGLESTRIDE) {
        *ndim_transfer = 1;
        *op_shape = op_transfersize;
        assert(**op_coords == 0);  /* initialized by caller currently */
        *op_strides = &NAD_STRIDES(axisdata)[iop];
        if ((*op_strides)[0] == 0 && (
                !(opitflags & NPY_OP_ITFLAG_CONTIG) ||
                (opitflags & NPY_OP_ITFLAG_WRITE))) {
            /*
             * If the user didn't force contig, optimize single element.
             * (Unless CONTIG was requested and this is not a write/reduce!)
             */
            *op_transfersize = 1;
            *buf_stride = 0;
        }
    }
    else {
        /* We do a full multi-dimensional copy */
        *op_shape = &NAD_SHAPE(axisdata);
        *op_coords = &NAD_INDEX(axisdata);
        *op_strides = &NAD_STRIDES(axisdata)[iop];
    }
}


/*
 * This gets called after the buffers have been exhausted, and
 * their data needs to be written back to the arrays.  The multi-index
 * must be positioned for the beginning of the buffer.
 */
NPY_NO_EXPORT int
npyiter_copy_from_buffers(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);
    int maskop = NIT_MASKOP(iter);

    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    NpyIter_AxisData *outer_axisdata = NULL;

    PyArray_Descr **dtypes = NIT_DTYPES(iter);
    npy_intp *strides = NBF_STRIDES(bufferdata);
    npy_intp transfersize = NBF_SIZE(bufferdata);

    char **dataptrs = NIT_DATAPTRS(iter);
    char **buffers = NBF_BUFFERS(bufferdata);
    char *buffer;

    npy_intp axisdata_incr = NIT_AXISDATA_SIZEOF(itflags, ndim, nop) /
                                NPY_SIZEOF_INTP;

    /* If we're past the end, nothing to copy */
    if (NBF_SIZE(bufferdata) == 0) {
        return 0;
    }

    if (itflags & NPY_ITFLAG_REDUCE) {
        npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
        outer_axisdata = NIT_INDEX_AXISDATA(axisdata, NBF_OUTERDIM(bufferdata));
        transfersize *= NBF_REDUCE_OUTERSIZE(bufferdata);
    }

    NPY_IT_DBG_PRINT("Iterator: Copying buffers to outputs\n");

    NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);
    for (iop = 0; iop < nop; ++iop) {
        if (op_itflags[iop]&NPY_OP_ITFLAG_BUFNEVER) {
            continue;
        }

        /* Currently, we always trash the buffer if there are references */
        if (PyDataType_REFCHK(dtypes[iop])) {
            NIT_OPITFLAGS(iter)[iop] &= ~NPY_OP_ITFLAG_BUF_REUSABLE;
        }

        buffer = buffers[iop];
        /*
         * Copy the data back to the arrays.  If the type has refs,
         * this function moves them so the buffer's refs are released.
         *
         * The flag USINGBUFFER is set when the buffer was used, so
         * only copy back when this flag is on.
         */
        if (transferinfo[iop].write.func != NULL) {
            NPY_IT_DBG_PRINT1("Iterator: Operand %d was buffered\n",
                                        (int)iop);

            npy_intp zero = 0;  /* used as coord for 1-D copies */
            int ndim_transfer;
            npy_intp op_transfersize;
            npy_intp src_stride;
            npy_intp *dst_strides;
            npy_intp *dst_coords = &zero;
            npy_intp *dst_shape;

            npyiter_fill_buffercopy_params(nop, iop, ndim, op_itflags[iop],
                transfersize, bufferdata, axisdata, outer_axisdata,
                &ndim_transfer, &op_transfersize, &src_stride,
                &dst_strides, &dst_shape, &dst_coords);

            NPY_IT_DBG_PRINT(
                "Iterator: Copying buffer to operand %d (%zd items):\n"
                "    transfer ndim: %d, inner stride: %zd, inner shape: %zd, buffer stride: %zd\n",
                iop, op_transfersize, ndim_transfer, dst_strides[0], dst_shape[0], src_stride);

            /* WRITEMASKED operand */
            if (op_itflags[iop] & NPY_OP_ITFLAG_WRITEMASKED) {
                npy_bool *maskptr;

                /*
                 * The mask pointer may be in the buffer or in
                 * the array, detect which one.
                 */
                if ((op_itflags[maskop]&NPY_OP_ITFLAG_BUFNEVER)) {
                    maskptr = (npy_bool *)dataptrs[maskop];
                }
                else {
                    maskptr = (npy_bool *)buffers[maskop];
                }

                if (PyArray_TransferMaskedStridedToNDim(ndim_transfer,
                        dataptrs[iop], dst_strides, axisdata_incr,
                        buffer, src_stride,
                        maskptr, strides[maskop],
                        dst_coords, axisdata_incr,
                        dst_shape, axisdata_incr,
                        op_transfersize, dtypes[iop]->elsize,
                        &transferinfo[iop].write) < 0) {
                    return -1;
                }
            }
            /* Regular operand */
            else {
                if (PyArray_TransferStridedToNDim(ndim_transfer,
                        dataptrs[iop], dst_strides, axisdata_incr,
                        buffer, src_stride,
                        dst_coords, axisdata_incr,
                        dst_shape, axisdata_incr,
                        op_transfersize, dtypes[iop]->elsize,
                        &transferinfo[iop].write) < 0) {
                    return -1;
                }
            }
        }
        /* If there's no copy back, we may have to decrement refs.  In
         * this case, the transfer is instead a function which clears
         * (DECREFs) the single input.
         *
         * The flag USINGBUFFER is set when the buffer was used, so
         * only decrement refs when this flag is on.
         */
        else if (transferinfo[iop].clear.func != NULL) {
            NPY_IT_DBG_PRINT1(
                    "Iterator: clearing refs of operand %d\n", (int)iop);
            npy_intp buf_stride = dtypes[iop]->elsize;
            // TODO: transfersize is too large for reductions
            if (transferinfo[iop].clear.func(
                    NULL, transferinfo[iop].clear.descr, buffer, transfersize,
                    buf_stride, transferinfo[iop].clear.auxdata) < 0) {
                /* Since this should only decrement, it should never error */
                assert(0);
                return -1;
            }
        }
    }

    NPY_IT_DBG_PRINT("Iterator: Finished copying buffers to outputs\n");
    return 0;
}

/*
 * This gets called after the iterator has been positioned to a multi-index
 * for the start of a buffer.  It decides which operands need a buffer,
 * and copies the data into the buffers.
 *
 * If passed, this function expects `prev_dataptrs` to be `NIT_USERPTRS`
 * (they are reset after querying `prev_dataptrs`).
 */
NPY_NO_EXPORT int
npyiter_copy_to_buffers(NpyIter *iter, char **prev_dataptrs)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    NpyIter_AxisData *outer_axisdata = NULL;

    PyArrayObject **operands = NIT_OPERANDS(iter);

    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    char **user_ptrs = NIT_USERPTRS(iter), **dataptrs = NIT_DATAPTRS(iter);
    char **buffers = NBF_BUFFERS(bufferdata);
    npy_intp iterindex, iterend, transfersize;

    npy_intp axisdata_incr = NIT_AXISDATA_SIZEOF(itflags, ndim, nop) /
                                NPY_SIZEOF_INTP;

    /* Fetch the maximum size we may wish to copy (or use if unbuffered) */
    iterindex = NIT_ITERINDEX(iter);
    iterend = NIT_ITEREND(iter);
    transfersize = NBF_BUFFERSIZE(bufferdata);
    outer_axisdata = NIT_INDEX_AXISDATA(axisdata, bufferdata->outerdim);
    npy_intp remaining_outersize = (
            outer_axisdata->shape - outer_axisdata->index);

    NPY_IT_DBG_PRINT("Iterator: Copying inputs to buffers\n");
    NPY_IT_DBG_PRINT("    Max transfersize=%zd, coresize=%zd\n",
                     transfersize, bufferdata->coresize);

    /*
     * If there is a coreoffset just copy to the end of a single coresize
     * NB: Also if the size is shrunk, we definitely won't set buffer re-use.
     */
    if (bufferdata->coreoffset) {
        prev_dataptrs = NULL;  /* No way we can re-use the buffers safely. */
        transfersize = bufferdata->coresize - bufferdata->coreoffset;
        NPY_IT_DBG_PRINT("    Shrunk transfersize due to coreoffset=%zd: %zd\n",
                         bufferdata->coreoffset, transfersize);
    }
    else if (transfersize > bufferdata->coresize * remaining_outersize) {
        /*
         * Shrink transfersize to not go beyond outer axis size.  If not
         * a reduction, it is unclear that this is necessary.
         */
        transfersize = bufferdata->coresize * remaining_outersize;
        NPY_IT_DBG_PRINT("    Shrunk transfersize outer size: %zd\n", transfersize);
    }

    /* And ensure that we don't go beyond the iterator end (if ranged) */
    if (transfersize > iterend - iterindex) {
        transfersize = iterend - iterindex;
        NPY_IT_DBG_PRINT("    Shrunk transfersize to itersize: %zd\n", transfersize);
    }

    bufferdata->size = transfersize;
    NBF_BUFITEREND(bufferdata) = iterindex + transfersize;

    if (transfersize == 0) {
        return 0;
    }

    NPY_IT_DBG_PRINT("Iterator: Buffer transfersize=%zd\n", transfersize);

    if (itflags & NPY_ITFLAG_REDUCE) {
        NBF_REDUCE_OUTERSIZE(bufferdata) = transfersize / bufferdata->coresize;
        if (NBF_REDUCE_OUTERSIZE(bufferdata) > 1) {
            /* WARNING: bufferdata->size does not include reduce-outersize */
            bufferdata->size = bufferdata->coresize;
            NBF_BUFITEREND(bufferdata) = iterindex + bufferdata->coresize;
        }
        NBF_REDUCE_POS(bufferdata) = 0;
    }

    NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);
    for (iop = 0; iop < nop; ++iop) {
        NPY_IT_DBG_PRINT("Iterator: buffer prep for op=%d @ %p inner-stride=%zd\n",
                         iop, dataptrs[iop], NBF_STRIDES(bufferdata)[iop]);

        if (op_itflags[iop]&NPY_OP_ITFLAG_BUFNEVER) {
            user_ptrs[iop] = dataptrs[iop];
            NBF_REDUCE_OUTERPTRS(bufferdata)[iop] = dataptrs[iop];
            NPY_IT_DBG_PRINT("    unbuffered op (skipping)\n");
            continue;
        }

        /*
         * We may be able to reuse buffers if the pointer is unchanged and
         * there is no coreoffset (which sets `prev_dataptrs = NULL` above).
         * We re-use `user_ptrs` for `prev_dataptrs` to simplify `iternext()`.
         */
        assert(prev_dataptrs == NULL || prev_dataptrs == user_ptrs);
        int unchanged_ptr_and_no_offset = (
            prev_dataptrs != NULL && prev_dataptrs[iop] == dataptrs[iop]);

        user_ptrs[iop] = buffers[iop];
        NBF_REDUCE_OUTERPTRS(bufferdata)[iop] = buffers[iop];

        if (!(op_itflags[iop]&NPY_OP_ITFLAG_READ)) {
            NPY_IT_DBG_PRINT("    non-reading op (skipping)\n");
            continue;
        }

        if (unchanged_ptr_and_no_offset && op_itflags[iop]&NPY_OP_ITFLAG_BUF_REUSABLE) {
            NPY_IT_DBG_PRINT2("Iterator: skipping operands %d "
                    "copy (%d items) because the data pointer didn't change\n",
                    (int)iop, (int)transfersize);
            continue;
        }
        else if (transfersize == NBF_BUFFERSIZE(bufferdata)
                    || (transfersize >= NBF_CORESIZE(bufferdata)
                        && op_itflags[iop]&NPY_OP_ITFLAG_REDUCE
                        && NAD_STRIDES(outer_axisdata)[iop] == 0)) {
            /*
             * If we have a full copy or a reduce with 0 stride outer and
             * a copy larger than the coresize, this is now re-usable.
             * NB: With a core-offset, we always copy less than the core-size.
             */
            NPY_IT_DBG_PRINT("    marking operand %d for buffer reuse\n", iop);
            NIT_OPITFLAGS(iter)[iop] |= NPY_OP_ITFLAG_BUF_REUSABLE;
        }
        else {
            NPY_IT_DBG_PRINT("    marking operand %d as not reusable\n", iop);
            NIT_OPITFLAGS(iter)[iop] &= ~NPY_OP_ITFLAG_BUF_REUSABLE;
        }

        npy_intp zero = 0;  /* used as coord for 1-D copies */
        int ndim_transfer;
        npy_intp op_transfersize;
        npy_intp dst_stride;
        npy_intp *src_strides;
        npy_intp *src_coords = &zero;
        npy_intp *src_shape;
        npy_intp src_itemsize = PyArray_DTYPE(operands[iop])->elsize;

        npyiter_fill_buffercopy_params(nop, iop, ndim, op_itflags[iop],
            transfersize, bufferdata, axisdata, outer_axisdata,
            &ndim_transfer, &op_transfersize, &dst_stride,
            &src_strides, &src_shape, &src_coords);

        /*
         * Copy data to the buffers if necessary.
         *
         * We always copy if the operand has references. In that case
         * a "write" function must be in use that either copies or clears
         * the buffer.
         * This write from buffer call does not check for skip-transfer
         * so we have to assume the buffer is cleared.  For dtypes that
         * do not have references, we can assume that the write function
         * will leave the source (buffer) unmodified.
         */
        NPY_IT_DBG_PRINT(
            "Iterator: Copying operand %d to buffer (%zd items):\n"
            "    transfer ndim: %d, inner stride: %zd, inner shape: %zd, buffer stride: %zd\n",
            iop, op_transfersize, ndim_transfer, src_strides[0], src_shape[0], dst_stride);

        if (PyArray_TransferNDimToStrided(
                ndim_transfer, buffers[iop], dst_stride,
                dataptrs[iop], src_strides, axisdata_incr,
                src_coords, axisdata_incr,
                src_shape, axisdata_incr,
                op_transfersize, src_itemsize,
                &transferinfo[iop].read) < 0) {
            return -1;
        }
    }

    NPY_IT_DBG_PRINT1("Iterator: Finished copying inputs to buffers "
                        "(buffered size is %zd)\n", transfersize);
    return 0;
}


/**
 * This function clears any references still held by the buffers and should
 * only be used to discard buffers if an error occurred.
 *
 * @param iter Iterator
 */
NPY_NO_EXPORT void
npyiter_clear_buffers(NpyIter *iter)
{
    int nop = iter->nop;
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);

    if (NBF_SIZE(bufferdata) == 0) {
        /* if the buffers are empty already, there is nothing to do */
        return;
    }

    /*
     * The iterator may be using a dtype with references (as of writing this
     * means Python objects, but does not need to stay that way).
     * In that case, further cleanup may be necessary and we clear buffers
     * explicitly.
     */
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type,  &value, &traceback);

    /* Cleanup any buffers with references */
    char **buffers = NBF_BUFFERS(bufferdata);

    NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);
    PyArray_Descr **dtypes = NIT_DTYPES(iter);
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    for (int iop = 0; iop < nop; ++iop, ++buffers) {
        if (transferinfo[iop].clear.func == NULL) {
            continue;
        }
        if (*buffers == 0) {
            continue;
        }
        assert(!(op_itflags[iop]&NPY_OP_ITFLAG_BUFNEVER));
        /* Buffer cannot be re-used (not that we should ever try!) */
        op_itflags[iop] &= ~NPY_OP_ITFLAG_BUF_REUSABLE;

        int itemsize = dtypes[iop]->elsize;
        if (transferinfo[iop].clear.func(NULL,
                dtypes[iop], *buffers, NBF_SIZE(bufferdata), itemsize,
                transferinfo[iop].clear.auxdata) < 0) {
            /* This should never fail; if it does write it out */
            PyErr_WriteUnraisable(NULL);
        }
    }
    /* Signal that the buffers are empty */
    NBF_SIZE(bufferdata) = 0;
    PyErr_Restore(type, value, traceback);
}


NPY_NO_EXPORT npy_bool
npyiter_has_writeback(NpyIter *iter)
{
    int iop, nop;
    npyiter_opitflags *op_itflags;
    if (iter == NULL) {
        return 0;
    }
    nop = NIT_NOP(iter);
    op_itflags = NIT_OPITFLAGS(iter);

    for (iop=0; iop<nop; iop++) {
        if (op_itflags[iop] & NPY_OP_ITFLAG_HAS_WRITEBACK) {
            return NPY_TRUE;
        }
    }
    return NPY_FALSE;
}
#undef NPY_ITERATOR_IMPLEMENTATION_CODE
