#ifndef __LOWLEVEL_STRIDED_LOOPS_H
#define __LOWLEVEL_STRIDED_LOOPS_H

/*
 * This function pointer is for functions that transfer an arbitrarily strided
 * input to a an arbitrarily strided output.  It may be a fully general
 * function, or a specialized function when the strides or item size
 * have special values.
 *
 * Examples of transfer functions are a straight copy, a byte-swap,
 * and a casting operation,
 *
 * The 'transferdata' parameter is slightly special, and must always contain
 * pointer to deallocation and copying routines at its beginning.  The function
 * PyArray_FreeStridedTransferData should be used to deallocate such
 * pointers, and calls the first function pointer, while the function
 * PyArray_CopyStridedTransferData should be used to copy it.
 * 
 */
typedef void (*PyArray_StridedTransferFn)(char *dst, npy_intp dst_stride,
                                    char *src, npy_intp src_stride,
                                    npy_intp N, npy_intp itemsize,
                                    void *transferdata);

/*
 * Deallocates a PyArray_StridedTransferFunction data object.  See
 * the comment with the function typedef for more details.
 */
NPY_NO_EXPORT void
PyArray_FreeStridedTransferData(void *transferdata);

/*
 * Copies a PyArray_StridedTransferFunction data object.  See
 * the comment with the function typedef for more details.
 */
NPY_NO_EXPORT void *
PyArray_CopyStridedTransferData(void *transferdata);

/*
 * Gives back a function pointer to a specialized function for copying
 * strided memory.  Returns NULL if there is a problem with the inputs.
 *
 * aligned:
 *      Should be 1 if the src and dst pointers are always aligned,
 *      0 otherwise.
 * src_stride:  
 *      Should be the src stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * dst_stride:  
 *      Should be the dst stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * itemsize:
 *      Should be the item size if it will always be the same, 0 otherwise.
 *
 */
NPY_NO_EXPORT PyArray_StridedTransferFn
PyArray_GetStridedCopyFn(npy_intp aligned, npy_intp src_stride,
                         npy_intp dst_stride, npy_intp itemsize);

/*
 * Gives back a function pointer to a specialized function for copying
 * and swapping strided memory.  This assumes each element is a single
 * value to be swapped.
 *
 * Parameters are as for PyArray_GetStridedCopyFn.
 */
NPY_NO_EXPORT PyArray_StridedTransferFn
PyArray_GetStridedCopySwapFn(npy_intp aligned, npy_intp src_stride,
                             npy_intp dst_stride, npy_intp itemsize);

/*
 * Gives back a function pointer to a specialized function for copying
 * and swapping strided memory.  This assumes each element is a pair
 * of values, each of which needs to be swapped.
 *
 * Parameters are as for PyArray_GetStridedCopyFn.
 */
NPY_NO_EXPORT PyArray_StridedTransferFn
PyArray_GetStridedCopySwapPairFn(npy_intp aligned, npy_intp src_stride,
                             npy_intp dst_stride, npy_intp itemsize);

/*
 * If it's possible, gives back a transfer function which casts and/or
 * byte swaps data with the dtype 'from' into data with the dtype 'to'.
 * If the outtransferdata is populated with a non-NULL value, it
 * must be deallocated with the ``PyArray_FreeStridedTransferData``
 * function when the transfer function is no longer required.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
PyArray_GetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *from, PyArray_Descr *to,
                            PyArray_StridedTransferFn *outstransfer,
                            void **outtransferdata);

/*
 * These two functions copy or convert the data of an n-dimensional array
 * to/from a 1-dimensional strided buffer.  These functions will only call
 * 'stransfer' with the provided dst_stride/src_stride and
 * dst_strides[0]/src_strides[0], so the caller can use those values to
 * specialize the function.
 *
 * The return value is the number of elements it couldn't copy.  A return value
 * of 0 means all elements were copied, a larger value means the end of
 * the n-dimensional array was reached before 'count' elements were copied.
 *
 * ndim:
 *      The number of dimensions of the n-dimensional array.
 * dst/src:
 *      The destination or src starting pointer.
 * dst_stride/src_stride:
 *      The stride of the 1-dimensional strided buffer
 * dst_strides/src_strides:
 *      The strides of the n-dimensional array.
 * dst_strides_inc/src_strides_inc:
 *      How much to add to the ..._strides pointer to get to the next stride.
 * coords:
 *      The starting coordinates in the n-dimensional array.
 * coords_inc:
 *      How much to add to the coords pointer to get to the next coordinate.
 * shape:
 *      The shape of the n-dimensional array.
 * shape_inc:
 *      How much to add to the shape pointer to get to the next shape entry.
 * count:
 *      How many elements to transfer
 * itemsize:
 *      How big each element is.  If transfering between elements of different
 *      sizes, for example a casting operation, the 'stransfer' function
 *      should be specialized for that, in which case 'stransfer' will ignore
 *      this parameter.
 * stransfer:
 *      The strided transfer function.
 * transferdata:
 *      An auxiliary data pointer passed to the strided transfer function.
 *      If a non-NULL value is returned, it must be deallocated with the
 *      function PyArray_FreeStridedTransferData.
 */
NPY_NO_EXPORT npy_intp
PyArray_TransferNDimToStrided(npy_intp ndim,
                char *dst, npy_intp dst_stride,
                char *src, npy_intp *src_strides, npy_intp src_strides_inc,
                npy_intp *coords, npy_intp coords_inc,
                npy_intp *shape, npy_intp shape_inc,
                npy_intp count, npy_intp itemsize,
                PyArray_StridedTransferFn stransfer,
                void *transferdata);

NPY_NO_EXPORT npy_intp
PyArray_TransferStridedToNDim(npy_intp ndim,
                char *dst, npy_intp *dst_strides, npy_intp dst_strides_inc,
                char *src, npy_intp src_stride,
                npy_intp *coords, npy_intp coords_inc,
                npy_intp *shape, npy_intp shape_inc,
                npy_intp count, npy_intp itemsize,
                PyArray_StridedTransferFn stransfer,
                void *transferdata);

#endif
