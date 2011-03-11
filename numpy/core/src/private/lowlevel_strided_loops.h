#ifndef __LOWLEVEL_STRIDED_LOOPS_H
#define __LOWLEVEL_STRIDED_LOOPS_H

/*
 * NOTE: This API should remain private for the time being, to allow
 *       for further refinement.  I think the 'aligned' mechanism
 *       needs changing, for example.
 */

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
typedef void (PyArray_StridedTransferFn)(char *dst, npy_intp dst_stride,
                                    char *src, npy_intp src_stride,
                                    npy_intp N, npy_intp src_itemsize,
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
NPY_NO_EXPORT PyArray_StridedTransferFn *
PyArray_GetStridedCopyFn(npy_intp aligned, npy_intp src_stride,
                         npy_intp dst_stride, npy_intp itemsize);

/*
 * Gives back a function pointer to a specialized function for copying
 * and swapping strided memory.  This assumes each element is a single
 * value to be swapped.
 *
 * For information on the 'aligned', 'src_stride' and 'dst_stride' parameters
 * see above.
 *
 * Parameters are as for PyArray_GetStridedCopyFn.
 */
NPY_NO_EXPORT PyArray_StridedTransferFn *
PyArray_GetStridedCopySwapFn(npy_intp aligned, npy_intp src_stride,
                             npy_intp dst_stride, npy_intp itemsize);

/*
 * Gives back a function pointer to a specialized function for copying
 * and swapping strided memory.  This assumes each element is a pair
 * of values, each of which needs to be swapped.
 *
 * For information on the 'aligned', 'src_stride' and 'dst_stride' parameters
 * see above.
 *
 * Parameters are as for PyArray_GetStridedCopyFn.
 */
NPY_NO_EXPORT PyArray_StridedTransferFn *
PyArray_GetStridedCopySwapPairFn(npy_intp aligned, npy_intp src_stride,
                             npy_intp dst_stride, npy_intp itemsize);

/*
 * Gives back a transfer function and transfer data pair which copies
 * the data from source to dest, truncating it if the data doesn't
 * fit, and padding with zero bytes if there's too much space.
 *
 * For information on the 'aligned', 'src_stride' and 'dst_stride' parameters
 * see above.
 *
 * Returns NPY_SUCCEED or NPY_FAIL
 */
NPY_NO_EXPORT int
PyArray_GetStridedZeroPadCopyFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp src_itemsize, npy_intp dst_itemsize,
                            PyArray_StridedTransferFn **outstransfer,
                            void **outtransferdata);

/*
 * For casts between built-in numeric types,
 * this produces a function pointer for casting from src_type_num
 * to dst_type_num.  If a conversion is unsupported, returns NULL
 * without setting a Python exception.
 */
NPY_NO_EXPORT PyArray_StridedTransferFn *
PyArray_GetStridedNumericCastFn(npy_intp aligned, npy_intp src_stride,
                             npy_intp dst_stride,
                             int src_type_num, int dst_type_num);

/*
 * If it's possible, gives back a transfer function which casts and/or
 * byte swaps data with the dtype 'src_dtype' into data with the dtype
 * 'dst_dtype'.  If the outtransferdata is populated with a non-NULL value,
 * it must be deallocated with the ``PyArray_FreeStridedTransferData``
 * function when the transfer function is no longer required.
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
 * src_dtype:
 *      The data type of source data.  If this is NULL, a transfer
 *      function which sets the destination to zeros is produced.
 * dst_dtype:
 *      The data type of destination data.  If this is NULL and
 *      move_references is 1, a transfer function which decrements
 *      source data references is produced.
 * move_references:
 *      If 0, the destination data gets new reference ownership.
 *      If 1, the references from the source data are moved to
 *      the destination data.
 * out_stransfer:
 *      The resulting transfer function is placed here.
 * out_transferdata:
 *      The auxiliary data for the transfer function is placed here.
 *      When finished with the transfer function, the caller must call
 *      ``PyArray_FreeStridedTransferData`` on this data.
 * out_needs_api:
 *      If this is non-NULL, and the transfer function produced needs
 *      to call into the (Python) API, this gets set to 1.  This
 *      remains untouched if no API access is required.
 *
 * WARNING: If you set move_references to 1, it is best that src_stride is
 *          never zero when calling the transfer function.  Otherwise, the
 *          first destination reference will get the value and all the rest
 *          will get NULL.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
PyArray_GetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedTransferFn **out_stransfer,
                            void **out_transferdata,
                            int *out_needs_api);

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
 * src_itemsize:
 *      How big each element is.  If transfering between elements of different
 *      sizes, for example a casting operation, the 'stransfer' function
 *      should be specialized for that, in which case 'stransfer' will use
 *      this parameter as the source item size.
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
                npy_intp count, npy_intp src_itemsize,
                PyArray_StridedTransferFn *stransfer,
                void *transferdata);

NPY_NO_EXPORT npy_intp
PyArray_TransferStridedToNDim(npy_intp ndim,
                char *dst, npy_intp *dst_strides, npy_intp dst_strides_inc,
                char *src, npy_intp src_stride,
                npy_intp *coords, npy_intp coords_inc,
                npy_intp *shape, npy_intp shape_inc,
                npy_intp count, npy_intp src_itemsize,
                PyArray_StridedTransferFn *stransfer,
                void *transferdata);

/*
 *            TRIVIAL ITERATION
 *
 * In some cases when the iteration order isn't important, iteration over
 * arrays is trivial.  This is the case when:
 *   * The array has 0 or 1 dimensions.
 *   * The array is C or Fortran contiguous.
 * Use of an iterator can be skipped when this occurs.  These macros assist
 * in detecting and taking advantage of the situation.  Note that it may
 * be worthwhile to further check if the stride is a contiguous stride
 * and take advantage of that.
 *
 * Here is example code for a single array:
 *
 *      if (PyArray_TRIVIALLY_ITERABLE(self) {
 *          char *data;
 *          npy_intp count, stride;
 *
 *          PyArray_PREPARE_TRIVIAL_ITERATION(self, count, data, stride);
 *
 *          while (count--) {
 *              // Use the data pointer
 *
 *              data += stride;
 *          }
 *      }
 *      else {
 *          // Create iterator, etc...
 *      }
 *
 * Here is example code for a pair of arrays:
 *
 *      if (PyArray_TRIVIALLY_ITERABLE_PAIR(a1, a2) {
 *          char *data1, *data2;
 *          npy_intp count, stride1, stride2;
 *
 *          PyArray_PREPARE_TRIVIAL_PAIR_ITERATION(a1, a2, count,
 *                                  data1, data2, stride1, stride2);
 *
 *          while (count--) {
 *              // Use the data1 and data2 pointers
 *
 *              data1 += stride1;
 *              data2 += stride2;
 *          }
 *      }
 *      else {
 *          // Create iterator, etc...
 *      }
 */

/*
 * Note: Equivalently iterable macro requires one of arr1 or arr2 be
 *       trivially iterable to be valid.
 */
#define PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2) ( \
                        PyArray_NDIM(arr1) == PyArray_NDIM(arr2) && \
                        PyArray_CompareLists(PyArray_DIMS(arr1), \
                                             PyArray_DIMS(arr2), \
                                             PyArray_NDIM(arr1)) && \
                        (arr1->flags&(NPY_CONTIGUOUS|NPY_FORTRAN)) == \
                                (arr2->flags&(NPY_CONTIGUOUS|NPY_FORTRAN)) \
                        )

#define PyArray_TRIVIALLY_ITERABLE(arr) ( \
                    PyArray_NDIM(arr) <= 1 || \
                    PyArray_CHKFLAGS(arr, NPY_CONTIGUOUS) || \
                    PyArray_CHKFLAGS(arr, NPY_FORTRAN) \
                    )
#define PyArray_PREPARE_TRIVIAL_ITERATION(arr, count, data, stride) \
                    count = PyArray_SIZE(arr), \
                    data = PyArray_BYTES(arr), \
                    stride = ((PyArray_NDIM(arr) == 0) ? 0 : \
                                (PyArray_CHKFLAGS(arr, NPY_FORTRAN) ? \
                                            PyArray_STRIDE(arr, 0) : \
                                            PyArray_STRIDE(arr, \
                                                PyArray_NDIM(arr)-1)))

#define PyArray_TRIVIALLY_ITERABLE_PAIR(arr1, arr2) (\
                    PyArray_TRIVIALLY_ITERABLE(arr1) && \
                        (PyArray_NDIM(arr2) == 0 || \
                         PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2) || \
                         (PyArray_NDIM(arr1) == 0 && \
                             PyArray_TRIVIALLY_ITERABLE(arr2) \
                         ) \
                        ) \
                    )
#define PyArray_PREPARE_TRIVIAL_PAIR_ITERATION(arr1, arr2, \
                                        count, \
                                        data1, data2, \
                                        stride1, stride2) { \
                    npy_intp size1 = PyArray_SIZE(arr1); \
                    npy_intp size2 = PyArray_SIZE(arr2); \
                    count = ((size1 > size2) || size1 == 0) ? size1 : size2; \
                    data1 = PyArray_BYTES(arr1); \
                    data2 = PyArray_BYTES(arr2); \
                    stride1 = (size1 == 1 ? 0 : \
                                (PyArray_CHKFLAGS(arr1, NPY_FORTRAN) ? \
                                            PyArray_STRIDE(arr1, 0) : \
                                            PyArray_STRIDE(arr1, \
                                                PyArray_NDIM(arr1)-1))); \
                    stride2 = (size2 == 1 ? 0 : \
                                (PyArray_CHKFLAGS(arr2, NPY_FORTRAN) ? \
                                            PyArray_STRIDE(arr2, 0) : \
                                            PyArray_STRIDE(arr2, \
                                                PyArray_NDIM(arr2)-1))); \
                }

#define PyArray_TRIVIALLY_ITERABLE_TRIPLE(arr1, arr2, arr3) (\
                PyArray_TRIVIALLY_ITERABLE(arr1) && \
                    ((PyArray_NDIM(arr2) == 0 && \
                        (PyArray_NDIM(arr3) == 0 || \
                            PyArray_EQUIVALENTLY_ITERABLE(arr1, arr3) \
                        ) \
                     ) || \
                     (PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2) && \
                        (PyArray_NDIM(arr3) == 0 || \
                            PyArray_EQUIVALENTLY_ITERABLE(arr1, arr3) \
                        ) \
                     ) || \
                     (PyArray_NDIM(arr1) == 0 && \
                        PyArray_TRIVIALLY_ITERABLE(arr2) && \
                            (PyArray_NDIM(arr3) == 0 || \
                                PyArray_EQUIVALENTLY_ITERABLE(arr2, arr3) \
                            ) \
                     ) \
                    ) \
                )

#define PyArray_PREPARE_TRIVIAL_TRIPLE_ITERATION(arr1, arr2, arr3, \
                                        count, \
                                        data1, data2, data3, \
                                        stride1, stride2, stride3) { \
                    npy_intp size1 = PyArray_SIZE(arr1); \
                    npy_intp size2 = PyArray_SIZE(arr2); \
                    npy_intp size3 = PyArray_SIZE(arr3); \
                    count = ((size1 > size2) || size1 == 0) ? size1 : size2; \
                    count = ((size3 > count) || size3 == 0) ? size3 : count; \
                    data1 = PyArray_BYTES(arr1); \
                    data2 = PyArray_BYTES(arr2); \
                    data3 = PyArray_BYTES(arr3); \
                    stride1 = (size1 == 1 ? 0 : \
                                (PyArray_CHKFLAGS(arr1, NPY_FORTRAN) ? \
                                            PyArray_STRIDE(arr1, 0) : \
                                            PyArray_STRIDE(arr1, \
                                                PyArray_NDIM(arr1)-1))); \
                    stride2 = (size2 == 1 ? 0 : \
                                (PyArray_CHKFLAGS(arr2, NPY_FORTRAN) ? \
                                            PyArray_STRIDE(arr2, 0) : \
                                            PyArray_STRIDE(arr2, \
                                                PyArray_NDIM(arr2)-1))); \
                    stride3 = (size3 == 1 ? 0 : \
                                (PyArray_CHKFLAGS(arr3, NPY_FORTRAN) ? \
                                            PyArray_STRIDE(arr3, 0) : \
                                            PyArray_STRIDE(arr3, \
                                                PyArray_NDIM(arr3)-1))); \
                }

#endif
