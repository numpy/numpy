#ifndef __LOWLEVEL_STRIDED_LOOPS_H
#define __LOWLEVEL_STRIDED_LOOPS_H
#include "common.h"
#include <npy_config.h>
#include "mem_overlap.h"

/* For PyArray_ macros used below */
#include "numpy/ndarrayobject.h"

/*
 * NOTE: This API should remain private for the time being, to allow
 *       for further refinement.  I think the 'aligned' mechanism
 *       needs changing, for example. 
 *
 *       Note: Updated in 2018 to distinguish "true" from "uint" alignment.
 */

/*
 * This function pointer is for unary operations that input an
 * arbitrarily strided one-dimensional array segment and output
 * an arbitrarily strided array segment of the same size.
 * It may be a fully general function, or a specialized function
 * when the strides or item size have particular known values.
 *
 * Examples of unary operations are a straight copy, a byte-swap,
 * and a casting operation,
 *
 * The 'transferdata' parameter is slightly special, following a
 * generic auxiliary data pattern defined in ndarraytypes.h
 * Use NPY_AUXDATA_CLONE and NPY_AUXDATA_FREE to deal with this data.
 *
 */
typedef void (PyArray_StridedUnaryOp)(char *dst, npy_intp dst_stride,
                                    char *src, npy_intp src_stride,
                                    npy_intp N, npy_intp src_itemsize,
                                    NpyAuxData *transferdata);

/*
 * This is for pointers to functions which behave exactly as
 * for PyArray_StridedUnaryOp, but with an additional mask controlling
 * which values are transformed.
 *
 * In particular, the 'i'-th element is operated on if and only if
 * mask[i*mask_stride] is true.
 */
typedef void (PyArray_MaskedStridedUnaryOp)(char *dst, npy_intp dst_stride,
                                    char *src, npy_intp src_stride,
                                    npy_bool *mask, npy_intp mask_stride,
                                    npy_intp N, npy_intp src_itemsize,
                                    NpyAuxData *transferdata);

/*
 * This function pointer is for binary operations that input two
 * arbitrarily strided one-dimensional array segments and output
 * an arbitrarily strided array segment of the same size.
 * It may be a fully general function, or a specialized function
 * when the strides or item size have particular known values.
 *
 * Examples of binary operations are the basic arithmetic operations,
 * logical operators AND, OR, and many others.
 *
 * The 'transferdata' parameter is slightly special, following a
 * generic auxiliary data pattern defined in ndarraytypes.h
 * Use NPY_AUXDATA_CLONE and NPY_AUXDATA_FREE to deal with this data.
 *
 */
typedef void (PyArray_StridedBinaryOp)(char *dst, npy_intp dst_stride,
                                    char *src0, npy_intp src0_stride,
                                    char *src1, npy_intp src1_stride,
                                    npy_intp N, NpyAuxData *transferdata);

/*
 * Gives back a function pointer to a specialized function for copying
 * strided memory.  Returns NULL if there is a problem with the inputs.
 *
 * aligned:
 *      Should be 1 if the src and dst pointers always point to
 *      locations at which a uint of equal size to dtype->elsize
 *      would be aligned, 0 otherwise.
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
NPY_NO_EXPORT PyArray_StridedUnaryOp *
PyArray_GetStridedCopyFn(int aligned,
                        npy_intp src_stride, npy_intp dst_stride,
                        npy_intp itemsize);

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
NPY_NO_EXPORT PyArray_StridedUnaryOp *
PyArray_GetStridedCopySwapFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp itemsize);

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
NPY_NO_EXPORT PyArray_StridedUnaryOp *
PyArray_GetStridedCopySwapPairFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp itemsize);

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
PyArray_GetStridedZeroPadCopyFn(int aligned, int unicode_swap,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp src_itemsize, npy_intp dst_itemsize,
                            PyArray_StridedUnaryOp **outstransfer,
                            NpyAuxData **outtransferdata);

/*
 * For casts between built-in numeric types,
 * this produces a function pointer for casting from src_type_num
 * to dst_type_num.  If a conversion is unsupported, returns NULL
 * without setting a Python exception.
 */
NPY_NO_EXPORT PyArray_StridedUnaryOp *
PyArray_GetStridedNumericCastFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            int src_type_num, int dst_type_num);

/*
 * Gets an operation which copies elements of the given dtype,
 * swapping if the dtype isn't in NBO.
 *
 * Returns NPY_SUCCEED or NPY_FAIL
 */
NPY_NO_EXPORT int
PyArray_GetDTypeCopySwapFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *dtype,
                            PyArray_StridedUnaryOp **outstransfer,
                            NpyAuxData **outtransferdata);

/*
 * If it's possible, gives back a transfer function which casts and/or
 * byte swaps data with the dtype 'src_dtype' into data with the dtype
 * 'dst_dtype'.  If the outtransferdata is populated with a non-NULL value,
 * it must be deallocated with the NPY_AUXDATA_FREE
 * function when the transfer function is no longer required.
 *
 * aligned:
 *      Should be 1 if the src and dst pointers always point to
 *      locations at which a uint of equal size to dtype->elsize
 *      would be aligned, 0 otherwise.
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
 *      NPY_AUXDATA_FREE on this data.
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
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api);

/*
 * This is identical to PyArray_GetDTypeTransferFunction, but returns a
 * transfer function which also takes a mask as a parameter.  The mask is used
 * to determine which values to copy, and data is transferred exactly when
 * mask[i*mask_stride] is true.
 *
 * If move_references is true, values which are not copied to the
 * destination will still have their source reference decremented.
 *
 * If mask_dtype is NPY_BOOL or NPY_UINT8, each full element is either
 * transferred or not according to the mask as described above. If
 * dst_dtype and mask_dtype are both struct dtypes, their names must
 * match exactly, and the dtype of each leaf field in mask_dtype must
 * be either NPY_BOOL or NPY_UINT8.
 */
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
                            int *out_needs_api);

/*
 * Casts the specified number of elements from 'src' with data type
 * 'src_dtype' to 'dst' with 'dst_dtype'. See
 * PyArray_GetDTypeTransferFunction for more details.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
PyArray_CastRawArrays(npy_intp count,
                      char *src, char *dst,
                      npy_intp src_stride, npy_intp dst_stride,
                      PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                      int move_references);

/*
 * These two functions copy or convert the data of an n-dimensional array
 * to/from a 1-dimensional strided buffer.  These functions will only call
 * 'stransfer' with the provided dst_stride/src_stride and
 * dst_strides[0]/src_strides[0], so the caller can use those values to
 * specialize the function.
 * Note that even if ndim == 0, everything needs to be set as if ndim == 1.
 *
 * The return value is the number of elements it couldn't copy.  A return value
 * of 0 means all elements were copied, a larger value means the end of
 * the n-dimensional array was reached before 'count' elements were copied.
 *
 * ndim:
 *      The number of dimensions of the n-dimensional array.
 * dst/src/mask:
 *      The destination, source or mask starting pointer.
 * dst_stride/src_stride/mask_stride:
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
 *      How big each element is.  If transferring between elements of different
 *      sizes, for example a casting operation, the 'stransfer' function
 *      should be specialized for that, in which case 'stransfer' will use
 *      this parameter as the source item size.
 * stransfer:
 *      The strided transfer function.
 * transferdata:
 *      An auxiliary data pointer passed to the strided transfer function.
 *      This follows the conventions of NpyAuxData objects.
 */
NPY_NO_EXPORT npy_intp
PyArray_TransferNDimToStrided(npy_intp ndim,
                char *dst, npy_intp dst_stride,
                char *src, npy_intp *src_strides, npy_intp src_strides_inc,
                npy_intp *coords, npy_intp coords_inc,
                npy_intp *shape, npy_intp shape_inc,
                npy_intp count, npy_intp src_itemsize,
                PyArray_StridedUnaryOp *stransfer,
                NpyAuxData *transferdata);

NPY_NO_EXPORT npy_intp
PyArray_TransferStridedToNDim(npy_intp ndim,
                char *dst, npy_intp *dst_strides, npy_intp dst_strides_inc,
                char *src, npy_intp src_stride,
                npy_intp *coords, npy_intp coords_inc,
                npy_intp *shape, npy_intp shape_inc,
                npy_intp count, npy_intp src_itemsize,
                PyArray_StridedUnaryOp *stransfer,
                NpyAuxData *transferdata);

NPY_NO_EXPORT npy_intp
PyArray_TransferMaskedStridedToNDim(npy_intp ndim,
                char *dst, npy_intp *dst_strides, npy_intp dst_strides_inc,
                char *src, npy_intp src_stride,
                npy_bool *mask, npy_intp mask_stride,
                npy_intp *coords, npy_intp coords_inc,
                npy_intp *shape, npy_intp shape_inc,
                npy_intp count, npy_intp src_itemsize,
                PyArray_MaskedStridedUnaryOp *stransfer,
                NpyAuxData *data);

NPY_NO_EXPORT int
mapiter_trivial_get(PyArrayObject *self, PyArrayObject *ind,
                       PyArrayObject *result);

NPY_NO_EXPORT int
mapiter_trivial_set(PyArrayObject *self, PyArrayObject *ind,
                       PyArrayObject *result);

NPY_NO_EXPORT int
mapiter_get(PyArrayMapIterObject *mit);

NPY_NO_EXPORT int
mapiter_set(PyArrayMapIterObject *mit);

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
 * You can use this together with NPY_RAW_ITER_START and
 * NPY_RAW_ITER_ONE_NEXT to handle the looping boilerplate of everything
 * but the innermost loop (which is for idim == 0).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareOneRawArrayIter(int ndim, npy_intp *shape,
                            char *data, npy_intp *strides,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_data, npy_intp *out_strides);

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
                            char **out_dataB, npy_intp *out_stridesB);

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
                            char **out_dataC, npy_intp *out_stridesC);

/*
 * Return number of elements that must be peeled from the start of 'addr' with
 * 'nvals' elements of size 'esize' in order to reach blockable alignment.
 * The required alignment in bytes is passed as the 'alignment' argument and
 * must be a power of two. This function is used to prepare an array for
 * blocking. See the 'npy_blocked_end' function documentation below for an
 * example of how this function is used.
 */
static NPY_INLINE npy_intp
npy_aligned_block_offset(const void * addr, const npy_uintp esize,
                         const npy_uintp alignment, const npy_uintp nvals)
{
    npy_uintp offset, peel;

    offset = (npy_uintp)addr & (alignment - 1);
    peel = offset ? (alignment - offset) / esize : 0;
    peel = (peel <= nvals) ? peel : nvals;
    assert(peel <= NPY_MAX_INTP);
    return (npy_intp)peel;
}

/*
 * Return upper loop bound for an array of 'nvals' elements
 * of size 'esize' peeled by 'offset' elements and blocking to
 * a vector size of 'vsz' in bytes
 *
 * example usage:
 * npy_intp i;
 * double v[101];
 * npy_intp esize = sizeof(v[0]);
 * npy_intp peel = npy_aligned_block_offset(v, esize, 16, n);
 * // peel to alignment 16
 * for (i = 0; i < peel; i++)
 *   <scalar-op>
 * // simd vectorized operation
 * for (; i < npy_blocked_end(peel, esize, 16, n); i += 16 / esize)
 *   <blocked-op>
 * // handle scalar rest
 * for(; i < n; i++)
 *   <scalar-op>
 */
static NPY_INLINE npy_intp
npy_blocked_end(const npy_uintp peel, const npy_uintp esize,
                const npy_uintp vsz, const npy_uintp nvals)
{
    npy_uintp ndiff = nvals - peel;
    npy_uintp res = (ndiff - ndiff % (vsz / esize));

    assert(nvals >= peel);
    assert(res <= NPY_MAX_INTP);
    return (npy_intp)(res);
}


/* byte swapping functions */
static NPY_INLINE npy_uint16
npy_bswap2(npy_uint16 x)
{
    return ((x & 0xffu) << 8) | (x >> 8);
}

/*
 * treat as int16 and byteswap unaligned memory,
 * some cpus don't support unaligned access
 */
static NPY_INLINE void
npy_bswap2_unaligned(char * x)
{
    char a = x[0];
    x[0] = x[1];
    x[1] = a;
}

static NPY_INLINE npy_uint32
npy_bswap4(npy_uint32 x)
{
#ifdef HAVE___BUILTIN_BSWAP32
    return __builtin_bswap32(x);
#else
    return ((x & 0xffu) << 24) | ((x & 0xff00u) << 8) |
           ((x & 0xff0000u) >> 8) | (x >> 24);
#endif
}

static NPY_INLINE void
npy_bswap4_unaligned(char * x)
{
    char a = x[0];
    x[0] = x[3];
    x[3] = a;
    a = x[1];
    x[1] = x[2];
    x[2] = a;
}

static NPY_INLINE npy_uint64
npy_bswap8(npy_uint64 x)
{
#ifdef HAVE___BUILTIN_BSWAP64
    return __builtin_bswap64(x);
#else
    return ((x & 0xffULL) << 56) |
           ((x & 0xff00ULL) << 40) |
           ((x & 0xff0000ULL) << 24) |
           ((x & 0xff000000ULL) << 8) |
           ((x & 0xff00000000ULL) >> 8) |
           ((x & 0xff0000000000ULL) >> 24) |
           ((x & 0xff000000000000ULL) >> 40) |
           ( x >> 56);
#endif
}

static NPY_INLINE void
npy_bswap8_unaligned(char * x)
{
    char a = x[0]; x[0] = x[7]; x[7] = a;
    a = x[1]; x[1] = x[6]; x[6] = a;
    a = x[2]; x[2] = x[5]; x[5] = a;
    a = x[3]; x[3] = x[4]; x[4] = a;
}


/* Start raw iteration */
#define NPY_RAW_ITER_START(idim, ndim, coord, shape) \
        memset((coord), 0, (ndim) * sizeof(coord[0])); \
        do {

/* Increment to the next n-dimensional coordinate for one raw array */
#define NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (data) -= ((shape)[idim] - 1) * (strides)[idim]; \
                } \
                else { \
                    (data) += (strides)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for two raw arrays */
#define NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, dataB, stridesB) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for three raw arrays */
#define NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for four raw arrays */
#define NPY_RAW_ITER_FOUR_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC, \
                              dataD, stridesD) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                    (dataD) -= ((shape)[idim] - 1) * (stridesD)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    (dataD) += (stridesD)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))


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

/**
 * Determine whether two arrays are safe for trivial iteration in cases where
 * some of the arrays may be modified.
 *
 * In-place iteration is safe if one of the following is true:
 *
 * - Both arrays are read-only
 * - The arrays do not have overlapping memory (based on a check that may be too
 *   strict)
 * - The strides match, and the non-read-only array base addresses are equal or
 *   before the read-only one, ensuring correct data dependency.
 */

#define PyArray_TRIVIALLY_ITERABLE_OP_NOREAD 0
#define PyArray_TRIVIALLY_ITERABLE_OP_READ 1

#define PyArray_TRIVIALLY_ITERABLE(arr) ( \
                    PyArray_NDIM(arr) <= 1 || \
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS) || \
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS) \
                    )

#define PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size, arr) ( \
        assert(PyArray_TRIVIALLY_ITERABLE(arr)), \
        size == 1 ? 0 : ((PyArray_NDIM(arr) == 1) ? \
                             PyArray_STRIDE(arr, 0) : PyArray_ITEMSIZE(arr)))

static NPY_INLINE int
PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK(PyArrayObject *arr1, PyArrayObject *arr2,
                                         int arr1_read, int arr2_read)
{
    npy_intp size1, size2, stride1, stride2;
    int arr1_ahead = 0, arr2_ahead = 0;

    if (arr1_read && arr2_read) {
        return 1;
    }

    if (solve_may_share_memory(arr1, arr2, 1) == 0) {
        return 1;
    }

    /*
     * Arrays overlapping in memory may be equivalently iterable if input
     * arrays stride ahead faster than output arrays.
     */

    size1 = PyArray_SIZE(arr1);
    size2 = PyArray_SIZE(arr2);

    stride1 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size1, arr1);
    stride2 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size2, arr2);

    /*
     * Arrays with zero stride are never "ahead" since the element is reused
     * (at this point we know the array extents overlap).
     */

    if (stride1 > 0) {
        arr1_ahead = (stride1 >= stride2 &&
                      PyArray_BYTES(arr1) >= PyArray_BYTES(arr2));
    }
    else if (stride1 < 0) {
        arr1_ahead = (stride1 <= stride2 &&
                      PyArray_BYTES(arr1) <= PyArray_BYTES(arr2));
    }

    if (stride2 > 0) {
        arr2_ahead = (stride2 >= stride1 &&
                      PyArray_BYTES(arr2) >= PyArray_BYTES(arr1));
    }
    else if (stride2 < 0) {
        arr2_ahead = (stride2 <= stride1 &&
                      PyArray_BYTES(arr2) <= PyArray_BYTES(arr1));
    }

    return (!arr1_read || arr1_ahead) && (!arr2_read || arr2_ahead);
}

#define PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) (            \
                        PyArray_NDIM(arr1) == PyArray_NDIM(arr2) && \
                        PyArray_CompareLists(PyArray_DIMS(arr1), \
                                             PyArray_DIMS(arr2), \
                                             PyArray_NDIM(arr1)) && \
                        (PyArray_FLAGS(arr1)&(NPY_ARRAY_C_CONTIGUOUS| \
                                      NPY_ARRAY_F_CONTIGUOUS)) & \
                                (PyArray_FLAGS(arr2)&(NPY_ARRAY_C_CONTIGUOUS| \
                                              NPY_ARRAY_F_CONTIGUOUS)) \
                        )

#define PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2, arr1_read, arr2_read) ( \
                        PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) && \
                        PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK( \
                            arr1, arr2, arr1_read, arr2_read))

#define PyArray_PREPARE_TRIVIAL_ITERATION(arr, count, data, stride) \
                    count = PyArray_SIZE(arr); \
                    data = PyArray_BYTES(arr); \
                    stride = ((PyArray_NDIM(arr) == 0) ? 0 : \
                                    ((PyArray_NDIM(arr) == 1) ? \
                                            PyArray_STRIDE(arr, 0) : \
                                            PyArray_ITEMSIZE(arr)));

#define PyArray_TRIVIALLY_ITERABLE_PAIR(arr1, arr2, arr1_read, arr2_read) (   \
                    PyArray_TRIVIALLY_ITERABLE(arr1) && \
                        (PyArray_NDIM(arr2) == 0 || \
                         PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) ||  \
                         (PyArray_NDIM(arr1) == 0 && \
                             PyArray_TRIVIALLY_ITERABLE(arr2) \
                         ) \
                        ) && \
                        PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK(arr1, arr2, arr1_read, arr2_read) \
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
                    stride1 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size1, arr1); \
                    stride2 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size2, arr2); \
                }

#define PyArray_TRIVIALLY_ITERABLE_TRIPLE(arr1, arr2, arr3, arr1_read, arr2_read, arr3_read) ( \
                PyArray_TRIVIALLY_ITERABLE(arr1) && \
                    ((PyArray_NDIM(arr2) == 0 && \
                        (PyArray_NDIM(arr3) == 0 || \
                            PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr3) \
                        ) \
                     ) || \
                     (PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) && \
                        (PyArray_NDIM(arr3) == 0 || \
                            PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr3) \
                        ) \
                     ) || \
                     (PyArray_NDIM(arr1) == 0 && \
                        PyArray_TRIVIALLY_ITERABLE(arr2) && \
                            (PyArray_NDIM(arr3) == 0 || \
                                PyArray_EQUIVALENTLY_ITERABLE_BASE(arr2, arr3) \
                            ) \
                     ) \
                    ) && \
                    PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK(arr1, arr2, arr1_read, arr2_read) && \
                    PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK(arr1, arr3, arr1_read, arr3_read) && \
                    PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK(arr2, arr3, arr2_read, arr3_read) \
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
                    stride1 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size1, arr1); \
                    stride2 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size2, arr2); \
                    stride3 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size3, arr3); \
                }

#endif
