#ifndef NUMPY_CORE_SRC_MULTIARRAY_SHAPE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_SHAPE_H_

#include "conversion_utils.h"

/*
 * Creates a sorted stride perm matching the KEEPORDER behavior
 * of the NpyIter object. Because this operates based on multiple
 * input strides, the 'stride' member of the npy_stride_sort_item
 * would be useless and we simply argsort a list of indices instead.
 *
 * The caller should have already validated that 'ndim' matches for
 * every array in the arrays list.
 */
NPY_NO_EXPORT void
PyArray_CreateMultiSortedStridePerm(int narrays, PyArrayObject **arrays,
                        int ndim, int *out_strideperm);

/*
 * Just like PyArray_Squeeze, but allows the caller to select
 * a subset of the size-one dimensions to squeeze out.
 */
NPY_NO_EXPORT PyObject *
PyArray_SqueezeSelected(PyArrayObject *self, npy_bool *axis_flags);

/*
 * Return matrix transpose (swap last two dimensions).
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixTranspose(PyArrayObject *ap);

NPY_NO_EXPORT PyObject *
_reshape_with_copy_arg(PyArrayObject *array, PyArray_Dims *newdims,
                       NPY_ORDER order, NPY_COPYMODE copy);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_SHAPE_H_ */
