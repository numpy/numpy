#ifndef _NPY_ARRAY_SHAPE_H_
#define _NPY_ARRAY_SHAPE_H_

/*
 * Builds a string representation of the shape given in 'vals'.
 * A negative value in 'vals' gets interpreted as newaxis.
 */
NPY_NO_EXPORT PyObject *
build_shape_string(npy_intp n, npy_intp *vals);

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


#endif
