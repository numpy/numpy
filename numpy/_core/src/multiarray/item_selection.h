#ifndef NUMPY_CORE_SRC_MULTIARRAY_ITEM_SELECTION_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ITEM_SELECTION_H_

/*
 * Counts the number of True values in a raw boolean array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
count_boolean_trues(int ndim, char *data, npy_intp const *ashape, npy_intp const *astrides);

/*
 * Gets a single item from the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 */
NPY_NO_EXPORT PyObject *
PyArray_MultiIndexGetItem(PyArrayObject *self, const npy_intp *multi_index);

/*
 * Sets a single item in the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_MultiIndexSetItem(PyArrayObject *self, const npy_intp *multi_index,
                                                PyObject *obj);

/*
 * Internal version of PyArray_SearchSorted that accepts an `op1` of any
 * dimensionality, searching each one dimensional slice along its last axis.
 */
NPY_NO_EXPORT PyObject *
PyArray_SearchSorted_int(PyArrayObject *op1, PyObject *op2,
                         NPY_SEARCHSIDE side, PyObject *perm);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ITEM_SELECTION_H_ */
