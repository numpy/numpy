#ifndef _NPY_ARRAY_SHAPE_H_
#define _NPY_ARRAY_SHAPE_H_

typedef struct {
    npy_intp perm, stride;
} _npy_stride_sort_item;

/*
 * This function populates the first PyArray_NDIM(arr) elements
 * of strideperm with sorted descending by their absolute values.
 * For example, the stride array (4, -2, 12) becomes
 * [(2, 12), (0, 4), (1, -2)].
 */
NPY_NO_EXPORT void
PyArray_CreateSortedStridePerm(PyArrayObject *arr,
                           _npy_stride_sort_item *strideperm);

#endif
