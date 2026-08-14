#include <numpy/ndarraytypes.h>
#include <stdlib.h>
#include <numpy/npy_math.h>
#include "npy_sort.h"
#include "dtypemeta.h"
#include "gil_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
npy_default_sort_loop(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *transferdata)
{
    PyArray_CompareFunc **cmps =
        (PyArray_CompareFunc **)context->method->static_data;

    PyArrayMethod_SortParameters *sort_params =
        (PyArrayMethod_SortParameters *)context->parameters;
    int descending = (sort_params->flags & NPY_SORT_DESCENDING) != 0;
    PyArray_CompareFunc *cmp = cmps[descending];
    PyArray_SortImpl *sort_func = NULL;

    if (cmp == NULL) {
        npy_gil_error(PyExc_ValueError,
                      "descending sort not supported for this DType");
        return -1;
    }

    switch (sort_params->flags & ~NPY_SORT_DESCENDING) {
        case NPY_SORT_DEFAULT:
            sort_func = npy_quicksort_impl;
            break;
        case NPY_SORT_STABLE:
            sort_func = npy_timsort_impl;
            break;
        default:
            npy_gil_error(PyExc_ValueError, "Invalid sort kind");
            return -1;
    }

    return sort_func(data[0], dimensions[0], context,
                     context->descriptors[0]->elsize, cmp);
}

NPY_NO_EXPORT int
npy_default_argsort_loop(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *transferdata)
{
    PyArray_CompareFunc **cmps =
        (PyArray_CompareFunc **)context->method->static_data;

    PyArrayMethod_SortParameters *sort_params =
        (PyArrayMethod_SortParameters *)context->parameters;
    int descending = (sort_params->flags & NPY_SORT_DESCENDING) != 0;
    PyArray_CompareFunc *cmp = cmps[descending];
    PyArray_ArgSortImpl *argsort_func = NULL;

    if (cmp == NULL) {
        npy_gil_error(PyExc_ValueError,
                      "descending sort not supported for this DType");
        return -1;
    }

    switch (sort_params->flags & ~NPY_SORT_DESCENDING) {
        case NPY_SORT_DEFAULT:
            argsort_func = npy_aquicksort_impl;
            break;
        case NPY_SORT_STABLE:
            argsort_func = npy_atimsort_impl;
            break;
        default:
            npy_gil_error(PyExc_ValueError, "Invalid sort kind");
            return -1;
    }

    return argsort_func(data[0], (npy_intp *)data[1], dimensions[0], context,
                        context->descriptors[0]->elsize, cmp);
}

#ifdef __cplusplus
}
#endif
