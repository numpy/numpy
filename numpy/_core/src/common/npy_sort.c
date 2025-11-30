#include <numpy/ndarraytypes.h>
#include <stdlib.h>
#include <numpy/npy_math.h>
#include "npy_sort.h"
#include "dtypemeta.h"

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
npy_default_sort_loop(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *transferdata)
{
    PyArray_CompareFunc *cmp = (PyArray_CompareFunc *)context->method->static_data;

    PyArrayMethod_SortParameters *sort_params =
        (PyArrayMethod_SortParameters *)context->parameters;
    PyArray_SortImpl *sort_func = NULL;
    
    switch (sort_params->flags) {
        case NPY_SORT_DEFAULT:
            sort_func = npy_quicksort_impl;
            break;
        case NPY_SORT_STABLE:
            sort_func = npy_mergesort_impl;
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Invalid sort kind");
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
    PyArray_CompareFunc *cmp = (PyArray_CompareFunc *)context->method->static_data;

    PyArrayMethod_SortParameters *sort_params =
        (PyArrayMethod_SortParameters *)context->parameters;
    PyArray_ArgSortImpl *argsort_func = NULL;
    
    switch (sort_params->flags) {
        case NPY_SORT_DEFAULT:
            argsort_func = npy_aquicksort_impl;
            break;
        case NPY_SORT_STABLE:
            argsort_func = npy_amergesort_impl;
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Invalid sort kind");
            return -1;
    }

    return argsort_func(data[0], (npy_intp *)data[1], dimensions[0], context,
                        context->descriptors[0]->elsize, cmp);
}

#ifdef __cplusplus
}
#endif
