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

NPY_NO_EXPORT NPY_CASTING
npy_default_sort_resolve_descriptors(
        PyArrayMethodObject *method,
        PyArray_DTypeMeta *const *dtypes,
        PyArray_Descr *const *input_descrs,
        PyArray_Descr **output_descrs,
        npy_intp *view_offset)
{
    output_descrs[0] = NPY_DT_CALL_ensure_canonical(input_descrs[0]);
    if (NPY_UNLIKELY(output_descrs[0] == NULL)) {
        return -1;
    }
    output_descrs[1] = NPY_DT_CALL_ensure_canonical(input_descrs[1]);
    if (NPY_UNLIKELY(output_descrs[1] == NULL)) {
        Py_XDECREF(output_descrs[0]);
        return -1;
    }

    return method->casting;
}

NPY_NO_EXPORT int
npy_default_get_sort_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArrayMethod_SortParameters *parameters = (PyArrayMethod_SortParameters *)context->parameters;
    *flags |= NPY_METH_NO_FLOATINGPOINT_ERRORS;

    if (PyDataType_REFCHK(context->descriptors[0])) {
        *flags |= NPY_METH_REQUIRES_PYAPI;
    }

    if ((parameters->flags == NPY_SORT_STABLE)
        || parameters->flags == NPY_SORT_DEFAULT) {
        *out_loop = (PyArrayMethod_StridedLoop *)npy_default_sort_loop;
    }
    else {
        PyErr_SetString(PyExc_RuntimeError, "unsupported sort kind");
        return -1;
    }
    return 0;
}

NPY_NO_EXPORT int
npy_default_get_argsort_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArrayMethod_SortParameters *parameters = (PyArrayMethod_SortParameters *)context->parameters;
    *flags |= NPY_METH_NO_FLOATINGPOINT_ERRORS;

    if (PyDataType_REFCHK(context->descriptors[0])) {
        *flags |= NPY_METH_REQUIRES_PYAPI;
    }

    if (parameters->flags == NPY_SORT_STABLE
        || parameters->flags == NPY_SORT_DEFAULT) {
        *out_loop = (PyArrayMethod_StridedLoop *)npy_default_argsort_loop;
    }
    else {
        PyErr_SetString(PyExc_RuntimeError, "unsupported sort kind");
        return -1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
