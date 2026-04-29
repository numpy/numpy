#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npysort_common.h"
#include "numpy_tag.h"
#include "npy_sort.h"

#include <cstdlib>

/*
 *****************************************************************************
 **                          SORTING ARRAYMETHODS                           **
 *****************************************************************************
 */

static NPY_CASTING
sort_resolve_descriptors(PyArrayMethodObject *method, PyArray_DTypeMeta *const *dtypes,
                         PyArray_Descr *const *input_descrs,
                         PyArray_Descr **output_descrs, npy_intp *view_offset)
{
    output_descrs[0] = NPY_DT_CALL_ensure_canonical(input_descrs[0]);
    if (NPY_UNLIKELY(output_descrs[0] == NULL)) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    Py_INCREF(output_descrs[0]);
    output_descrs[1] = output_descrs[0];

    return method->casting;
}

static NPY_CASTING
argsort_resolve_descriptors(PyArrayMethodObject *method,
                            PyArray_DTypeMeta *const *dtypes,
                            PyArray_Descr *const *input_descrs,
                            PyArray_Descr **output_descrs, npy_intp *view_offset)
{
    output_descrs[0] = NPY_DT_CALL_ensure_canonical(input_descrs[0]);
    if (NPY_UNLIKELY(output_descrs[0] == NULL)) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    output_descrs[1] = PyArray_DescrFromType(NPY_INTP);
    if (NPY_UNLIKELY(output_descrs[1] == NULL)) {
        Py_XDECREF(output_descrs[0]);
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    return method->casting;
}

template <typename Tag, typename type>
static int
sort_loop_(PyArrayMethod_Context *context, char *const data[],
           npy_intp const dimensions[], npy_intp const strides[],
           NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArrayMethod_SortParameters *params =
            (PyArrayMethod_SortParameters *)context->parameters;
    switch ((int)params->flags) {
        case NPY_SORT_DEFAULT:
            return quicksort_<Tag, type, false>((type *)data[0], dimensions[0]);
        case _NPY_SORT_HEAPSORT:
        case NPY_SORT_STABLE:
            return timsort_<Tag, type, false>((type *)data[0], dimensions[0]);
        case NPY_SORT_DEFAULT | NPY_SORT_DESCENDING:
            return quicksort_<Tag, type, true>((type *)data[0], dimensions[0]);
        case _NPY_SORT_HEAPSORT | NPY_SORT_DESCENDING:
        case NPY_SORT_STABLE | NPY_SORT_DESCENDING:
            return timsort_<Tag, type, true>((type *)data[0], dimensions[0]);
        default:
            PyErr_Format(PyExc_RuntimeError, "unknown sort kind %d",
                         (int)params->flags);
            return -1;
    }
}

template <typename Tag, typename type>
static int
argsort_loop_(PyArrayMethod_Context *context, char *const data[],
              npy_intp const dimensions[], npy_intp const strides[],
              NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArrayMethod_SortParameters *params =
            (PyArrayMethod_SortParameters *)context->parameters;
    switch ((int)params->flags) {
        case NPY_SORT_DEFAULT:
            return aquicksort_<Tag, type, false>((type *)data[0], (npy_intp *)data[1],
                                                 dimensions[0]);
        case _NPY_SORT_HEAPSORT:
        case NPY_SORT_STABLE:
            return atimsort_<Tag, type, false>((type *)data[0], (npy_intp *)data[1],
                                               dimensions[0]);
        case NPY_SORT_DEFAULT | NPY_SORT_DESCENDING:
            return aquicksort_<Tag, type, true>((type *)data[0], (npy_intp *)data[1],
                                                dimensions[0]);
        case _NPY_SORT_HEAPSORT | NPY_SORT_DESCENDING:
        case NPY_SORT_STABLE | NPY_SORT_DESCENDING:
            return atimsort_<Tag, type, true>((type *)data[0], (npy_intp *)data[1],
                                              dimensions[0]);
        default:
            PyErr_Format(PyExc_RuntimeError, "unknown sort kind %d",
                         (int)params->flags);
            return -1;
    }
}

template <typename Tag, typename type>
static int
sort_loop_string_(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArrayMethod_SortParameters *params =
            (PyArrayMethod_SortParameters *)context->parameters;
    int elsize = (int)strides[0];
    switch ((int)params->flags) {
        case NPY_SORT_DEFAULT:
            return string_quicksort_<Tag, type, false>((type *)data[0], dimensions[0],
                                                       elsize);
        case _NPY_SORT_HEAPSORT:
        case NPY_SORT_STABLE:
            return string_timsort_<Tag, type, false>((type *)data[0], dimensions[0],
                                                     elsize);
        case NPY_SORT_DEFAULT | NPY_SORT_DESCENDING:
            return string_quicksort_<Tag, type, true>((type *)data[0], dimensions[0],
                                                      elsize);
        case _NPY_SORT_HEAPSORT | NPY_SORT_DESCENDING:
        case NPY_SORT_STABLE | NPY_SORT_DESCENDING:
            return string_timsort_<Tag, type, true>((type *)data[0], dimensions[0],
                                                    elsize);
        default:
            PyErr_Format(PyExc_RuntimeError, "unknown sort kind %d",
                         (int)params->flags);
            return -1;
    }
}

template <typename Tag, typename type>
static int
argsort_loop_string_(PyArrayMethod_Context *context, char *const data[],
                     npy_intp const dimensions[], npy_intp const strides[],
                     NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArrayMethod_SortParameters *params =
            (PyArrayMethod_SortParameters *)context->parameters;
    int elsize = (int)strides[0];
    switch ((int)params->flags) {
        case NPY_SORT_DEFAULT:
            return string_aquicksort_<Tag, type, false>(
                    (type *)data[0], (npy_intp *)data[1], dimensions[0], elsize);
        case _NPY_SORT_HEAPSORT:
        case NPY_SORT_STABLE:
            return string_atimsort_<Tag, type, false>(
                    (type *)data[0], (npy_intp *)data[1], dimensions[0], elsize);
        case NPY_SORT_DEFAULT | NPY_SORT_DESCENDING:
            return string_aquicksort_<Tag, type, true>(
                    (type *)data[0], (npy_intp *)data[1], dimensions[0], elsize);
        case _NPY_SORT_HEAPSORT | NPY_SORT_DESCENDING:
        case NPY_SORT_STABLE | NPY_SORT_DESCENDING:
            return string_atimsort_<Tag, type, true>(
                    (type *)data[0], (npy_intp *)data[1], dimensions[0], elsize);
        default:
            PyErr_Format(PyExc_RuntimeError, "unknown sort kind %d",
                         (int)params->flags);
            return -1;
    }
}

template <typename Tag, typename type>
NPY_NO_EXPORT int
make_sorts_(PyArray_DTypeMeta *dtypemeta)
{
    PyArray_DTypeMeta *sort_dtypes[2] = {dtypemeta, dtypemeta};
    PyType_Slot sort_slots[3] = {
            {NPY_METH_resolve_descriptors, reinterpret_cast<void *>(sort_resolve_descriptors)},
            {NPY_METH_strided_loop, reinterpret_cast<void *>(sort_loop_<Tag, type>)},
            {0, NULL}};
    PyArrayMethod_Spec sort_spec = {
            .name = "@name@_sort",
            .nin = 1,
            .nout = 1,
            .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
            .dtypes = sort_dtypes,
            .slots = sort_slots,
    };
    PyBoundArrayMethodObject *sort_method = PyArrayMethod_FromSpec_int(&sort_spec, 1);
    if (sort_method == NULL) {
        return -1;
    }
    NPY_DT_SLOTS(dtypemeta)->sort_meth = sort_method->method;
    Py_INCREF(sort_method->method);
    Py_DECREF(sort_method);

    PyArray_DTypeMeta *argsort_dtypes[2] = {dtypemeta, &PyArray_IntpDType};
    PyType_Slot argsort_slots[3] = {
            {NPY_METH_resolve_descriptors, reinterpret_cast<void *>(argsort_resolve_descriptors)},
            {NPY_METH_strided_loop, reinterpret_cast<void *>(argsort_loop_<Tag, type>)},
            {0, NULL}};
    PyArrayMethod_Spec argsort_spec = {
            .name = "@name@_argsort",
            .nin = 1,
            .nout = 1,
            .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
            .dtypes = argsort_dtypes,
            .slots = argsort_slots,
    };
    PyBoundArrayMethodObject *argsort_method =
            PyArrayMethod_FromSpec_int(&argsort_spec, 1);
    if (argsort_method == NULL) {
        return -1;
    }
    NPY_DT_SLOTS(dtypemeta)->argsort_meth = argsort_method->method;
    Py_INCREF(argsort_method->method);
    Py_DECREF(argsort_method);

    return 0;
}

extern "C" {
NPY_NO_EXPORT int
register_bool_sorts()
{
    return make_sorts_<npy::bool_tag, npy_bool>(&PyArray_BoolDType);
}
NPY_NO_EXPORT int
register_byte_sorts()
{
    return make_sorts_<npy::byte_tag, npy_byte>(&PyArray_ByteDType);
}
NPY_NO_EXPORT int
register_ubyte_sorts()
{
    return make_sorts_<npy::ubyte_tag, npy_ubyte>(&PyArray_UByteDType);
}
NPY_NO_EXPORT int
register_short_sorts()
{
    return make_sorts_<npy::short_tag, npy_short>(&PyArray_ShortDType);
}
NPY_NO_EXPORT int
register_ushort_sorts()
{
    return make_sorts_<npy::ushort_tag, npy_ushort>(&PyArray_UShortDType);
}
NPY_NO_EXPORT int
register_int_sorts()
{
    return make_sorts_<npy::int_tag, npy_int>(&PyArray_IntDType);
}
NPY_NO_EXPORT int
register_uint_sorts()
{
    return make_sorts_<npy::uint_tag, npy_uint>(&PyArray_UIntDType);
}
NPY_NO_EXPORT int
register_long_sorts()
{
    return make_sorts_<npy::long_tag, npy_long>(&PyArray_LongDType);
}
NPY_NO_EXPORT int
register_ulong_sorts()
{
    return make_sorts_<npy::ulong_tag, npy_ulong>(&PyArray_ULongDType);
}
NPY_NO_EXPORT int
register_longlong_sorts()
{
    return make_sorts_<npy::longlong_tag, npy_longlong>(&PyArray_LongLongDType);
}
NPY_NO_EXPORT int
register_ulonglong_sorts()
{
    return make_sorts_<npy::ulonglong_tag, npy_ulonglong>(&PyArray_ULongLongDType);
}
NPY_NO_EXPORT int
register_float_sorts()
{
    return make_sorts_<npy::float_tag, npy_float>(&PyArray_FloatDType);
}
NPY_NO_EXPORT int
register_double_sorts()
{
    return make_sorts_<npy::double_tag, npy_double>(&PyArray_DoubleDType);
}
NPY_NO_EXPORT int
register_longdouble_sorts()
{
    return make_sorts_<npy::longdouble_tag, npy_longdouble>(&PyArray_LongDoubleDType);
}
NPY_NO_EXPORT int
register_cfloat_sorts()
{
    return make_sorts_<npy::cfloat_tag, npy_cfloat>(&PyArray_CFloatDType);
}
NPY_NO_EXPORT int
register_cdouble_sorts()
{
    return make_sorts_<npy::cdouble_tag, npy_cdouble>(&PyArray_CDoubleDType);
}
NPY_NO_EXPORT int
register_clongdouble_sorts()
{
    return make_sorts_<npy::clongdouble_tag, npy_clongdouble>(
            &PyArray_CLongDoubleDType);
}
NPY_NO_EXPORT int
register_datetime_sorts()
{
    return make_sorts_<npy::datetime_tag, npy_datetime>(&PyArray_DatetimeDType);
}
NPY_NO_EXPORT int
register_timedelta_sorts()
{
    return make_sorts_<npy::timedelta_tag, npy_timedelta>(&PyArray_TimedeltaDType);
}
NPY_NO_EXPORT int
register_string_sorts()
{
    return 0;
}
NPY_NO_EXPORT int
register_unicode_sorts()
{
    return 0;
}
NPY_NO_EXPORT int
register_half_sorts()
{
    return make_sorts_<npy::half_tag, npy_half>(&PyArray_HalfDType);
}
}