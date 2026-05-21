#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"
#include "numpy_tag.h"
#include "gil_utils.h"
#include "quicksort.hpp"
#include "radixsort.hpp"
#include "timsort.hpp"

#include <cstdlib>

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
    if (input_descrs[1] == NULL) {
        output_descrs[1] = PyArray_DescrFromType(NPY_INTP);
    }
    else {
        output_descrs[1] = NPY_DT_CALL_ensure_canonical(input_descrs[1]);
    }
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
    constexpr bool use_radixsort = (
        std::is_same_v<Tag, npy::bool_tag> ||
        std::is_same_v<Tag, npy::ubyte_tag> ||
        std::is_same_v<Tag, npy::byte_tag> ||
        std::is_same_v<Tag, npy::ushort_tag> ||
        std::is_same_v<Tag, npy::short_tag>
    );

    PyArrayMethod_SortParameters *params =
            (PyArrayMethod_SortParameters *)context->parameters;
    switch ((int)params->flags) {
        case NPY_SORT_DEFAULT:
            return quicksort_<Tag, type, false>((type *)data[0], dimensions[0]);
        case NPY_SORT_STABLE:
        if constexpr (use_radixsort) {
            return radixsort<type>(data[0], dimensions[0]);
        }
        else {
            return timsort_<Tag, type, false>((type *)data[0], dimensions[0]);
        }
        case NPY_SORT_DEFAULT | NPY_SORT_DESCENDING:
            return quicksort_<Tag, type, true>((type *)data[0], dimensions[0]);
        case NPY_SORT_STABLE | NPY_SORT_DESCENDING:
            return timsort_<Tag, type, true>((type *)data[0], dimensions[0]);
        default:
            npy_gil_error(PyExc_RuntimeError, "unknown sort kind %d",
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
    constexpr bool use_radixsort = (
        std::is_same_v<Tag, npy::bool_tag> ||
        std::is_same_v<Tag, npy::ubyte_tag> ||
        std::is_same_v<Tag, npy::byte_tag> ||
        std::is_same_v<Tag, npy::ushort_tag> ||
        std::is_same_v<Tag, npy::short_tag>
    );

    PyArrayMethod_SortParameters *params =
            (PyArrayMethod_SortParameters *)context->parameters;
    switch ((int)params->flags) {
        case NPY_SORT_DEFAULT:
            return aquicksort_<Tag, type, false>((type *)data[0], (npy_intp *)data[1],
                                                 dimensions[0]);
        case NPY_SORT_STABLE:
        if constexpr (use_radixsort) {
            return aradixsort<type>(data[0], (npy_intp *)data[1], dimensions[0]);
        }
        else {
            return atimsort_<Tag, type, false>((type *)data[0], (npy_intp *)data[1],
                                               dimensions[0]);
        }
        case NPY_SORT_DEFAULT | NPY_SORT_DESCENDING:
            return aquicksort_<Tag, type, true>((type *)data[0], (npy_intp *)data[1],
                                                dimensions[0]);
        case NPY_SORT_STABLE | NPY_SORT_DESCENDING:
            return atimsort_<Tag, type, true>((type *)data[0], (npy_intp *)data[1],
                                              dimensions[0]);
        default:
            npy_gil_error(PyExc_RuntimeError, "unknown sort kind %d",
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
    int elsize = context->descriptors[0]->elsize;
    switch ((int)params->flags) {
        case NPY_SORT_DEFAULT:
            return string_quicksort_<Tag, type, false>((type *)data[0], dimensions[0],
                                                       elsize);
        case NPY_SORT_STABLE:
            return string_timsort_<Tag, type, false>((type *)data[0], dimensions[0],
                                                     elsize);
        case NPY_SORT_DEFAULT | NPY_SORT_DESCENDING:
            return string_quicksort_<Tag, type, true>((type *)data[0], dimensions[0],
                                                      elsize);
        case NPY_SORT_STABLE | NPY_SORT_DESCENDING:
            return string_timsort_<Tag, type, true>((type *)data[0], dimensions[0],
                                                    elsize);
        default:
            npy_gil_error(PyExc_RuntimeError, "unknown sort kind %d",
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
    int elsize = context->descriptors[0]->elsize;
    switch ((int)params->flags) {
        case NPY_SORT_DEFAULT:
            return string_aquicksort_<Tag, type, false>(
                    (type *)data[0], (npy_intp *)data[1], dimensions[0], elsize);
        case NPY_SORT_STABLE:
            return string_atimsort_<Tag, type, false>(
                    (type *)data[0], (npy_intp *)data[1], dimensions[0], elsize);
        case NPY_SORT_DEFAULT | NPY_SORT_DESCENDING:
            return string_aquicksort_<Tag, type, true>(
                    (type *)data[0], (npy_intp *)data[1], dimensions[0], elsize);
        case NPY_SORT_STABLE | NPY_SORT_DESCENDING:
            return string_atimsort_<Tag, type, true>(
                    (type *)data[0], (npy_intp *)data[1], dimensions[0], elsize);
        default:
            npy_gil_error(PyExc_RuntimeError, "unknown sort kind %d",
                          (int)params->flags);
            return -1;
    }
}

template <typename Tag>
NPY_NO_EXPORT int
make_sorts_(PyArray_DTypeMeta *dtypemeta, const char *name)
{
    using type = typename Tag::type;
    constexpr bool use_radixsort = (
        std::is_same_v<Tag, npy::bool_tag> ||
        std::is_same_v<Tag, npy::ubyte_tag> ||
        std::is_same_v<Tag, npy::byte_tag> ||
        std::is_same_v<Tag, npy::ushort_tag> ||
        std::is_same_v<Tag, npy::short_tag>
    );

    NPY_DT_SLOTS(dtypemeta)->f.sort[0] = quicksort_impl<Tag, type>;
    NPY_DT_SLOTS(dtypemeta)->f.sort[1] = heapsort_impl<Tag, type>;
    if constexpr (use_radixsort) {
        NPY_DT_SLOTS(dtypemeta)->f.sort[2] = radixsort_impl<Tag, type>;
    }
    else {
        NPY_DT_SLOTS(dtypemeta)->f.sort[2] = timsort_impl<Tag, type>;
    }

    NPY_DT_SLOTS(dtypemeta)->f.argsort[0] = aquicksort_impl<Tag, type>;
    NPY_DT_SLOTS(dtypemeta)->f.argsort[1] = aheapsort_impl<Tag, type>;
    if constexpr (use_radixsort) {
        NPY_DT_SLOTS(dtypemeta)->f.argsort[2] = aradixsort_impl<Tag, type>;
    }
    else {
        NPY_DT_SLOTS(dtypemeta)->f.argsort[2] = atimsort_impl<Tag, type>;
    }

    std::string sort_name = std::string(name) + "_sort";
    PyArray_DTypeMeta *sort_dtypes[2] = {dtypemeta, dtypemeta};
    PyType_Slot sort_slots[3] = {
            {NPY_METH_resolve_descriptors,
             reinterpret_cast<void *>(sort_resolve_descriptors)},
            {NPY_METH_strided_loop, reinterpret_cast<void *>(sort_loop_<Tag, type>)},
            {0, NULL}};
    PyArrayMethod_Spec sort_spec = {
            sort_name.c_str(),
            1,
            1,
            NPY_NO_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS,
            sort_dtypes,
            sort_slots,
    };
    PyBoundArrayMethodObject *sort_method = PyArrayMethod_FromSpec_int(&sort_spec, 1);
    if (sort_method == NULL) {
        return -1;
    }
    NPY_DT_SLOTS(dtypemeta)->sort_meth = sort_method->method;
    Py_INCREF(sort_method->method);
    Py_DECREF(sort_method);

    std::string argsort_name = std::string(name) + "_argsort";
    PyArray_DTypeMeta *argsort_dtypes[2] = {dtypemeta, &PyArray_IntpDType};
    PyType_Slot argsort_slots[3] = {
            {NPY_METH_resolve_descriptors,
             reinterpret_cast<void *>(argsort_resolve_descriptors)},
            {NPY_METH_strided_loop, reinterpret_cast<void *>(argsort_loop_<Tag, type>)},
            {0, NULL}};
    PyArrayMethod_Spec argsort_spec = {
            argsort_name.c_str(),
            1,
            1,
            NPY_NO_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS,
            argsort_dtypes,
            argsort_slots,
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

template <typename Tag>
NPY_NO_EXPORT int
make_string_sorts_(PyArray_DTypeMeta *dtypemeta, const char *name)
{
    using type = typename Tag::type;

    std::string sort_name = std::string(name) + "_sort";
    PyArray_DTypeMeta *sort_dtypes[2] = {dtypemeta, dtypemeta};
    PyType_Slot sort_slots[3] = {
            {NPY_METH_resolve_descriptors,
             reinterpret_cast<void *>(sort_resolve_descriptors)},
            {NPY_METH_strided_loop,
             reinterpret_cast<void *>(sort_loop_string_<Tag, type>)},
            {0, NULL}};
    PyArrayMethod_Spec sort_spec = {
            sort_name.c_str(),
            1,
            1,
            NPY_NO_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS,
            sort_dtypes,
            sort_slots,
    };
    PyBoundArrayMethodObject *sort_method = PyArrayMethod_FromSpec_int(&sort_spec, 1);
    if (sort_method == NULL) {
        return -1;
    }
    NPY_DT_SLOTS(dtypemeta)->sort_meth = sort_method->method;
    Py_INCREF(sort_method->method);
    Py_DECREF(sort_method);

    std::string argsort_name = std::string(name) + "_argsort";
    PyArray_DTypeMeta *argsort_dtypes[2] = {dtypemeta, &PyArray_IntpDType};
    PyType_Slot argsort_slots[3] = {
            {NPY_METH_resolve_descriptors,
             reinterpret_cast<void *>(argsort_resolve_descriptors)},
            {NPY_METH_strided_loop,
             reinterpret_cast<void *>(argsort_loop_string_<Tag, type>)},
            {0, NULL}};
    PyArrayMethod_Spec argsort_spec = {
            argsort_name.c_str(),
            1,
            1,
            NPY_NO_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS,
            argsort_dtypes,
            argsort_slots,
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

int register_all_sorts() {
    int r = 0;

    r += make_sorts_<npy::bool_tag>(&PyArray_BoolDType, "bool");
    r += make_sorts_<npy::byte_tag>(&PyArray_ByteDType, "byte");
    r += make_sorts_<npy::ubyte_tag>(&PyArray_UByteDType, "ubyte");
    r += make_sorts_<npy::short_tag>(&PyArray_ShortDType, "short");
    r += make_sorts_<npy::ushort_tag>(&PyArray_UShortDType, "ushort");
    r += make_sorts_<npy::int_tag>(&PyArray_IntDType, "int");
    r += make_sorts_<npy::uint_tag>(&PyArray_UIntDType, "uint");
    r += make_sorts_<npy::long_tag>(&PyArray_LongDType, "long");
    r += make_sorts_<npy::ulong_tag>(&PyArray_ULongDType, "ulong");
    r += make_sorts_<npy::longlong_tag>(&PyArray_LongLongDType, "longlong");
    r += make_sorts_<npy::ulonglong_tag>(&PyArray_ULongLongDType, "ulonglong");
    r += make_sorts_<npy::float_tag>(&PyArray_FloatDType, "float");
    r += make_sorts_<npy::double_tag>(&PyArray_DoubleDType, "double");
    r += make_sorts_<npy::longdouble_tag>(&PyArray_LongDoubleDType, "longdouble");
    r += make_sorts_<npy::cfloat_tag>(&PyArray_CFloatDType, "cfloat");
    r += make_sorts_<npy::cdouble_tag>(&PyArray_CDoubleDType, "cdouble");
    r += make_sorts_<npy::clongdouble_tag>(&PyArray_CLongDoubleDType, "clongdouble");
    r += make_sorts_<npy::datetime_tag>(&PyArray_DatetimeDType, "datetime");
    r += make_sorts_<npy::timedelta_tag>(&PyArray_TimedeltaDType, "timedelta");
    r += make_string_sorts_<npy::string_tag>(&PyArray_BytesDType, "string");
    r += make_string_sorts_<npy::unicode_tag>(&PyArray_UnicodeDType, "unicode");
    r += make_sorts_<npy::half_tag>(&PyArray_HalfDType, "half");

    return r;
}
