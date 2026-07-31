#ifndef NPYSORT_SELECTION_METHODS_CPP
#define NPYSORT_SELECTION_METHODS_CPP

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_partition.h"
#include "npysort_common.h"
#include "numpy_tag.hpp"
#include "selection.hpp"
#include "gil_utils.h"

#include <cstdlib>

static NPY_CASTING
partition_resolve_descriptors(PyArrayMethodObject *method, PyArray_DTypeMeta *const *dtypes,
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

    /* We reuse the input descriptor for the output as PyArray_Partition uses
       the same descriptor for both input and output. This can work because
       partitioning is in-place. */
    Py_INCREF(output_descrs[0]);
    output_descrs[2] = output_descrs[0];

    return method->casting;
}

static NPY_CASTING
argpartition_resolve_descriptors(PyArrayMethodObject *method,
                                 PyArray_DTypeMeta *const *dtypes,
                                 PyArray_Descr *const *input_descrs,
                                 PyArray_Descr **output_descrs, npy_intp *view_offset)
{
    output_descrs[0] = NPY_DT_CALL_ensure_canonical(input_descrs[0]);
    if (NPY_UNLIKELY(output_descrs[0] == NULL)) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    /* Both the kth array and output indices are NPY_INTP, so use the same logic for both */
    for (int i = 1; i <= 2; i++) {
        if (input_descrs[i] == NULL) {
            output_descrs[i] = PyArray_DescrFromType(NPY_INTP);
        }
        else {
            output_descrs[i] = NPY_DT_CALL_ensure_canonical(input_descrs[i]);
        }
        if (NPY_UNLIKELY(output_descrs[i] == NULL)) {
            Py_XDECREF(output_descrs[0]);
            Py_XDECREF(output_descrs[1]);
            return _NPY_ERROR_OCCURRED_IN_CAST;
        }
    }

    return method->casting;
}

template <typename Tag, typename type>
static int
partition_loop_(PyArrayMethod_Context *context, char *const data[],
                npy_intp const dimensions[], npy_intp const strides[],
                NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArrayMethod_PartitionParameters *params =
            (PyArrayMethod_PartitionParameters *)context->parameters;
    PyArray_PartitionFunc *func = NULL;

    switch ((int)params->flags) {
        case NPY_SELECT_DEFAULT:
            func = introselect_noarg<Tag, false>;
            break;
        case NPY_SELECT_DESCENDING:
            func = introselect_noarg<Tag, true>;
            break;
        default:
            npy_gil_error(PyExc_RuntimeError, "unknown partition kind %d",
                          (int)params->flags);
            return -1;
    }

    npy_intp pivots[NPY_MAX_PIVOT_STACK];
    npy_intp npiv = 0;
    type *ip = (type *)data[0];
    npy_intp *kth = (npy_intp *)data[1];
    for (npy_intp i = 0; i < dimensions[1]; ++i) {
        if (func(ip, dimensions[0], kth[i], pivots, &npiv, dimensions[1], NULL) < 0) {
            return -1;
        }
    }
    return 0;
}

template <typename Tag, typename type>
static int
argpartition_loop_(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArrayMethod_PartitionParameters *params =
            (PyArrayMethod_PartitionParameters *)context->parameters;
    PyArray_ArgPartitionFunc *func = NULL;

    switch ((int)params->flags) {
        case NPY_SELECT_DEFAULT:
            func = introselect_arg<Tag, false>;
            break;
        case NPY_SELECT_DESCENDING:
            func = introselect_arg<Tag, true>;
            break;
        default:
            npy_gil_error(PyExc_RuntimeError, "unknown partition kind %d",
                          (int)params->flags);
            return -1;
    }

    npy_intp pivots[NPY_MAX_PIVOT_STACK];
    npy_intp npiv = 0;
    type *ip = (type *)data[0];
    npy_intp *kth = (npy_intp *)data[1];
    npy_intp *indices = (npy_intp *)data[2];
    for (npy_intp i = 0; i < dimensions[1]; ++i) {
        if (func(ip, indices, dimensions[0], kth[i], pivots, &npiv, dimensions[1], NULL) < 0) {
            return -1;
        }
    }
    return 0;
}

template <typename Tag>
NPY_NO_EXPORT int
make_partitions_(PyArray_DTypeMeta *dtypemeta, const char *name)
{
    using type = typename Tag::type;

    NPY_ARRAYMETHOD_FLAGS meth_flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    if constexpr (std::is_same_v<Tag, npy::object_tag>) {
        // lock the GIL for object partitions
        meth_flags = (NPY_ARRAYMETHOD_FLAGS)(meth_flags | NPY_METH_REQUIRES_PYAPI);
    }

    std::string partition_name = std::string(name) + "_partition";
    PyArray_DTypeMeta *partition_dtypes[3] = {dtypemeta, &PyArray_IntpDType, dtypemeta};
    PyType_Slot partition_slots[3] = {
            {NPY_METH_resolve_descriptors,
             reinterpret_cast<void *>(partition_resolve_descriptors)},
            {NPY_METH_strided_loop, reinterpret_cast<void *>(partition_loop_<Tag, type>)},
            {0, NULL}};
    PyArrayMethod_Spec partition_spec = {
            partition_name.c_str(),
            2,
            1,
            NPY_NO_CASTING,
            meth_flags,
            partition_dtypes,
            partition_slots,
    };
    PyBoundArrayMethodObject *part_method = PyArrayMethod_FromSpec_int(&partition_spec, 1);
    if (part_method == NULL) {
        return -1;
    }
    NPY_DT_SLOTS(dtypemeta)->part_meth = part_method->method;
    Py_INCREF(part_method->method);
    Py_DECREF(part_method);

    std::string argpartition_name = std::string(name) + "_argpartition";
    PyArray_DTypeMeta *argpartition_dtypes[3] = {dtypemeta, &PyArray_IntpDType, &PyArray_IntpDType};
    PyType_Slot argpartition_slots[3] = {
            {NPY_METH_resolve_descriptors,
             reinterpret_cast<void *>(argpartition_resolve_descriptors)},
            {NPY_METH_strided_loop, reinterpret_cast<void *>(argpartition_loop_<Tag, type>)},
            {0, NULL}};
    PyArrayMethod_Spec argpartition_spec = {
            argpartition_name.c_str(),
            2,
            1,
            NPY_NO_CASTING,
            meth_flags,
            argpartition_dtypes,
            argpartition_slots,
    };
    PyBoundArrayMethodObject *argpart_method =
            PyArrayMethod_FromSpec_int(&argpartition_spec, 1);
    if (argpart_method == NULL) {
        return -1;
    }
    NPY_DT_SLOTS(dtypemeta)->argpart_meth = argpart_method->method;
    Py_INCREF(argpart_method->method);
    Py_DECREF(argpart_method);

    return 0;
}

int register_all_partitions() {
     // TODO: Support object, string, and unicode dtypes.
    if (make_partitions_<npy::bool_tag>(&PyArray_BoolDType, "bool") < 0 ||
        make_partitions_<npy::byte_tag>(&PyArray_ByteDType, "byte") < 0 ||
        make_partitions_<npy::ubyte_tag>(&PyArray_UByteDType, "ubyte") < 0 ||
        make_partitions_<npy::short_tag>(&PyArray_ShortDType, "short") < 0 ||
        make_partitions_<npy::ushort_tag>(&PyArray_UShortDType, "ushort") < 0 ||
        make_partitions_<npy::int_tag>(&PyArray_IntDType, "int") < 0 ||
        make_partitions_<npy::uint_tag>(&PyArray_UIntDType, "uint") < 0 ||
        make_partitions_<npy::long_tag>(&PyArray_LongDType, "long") < 0 ||
        make_partitions_<npy::ulong_tag>(&PyArray_ULongDType, "ulong") < 0 ||
        make_partitions_<npy::longlong_tag>(&PyArray_LongLongDType, "longlong") < 0 ||
        make_partitions_<npy::ulonglong_tag>(&PyArray_ULongLongDType, "ulonglong") < 0 ||
        make_partitions_<npy::float_tag>(&PyArray_FloatDType, "float") < 0 ||
        make_partitions_<npy::double_tag>(&PyArray_DoubleDType, "double") < 0 ||
        make_partitions_<npy::longdouble_tag>(&PyArray_LongDoubleDType, "longdouble") < 0 ||
        make_partitions_<npy::cfloat_tag>(&PyArray_CFloatDType, "cfloat") < 0 ||
        make_partitions_<npy::cdouble_tag>(&PyArray_CDoubleDType, "cdouble") < 0 ||
        make_partitions_<npy::clongdouble_tag>(&PyArray_CLongDoubleDType, "clongdouble") < 0 ||
        make_partitions_<npy::datetime_tag>(&PyArray_DatetimeDType, "datetime") < 0 ||
        make_partitions_<npy::timedelta_tag>(&PyArray_TimedeltaDType, "timedelta") < 0 ||
        make_partitions_<npy::half_tag>(&PyArray_HalfDType, "half") < 0) {
        return -1;
    }
    return 0;
}

#endif
