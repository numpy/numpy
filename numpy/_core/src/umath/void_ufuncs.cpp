#include <Python.h>

#include <cstring>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "dispatching.h"
#include "dtypemeta.h"
#include "void_ufuncs.h"


enum class BITWISE_OP {
    AND,
    OR,
    XOR,
};


template <BITWISE_OP operation>
static int
void_bitwise_loop(
        PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *NPY_UNUSED(auxdata))
{
    const npy_intp input_itemsize = context->descriptors[0]->elsize;
    const npy_intp output_itemsize = context->descriptors[2]->elsize;
    const npy_intp operation_itemsize = input_itemsize < output_itemsize ?
            input_itemsize : output_itemsize;
    npy_ubyte *left = (npy_ubyte *)data[0];
    npy_ubyte *right = (npy_ubyte *)data[1];
    npy_ubyte *out = (npy_ubyte *)data[2];

    for (npy_intp i = 0; i < dimensions[0]; i++) {
        npy_intp j = 0;
        for (; j < operation_itemsize; j++) {
            if constexpr (operation == BITWISE_OP::AND) {
                out[j] = left[j] & right[j];
            }
            else if constexpr (operation == BITWISE_OP::OR) {
                out[j] = left[j] | right[j];
            }
            else {
                out[j] = left[j] ^ right[j];
            }
        }
        for (; j < output_itemsize; j++) {
            out[j] = 0;
        }
        left += strides[0];
        right += strides[1];
        out += strides[2];
    }
    return 0;
}


template <BITWISE_OP operation>
static int
void_bitwise_reduction_initial(
        PyArrayMethod_Context *context,
        npy_bool NPY_UNUSED(reduction_is_empty), void *initial)
{
    constexpr int identity = operation == BITWISE_OP::AND ? 0xff : 0x00;
    if (context->descriptors[0]->elsize > 0) {
        std::memset(initial, identity, context->descriptors[0]->elsize);
    }
    return 1;
}


static NPY_CASTING
void_bitwise_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    for (int i = 0; i < 3; i++) {
        if (given_descrs[i] != nullptr &&
                (PyDataType_HASFIELDS(given_descrs[i]) ||
                 PyDataType_HASSUBARRAY(given_descrs[i]))) {
            PyErr_SetString(PyExc_TypeError,
                    "bitwise operations are not supported for structured "
                    "void dtypes");
            return _NPY_ERROR_OCCURRED_IN_CAST;
        }
    }

    if (given_descrs[0]->elsize != given_descrs[1]->elsize) {
        PyErr_Format(PyExc_TypeError,
                "bitwise operations require void inputs with matching item "
                "sizes, got %zd and %zd",
                given_descrs[0]->elsize, given_descrs[1]->elsize);
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    if (given_descrs[2] == nullptr) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[2] = given_descrs[0];
        return NPY_NO_CASTING;
    }

    Py_INCREF(given_descrs[2]);
    loop_descrs[2] = given_descrs[2];
    if (given_descrs[0]->elsize < given_descrs[2]->elsize) {
        return NPY_SAFE_CASTING;
    }
    if (given_descrs[0]->elsize > given_descrs[2]->elsize) {
        return NPY_SAME_KIND_CASTING;
    }
    return NPY_NO_CASTING;
}


static int
add_void_bitwise_loop(
        PyObject *umath, const char *ufunc_name, const char *loop_name,
        PyArrayMethod_StridedLoop *loop,
        PyArrayMethod_GetReductionInitial *reduction_initial)
{
    PyObject *ufunc = nullptr;
    int result = PyDict_GetItemStringRef(umath, ufunc_name, &ufunc);
    if (result < 0) {
        return -1;
    }
    if (result == 0) {
        PyErr_Format(PyExc_RuntimeError, "missing ufunc '%s'", ufunc_name);
        return -1;
    }

    PyArray_DTypeMeta *dtypes[] = {
        &PyArray_VoidDType, &PyArray_VoidDType, &PyArray_VoidDType,
    };
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, (void *)loop},
        {NPY_METH_resolve_descriptors,
                (void *)&void_bitwise_resolve_descriptors},
        {NPY_METH_get_reduction_initial, (void *)reduction_initial},
        {0, nullptr},
    };
    PyArrayMethod_Spec spec = {};
    spec.name = loop_name;
    spec.nin = 2;
    spec.nout = 1;
    spec.casting = NPY_SAME_KIND_CASTING;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    spec.dtypes = dtypes;
    spec.slots = slots;
    result = PyUFunc_AddLoopFromSpec_int(ufunc, &spec, 1);
    Py_DECREF(ufunc);
    return result;
}


NPY_NO_EXPORT int
init_void_ufuncs(PyObject *umath)
{
    if (add_void_bitwise_loop(
            umath, "bitwise_and", "void_bitwise_and",
            void_bitwise_loop<BITWISE_OP::AND>,
            void_bitwise_reduction_initial<BITWISE_OP::AND>) < 0) {
        return -1;
    }
    if (add_void_bitwise_loop(
            umath, "bitwise_or", "void_bitwise_or",
            void_bitwise_loop<BITWISE_OP::OR>,
            void_bitwise_reduction_initial<BITWISE_OP::OR>) < 0) {
        return -1;
    }
    if (add_void_bitwise_loop(
            umath, "bitwise_xor", "void_bitwise_xor",
            void_bitwise_loop<BITWISE_OP::XOR>,
            void_bitwise_reduction_initial<BITWISE_OP::XOR>) < 0) {
        return -1;
    }
    return 0;
}
