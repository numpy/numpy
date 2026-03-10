/*
 * real/imag ufuncs: 1 input (complex), 1 output (float).
 * No dedicated loops: either a view is returned (view_offset path in ufunc
 * fast path) or the get_loop returns a copy loop for the real/imag part.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include <Python.h>
#include "npy_pycompat.h"  // PyObject_GetOptionalAttr
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "dispatching.h"

#include "numpyos.h"
#include "dtypemeta.h"
#include "dtype_transfer.h"
#include "lowlevel_strided_loops.h"
#include "array_method.h"

#include "real_imag_ufuncs.h"


template <int real_num, bool real_part>
static NPY_CASTING
complex_to_real_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    loop_descrs[1] = PyArray_DescrFromType(real_num);

    if (PyDataType_ISBYTESWAPPED(loop_descrs[0])) {
        Py_SETREF(
            loop_descrs[1], PyArray_DescrNewByteorder(loop_descrs[1], NPY_SWAP));
        if (loop_descrs[1] == NULL) {
            Py_DECREF(loop_descrs[0]);
            return _NPY_ERROR_OCCURRED_IN_CAST;
        }
    }
    if constexpr (real_part) {
        *view_offset = 0;
    }
    else {
        *view_offset = loop_descrs[1]->elsize;
    }
    return NPY_NO_CASTING;
}


/* We shouldn't normally use it, but define a simple loop anyway. */
template <typename real_type, bool real_part>
static int extract_complex_part_loop(
        PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp istride = strides[0];
    npy_intp ostride = strides[1];

    while (N--) {
        real_type value;
        if constexpr (real_part) {
            value = *reinterpret_cast<real_type *>(in);
        }
        else {
            value = *reinterpret_cast<real_type *>(in + sizeof(real_type));
        }
        *reinterpret_cast<real_type *>(out) = value;
        in += istride;
        out += ostride;
    }
    return 0;
}


template <auto component>
static int
object_get_comp_strided_loop(
        PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp istride = strides[0];
    npy_intp ostride = strides[1];

    while (N--) {
        PyObject *obj = *reinterpret_cast<PyObject **>(in);
        PyObject *attr;
        if (PyObject_GetOptionalAttr(obj, npy_interned_str.*component, &attr) < 0) {
            return -1;
        }
        if (attr == NULL) {
            if constexpr (component == &npy_interned_str_struct::real) {
                attr = Py_NewRef(obj);  // just use the old object...
            }
            else {
                // Use long zero as a best bet (also historical value)
                attr = PyLong_FromLong(0);
                if (attr == NULL) {
                    return -1;
                }
            }
        }

        Py_XSETREF((*reinterpret_cast<PyObject **>(out)), attr);
        in += istride;
        out += ostride;
    }
    return 0;
}


template <int complex_num, typename real_type, int real_num, bool real_part>
static int
register_one_for_type(const char *name)
{
    PyArray_DTypeMeta *dtypes[2] = {
        PyArray_DTypeFromTypeNum(complex_num),
        PyArray_DTypeFromTypeNum(real_num),
    };
    PyType_Slot meth_slots[] = {
        {NPY_METH_resolve_descriptors, (void *)&complex_to_real_resolve_descriptors<real_num, real_part>},
        {NPY_METH_strided_loop, (void *)&extract_complex_part_loop<real_type, real_part>},
        {0, NULL}
    };
    PyArrayMethod_Spec meth_spec = {
        .nin = 1,
        .nout = 1,
        .dtypes = dtypes,
        .slots = meth_slots,
    };
    PyUFunc_LoopSlot slots[] = {
        {.name = name, .spec = &meth_spec},
        {0, nullptr}
    };
    int res = PyUFunc_AddLoopsFromSpecs(slots);
    Py_DECREF(dtypes[0]);
    Py_DECREF(dtypes[1]);
    return res;
}


template <int complex_num, typename real_type, int real_num>
static int
register_both_for_type() {
    if (register_one_for_type<complex_num, real_type, real_num, true>(".real") < 0) {
        return -1;
    }
    if (register_one_for_type<complex_num, real_type, real_num, false>(".imag") < 0) {
        return -1;
    }
    return 0;
}


template <auto component>
static int
register_one_object_loop(const char *name)
{
    PyArray_DTypeMeta *dtypes[2] = {&PyArray_ObjectDType, &PyArray_ObjectDType};
    PyType_Slot meth_slots[] = {
        {NPY_METH_strided_loop, (void *)&object_get_comp_strided_loop<component>},
        {0, nullptr}
    };
    PyArrayMethod_Spec meth_spec = {
        .nin = 1,
        .nout = 1,
        .dtypes = dtypes,
        .slots = meth_slots,
    };
    PyUFunc_LoopSlot slots[] = {
        {.name = name, .spec = &meth_spec},
        {0, nullptr}
    };
    return PyUFunc_AddLoopsFromSpecs(slots);
}


template <auto slot>
static int
real_imag_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    PyBoundArrayMethodObject *meth = NPY_DT_SLOTS(op_dtypes[0])->*slot;
    if (meth == NULL) {
        return -1;  // nothing to do.
    }
    if (signature[1] != NULL && signature[1] != meth->dtypes[1]) {
        // out signature requested, but not compatible (may be unreachable).
        return -1;
    }

    /*
     * Dynamically add the loop to the ufunc, since it seem it was missing.
     */
    PyObject *DType_tuple = PyTuple_FromArray((PyObject **)meth->dtypes, 2);
    if (DType_tuple == NULL) {
        return -1;
    }
    PyObject *info = PyTuple_Pack(2, DType_tuple, meth->method);
    Py_DECREF(DType_tuple);
    if (info == NULL) {
        return -1;
    }
    int res = PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);
    Py_DECREF(info);
    if (res < 0) {
        return -1;
    }
    new_op_dtypes[0] = NPY_DT_NewRef(meth->dtypes[0]);
    new_op_dtypes[1] = NPY_DT_NewRef(meth->dtypes[1]);
    return 1;
}



template <auto slot>
static int
add_promoter_for_slot(PyObject *ufunc)
{
    PyObject *promoter = PyCapsule_New(
        (void *)real_imag_promoter<slot>, "numpy._ufunc_promoter", NULL);
    if (promoter == NULL) {
        return -1;
    }
    PyObject *dtypes[2] = {(PyObject *)&PyArrayDescr_Type, (PyObject *)&PyArrayDescr_Type};
    PyObject *info = PyTuple_FromArray(dtypes, 2);
    if (info == NULL) {
        Py_DECREF(promoter);
        return -1;
    }
    int res = PyUFunc_AddPromoter(ufunc, info, promoter);
    Py_DECREF(info);
    Py_DECREF(promoter);
    return res;
}


NPY_NO_EXPORT int
init_real_imag_ufuncs(PyObject *umath)
{
    int res = -1;
    PyObject *real_ufunc = PyObject_GetAttr(umath, npy_interned_str.real);
    PyObject *imag_ufunc = PyObject_GetAttr(umath, npy_interned_str.imag);
    if (real_ufunc == NULL || imag_ufunc == NULL) {
        goto finish;
    }

    if (register_both_for_type<NPY_CFLOAT, npy_float32, NPY_FLOAT>() < 0) {
        goto finish;
    }
    if (register_both_for_type<NPY_CDOUBLE, npy_float64, NPY_DOUBLE>() < 0) {
        goto finish;
    }
    if (register_both_for_type<NPY_CLONGDOUBLE, npy_longdouble, NPY_LONGDOUBLE>() < 0) {
        goto finish;
    }
    if (register_one_object_loop<&npy_interned_str_struct::real>(".real") < 0) {
        goto finish;
    }
    if (register_one_object_loop<&npy_interned_str_struct::imag>(".imag") < 0) {
        goto finish;
    }

    /*
     * The above actually only adds the method to the DType itself. We deal with
     * the ufunc by adding a general fall-back method that
     */
    if (add_promoter_for_slot<&NPY_DType_Slots::real_meth>(real_ufunc) < 0) {
        goto finish;
    }
    if (add_promoter_for_slot<&NPY_DType_Slots::imag_meth>(imag_ufunc) < 0) {
        goto finish;
    }
    res = 0;
  finish:
    Py_XDECREF(real_ufunc);
    Py_XDECREF(imag_ufunc);

    return res;
}
