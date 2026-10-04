/*
 * Registration of the fused `minimummaximum` reduction loops.
 *
 * The loops themselves are defined next to the `minimum`/`maximum` loops
 * (loops_minmax.dispatch.c.src for the SIMD dtypes and loops.c.src for the
 * rest); this file only attaches the `get_reduction_loop` slot to each
 * dtype's ArrayMethod after the ufunc has been created.
 */
#include <Python.h>

#include "npy_pycompat.h"
#include "object.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "array_method.h"
#include "dispatching.h"
#include "dtypemeta.h"
#include "module_state.h"

#include "loops.h"
#include "minmax.h"


/*
 * `get_reduction_loop` slot for `minimummaximum`: return the dedicated
 * (nout+1)->nout reduction loop for the resolved dtype.
 */
#include "loops_minmax.dispatch.h"
static int
minimummaximum_get_reduction_loop(
        PyArrayMethod_Context *context,
        int NPY_UNUSED(aligned), int NPY_UNUSED(move_references),
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArrayMethod_StridedLoop *loop = NULL;
    NPY_ARRAYMETHOD_FLAGS f = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    switch (context->descriptors[0]->type_num) {
#define SIMD_CASE(TYPE) \
        case NPY_##TYPE: \
            NPY_CPU_DISPATCH_CALL(loop = TYPE##_minimummaximum_reduce); \
            break;
        SIMD_CASE(BYTE)
        SIMD_CASE(UBYTE)
        SIMD_CASE(SHORT)
        SIMD_CASE(USHORT)
        SIMD_CASE(INT)
        SIMD_CASE(UINT)
        SIMD_CASE(LONG)
        SIMD_CASE(ULONG)
        SIMD_CASE(LONGLONG)
        SIMD_CASE(ULONGLONG)
        SIMD_CASE(FLOAT)
        SIMD_CASE(DOUBLE)
        SIMD_CASE(LONGDOUBLE)
#undef SIMD_CASE
        case NPY_BOOL:
            loop = &BOOL_minimummaximum_reduce;
            break;
        case NPY_HALF:
            loop = &HALF_minimummaximum_reduce;
            break;
        case NPY_CFLOAT:
            loop = &CFLOAT_minimummaximum_reduce;
            break;
        case NPY_CDOUBLE:
            loop = &CDOUBLE_minimummaximum_reduce;
            break;
        case NPY_CLONGDOUBLE:
            loop = &CLONGDOUBLE_minimummaximum_reduce;
            break;
        case NPY_DATETIME:
            loop = &DATETIME_minimummaximum_reduce;
            break;
        case NPY_TIMEDELTA:
            loop = &TIMEDELTA_minimummaximum_reduce;
            break;
        case NPY_OBJECT:
            loop = &OBJECT_minimummaximum_reduce;
            f = NPY_METH_REQUIRES_PYAPI;
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError,
                    "minimummaximum reduction: unsupported dtype");
            return -1;
    }
    *out_loop = loop;
    *out_transferdata = NULL;
    *flags = f;
    return 0;
}


/*
 * Promotes to the common DType of the inputs.
 * A dtype without a `minimummaximum` loop reports no loop,
 * instead of reaching a loop of another dtype by casting.
 */
static int
minimummaximum_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[],
        PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    /* A fixed output DType fixes the operation DType. */
    PyArray_DTypeMeta *common = signature[2] != NULL ? signature[2] : signature[3];
    if (common != NULL) {
        Py_INCREF(common);
    }
    else {
        common = PyArray_PromoteDTypeSequence(2, (PyArray_DTypeMeta **)op_dtypes);
        if (common == NULL) {
            if (PyErr_ExceptionMatches(
                        _npy_module_state->static_pydata.DTypePromotionError)) {
                /* Promotion failing means there is no loop */
                PyErr_Clear();
            }
            return -1;
        }
    }

    for (int i = 0; i < 4; i++) {
        PyArray_DTypeMeta *dt = signature[i] != NULL ? signature[i] : common;
        Py_INCREF(dt);
        new_op_dtypes[i] = dt;
    }
    Py_DECREF(common);
    return 0;
}


static int
register_minimummaximum_promoter(PyObject *ufunc)
{
    PyObject *none_tuple = PyTuple_Pack(4, Py_None, Py_None, Py_None, Py_None);
    if (none_tuple == NULL) {
        return -1;
    }
    PyObject *promoter = PyCapsule_New(
            (void *)&minimummaximum_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter == NULL) {
        Py_DECREF(none_tuple);
        return -1;
    }
    int res = PyUFunc_AddPromoter(ufunc, none_tuple, promoter);
    Py_DECREF(none_tuple);
    Py_DECREF(promoter);
    return res;
}


/*
* `resolve_descriptors` for the datetime and timedelta loops that resolves
* to the common unit of the inputs.
* The other loops are not parametric and use the default legacy resolution.
*/
static NPY_CASTING
minimummaximum_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[],
        npy_intp *NPY_UNUSED(view_offset))
{
    PyArray_Descr *common = PyArray_PromoteTypes(given_descrs[0], given_descrs[1]);
    if (common == NULL) {
        return (NPY_CASTING)-1;
    }

    NPY_CASTING casting = NPY_NO_CASTING;
    for (int i = 0; i < 4; i++) {
        if (given_descrs[i] != NULL && given_descrs[i] != common) {
            casting = NPY_SAFE_CASTING;
        }
        Py_INCREF(common);
        loop_descrs[i] = common;
    }
    Py_DECREF(common);
    return casting;
}


NPY_NO_EXPORT int
init_minimummaximum(PyObject *umath)
{
    static const int typenums[] = {
        NPY_BOOL,
        NPY_BYTE, NPY_UBYTE, NPY_SHORT, NPY_USHORT, NPY_INT, NPY_UINT,
        NPY_LONG, NPY_ULONG, NPY_LONGLONG, NPY_ULONGLONG,
        NPY_HALF, NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
        NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
        NPY_DATETIME, NPY_TIMEDELTA,
        NPY_OBJECT,
    };

    PyObject *ufunc = NULL;
    int res = PyDict_GetItemStringRef(umath, "minimummaximum", &ufunc);
    if (res < 0) {
        return -1;
    }
    if (res == 0) {
        PyErr_SetString(PyExc_RuntimeError,
                "internal NumPy error: minimummaximum ufunc not found");
        return -1;
    }

    for (size_t k = 0; k < sizeof(typenums) / sizeof(typenums[0]); k++) {
        PyArray_DTypeMeta *dt = PyArray_DTypeFromTypeNum(typenums[k]);
        if (dt == NULL) {
            goto fail;
        }
        PyObject *info = get_info_no_cast((PyUFuncObject *)ufunc, dt, 4);
        Py_DECREF(dt);
        if (info == NULL) {
            goto fail;
        }
        if (info == Py_None || !PyObject_TypeCheck(info, &PyArrayMethod_Type)) {
            PyErr_SetString(PyExc_RuntimeError,
                    "internal NumPy error: minimummaximum loop not found");
            goto fail;
        }
        PyArrayMethodObject *meth = (PyArrayMethodObject *)info;
        /*
         * `minimummaximum` is reorderable (like `minimum`/`maximum`), but the
         * legacy ArrayMethod only sets that flag for nin==2/nout==1 loops, so
         * set it here to allow multi-axis reductions.
         */
        meth->flags = (NPY_ARRAYMETHOD_FLAGS)(
                meth->flags | NPY_METH_IS_REORDERABLE);
        meth->get_reduction_loop = &minimummaximum_get_reduction_loop;
        if (typenums[k] == NPY_DATETIME || typenums[k] == NPY_TIMEDELTA) {
            meth->resolve_descriptors = &minimummaximum_resolve_descriptors;
        }
    }

    if (register_minimummaximum_promoter(ufunc) < 0) {
        goto fail;
    }

    Py_DECREF(ufunc);
    return 0;

  fail:
    Py_DECREF(ufunc);
    return -1;
}
