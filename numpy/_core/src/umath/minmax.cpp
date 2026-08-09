/*
 * Registration of the fused `minimummaximum` reduction loops.
 *
 * The loops themselves are defined next to the `minimum`/`maximum` loops
 * (loops_minmax.dispatch.c.src for the SIMD dtypes and loops.c.src for the
 * rest); this file only attaches the `get_reduction_loop` slot to each
 * dtype's ArrayMethod after the ufunc has been created.
 */
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "npy_cpu_dispatch.h"
#include "array_method.h"
#include "dispatching.h"
#include "dtypemeta.h"

#include "loops.h"
#include "minmax.h"


/*
 * `get_reduction_loop` slot for `minimummaximum`: return the dedicated
 * (nout+1)->nout reduction loop for the resolved dtype.
 */
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

    PyUFuncObject *ufunc =
            (PyUFuncObject *)PyDict_GetItemString(umath, "minimummaximum");
    if (ufunc == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "internal NumPy error: minimummaximum ufunc not found");
        return -1;
    }

    for (size_t k = 0; k < sizeof(typenums) / sizeof(typenums[0]); k++) {
        PyArray_DTypeMeta *dt = PyArray_DTypeFromTypeNum(typenums[k]);
        if (dt == NULL) {
            return -1;
        }
        PyObject *info = get_info_no_cast(ufunc, dt, 4);
        Py_DECREF(dt);
        if (info == NULL) {
            return -1;
        }
        if (info == Py_None || !PyObject_TypeCheck(info, &PyArrayMethod_Type)) {
            PyErr_SetString(PyExc_RuntimeError,
                    "internal NumPy error: minimummaximum loop not found");
            return -1;
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
    }

    return 0;
}
