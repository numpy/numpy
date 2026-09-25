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

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "array_method.h"
#include "dispatching.h"
#include "dtypemeta.h"

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
 * Promotes to the common DType of the inputs.  A dtype without a
 * `minimummaximum` loop therefore reports no loop, rather than reaching a
 * loop of another dtype by casting.
 */
static int
minimummaximum_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *const op_dtypes[],
        PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    PyUFuncObject *minmax = (PyUFuncObject *)ufunc;
    PyArray_DTypeMeta *common = NULL;

    /* A homogeneous output signature fixes the operation DType. */
    for (int iop = minmax->nin; iop < minmax->nargs; iop++) {
        if (signature[iop] == NULL) {
            continue;
        }
        if (common == NULL) {
            common = signature[iop];
        }
        else if (common != signature[iop]) {
            common = NULL;
            break;
        }
    }
    if (common != NULL) {
        Py_INCREF(common);
    }
    else {
        PyArray_DTypeMeta *inputs[NPY_MAXARGS];
        int ninput = 0;
        for (int iop = 0; iop < minmax->nin; iop++) {
            if (op_dtypes[iop] != NULL) {
                inputs[ninput++] = op_dtypes[iop];
            }
        }
        if (ninput == 0) {
            /* Nothing to promote from, so there is no loop */
            return -1;
        }
        common = PyArray_PromoteDTypeSequence(ninput, inputs);
        if (common == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                /* Promotion failing means there is no loop */
                PyErr_Clear();
            }
            return -1;
        }
    }

    for (int iop = 0; iop < minmax->nargs; iop++) {
        PyArray_DTypeMeta *dt = signature[iop] != NULL ? signature[iop] : common;
        Py_INCREF(dt);
        new_op_dtypes[iop] = dt;
    }
    Py_DECREF(common);
    return 0;
}


static int
register_minimummaximum_promoter(PyObject *ufunc)
{
    PyObject *dtypes = PyTuple_Pack(4, Py_None, Py_None, Py_None, Py_None);
    if (dtypes == NULL) {
        return -1;
    }
    PyObject *promoter = PyCapsule_New(
            (void *)&minimummaximum_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter == NULL) {
        Py_DECREF(dtypes);
        return -1;
    }
    int res = PyUFunc_AddPromoter(ufunc, dtypes, promoter);
    Py_DECREF(promoter);
    Py_DECREF(dtypes);
    return res;
}


/*
 * `resolve_descriptors` for the datetime and timedelta loops, which operate on
 * the common unit of the inputs.  The other loops are not parametric and use
 * the default legacy resolution.
 */
static NPY_CASTING
minimummaximum_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[],
        npy_intp *NPY_UNUSED(view_offset))
{
    int nargs = self->nin + self->nout;
    PyArray_Descr *common = NULL;

    for (int iop = 0; iop < self->nin; iop++) {
        if (given_descrs[iop] == NULL) {
            continue;
        }
        if (common == NULL) {
            Py_INCREF(given_descrs[iop]);
            common = given_descrs[iop];
        }
        else {
            Py_SETREF(common,
                    PyArray_PromoteTypes(common, given_descrs[iop]));
            if (common == NULL) {
                return (NPY_CASTING)-1;
            }
        }
    }
    if (common == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "minimummaximum requires at least one input descriptor");
        return (NPY_CASTING)-1;
    }
    Py_SETREF(common, NPY_DT_CALL_ensure_canonical(common));
    if (common == NULL) {
        return (NPY_CASTING)-1;
    }

    NPY_CASTING casting = NPY_NO_CASTING;
    for (int iop = 0; iop < nargs; iop++) {
        if (given_descrs[iop] != NULL && given_descrs[iop] != common) {
            casting = NPY_SAFE_CASTING;
        }
        Py_INCREF(common);
        loop_descrs[iop] = common;
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
