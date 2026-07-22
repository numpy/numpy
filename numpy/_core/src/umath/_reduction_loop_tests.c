/* -*- c -*- */

/*
 * _reduction_loop_tests.c
 * Two minimal float64-only `minimummaximum` ufuncs exercising the
 * NPY_METH_get_reduction_loop ArrayMethod slot: a 2-in/2-out forward
 * loop plus a 3-in/2-out reduction loop, so `.reduce(a)` returns
 * (min, max). `minimummaximum_with_identity` additionally registers
 * NPY_METH_get_multi_reduction_initials (+inf/-inf).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#if defined(NPY_INTERNAL_BUILD)
#undef NPY_INTERNAL_BUILD
#endif
// for NPY_METH_get_reduction_loop
#define NPY_TARGET_VERSION NPY_2_6_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"
#include "numpy/npy_math.h"


static inline double
nan_min(double a, double b)
{
    return (a <= b || npy_isnan(a)) ? a : b;
}

static inline double
nan_max(double a, double b)
{
    return (a >= b || npy_isnan(a)) ? a : b;
}


static int
double_minimummaximum_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp n = dimensions[0];
    char *in1 = data[0], *in2 = data[1];
    char *out1 = data[2], *out2 = data[3];
    npy_intp is1 = strides[0], is2 = strides[1];
    npy_intp os1 = strides[2], os2 = strides[3];

    for (npy_intp i = 0; i < n; i++) {
        double a = *(double *)in1;
        double b = *(double *)in2;
        *(double *)out1 = nan_min(a, b);
        *(double *)out2 = nan_max(a, b);
        in1 += is1; in2 += is2; out1 += os1; out2 += os2;
    }
    return 0;
}


static int
double_minimummaximum_reduce_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp n = dimensions[0];
    char *acc_min = data[0], *acc_max = data[1], *x = data[2];
    char *out_min = data[3], *out_max = data[4];
    npy_intp s_amin = strides[0], s_amax = strides[1], s_x = strides[2];
    npy_intp s_omin = strides[3], s_omax = strides[4];

    for (npy_intp i = 0; i < n; i++) {
        double cur_min = *(double *)acc_min;
        double cur_max = *(double *)acc_max;
        double val = *(double *)x;
        *(double *)out_min = nan_min(cur_min, val);
        *(double *)out_max = nan_max(cur_max, val);
        acc_min += s_amin; acc_max += s_amax; x += s_x;
        out_min += s_omin; out_max += s_omax;
    }
    return 0;
}


static int
minimummaximum_get_reduction_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        int NPY_UNUSED(aligned), int NPY_UNUSED(move_references),
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *out_loop = &double_minimummaximum_reduce_loop;
    *out_transferdata = NULL;
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    return 0;
}


static int
minimummaximum_get_multi_reduction_initials(
        PyArrayMethod_Context *NPY_UNUSED(context),
        npy_bool NPY_UNUSED(reduction_is_empty), void **initials)
{
    *(double *)initials[0] = NPY_INFINITY;
    *(double *)initials[1] = -NPY_INFINITY;
    return 1;
}


static int
minimummaximum_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const NPY_UNUSED(op_dtypes[]),
        PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    PyArray_Descr *double_descr = PyArray_DescrFromType(NPY_DOUBLE);
    if (double_descr == NULL) {
        return -1;
    }
    PyArray_DTypeMeta *double_dt = NPY_DTYPE(double_descr);

    for (int i = 0; i < 4; i++) {
        PyArray_DTypeMeta *dt = signature[i] != NULL ? signature[i] : double_dt;
        Py_INCREF(dt);
        new_op_dtypes[i] = dt;
    }
    Py_DECREF(double_descr);
    return 0;
}


static int
register_minimummaximum_promoter(PyObject *minimummaximum)
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
    int res = PyUFunc_AddPromoter(minimummaximum, none_tuple, promoter);
    Py_DECREF(none_tuple);
    Py_DECREF(promoter);
    return res;
}


static int
add_minimummaximum(PyObject *module, const char *name, int with_identity)
{
    PyObject *minimummaximum = PyUFunc_FromFuncAndData(
            NULL, NULL, NULL, 0, 2, 2, PyUFunc_None, name, NULL, 0);
    if (minimummaximum == NULL) {
        return -1;
    }

    PyArray_Descr *double_descr = PyArray_DescrFromType(NPY_DOUBLE);
    if (double_descr == NULL) {
        Py_DECREF(minimummaximum);
        return -1;
    }
    PyArray_DTypeMeta *dt = NPY_DTYPE(double_descr);
    PyArray_DTypeMeta *dtypes[4] = {dt, dt, dt, dt};

    PyType_Slot slots[4] = {
        {NPY_METH_strided_loop, (void *)&double_minimummaximum_loop},
        {NPY_METH_get_reduction_loop, (void *)&minimummaximum_get_reduction_loop},
        {0, NULL},
        {0, NULL},
    };
    if (with_identity) {
        slots[2].slot = NPY_METH_get_multi_reduction_initials;
        slots[2].pfunc = (void *)&minimummaximum_get_multi_reduction_initials;
    }

    PyArrayMethod_Spec spec = {
        .name = name,
        .nin = 2,
        .nout = 2,
        .casting = NPY_NO_CASTING,
        .flags = NPY_METH_IS_REORDERABLE | NPY_METH_NO_FLOATINGPOINT_ERRORS,
        .dtypes = dtypes,
        .slots = slots,
    };

    int res = PyUFunc_AddLoopFromSpec(minimummaximum, &spec);
    Py_DECREF(double_descr);
    if (res < 0 || register_minimummaximum_promoter(minimummaximum) < 0
            || PyModule_AddObject(module, name, minimummaximum) < 0) {
        Py_XDECREF(minimummaximum);
        return -1;
    }
    return 0;
}


static PyMethodDef ReductionLoopTestsMethods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_reduction_loop_tests",
    NULL,
    -1,
    ReductionLoopTestsMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__reduction_loop_tests(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    if (PyArray_ImportNumPyAPI() < 0) {
        Py_DECREF(m);
        return NULL;
    }
    if (PyUFunc_ImportUFuncAPI() < 0) {
        Py_DECREF(m);
        return NULL;
    }

    if (add_minimummaximum(m, "minimummaximum", 0) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    if (add_minimummaximum(m, "minimummaximum_with_identity", 1) < 0) {
        Py_DECREF(m);
        return NULL;
    }

#ifdef Py_GIL_DISABLED
    // signal this module supports running with the GIL disabled
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    return m;
}
