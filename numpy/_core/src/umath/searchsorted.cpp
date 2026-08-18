/**
 * This module provides the inner loops for the searchsorted gufuncs
 */

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"

#include "array_method.h"
#include "dispatching.h"
#include "dtypemeta.h"
#include "ufunc_object.h"
#include <string.h>

#include "npy_binsearch.h"
#include "common.h"


#include "searchsorted.h"

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))


/*
 * Comparing against a NaN key raises FE_INVALID, which searchsorted has never
 * reported.  NPY_METH_NO_FLOATINGPOINT_ERRORS does not help: for a gufunc it
 * is combined with the iterator transfer flags, which are zero when nothing
 * is buffered, and the combination only keeps the flag if both sides have it.
 * So drop what the searches raise and put back what the caller had pending.
 */
static inline int
searchsorted_save_floatstatus(char *param)
{
    return npy_clear_floatstatus_barrier(param);
}

static inline void
searchsorted_restore_floatstatus(char *param, int saved)
{
    npy_clear_floatstatus_barrier(param);
    if (saved & NPY_FPE_DIVIDEBYZERO) {
        npy_set_floatstatus_divbyzero();
    }
    if (saved & NPY_FPE_OVERFLOW) {
        npy_set_floatstatus_overflow();
    }
    if (saved & NPY_FPE_UNDERFLOW) {
        npy_set_floatstatus_underflow();
    }
    if (saved & NPY_FPE_INVALID) {
        npy_set_floatstatus_invalid();
    }
}


/*
 * The comparison callback only reads the descriptor off the array it is
 * handed, and both operands carry the same one, so one 0-d dummy is enough.
 */
static int
searchsorted_compare(const void *a, const void *b, PyArrayObject *arr_a,
                     PyArrayObject *NPY_UNUSED(arr_b))
{
    PyArray_CompareFunc *compare =
            PyDataType_GetArrFuncs(PyArray_DESCR(arr_a))->compare;
    return compare(a, b, arr_a);
}


static PyArrayObject *
searchsorted_descr_carrier(PyArray_Descr *descr)
{
    Py_INCREF(descr);
    return (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, descr, 0, NULL, NULL, NULL, 0, NULL);
}


/*
 * `generic` selects the legacy comparison function, used by the dtypes
 * without a dedicated kernel.  It needs a descriptor carrier because the
 * comparison callback reads the descriptor off the array it is handed.
 * `side` picks the kernel at compile time, as PyArray_SearchSorted does.
 */
template <bool generic, NPY_SEARCHSIDE side>
static int
searchsorted_loop(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr *descr = context->descriptors[0];
    PyArray_BinSearchFunc *bs = get_binsearch_func(descr, side);
    if (bs == NULL) {
        /* Unreachable, but the typed loop may run without the GIL. */
        NPY_ALLOW_C_API_DEF;
        NPY_ALLOW_C_API;
        PyErr_SetString(PyExc_TypeError, "compare not supported for type");
        NPY_DISABLE_C_API;
        return -1;
    }
    PyArrayObject *carrier = NULL;
    if (generic) {
        carrier = searchsorted_descr_carrier(descr);
        if (carrier == NULL) {
            return -1;
        }
    }

    npy_intp n_outer = dimensions[0];
    npy_intp n = dimensions[1];
    npy_intp m = dimensions[2];

    /* outer strides for a, v, out, then their core strides */
    npy_intp s_a = strides[0], s_v = strides[1], s_out = strides[2];
    npy_intp a_str = strides[3], v_str = strides[4], out_str = strides[5];

    char *a_o = data[0], *v_o = data[1], *out_o = data[2];

    int saved = searchsorted_save_floatstatus((char *)&n_outer);
    for (npy_intp k = 0; k < n_outer; k++,
                                      a_o += s_a, v_o += s_v, out_o += s_out) {
        bs(a_o, v_o, out_o, n, m, a_str, v_str, out_str, carrier, carrier,
           generic ? &searchsorted_compare : NULL);
    }
    searchsorted_restore_floatstatus((char *)&n_outer, saved);
    Py_XDECREF(carrier);
    return (generic && PyErr_Occurred()) ? -1 : 0;
}


template <bool generic, NPY_SEARCHSIDE side>
static int
searchsorted_sorter_loop(PyArrayMethod_Context *context, char *const data[],
                         npy_intp const dimensions[], npy_intp const strides[],
                         NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr *descr = context->descriptors[0];
    PyArray_ArgBinSearchFunc *bs = get_argbinsearch_func(descr, side);
    if (bs == NULL) {
        /* Unreachable, but the typed loop may run without the GIL. */
        NPY_ALLOW_C_API_DEF;
        NPY_ALLOW_C_API;
        PyErr_SetString(PyExc_TypeError, "compare not supported for type");
        NPY_DISABLE_C_API;
        return -1;
    }
    PyArrayObject *carrier = NULL;
    if (generic) {
        carrier = searchsorted_descr_carrier(descr);
        if (carrier == NULL) {
            return -1;
        }
    }

    npy_intp n_outer = dimensions[0];
    npy_intp n = dimensions[1];
    npy_intp m = dimensions[2];

    /* outer strides for a, v, sorter, out, then their core strides */
    npy_intp s_a = strides[0], s_v = strides[1];
    npy_intp s_sort = strides[2], s_out = strides[3];
    npy_intp a_str = strides[4], v_str = strides[5];
    npy_intp sort_str = strides[6], out_str = strides[7];

    char *a_o = data[0], *v_o = data[1], *sort_o = data[2], *out_o = data[3];

    int ret = 0;
    int saved = searchsorted_save_floatstatus((char *)&n_outer);
    for (npy_intp k = 0; k < n_outer; k++, a_o += s_a, v_o += s_v,
                                          sort_o += s_sort, out_o += s_out) {
        if (bs(a_o, v_o, sort_o, out_o, n, m, a_str, v_str, sort_str, out_str,
               carrier, carrier,
               generic ? &searchsorted_compare : NULL) < 0) {
            /* The kernel reports an out of bounds sorter entry but does not
             * set an error itself. */
            NPY_ALLOW_C_API_DEF;
            NPY_ALLOW_C_API;
            PyErr_SetString(PyExc_ValueError, "Sorter index out of range.");
            NPY_DISABLE_C_API;
            ret = -1;
            break;
        }
    }
    searchsorted_restore_floatstatus((char *)&n_outer, saved);
    Py_XDECREF(carrier);
    return (ret == 0 && generic && PyErr_Occurred()) ? -1 : ret;
}


/*
 * `generic` picks the legacy comparison function over the dedicated kernels
 * from binsearch.cpp.
 */
static int
add_searchsorted_loop(PyObject *ufunc, PyArray_DTypeMeta *dt, int with_sorter,
                      NPY_SEARCHSIDE side, int generic)
{
    PyArray_DTypeMeta *intp_dt = PyArray_DTypeFromTypeNum(NPY_INTP);
    if (intp_dt == NULL) {
        return -1;
    }

    /* {a, v, out} or {a, v, sorter, out} */
    PyArray_DTypeMeta *dtypes[4] = {dt, dt, intp_dt, intp_dt};

    void *loop;
    if (side == NPY_SEARCHLEFT) {
        if (generic) {
            loop = (void *)(with_sorter
                    ? searchsorted_sorter_loop<true, NPY_SEARCHLEFT>
                    : searchsorted_loop<true, NPY_SEARCHLEFT>);
        }
        else {
            loop = (void *)(with_sorter
                    ? searchsorted_sorter_loop<false, NPY_SEARCHLEFT>
                    : searchsorted_loop<false, NPY_SEARCHLEFT>);
        }
    }
    else {
        if (generic) {
            loop = (void *)(with_sorter
                    ? searchsorted_sorter_loop<true, NPY_SEARCHRIGHT>
                    : searchsorted_loop<true, NPY_SEARCHRIGHT>);
        }
        else {
            loop = (void *)(with_sorter
                    ? searchsorted_sorter_loop<false, NPY_SEARCHRIGHT>
                    : searchsorted_loop<false, NPY_SEARCHRIGHT>);
        }
    }
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, loop},
        {0, NULL}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = with_sorter ? "searchsorted_sorter_loop" : "searchsorted_loop";
    spec.nin = with_sorter ? 3 : 2;
    spec.nout = 1;
    spec.casting = NPY_NO_CASTING;
    /* Discarded by the gufunc machinery, see searchsorted_restore_floatstatus */
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    if (generic) {
        /* the legacy comparison function may call into Python */
        spec.flags = (NPY_ARRAYMETHOD_FLAGS)(spec.flags
                                             | NPY_METH_REQUIRES_PYAPI);
    }
    spec.dtypes = dtypes;
    spec.slots = slots;

    int ret = PyUFunc_AddLoopFromSpec_int(ufunc, &spec, 1);
    Py_DECREF(intp_dt);
    return ret;
}


static int
add_searchsorted_loop_for_typenum(PyObject *ufunc, int typenum,
                                  int with_sorter, NPY_SEARCHSIDE side,
                                  int generic)
{
    PyArray_DTypeMeta *dt = PyArray_DTypeFromTypeNum(typenum);
    if (dt == NULL) {
        return -1;
    }
    int ret = add_searchsorted_loop(ufunc, dt, with_sorter, side, generic);
    Py_DECREF(dt);
    return ret;
}


/* The four gufuncs differ only in side and in whether a sorter is taken. */
static NPY_SEARCHSIDE
searchsorted_ufunc_side(PyUFuncObject *ufunc)
{
    const char *name = ufunc_get_name_cstr(ufunc);
    return strstr(name, "right") != NULL ? NPY_SEARCHRIGHT : NPY_SEARCHLEFT;
}


/*
 * Dtypes registered at run time cannot be given a loop up front, so install
 * one, using the legacy comparison function, on first use.
 */
static int
searchsorted_promoter(PyObject *ufunc, PyArray_DTypeMeta *const op_dtypes[],
                      PyArray_DTypeMeta *const NPY_UNUSED(signature[]),
                      PyArray_DTypeMeta *new_op_dtypes[])
{
    PyUFuncObject *ufunc_obj = (PyUFuncObject *)ufunc;
    PyArray_DTypeMeta *dt = op_dtypes[0];

    /* The wrapper casts both searched operands to one descriptor. */
    if (dt == NULL || op_dtypes[1] != dt || !NPY_DT_is_legacy(dt)
            || NPY_DT_SLOTS(dt)->f.compare == NULL) {
        return -1;
    }
    if (add_searchsorted_loop(ufunc, dt, ufunc_obj->nin == 3,
                              searchsorted_ufunc_side(ufunc_obj), 1) < 0) {
        return -1;
    }

    PyArray_DTypeMeta *intp_dt = PyArray_DTypeFromTypeNum(NPY_INTP);
    if (intp_dt == NULL) {
        return -1;
    }
    new_op_dtypes[0] = (PyArray_DTypeMeta *)Py_NewRef(dt);
    new_op_dtypes[1] = (PyArray_DTypeMeta *)Py_NewRef(dt);
    for (int i = 2; i < ufunc_obj->nargs; i++) {
        new_op_dtypes[i] = (PyArray_DTypeMeta *)Py_NewRef(intp_dt);
    }
    Py_DECREF(intp_dt);
    return 0;
}


static int
add_searchsorted_promoter(PyObject *ufunc)
{
    PyUFuncObject *ufunc_obj = (PyUFuncObject *)ufunc;
    PyObject *dtypes[NPY_MAXARGS];

    for (int i = 0; i < ufunc_obj->nargs; i++) {
        dtypes[i] = (PyObject *)&PyArrayDescr_Type;
    }
    PyObject *info = PyArray_TupleFromItems(ufunc_obj->nargs, dtypes, 0);
    if (info == NULL) {
        return -1;
    }
    PyObject *promoter = PyCapsule_New((void *)&searchsorted_promoter,
                                       "numpy._ufunc_promoter", NULL);
    if (promoter == NULL) {
        Py_DECREF(info);
        return -1;
    }
    int res = PyUFunc_AddPromoter(ufunc, info, promoter);
    Py_DECREF(info);
    Py_DECREF(promoter);
    return res;
}


NPY_NO_EXPORT int
init_searchsorted_ufuncs(PyObject *umath)
{
    /* Keep in sync with the tag list in npysort/binsearch.cpp. */
    static const int typed_typenums[] = {
        NPY_BOOL,
        NPY_BYTE, NPY_UBYTE, NPY_SHORT, NPY_USHORT,
        NPY_INT, NPY_UINT, NPY_LONG, NPY_ULONG,
        NPY_LONGLONG, NPY_ULONGLONG,
        NPY_HALF, NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
        NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
        NPY_DATETIME, NPY_TIMEDELTA,
    };

    /*
     * Searched through the legacy comparison function, so that `axis` is not
     * limited to the dtypes above.  StringDType still falls back, its
     * comparison needs its allocators held.
     */
    static const int generic_typenums[] = {
        NPY_OBJECT, NPY_STRING, NPY_UNICODE, NPY_VOID,
    };

    static const struct {
        const char *name;
        int with_sorter;
        NPY_SEARCHSIDE side;
    } ufuncs[] = {
        {"_searchsorted_left", 0, NPY_SEARCHLEFT},
        {"_searchsorted_right", 0, NPY_SEARCHRIGHT},
        {"_searchsorted_left_sorter", 1, NPY_SEARCHLEFT},
        {"_searchsorted_right_sorter", 1, NPY_SEARCHRIGHT},
    };

    for (size_t u = 0; u < ARRAY_SIZE(ufuncs); u++) {
        PyObject *name = PyUnicode_FromString(ufuncs[u].name);
        if (name == NULL) {
            return -1;
        }
        PyObject *ufunc = PyObject_GetItem(umath, name);
        Py_DECREF(name);
        if (ufunc == NULL) {
            return -1;
        }
        int res = 0;
        for (size_t i = 0;
                res == 0 && i < ARRAY_SIZE(typed_typenums); i++) {
            res = add_searchsorted_loop_for_typenum(
                    ufunc, typed_typenums[i], ufuncs[u].with_sorter,
                    ufuncs[u].side, 0);
        }
        for (size_t i = 0;
                res == 0 && i < ARRAY_SIZE(generic_typenums); i++) {
            res = add_searchsorted_loop_for_typenum(
                    ufunc, generic_typenums[i], ufuncs[u].with_sorter,
                    ufuncs[u].side, 1);
        }
        if (res == 0) {
            res = add_searchsorted_promoter(ufunc);
        }
        Py_DECREF(ufunc);
        if (res < 0) {
            return -1;
        }
    }
    return 0;
}
