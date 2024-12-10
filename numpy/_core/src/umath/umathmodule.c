/* -*- c -*- */
/* vim:syntax=c */

/*
 * _UMATHMODULE IS needed in __ufunc_api.h, included from numpy/ufuncobject.h.
 * This is a mess and it would be nice to fix it. It has nothing to do with
 * __ufunc_api.c
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "npy_config.h"
#include "npy_cpu_features.h"
#include "npy_cpu_dispatch.h"
#include "numpy/npy_cpu.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "npy_pycompat.h"
#include "npy_argparse.h"
#include "abstract.h"

#include "numpy/npy_math.h"
#include "number.h"
#include "dispatching.h"
#include "string_ufuncs.h"
#include "stringdtype_ufuncs.h"
#include "special_integer_comparisons.h"
#include "extobj.h"  /* for _extobject_contextvar exposure */
#include "ufunc_type_resolution.h"
#include "legacy_array_method.h"

/* Automatically generated code to define all ufuncs: */
#include "funcs.inc"
#include "__umath_generated.c"


static int
frompyfunc_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *const NPY_UNUSED(op_dtypes[]),
        PyArray_DTypeMeta *const NPY_UNUSED(signature[]),
        PyArray_DTypeMeta *new_op_dtypes[]);

static int
frompyfunc_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *const NPY_UNUSED(op_dtypes[]),
        PyArray_DTypeMeta *const NPY_UNUSED(signature[]),
        PyArray_DTypeMeta *new_op_dtypes[])
{
    int i, nop = ((PyUFuncObject *)ufunc)->nin + ((PyUFuncObject *)ufunc)->nout;

    new_op_dtypes[0] = PyArray_DTypeFromTypeNum(NPY_OBJECT);
    if (new_op_dtypes[0] == NULL) {
        return -1;
    }

    for (i = 1; i < nop; ++i) {
        Py_INCREF(new_op_dtypes[0]);
        new_op_dtypes[i] = new_op_dtypes[0];
    }

    return 0;
}

static int
pyfunc_loop(PyArrayMethod_Context *context,
                char *const data[], npy_intp const dimensions[],
                npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    int nin = context->method->nin;
    int nout = context->method->nout;
    int ntot = nin + nout;
    PyObject *pyfunc = context->method->static_data;
    char *ptrs[NPY_MAXARGS];
    PyObject *arglist, *result;
    PyObject *in, **op;
    npy_intp i;

    for(i = 0; i < ntot; i++) {
        ptrs[i] = data[i];
    }
    while(N--) {
        arglist = PyTuple_New(nin);
        if (arglist == NULL) {
            return -1;
        }
        for(i = 0; i < nin; i++) {
            in = *((PyObject **)ptrs[i]);
            /* We allow NULL, but try to guarantee non-NULL to downstream */
            assert(in != NULL);
            if (in == NULL) {
                in = Py_None;
            }
            PyTuple_SET_ITEM(arglist, i, in);
            Py_INCREF(in);
        }
        result = PyObject_CallObject(pyfunc, arglist);
        Py_DECREF(arglist);
        if (result == NULL) {
            return -1;
        }
        if (nout == 0  && result == Py_None) {
            /* No output expected, no output received, continue */
            Py_DECREF(result);
        }
        else if (nout == 1) {
            /* Single output expected, assign and continue */
            op = (PyObject **)ptrs[nin];
            Py_XDECREF(*op);
            *op = result;
        }
        else if (PyTuple_Check(result) && nout == PyTuple_Size(result)) {
            /*
             * Multiple returns match expected number of outputs, assign
             * and continue. Will also gobble empty tuples if nout == 0.
             */
            for(i = 0; i < nout; i++) {
                op = (PyObject **)ptrs[i+nin];
                Py_XDECREF(*op);
                *op = PyTuple_GET_ITEM(result, i);
                Py_INCREF(*op);
            }
            Py_DECREF(result);
        }
        else {
            /* Mismatch between returns and expected outputs, exit */
            Py_DECREF(result);
            return -1;
        }
        for(i = 0; i < ntot; i++) {
            ptrs[i] += strides[i];
        }
    }

    return 0;
}


PyObject *
ufunc_frompyfunc(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds) {
    PyObject *function, *pyname = NULL;
    int nin, nout, nargs;
    PyUFuncObject *self;
    const char *fname = NULL;
    char *str, *doc;
    Py_ssize_t fname_len = -1;
    PyObject *identity = NULL;  /* note: not the same semantics as Py_None */
    static char *kwlist[] = {"", "nin", "nout", "identity", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oii|$O:frompyfunc", kwlist,
                &function, &nin, &nout, &identity)) {
        return NULL;
    }
    if (!PyCallable_Check(function)) {
        PyErr_SetString(PyExc_TypeError, "function must be callable");
        return NULL;
    }

    nargs = nin + nout;

    pyname = PyObject_GetAttrString(function, "__name__");
    if (pyname) {
        fname = PyUnicode_AsUTF8AndSize(pyname, &fname_len);
    }
    if (fname == NULL) {
        PyErr_Clear();
        fname = "?";
        fname_len = 1;
    }

    str = (char *)PyArray_malloc(fname_len + 14);
    if (str == NULL) {
        Py_XDECREF(pyname);
        return PyErr_NoMemory();
    }
    memcpy(str, fname, fname_len);
    memcpy(str+fname_len, " (vectorized)", 14);
    Py_XDECREF(pyname);

    /* Do a better job someday */
    doc = "dynamic ufunc based on a python function";

    /* Create the ufunc but without any loops by not passing types*/
    self = (PyUFuncObject *)PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
            NULL, NULL,
            NULL, /* ntypes */ 0, nin, nout, identity ? PyUFunc_IdentityValue : PyUFunc_None,
            str, doc, /* unused */ 0, NULL, identity);

    if (self == NULL) {
        PyArray_free(str);
        return NULL;
    }

    PyArray_DTypeMeta *op_dtypes[NPY_MAXARGS];
    for (int arg = 0; arg < nargs; arg++) {
        op_dtypes[arg] = PyArray_DTypeFromTypeNum(NPY_OBJECT);
        /* These DTypes are immortal and adding INCREFs: so borrow it */
        Py_DECREF(op_dtypes[arg]);
    }

    char method_name[101];
    snprintf(method_name, 100, "ArrayMethod_implementation_for_%s", str);

    NPY_ARRAYMETHOD_FLAGS flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS;

    PyArrayMethod_GetReductionInitial *get_reduction_intial = NULL;
    if (nin == 2 && nout == 1) {
        npy_bool reorderable = NPY_FALSE;
        PyObject *identity_obj = PyUFunc_GetDefaultIdentity(
                self, &reorderable);
        if (identity_obj == NULL) {
            PyArray_free(str);
            return NULL;
        }
        /*
         * TODO: For object, "reorderable" is needed(?), because otherwise
         *       we disable multi-axis reductions `arr.sum(0, 1)`. But for
         *       `arr = array([["a", "b"], ["c", "d"]], dtype="object")`
         *       it isn't actually reorderable (order changes result).
         */
        if (reorderable) {
            flags |= NPY_METH_IS_REORDERABLE;
        }
        if (identity_obj != Py_None) {
            get_reduction_intial = &get_initial_from_ufunc;
        }
    }

    PyType_Slot slots[4] = {
        {NPY_METH_strided_loop, &pyfunc_loop},
        {NPY_METH_get_reduction_initial, get_reduction_intial},
        {_NPY_METH_static_data, function},
        {0, NULL},
    };
    PyArrayMethod_Spec spec = {
        .name = method_name,
        .nin = nin,
        .nout = nout,
        .casting = NPY_NO_CASTING,
        .flags = flags,
        .dtypes = op_dtypes,
        .slots = slots,
    };
    int res = PyUFunc_AddLoopFromSpec_int((PyObject *)self, &spec, 0);
    if (res < 0) {
        PyArray_free(str);
        return NULL;
    }

    PyObject *promoter_obj = PyCapsule_New((void *) frompyfunc_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter_obj == NULL) {
        PyArray_free(str);
        return NULL;
    }

    PyObject *dtypes_tuple = PyTuple_New(nargs);
    if (dtypes_tuple == NULL) {
        Py_DECREF(promoter_obj);
        PyArray_free(str);
        return NULL;
    }
    for (int i = 0; i < nargs; i++) {
        PyTuple_SET_ITEM(dtypes_tuple, i, Py_None);
    }

    PyObject *info = PyTuple_Pack(2, dtypes_tuple, promoter_obj);
    Py_DECREF(dtypes_tuple);
    Py_DECREF(promoter_obj);
    if (info == NULL) {
        PyArray_free(str);
        return NULL;
    }

    res = PyUFunc_AddLoop(self, info, 0);
    if (res < 0) {
        PyArray_free(str);
        return NULL;
    }

    Py_INCREF(function);
    self->obj = function;
    self->ptr = str;

    PyObject_GC_Track(self);

    return (PyObject *)self;
}

/* docstring in numpy.add_newdocs.py */
PyObject *
add_newdoc_ufunc(PyObject *NPY_UNUSED(dummy), PyObject *args)
{

    /* 2024-11-12, NumPy 2.2 */
    if (DEPRECATE("_add_newdoc_ufunc is deprecated. "
                  "Use `ufunc.__doc__ = newdoc` instead.") < 0) {
        return NULL;
    }

    PyUFuncObject *ufunc;
    PyObject *str;
    if (!PyArg_ParseTuple(args, "O!O!:_add_newdoc_ufunc", &PyUFunc_Type, &ufunc,
                                        &PyUnicode_Type, &str)) {
        return NULL;
    }
    if (ufunc->doc != NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot change docstring of ufunc with non-NULL docstring");
        return NULL;
    }

    PyObject *tmp = PyUnicode_AsUTF8String(str);
    if (tmp == NULL) {
        return NULL;
    }
    char *docstr = PyBytes_AS_STRING(tmp);

    /*
     * This introduces a memory leak, as the memory allocated for the doc
     * will not be freed even if the ufunc itself is deleted. In practice
     * this should not be a problem since the user would have to
     * repeatedly create, document, and throw away ufuncs.
     */
    char *newdocstr = malloc(strlen(docstr) + 1);
    if (!newdocstr) {
        Py_DECREF(tmp);
        return PyErr_NoMemory();
    }
    strcpy(newdocstr, docstr);
    ufunc->doc = newdocstr;

    Py_DECREF(tmp);
    Py_RETURN_NONE;
}


/*
 *****************************************************************************
 **                            SETUP UFUNCS                                 **
 *****************************************************************************
 */

/* Setup the umath part of the module */

int initumath(PyObject *m)
{
    PyObject *d, *s, *s2;
    int UFUNC_FLOATING_POINT_SUPPORT = 1;

#ifdef NO_UFUNC_FLOATING_POINT_SUPPORT
    UFUNC_FLOATING_POINT_SUPPORT = 0;
#endif

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    if (InitOperators(d) < 0) {
        return -1;
    }

    PyDict_SetItemString(d, "pi", s = PyFloat_FromDouble(NPY_PI));
    Py_DECREF(s);
    PyDict_SetItemString(d, "e", s = PyFloat_FromDouble(NPY_E));
    Py_DECREF(s);
    PyDict_SetItemString(d, "euler_gamma", s = PyFloat_FromDouble(NPY_EULER));
    Py_DECREF(s);

#define ADDCONST(str) PyModule_AddIntConstant(m, #str, UFUNC_##str)
#define ADDSCONST(str) PyModule_AddStringConstant(m, "UFUNC_" #str, UFUNC_##str)

    ADDCONST(FPE_DIVIDEBYZERO);
    ADDCONST(FPE_OVERFLOW);
    ADDCONST(FPE_UNDERFLOW);
    ADDCONST(FPE_INVALID);

    ADDCONST(FLOATING_POINT_SUPPORT);

    ADDSCONST(PYVALS_NAME);

#undef ADDCONST
#undef ADDSCONST
    PyModule_AddIntConstant(m, "UFUNC_BUFSIZE_DEFAULT", (long)NPY_BUFSIZE);

    Py_INCREF(npy_static_pydata.npy_extobj_contextvar);
    PyModule_AddObject(m, "_extobj_contextvar", npy_static_pydata.npy_extobj_contextvar);

    PyModule_AddObject(m, "PINF", PyFloat_FromDouble(NPY_INFINITY));
    PyModule_AddObject(m, "NINF", PyFloat_FromDouble(-NPY_INFINITY));
    PyModule_AddObject(m, "PZERO", PyFloat_FromDouble(NPY_PZERO));
    PyModule_AddObject(m, "NZERO", PyFloat_FromDouble(NPY_NZERO));
    PyModule_AddObject(m, "NAN", PyFloat_FromDouble(NPY_NAN));

    s = PyDict_GetItemString(d, "divide");
    PyDict_SetItemString(d, "true_divide", s);

    s = PyDict_GetItemString(d, "conjugate");
    s2 = PyDict_GetItemString(d, "remainder");

    /* Setup the array object's numerical structures with appropriate
       ufuncs in d*/
    if (_PyArray_SetNumericOps(d) < 0) {
        return -1;
    }

    PyDict_SetItemString(d, "conj", s);
    PyDict_SetItemString(d, "mod", s2);

    /*
     * Set up promoters for logical functions
     * TODO: This should probably be done at a better place, or even in the
     *       code generator directly.
     */
    int res = PyDict_GetItemStringRef(d, "logical_and", &s);
    if (res <= 0) {
        return -1;
    }
    if (install_logical_ufunc_promoter(s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    res = PyDict_GetItemStringRef(d, "logical_or", &s);
    if (res <= 0) {
        return -1;
    }
    if (install_logical_ufunc_promoter(s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    res = PyDict_GetItemStringRef(d, "logical_xor", &s);
    if (res <= 0) {
        return -1;
    }
    if (install_logical_ufunc_promoter(s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    if (init_string_ufuncs(d) < 0) {
        return -1;
    }

    if (init_stringdtype_ufuncs(m) < 0) {
        return -1;
    }

    if (init_special_int_comparisons(d) < 0) {
        return -1;
    }

    if (init_argparse_mutex() < 0) {
        return -1;
    }

    return 0;
}
