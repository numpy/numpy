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

/* Automatically generated code to define all ufuncs: */
#include "funcs.inc"
#include "__umath_generated.c"


static PyUFuncGenericFunction pyfunc_functions[] = {PyUFunc_On_Om};

static int
object_ufunc_type_resolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int i, nop = ufunc->nin + ufunc->nout;

    out_dtypes[0] = PyArray_DescrFromType(NPY_OBJECT);
    if (out_dtypes[0] == NULL) {
        return -1;
    }

    for (i = 1; i < nop; ++i) {
        Py_INCREF(out_dtypes[0]);
        out_dtypes[i] = out_dtypes[0];
    }

    return 0;
}


PyObject *
ufunc_frompyfunc(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds) {
    PyObject *function, *pyname = NULL;
    int nin, nout, i, nargs;
    PyUFunc_PyFuncData *fdata;
    PyUFuncObject *self;
    const char *fname = NULL;
    char *str, *types, *doc;
    Py_ssize_t fname_len = -1;
    void * ptr, **data;
    int offset[2];
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

    /*
     * ptr will be assigned to self->ptr, holds a pointer for enough memory for
     * self->data[0] (fdata)
     * self->data
     * self->name
     * self->types
     *
     * To be safest, all of these need their memory aligned on void * pointers
     * Therefore, we may need to allocate extra space.
     */
    offset[0] = sizeof(PyUFunc_PyFuncData);
    i = (sizeof(PyUFunc_PyFuncData) % sizeof(void *));
    if (i) {
        offset[0] += (sizeof(void *) - i);
    }
    offset[1] = nargs;
    i = (nargs % sizeof(void *));
    if (i) {
        offset[1] += (sizeof(void *)-i);
    }
    ptr = PyArray_malloc(offset[0] + offset[1] + sizeof(void *) +
                            (fname_len + 14));
    if (ptr == NULL) {
        Py_XDECREF(pyname);
        return PyErr_NoMemory();
    }
    fdata = (PyUFunc_PyFuncData *)(ptr);
    fdata->callable = function;
    fdata->nin = nin;
    fdata->nout = nout;

    data = (void **)(((char *)ptr) + offset[0]);
    data[0] = (void *)fdata;
    types = (char *)data + sizeof(void *);
    for (i = 0; i < nargs; i++) {
        types[i] = NPY_OBJECT;
    }
    str = types + offset[1];
    memcpy(str, fname, fname_len);
    memcpy(str+fname_len, " (vectorized)", 14);
    Py_XDECREF(pyname);

    /* Do a better job someday */
    doc = "dynamic ufunc based on a python function";

    self = (PyUFuncObject *)PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
            (PyUFuncGenericFunction *)pyfunc_functions, data,
            types, /* ntypes */ 1, nin, nout, identity ? PyUFunc_IdentityValue : PyUFunc_None,
            str, doc, /* unused */ 0, NULL, identity);

    if (self == NULL) {
        PyArray_free(ptr);
        return NULL;
    }
    Py_INCREF(function);
    self->obj = function;
    self->ptr = ptr;

    self->type_resolver = &object_ufunc_type_resolver;
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

    s = PyDict_GetItemString(d, "divide"); // noqa: borrowed-ref OK
    PyDict_SetItemString(d, "true_divide", s);

    s = PyDict_GetItemString(d, "conjugate"); // noqa: borrowed-ref OK
    s2 = PyDict_GetItemString(d, "remainder"); // noqa: borrowed-ref OK

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
