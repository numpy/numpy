/* -*- c -*- */

/*
 * vim:syntax=c
 */

/*
 *****************************************************************************
 **                            INCLUDES                                     **
 *****************************************************************************
 */

/*
 * _UMATHMODULE IS needed in __ufunc_api.h, included from numpy/ufuncobject.h.
 * This is a mess and it would be nice to fix it. It has nothing to do with
 * __ufunc_api.c
 */
#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"

#include "npy_config.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "abstract.h"

#include "numpy/npy_math.h"
#include "number.h"

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

static int
object_ufunc_loop_selector(PyUFuncObject *ufunc,
                            PyArray_Descr **NPY_UNUSED(dtypes),
                            PyUFuncGenericFunction *out_innerloop,
                            void **out_innerloopdata,
                            int *out_needs_api)
{
    *out_innerloop = ufunc->functions[0];
    *out_innerloopdata = ufunc->data[0];
    *out_needs_api = 1;

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
    self->legacy_inner_loop_selector = &object_ufunc_loop_selector;
    PyObject_GC_Track(self);

    return (PyObject *)self;
}

/* docstring in numpy.add_newdocs.py */
PyObject *
add_newdoc_ufunc(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
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

NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_out = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_where = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_axes = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_axis = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_keepdims = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_casting = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_order = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_dtype = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_subok = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_signature = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_sig = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_extobj = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_array_prepare = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_array_wrap = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_array_finalize = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_ufunc = NULL;
NPY_VISIBILITY_HIDDEN PyObject *npy_um_str_pyvals_name = NULL;

/* intern some strings used in ufuncs, returns 0 on success */
static int
intern_strings(void)
{
    if (!(npy_um_str_out = PyUnicode_InternFromString("out"))) return -1;
    if (!(npy_um_str_where = PyUnicode_InternFromString("where"))) return -1;
    if (!(npy_um_str_axes = PyUnicode_InternFromString("axes"))) return -1;
    if (!(npy_um_str_axis = PyUnicode_InternFromString("axis"))) return -1;
    if (!(npy_um_str_keepdims = PyUnicode_InternFromString("keepdims"))) return -1;
    if (!(npy_um_str_casting = PyUnicode_InternFromString("casting"))) return -1;
    if (!(npy_um_str_order = PyUnicode_InternFromString("order"))) return -1;
    if (!(npy_um_str_dtype = PyUnicode_InternFromString("dtype"))) return -1;
    if (!(npy_um_str_subok = PyUnicode_InternFromString("subok"))) return -1;
    if (!(npy_um_str_signature = PyUnicode_InternFromString("signature"))) return -1;
    if (!(npy_um_str_sig = PyUnicode_InternFromString("sig"))) return -1;
    if (!(npy_um_str_extobj = PyUnicode_InternFromString("extobj"))) return -1;
    if (!(npy_um_str_array_prepare = PyUnicode_InternFromString("__array_prepare__"))) return -1;
    if (!(npy_um_str_array_wrap = PyUnicode_InternFromString("__array_wrap__"))) return -1;
    if (!(npy_um_str_array_finalize = PyUnicode_InternFromString("__array_finalize__"))) return -1;
    if (!(npy_um_str_ufunc = PyUnicode_InternFromString("__array_ufunc__"))) return -1;
    if (!(npy_um_str_pyvals_name = PyUnicode_InternFromString(UFUNC_PYVALS_NAME))) return -1;
    return 0;
}

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

    PyDict_SetItemString(d, "pi", s = PyFloat_FromDouble(NPY_PI));
    Py_DECREF(s);
    PyDict_SetItemString(d, "e", s = PyFloat_FromDouble(NPY_E));
    Py_DECREF(s);
    PyDict_SetItemString(d, "euler_gamma", s = PyFloat_FromDouble(NPY_EULER));
    Py_DECREF(s);

#define ADDCONST(str) PyModule_AddIntConstant(m, #str, UFUNC_##str)
#define ADDSCONST(str) PyModule_AddStringConstant(m, "UFUNC_" #str, UFUNC_##str)

    ADDCONST(ERR_IGNORE);
    ADDCONST(ERR_WARN);
    ADDCONST(ERR_CALL);
    ADDCONST(ERR_RAISE);
    ADDCONST(ERR_PRINT);
    ADDCONST(ERR_LOG);
    ADDCONST(ERR_DEFAULT);

    ADDCONST(SHIFT_DIVIDEBYZERO);
    ADDCONST(SHIFT_OVERFLOW);
    ADDCONST(SHIFT_UNDERFLOW);
    ADDCONST(SHIFT_INVALID);

    ADDCONST(FPE_DIVIDEBYZERO);
    ADDCONST(FPE_OVERFLOW);
    ADDCONST(FPE_UNDERFLOW);
    ADDCONST(FPE_INVALID);

    ADDCONST(FLOATING_POINT_SUPPORT);

    ADDSCONST(PYVALS_NAME);

#undef ADDCONST
#undef ADDSCONST
    PyModule_AddIntConstant(m, "UFUNC_BUFSIZE_DEFAULT", (long)NPY_BUFSIZE);

    PyModule_AddObject(m, "PINF", PyFloat_FromDouble(NPY_INFINITY));
    PyModule_AddObject(m, "NINF", PyFloat_FromDouble(-NPY_INFINITY));
    PyModule_AddObject(m, "PZERO", PyFloat_FromDouble(NPY_PZERO));
    PyModule_AddObject(m, "NZERO", PyFloat_FromDouble(NPY_NZERO));
    PyModule_AddObject(m, "NAN", PyFloat_FromDouble(NPY_NAN));

    s = PyDict_GetItemString(d, "true_divide");
    PyDict_SetItemString(d, "divide", s);

    s = PyDict_GetItemString(d, "conjugate");
    s2 = PyDict_GetItemString(d, "remainder");
    /* Setup the array object's numerical structures with appropriate
       ufuncs in d*/
    _PyArray_SetNumericOps(d);

    PyDict_SetItemString(d, "conj", s);
    PyDict_SetItemString(d, "mod", s2);

    if (intern_strings() < 0) {
        PyErr_SetString(PyExc_RuntimeError,
           "cannot intern umath strings while initializing _multiarray_umath.");
        return -1;
    }

    return 0;
}
