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
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"

#include "npy_config.h"
#ifdef ENABLE_SEPARATE_COMPILATION
#define PY_ARRAY_UNIQUE_SYMBOL _npy_umathmodule_ARRAY_API
#endif

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "abstract.h"

#include "numpy/npy_math.h"

/*
 *****************************************************************************
 **                    INCLUDE GENERATED CODE                               **
 *****************************************************************************
 */
#include "funcs.inc"
#include "loops.h"
#include "ufunc_object.h"
#include "ufunc_type_resolution.h"
#include "__umath_generated.c"
#include "__ufunc_api.c"

static PyUFuncGenericFunction pyfunc_functions[] = {PyUFunc_On_Om};

static int
object_ufunc_type_resolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int i, nop = ufunc->nin + ufunc->nout;
    PyArray_Descr *obj_dtype;

    obj_dtype = PyArray_DescrFromType(NPY_OBJECT);
    if (obj_dtype == NULL) {
        return -1;
    }

    for (i = 0; i < nop; ++i) {
        Py_INCREF(obj_dtype);
        out_dtypes[i] = obj_dtype;
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

static PyObject *
ufunc_frompyfunc(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *NPY_UNUSED(kwds)) {
    /* Keywords are ignored for now */

    PyObject *function, *pyname = NULL;
    int nin, nout, i;
    PyUFunc_PyFuncData *fdata;
    PyUFuncObject *self;
    char *fname, *str;
    Py_ssize_t fname_len = -1;
    int offset[2];

    if (!PyArg_ParseTuple(args, "Oii", &function, &nin, &nout)) {
        return NULL;
    }
    if (!PyCallable_Check(function)) {
        PyErr_SetString(PyExc_TypeError, "function must be callable");
        return NULL;
    }
    self = PyArray_malloc(sizeof(PyUFuncObject));
    if (self == NULL) {
        return NULL;
    }
    PyObject_Init((PyObject *)self, &PyUFunc_Type);

    self->userloops = NULL;
    self->nin = nin;
    self->nout = nout;
    self->nargs = nin + nout;
    self->identity = PyUFunc_None;
    self->functions = pyfunc_functions;
    self->ntypes = 1;
    self->check_return = 0;

    /* generalized ufunc */
    self->core_enabled = 0;
    self->core_num_dim_ix = 0;
    self->core_num_dims = NULL;
    self->core_dim_ixs = NULL;
    self->core_offsets = NULL;
    self->core_signature = NULL;
    self->op_flags = PyArray_malloc(sizeof(npy_uint32)*self->nargs);
    if (self->op_flags == NULL) {
        return PyErr_NoMemory();
    }
    memset(self->op_flags, 0, sizeof(npy_uint32)*self->nargs);
    self->iter_flags = 0;

    self->type_resolver = &object_ufunc_type_resolver;
    self->legacy_inner_loop_selector = &object_ufunc_loop_selector;

    pyname = PyObject_GetAttrString(function, "__name__");
    if (pyname) {
        (void) PyString_AsStringAndSize(pyname, &fname, &fname_len);
    }
    if (PyErr_Occurred()) {
        fname = "?";
        fname_len = 1;
        PyErr_Clear();
    }

    /*
     * self->ptr holds a pointer for enough memory for
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
    offset[1] = self->nargs;
    i = (self->nargs % sizeof(void *));
    if (i) {
        offset[1] += (sizeof(void *)-i);
    }
    self->ptr = PyArray_malloc(offset[0] + offset[1] + sizeof(void *) +
                            (fname_len + 14));
    if (self->ptr == NULL) {
        Py_XDECREF(pyname);
        return PyErr_NoMemory();
    }
    Py_INCREF(function);
    self->obj = function;
    fdata = (PyUFunc_PyFuncData *)(self->ptr);
    fdata->nin = nin;
    fdata->nout = nout;
    fdata->callable = function;

    self->data = (void **)(((char *)self->ptr) + offset[0]);
    self->data[0] = (void *)fdata;
    self->types = (char *)self->data + sizeof(void *);
    for (i = 0; i < self->nargs; i++) {
        self->types[i] = NPY_OBJECT;
    }
    str = self->types + offset[1];
    memcpy(str, fname, fname_len);
    memcpy(str+fname_len, " (vectorized)", 14);
    self->name = str;

    Py_XDECREF(pyname);

    /* Do a better job someday */
    self->doc = "dynamic ufunc based on a python function";

    return (PyObject *)self;
}

/*
 *****************************************************************************
 **                            SETUP UFUNCS                                 **
 *****************************************************************************
 */

/* Less automated additions to the ufuncs */

static PyUFuncGenericFunction frexp_functions[] = {
#ifdef HAVE_FREXPF
    HALF_frexp,
    FLOAT_frexp,
#endif
    DOUBLE_frexp
#ifdef HAVE_FREXPL
    ,LONGDOUBLE_frexp
#endif
};

static void * blank3_data[] = { (void *)NULL, (void *)NULL, (void *)NULL};
static void * blank6_data[] = { (void *)NULL, (void *)NULL, (void *)NULL,
                                (void *)NULL, (void *)NULL, (void *)NULL};
static char frexp_signatures[] = {
#ifdef HAVE_FREXPF
    NPY_HALF, NPY_HALF, NPY_INT,
    NPY_FLOAT, NPY_FLOAT, NPY_INT,
#endif
    NPY_DOUBLE, NPY_DOUBLE, NPY_INT
#ifdef HAVE_FREXPL
    ,NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_INT
#endif
};

#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
#define LDEXP_LONG(typ) typ##_ldexp
#else
#define LDEXP_LONG(typ) typ##_ldexp_long
#endif

static PyUFuncGenericFunction ldexp_functions[] = {
#ifdef HAVE_LDEXPF
    HALF_ldexp,
    FLOAT_ldexp,
    LDEXP_LONG(HALF),
    LDEXP_LONG(FLOAT),
#endif
    DOUBLE_ldexp,
    LDEXP_LONG(DOUBLE)
#ifdef HAVE_LDEXPL
    ,
    LONGDOUBLE_ldexp,
    LDEXP_LONG(LONGDOUBLE)
#endif
};

static char ldexp_signatures[] = {
#ifdef HAVE_LDEXPF
    NPY_HALF, NPY_INT, NPY_HALF,
    NPY_FLOAT, NPY_INT, NPY_FLOAT,
    NPY_HALF, NPY_LONG, NPY_HALF,
    NPY_FLOAT, NPY_LONG, NPY_FLOAT,
#endif
    NPY_DOUBLE, NPY_INT, NPY_DOUBLE,
    NPY_DOUBLE, NPY_LONG, NPY_DOUBLE
#ifdef HAVE_LDEXPL
    ,NPY_LONGDOUBLE, NPY_INT, NPY_LONGDOUBLE
    ,NPY_LONGDOUBLE, NPY_LONG, NPY_LONGDOUBLE
#endif
};

static void
InitOtherOperators(PyObject *dictionary) {
    PyObject *f;
    int num;

    num = sizeof(frexp_functions) / sizeof(frexp_functions[0]);
    f = PyUFunc_FromFuncAndData(frexp_functions, blank3_data,
                                frexp_signatures, num,
                                1, 2, PyUFunc_None, "frexp",
                                "Split the number, x, into a normalized"\
                                " fraction (y1) and exponent (y2)",0);
    PyDict_SetItemString(dictionary, "frexp", f);
    Py_DECREF(f);

    num = sizeof(ldexp_functions) / sizeof(ldexp_functions[0]);
    f = PyUFunc_FromFuncAndData(ldexp_functions, blank6_data, ldexp_signatures, num,
                                2, 1, PyUFunc_None, "ldexp",
                                "Compute y = x1 * 2**x2.",0);
    PyDict_SetItemString(dictionary, "ldexp", f);
    Py_DECREF(f);

#if defined(NPY_PY3K)
    f = PyDict_GetItemString(dictionary, "true_divide");
    PyDict_SetItemString(dictionary, "divide", f);
#endif
    return;
}

/* Setup the umath module */
/* Remove for time being, it is declared in __ufunc_api.h */
/*static PyTypeObject PyUFunc_Type;*/

static struct PyMethodDef methods[] = {
    {"frompyfunc",
        (PyCFunction) ufunc_frompyfunc,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"seterrobj",
        (PyCFunction) ufunc_seterr,
        METH_VARARGS, NULL},
    {"geterrobj",
        (PyCFunction) ufunc_geterr,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}                /* sentinel */
};


#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "umath",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#include <stdio.h>

#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC PyInit_umath(void)
#else
#define RETVAL
PyMODINIT_FUNC initumath(void)
#endif
{
    PyObject *m, *d, *s, *s2, *c_api;
    int UFUNC_FLOATING_POINT_SUPPORT = 1;

#ifdef NO_UFUNC_FLOATING_POINT_SUPPORT
    UFUNC_FLOATING_POINT_SUPPORT = 0;
#endif
    /* Create the module and add the functions */
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("umath", methods);
#endif
    if (!m) {
        return RETVAL;
    }

    /* Import the array */
    if (_import_array() < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ImportError,
                            "umath failed: Could not import array core.");
        }
        return RETVAL;
    }

    /* Initialize the types */
    if (PyType_Ready(&PyUFunc_Type) < 0)
        return RETVAL;

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    c_api = NpyCapsule_FromVoidPtr((void *)PyUFunc_API, NULL);
    if (PyErr_Occurred()) {
        goto err;
    }
    PyDict_SetItemString(d, "_UFUNC_API", c_api);
    Py_DECREF(c_api);
    if (PyErr_Occurred()) {
        goto err;
    }

    s = PyString_FromString("0.4.0");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    /* Load the ufunc operators into the array module's namespace */
    InitOperators(d);

    InitOtherOperators(d);

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
    ADDCONST(ERR_DEFAULT2);

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

    s = PyDict_GetItemString(d, "conjugate");
    s2 = PyDict_GetItemString(d, "remainder");
    /* Setup the array object's numerical structures with appropriate
       ufuncs in d*/
    PyArray_SetNumericOps(d);

    PyDict_SetItemString(d, "conj", s);
    PyDict_SetItemString(d, "mod", s2);

    return RETVAL;

 err:
    /* Check for errors */
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load umath module.");
    }
    return RETVAL;
}
