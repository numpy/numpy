#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"

#include "npy_pycompat.h"
#include "npy_config.h"
#include "numpy/npy_math.h"

#define CACOS 1
#define CASIN 1
#define CATAN 1
#define CACOSH 1
#define CASINH 1
#define CATANH 1
#define CCOS 1
#define CSIN 1
#define CTAN 1
#define CCOSH 1
#define CSINH 1
#define CTANH 1
#define CEXP 1
#define CLOG 1
#define CPOW 1
#define CSQRT 1
#define HAVE_NUMPY 1

#define FLOAT 1
#include "../../test_c99complex.c"
#undef FLOAT

#define DOUBLE 1
#include "../../test_c99complex.c"
#undef DOUBLE

#define LONGDOUBLE 1
#include "../../test_c99complex.c"
#undef LONGDOUBLE

#define TESTFUNC_INT(func, suffix) \
    static PyObject * CONCAT3(_test_, func, suffix)(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args)) \
    { \
        PyObject *errs; \
        errs = CONCAT3(test_, func, suffix)(); \
        if (errs == NULL) { \
            return errs; \
        } \
        if (PySequence_Size(errs) == 0) { \
            Py_DECREF(errs); \
            Py_INCREF(Py_None); \
            return Py_None; \
        } \
        else { \
            PyErr_SetObject(PyExc_AssertionError, errs); \
            return NULL; \
        } \
    }

#define TESTFUNC(func) \
    TESTFUNC_INT(func, f) \
    TESTFUNC_INT(func, ) \
    TESTFUNC_INT(func, l)

#define TESTMETHODDEF_INT(func, suffix) \
    {STRINGIZE(CONCAT3(test_, func, suffix)), CONCAT3(_test_, func, suffix), METH_VARARGS, ""}

#define TESTMETHODDEF(func) \
    TESTMETHODDEF_INT(func, f), \
    TESTMETHODDEF_INT(func, ), \
    TESTMETHODDEF_INT(func, l)

TESTFUNC(cacos)
TESTFUNC(casin)
TESTFUNC(catan)
TESTFUNC(cacosh)
TESTFUNC(casinh)
TESTFUNC(catanh)
TESTFUNC(ccos)
TESTFUNC(csin)
TESTFUNC(ctan)
TESTFUNC(ccosh)
TESTFUNC(csinh)
TESTFUNC(ctanh)
TESTFUNC(cexp)
TESTFUNC(clog)
TESTFUNC(cpow)
TESTFUNC(csqrt)

static PyMethodDef methods[] = {
     TESTMETHODDEF(cacos),
     TESTMETHODDEF(casin),
     TESTMETHODDEF(catan),
     TESTMETHODDEF(cacosh),
     TESTMETHODDEF(casinh),
     TESTMETHODDEF(catanh),
     TESTMETHODDEF(ccos),
     TESTMETHODDEF(csin),
     TESTMETHODDEF(ctan),
     TESTMETHODDEF(ccosh),
     TESTMETHODDEF(csinh),
     TESTMETHODDEF(ctanh),
     TESTMETHODDEF(cexp),
     TESTMETHODDEF(clog),
     TESTMETHODDEF(cpow),
     TESTMETHODDEF(csqrt),
     {NULL, NULL, 0, NULL}
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "npymath_tests",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit_npymath_tests(void)
#else
PyMODINIT_FUNC
initnpymath_tests(void)
#endif
{
#if defined(NPY_PY3K)
    return PyModule_Create(&moduledef);
#else
    Py_InitModule("npymath_tests", methods);
#endif
}

