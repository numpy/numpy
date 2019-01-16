#include "Python.h"

/*
 * This is a dummy module. It will be used to ruin the import of multiarray
 * during testing. It exports two entry points, one to make the build happy,
 * and a multiarray one for the actual test. The content of the module is
 * irrelevant to the test.
 *
 * The code is from
 * https://docs.python.org/3/howto/cporting.html
 * or
 * https://github.com/python/cpython/blob/v3.7.0/Doc/howto/cporting.rst
 */

#if defined _WIN32 || defined __CYGWIN__ || defined __MINGW32__
  #if defined __GNUC__ || defined __clang__
    #define DLL_PUBLIC __attribute__ ((dllexport))
  #else
    #define DLL_PUBLIC __declspec(dllexport)
  #endif
#elif defined __GNUC__  || defined __clang__
  #define DLL_PUBLIC __attribute__ ((visibility ("default")))
#else
    /* Enhancement: error now instead ? */
    #define DLL_PUBLIC
#endif

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef multiarray_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int multiarray_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int multiarray_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "multiarray",
        NULL,
        sizeof(struct module_state),
        multiarray_methods,
        NULL,
        multiarray_traverse,
        multiarray_clear,
        NULL
};

#define INITERROR return NULL

DLL_PUBLIC PyObject *
PyInit_multiarray(void)

#else
#define INITERROR return

void
DLL_PUBLIC initmultiarray(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("multiarray", multiarray_methods);
#endif
    struct module_state *st;
    if (module == NULL)
        INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("multiarray.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

/*
 * Define a dummy entry point to make MSVC happy
 * Python's build system will export this function automatically
 */
#if PY_MAJOR_VERSION >= 3

PyObject *
PyInit__multiarray_module_test(void)
{
    return PyInit_multiarray();
}

#else

void
init_multiarray_module_test(void)
{
    initmultiarray();
}

#endif                                                    
