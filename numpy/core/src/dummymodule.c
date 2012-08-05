/* -*- c -*- */

/*
 * This is a dummy module whose purpose is to get distutils to generate the
 * configuration files before the libraries are made.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <npy_pycompat.h>

static struct PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}
};


#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "dummy",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

/* Initialization function for the module */
#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit__dummy(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC
init_dummy(void) {
    Py_InitModule("_dummy", methods);
}
#endif
