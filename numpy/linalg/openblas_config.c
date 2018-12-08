/* -*- c -*- */

#include <Python.h>
#include <npy_cblas.h>

static PyObject *
_openblas_info(PyObject *self, PyObject *args) {
    char *result;
    result = openblas_get_config();
    return PyBytes_FromString(result);
}

static PyMethodDef methods[] = {
    {"_openblas_info", _openblas_info, METH_VARARGS,
     "Call openblas_get_config"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "openblas_config",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_openblas_config(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
