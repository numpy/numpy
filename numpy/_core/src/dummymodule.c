/* -*- c -*- */

/*
 * This is a dummy module whose purpose is to get distutils to generate the
 * configuration files before the libraries are made.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef_Slot dummy_slots[] = {
#if PY_VERSION_HEX >= 0x030c00f0  // Python 3.12+
    // signal that this module can be imported in isolated subinterpreters
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d00f0  // Python 3.13+
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "dummy",
    .m_size = 0,
    .m_methods = methods,
    .m_slots = dummy_slots,
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__dummy(void) {
    return PyModuleDef_Init(&moduledef);
}
