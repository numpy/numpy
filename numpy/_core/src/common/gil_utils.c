#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/ndarraytypes.h>

#include <stdarg.h>

NPY_NO_EXPORT void
npy_gil_error(PyObject *type, const char *format, ...)
{
    va_list va;
    va_start(va, format);
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    if (!PyErr_Occurred()) {
        PyErr_FormatV(type, format, va);
    }
    NPY_DISABLE_C_API;
}
