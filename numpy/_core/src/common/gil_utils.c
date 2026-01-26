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
#if !defined(PYPY_VERSION)
        PyErr_FormatV(type, format, va);
#else
        PyObject *exc_str = PyUnicode_FromFormatV(format, va);
        if (exc_str == NULL) {
            // no reason to have special handling for this error case, since
            // this function sets an error anyway
            NPY_DISABLE_C_API;
            va_end(va);
            return;
        }
        PyErr_SetObject(type, exc_str);
        Py_DECREF(exc_str);
#endif
    }
    NPY_DISABLE_C_API;
    va_end(va);
}

// Acquire the GIL before emitting a warning containing a message of
// the given category and stacklevel.
NPY_NO_EXPORT int
npy_gil_warning(PyObject *category, int stacklevel, const char *message)
{
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    int result = PyErr_WarnEx(category, message, stacklevel);
    NPY_DISABLE_C_API;
    return result;
}
