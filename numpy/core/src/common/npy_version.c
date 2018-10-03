#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>

#include "npy_config.h"
#include "npy_pycompat.h"

/*NUMPY_API
 *
 * Included at the very first so not auto-grabbed and thus not labeled.
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCVersion(void)
{
    return (unsigned int)NPY_ABI_VERSION;
}

/*NUMPY_API
 * Returns the built-in (at compilation time) C API version
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCFeatureVersion(void)
{
    return (unsigned int)NPY_API_VERSION;
}

/*
 * Return equivalent of PY_VERSION_HEX for NumPy's version
 */
NPY_NO_EXPORT long
get_numpy_version_as_hex(void)
{
    static long ver = 0;
    PyObject *_version, *_func, *result, *tmp;
    char *buf, *endbuf;
    Py_ssize_t len;
    int retval;
    if (ver != 0) {
        return ver;
    }
    _version = PyImport_ImportModule("numpy.version");
    if (_version == NULL) {
        return -1L;
    }
    _func = PyObject_GetAttrString(_version, "get_numpy_version_as_hex");
    Py_DECREF(_version);
    if (_func == NULL) {
        return -1L;
    }
    result = PyObject_CallFunction(_func, NULL);
    Py_DECREF(_func);
    if (result == NULL) {
        return -1L;
    }
#if defined(NPY_PY3K)
    /* FIXME: XXX -- should it use UTF-8 here? */
    tmp = PyUnicode_AsUTF8String(result);
    Py_DECREF(result);
    if (tmp == NULL) {
        return -1;
    }
#else
    tmp = result;
#endif
    retval = PyBytes_AsStringAndSize(tmp, &buf, &len);
    Py_INCREF(tmp);
    if (retval < 0) {
        return -1L;
    }
    ver = strtol(buf, &endbuf, 16);
    if (buf == endbuf || ver == LONG_MAX || ver <= 0) {
        PyErr_SetString(PyExc_ValueError,
                "invalid hex value from version.get_numpy_version_as_hex()");
        ver = 0L;
        return -1L;
    }
    return ver;
}
