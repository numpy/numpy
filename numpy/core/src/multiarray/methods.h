#ifndef _NPY_ARRAY_METHODS_H_
#define _NPY_ARRAY_METHODS_H_

#include "npy_import.h"

extern NPY_NO_EXPORT PyMethodDef array_methods[];

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);

/* Pathlib support */
static inline PyObject *
NpyPath_PathlikeToFspath(PyObject *file)
{
    static PyObject *os_PathLike = NULL;
    static PyObject *os_fspath = NULL;
    npy_cache_import("numpy.compat", "os_PathLike", &os_PathLike);
    if (os_PathLike == NULL) {
        return NULL;
    }
    npy_cache_import("numpy.compat", "os_fspath", &os_fspath);
    if (os_fspath == NULL) {
        return NULL;
    }

    if (!PyObject_IsInstance(file, os_PathLike)) {
        return file;
    }
    return PyObject_CallFunctionObjArgs(os_fspath, file, NULL);
}

#endif
