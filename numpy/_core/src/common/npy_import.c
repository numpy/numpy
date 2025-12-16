#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "numpy/ndarraytypes.h"
#include "npy_import.h"
#include "npy_atomic.h"


NPY_VISIBILITY_HIDDEN npy_runtime_imports_struct npy_runtime_imports;

NPY_NO_EXPORT int
init_import_mutex(void) {
#if PY_VERSION_HEX < 0x30d00b3
    npy_runtime_imports.import_mutex = PyThread_allocate_lock();
    if (npy_runtime_imports.import_mutex == NULL) {
        PyErr_NoMemory();
        return -1;
    }
#endif
    return 0;
}


/*! \brief Import a Python object from an entry point string.

 * The name should be of the form "(module ':')? (object '.')* attr".
 * If no module is present, it is assumed to be "numpy".
 * On error, returns NULL.
 */
NPY_NO_EXPORT PyObject*
npy_import_entry_point(const char *entry_point) {
    PyObject *result;
    const char *item;

    const char *colon = strchr(entry_point, ':');
    if (colon) { // there is a module.
        result = PyUnicode_FromStringAndSize(entry_point, colon - entry_point);
        if (result != NULL) {
            Py_SETREF(result, PyImport_Import(result));
        }
        item = colon + 1;
    }
    else {
        result = PyImport_ImportModule("numpy");
        item = entry_point;
    }

    const char *dot = item - 1;
    while (result != NULL && dot != NULL) {
        item = dot + 1;
        dot = strchr(item, '.');
        PyObject *string = PyUnicode_FromStringAndSize(
            item, dot ? dot - item : strlen(item));
        if (string == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        Py_SETREF(result, PyObject_GetAttr(result, string));
        Py_DECREF(string);
    }
    return result;
}
