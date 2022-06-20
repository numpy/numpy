#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"  // For NPY_NO_EXPORT

//
// Convert a Python sequence to a C array of Py_ssize_t integers.
//
// `seq` must be a Python sequence of length `len`.  It is assumed that the
// caller has already checked the length of the sequence, so the length is an
// argument instead of being inferred from `seq` itself.
//
// `errtext` must be provided, and it must contain one occurrence of the
// format code sequence '%s'.  This text is used as the text of the TypeError
// that is raised when an element of `seq` is found that cannot be converted
// to an a Py_ssize_t integer.  The '%s' format code will be replaced with
// the type of the object that failed the conversion.
//
// Returns NULL with an exception set if the conversion fails.
//
// The memory for the array is allocated with PyMem_Calloc.
// The caller must free the memory with PyMem_FREE or PyMem_Free.
//
NPY_NO_EXPORT Py_ssize_t *
seq_to_ssize_c_array(Py_ssize_t len, PyObject *seq, char *errtext)
{
    Py_ssize_t *arr = PyMem_Calloc(len, sizeof(Py_ssize_t));
    if (arr == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t i = 0; i < len; ++i) {
        PyObject *tmp = PySequence_GetItem(seq, i);
        if (tmp == NULL) {
            PyMem_Free(arr);
            return NULL;
        }
        arr[i] = PyNumber_AsSsize_t(tmp, PyExc_OverflowError);
        if (arr[i] == -1 && PyErr_Occurred()) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Format(PyExc_TypeError, errtext, Py_TYPE(tmp)->tp_name);
            }
            Py_DECREF(tmp);
            PyMem_Free(arr);
            return NULL;
        }
        Py_DECREF(tmp);
    }
    return arr;
}
