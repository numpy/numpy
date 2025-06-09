#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <numpy/ndarraytypes.h>


NPY_NO_EXPORT PyObject *
array_device(PyObject *NPY_UNUSED(self), void *NPY_UNUSED(ignored))
{
    return PyUnicode_FromString("cpu");
}

NPY_NO_EXPORT PyObject *
array_to_device(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"", "stream", NULL};
    char *device = "";
    PyObject *stream = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|$O:to_device", kwlist,
                                     &device,
                                     &stream)) {
        return NULL;
    }

    if (stream != Py_None) {
        PyErr_SetString(PyExc_ValueError,
                        "The stream argument in to_device() "
                        "is not supported");
        return NULL;
    }

    if (strcmp(device, "cpu") != 0) {
        PyErr_Format(PyExc_ValueError,
                     "Unsupported device: %s. Only 'cpu' is accepted.", device);
        return NULL;
    }

    Py_INCREF(self);
    return self;
}

NPY_NO_EXPORT PyObject *
array_array_namespace(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"api_version", NULL};
    PyObject *array_api_version = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|$O:__array_namespace__", kwlist,
                                     &array_api_version)) {
        return NULL;
    }

    if (array_api_version != Py_None) {
        if (!PyUnicode_Check(array_api_version))
        {
            PyErr_Format(PyExc_ValueError,
                "Only None and strings are allowed as the Array API version, "
                "but received: %S.", array_api_version);
            return NULL;
        } else if (PyUnicode_CompareWithASCIIString(array_api_version, "2021.12") != 0 &&
            PyUnicode_CompareWithASCIIString(array_api_version, "2022.12") != 0 &&
            PyUnicode_CompareWithASCIIString(array_api_version, "2023.12") != 0 &&
            PyUnicode_CompareWithASCIIString(array_api_version, "2024.12") != 0)
        {
            PyErr_Format(PyExc_ValueError,
                "Version \"%U\" of the Array API Standard is not supported.",
                array_api_version);
            return NULL;
        }
    }

    PyObject *numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module == NULL){
        return NULL;
    }

    return numpy_module;
}
