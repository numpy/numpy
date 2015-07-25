#ifndef __GET_ATTR_STRING_H
#define __GET_ATTR_STRING_H

static NPY_INLINE int
_is_basic_python_type(PyObject * obj)
{
    if (obj == Py_None ||
            PyBool_Check(obj) ||
            /* Basic number types */
#if !defined(NPY_PY3K)
            PyInt_CheckExact(obj) ||
            PyString_CheckExact(obj) ||
#endif
            PyLong_CheckExact(obj) ||
            PyFloat_CheckExact(obj) ||
            PyComplex_CheckExact(obj) ||
            /* Basic sequence types */
            PyList_CheckExact(obj) ||
            PyTuple_CheckExact(obj) ||
            PyDict_CheckExact(obj) ||
            PyAnySet_CheckExact(obj) ||
            PyUnicode_CheckExact(obj) ||
            PyBytes_CheckExact(obj) ||
            PySlice_Check(obj)) {

        return 1;
    }

    return 0;
}

/*
 * PyArray_GetAttrString_SuppressException:
 *
 * Stripped down version of PyObject_GetAttrString,
 * avoids lookups for None, tuple, and List objects,
 * and doesn't create a PyErr since this code ignores it.
 *
 * This can be much faster then PyObject_GetAttrString where
 * exceptions are not used by caller.
 *
 * 'obj' is the object to search for attribute.
 *
 * 'name' is the attribute to search for.
 *
 * Returns attribute value on success, 0 on failure.
 */
static PyObject *
PyArray_GetAttrString_SuppressException(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *res = (PyObject *)NULL;

    /* We do not need to check for special attributes on trivial types */
    if (_is_basic_python_type(obj)) {
        return NULL;
    }

    /* Attribute referenced by (char *)name */
    if (tp->tp_getattr != NULL) {
        res = (*tp->tp_getattr)(obj, name);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    /* Attribute referenced by (PyObject *)name */
    else if (tp->tp_getattro != NULL) {
#if defined(NPY_PY3K)
        PyObject *w = PyUnicode_InternFromString(name);
#else
        PyObject *w = PyString_InternFromString(name);
#endif
        if (w == NULL) {
            return (PyObject *)NULL;
        }
        res = (*tp->tp_getattro)(obj, w);
        Py_DECREF(w);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    return res;
}

#endif
