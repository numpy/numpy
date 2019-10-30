#ifndef __GET_ATTR_STRING_H
#define __GET_ATTR_STRING_H

static NPY_INLINE npy_bool
_is_basic_python_type(PyTypeObject *tp)
{
    return (
        /* Basic number types */
        tp == &PyBool_Type ||
#if !defined(NPY_PY3K)
        tp == &PyInt_Type ||
#endif
        tp == &PyLong_Type ||
        tp == &PyFloat_Type ||
        tp == &PyComplex_Type ||

        /* Basic sequence types */
        tp == &PyList_Type ||
        tp == &PyTuple_Type ||
        tp == &PyDict_Type ||
        tp == &PySet_Type ||
        tp == &PyFrozenSet_Type ||
        tp == &PyUnicode_Type ||
        tp == &PyBytes_Type ||
#if !defined(NPY_PY3K)
        tp == &PyString_Type ||
#endif

        /* other builtins */
        tp == &PySlice_Type ||
        tp == Py_TYPE(Py_None) ||
        tp == Py_TYPE(Py_Ellipsis) ||
        tp == Py_TYPE(Py_NotImplemented) ||

        /* TODO: ndarray, but we can't see PyArray_Type here */

        /* sentinel to swallow trailing || */
        NPY_FALSE
    );
}

/*
 * Stripped down version of PyObject_GetAttrString(obj, name) that does not
 * raise PyExc_AttributeError.
 *
 * This allows it to avoid creating then discarding exception objects when
 * performing lookups on objects without any attributes.
 *
 * Returns attribute value on success, NULL without an exception set if
 * there is no such attribute, and NULL with an exception on failure.
 */
static NPY_INLINE PyObject *
maybe_get_attr(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *res = (PyObject *)NULL;

    /* Attribute referenced by (char *)name */
    if (tp->tp_getattr != NULL) {
        res = (*tp->tp_getattr)(obj, name);
        if (res == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
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
        if (res == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
        }
    }
    return res;
}

/*
 * Lookup a special method, following the python approach of looking up
 * on the type object, rather than on the instance itself.
 *
 * Assumes that the special method is a numpy-specific one, so does not look
 * at builtin types, nor does it look at a base ndarray.
 *
 * In future, could be made more like _Py_LookupSpecial
 */
static NPY_INLINE PyObject *
PyArray_LookupSpecial(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* We do not need to check for special attributes on trivial types */
    if (_is_basic_python_type(tp)) {
        return NULL;
    }
    return maybe_get_attr((PyObject *)tp, name);
}

/*
 * PyArray_LookupSpecial_OnInstance:
 *
 * Implements incorrect special method lookup rules, that break the python
 * convention, and looks on the instance, not the type.
 *
 * Kept for backwards compatibility. In future, we should deprecate this.
 */
static NPY_INLINE PyObject *
PyArray_LookupSpecial_OnInstance(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* We do not need to check for special attributes on trivial types */
    if (_is_basic_python_type(tp)) {
        return NULL;
    }

    return maybe_get_attr(obj, name);
}

#endif
