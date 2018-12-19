#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "npy_pycompat.h"
#include "get_attr_string.h"
#include "npy_import.h"
#include "multiarraymodule.h"


/* Return the ndarray.__array_function__ method. */
static PyObject *
get_ndarray_array_function(void)
{
    PyObject* method = PyObject_GetAttrString((PyObject *)&PyArray_Type,
                                              "__array_function__");
    assert(method != NULL);
    return method;
}


/*
 * Get an object's __array_function__ method in the fastest way possible.
 * Never raises an exception. Returns NULL if the method doesn't exist.
 */
static PyObject *
get_array_function(PyObject *obj)
{
    static PyObject *ndarray_array_function = NULL;

    if (ndarray_array_function == NULL) {
        ndarray_array_function = get_ndarray_array_function();
    }

    /* Fast return for ndarray */
    if (PyArray_CheckExact(obj)) {
        Py_INCREF(ndarray_array_function);
        return ndarray_array_function;
    }

    return PyArray_LookupSpecial(obj, "__array_function__");
}


/*
 * Like list.insert(), but for C arrays of PyObject*. Skips error checking.
 */
static void
pyobject_array_insert(PyObject **array, int length, int index, PyObject *item)
{
    int j;

    for (j = length; j > index; j--) {
        array[j] = array[j - 1];
    }
    array[index] = item;
}


/*
 * Collects arguments with __array_function__ and their corresponding methods
 * in the order in which they should be tried (i.e., skipping redundant types).
 * `relevant_args` is expected to have been produced by PySequence_Fast.
 * Returns the number of arguments, or -1 on failure. 
 */
static int
get_implementing_args_and_methods(PyObject *relevant_args,
                                  PyObject **implementing_args,
                                  PyObject **methods)
{
    int num_implementing_args = 0;
    Py_ssize_t i;
    int j;

    PyObject **items = PySequence_Fast_ITEMS(relevant_args);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(relevant_args);

    for (i = 0; i < length; i++) {
        int new_class = 1;
        PyObject *argument = items[i];

        /* Have we seen this type before? */
        for (j = 0; j < num_implementing_args; j++) {
            if (Py_TYPE(argument) == Py_TYPE(implementing_args[j])) {
                new_class = 0;
                break;
            }
        }
        if (new_class) {
            PyObject *method = get_array_function(argument);

            if (method != NULL) {
                int arg_index;

                if (num_implementing_args >= NPY_MAXARGS) {
                    PyErr_Format(
                        PyExc_TypeError,
                        "maximum number (%d) of distinct argument types " \
                        "implementing __array_function__ exceeded",
                        NPY_MAXARGS);
                    Py_DECREF(method);
                    goto fail;
                }

                /* "subclasses before superclasses, otherwise left to right" */
                arg_index = num_implementing_args;
                for (j = 0; j < num_implementing_args; j++) {
                    PyObject *other_type;
                    other_type = (PyObject *)Py_TYPE(implementing_args[j]);
                    if (PyObject_IsInstance(argument, other_type)) {
                        arg_index = j;
                        break;
                    }
                }
                Py_INCREF(argument);
                pyobject_array_insert(implementing_args, num_implementing_args,
                                      arg_index, argument);
                pyobject_array_insert(methods, num_implementing_args,
                                      arg_index, method);
                ++num_implementing_args;
            }
        }
    }
    return num_implementing_args;

fail:
    for (j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(methods[j]);
    }
    return -1;
}


/*
 * Is this object ndarray.__array_function__?
 */
static int
is_default_array_function(PyObject *obj)
{
    static PyObject *ndarray_array_function = NULL;

    if (ndarray_array_function == NULL) {
        ndarray_array_function = get_ndarray_array_function();
    }
    return obj == ndarray_array_function;
}


/*
 * Core implementation of ndarray.__array_function__. This is exposed
 * separately so we can avoid the overhead of a Python method call from
 * within `implement_array_function`.
 */
NPY_NO_EXPORT PyObject *
array_function_method_impl(PyObject *func, PyObject *types, PyObject *args,
                           PyObject *kwargs)
{
    Py_ssize_t j;
    PyObject *implementation, *result;

    PyObject **items = PySequence_Fast_ITEMS(types);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(types);

    for (j = 0; j < length; j++) {
        int is_subclass = PyObject_IsSubclass(
            items[j], (PyObject *)&PyArray_Type);
        if (is_subclass == -1) {
            return NULL;
        }
        if (!is_subclass) {
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
    }

    implementation = PyObject_GetAttr(func, npy_ma_str_wrapped);
    if (implementation == NULL) {
        return NULL;
    }
    result = PyObject_Call(implementation, args, kwargs);
    Py_DECREF(implementation);
    return result;
}


/*
 * Calls __array_function__ on the provided argument, with a fast-path for
 * ndarray.
 */
static PyObject *
call_array_function(PyObject* argument, PyObject* method,
                    PyObject* public_api, PyObject* types,
                    PyObject* args, PyObject* kwargs)
{
    if (is_default_array_function(method)) {
        return array_function_method_impl(public_api, types, args, kwargs);
    }
    else {
        return PyObject_CallFunctionObjArgs(
            method, argument, public_api, types, args, kwargs, NULL);
    }
}


/*
 * Implements the __array_function__ protocol for a function, as described in
 * in NEP-18. See numpy.core.overrides for a full docstring.
 */
NPY_NO_EXPORT PyObject *
array_implement_array_function(
    PyObject *NPY_UNUSED(dummy), PyObject *positional_args)
{
    PyObject *implementation, *public_api, *relevant_args, *args, *kwargs;

    PyObject *types = NULL;
    PyObject *implementing_args[NPY_MAXARGS];
    PyObject *array_function_methods[NPY_MAXARGS];

    int j, any_overrides;
    int num_implementing_args = 0;
    PyObject *result = NULL;

    static PyObject *errmsg_formatter = NULL;

    if (!PyArg_UnpackTuple(
            positional_args, "implement_array_function", 5, 5,
            &implementation, &public_api, &relevant_args, &args, &kwargs)) {
        return NULL;
    }

    relevant_args = PySequence_Fast(
        relevant_args,
        "dispatcher for __array_function__ did not return an iterable");
    if (relevant_args == NULL) {
        return NULL;
    }

    /* Collect __array_function__ implementations */
    num_implementing_args = get_implementing_args_and_methods(
        relevant_args, implementing_args, array_function_methods);
    if (num_implementing_args == -1) {
        goto cleanup;
    }

    /*
     * Handle the typical case of no overrides. This is merely an optimization
     * if some arguments are ndarray objects, but is also necessary if no
     * arguments implement __array_function__ at all (e.g., if they are all
     * built-in types).
     */
    any_overrides = 0;
    for (j = 0; j < num_implementing_args; j++) {
        if (!is_default_array_function(array_function_methods[j])) {
            any_overrides = 1;
            break;
        }
    }
    if (!any_overrides) {
        result = PyObject_Call(implementation, args, kwargs);
        goto cleanup;
    }

    /*
     * Create a Python object for types.
     * We use a tuple, because it's the fastest Python collection to create
     * and has the bonus of being immutable.
     */
    types = PyTuple_New(num_implementing_args);
    if (types == NULL) {
        goto cleanup;
    }
    for (j = 0; j < num_implementing_args; j++) {
        PyObject *arg_type = (PyObject *)Py_TYPE(implementing_args[j]);
        Py_INCREF(arg_type);
        PyTuple_SET_ITEM(types, j, arg_type);
    }

    /* Call __array_function__ methods */
    for (j = 0; j < num_implementing_args; j++) {
        PyObject *argument = implementing_args[j];
        PyObject *method = array_function_methods[j];

        /*
         * We use `public_api` instead of `implementation` here so
         * __array_function__ implementations can do equality/identity
         * comparisons.
         */
        result = call_array_function(
            argument, method, public_api, types, args, kwargs);

        if (result == Py_NotImplemented) {
            /* Try the next one */
            Py_DECREF(result);
            result = NULL;
        }
        else {
            /* Either a good result, or an exception was raised. */
            goto cleanup;
        }
    }

    /* No acceptable override found, raise TypeError. */
    npy_cache_import("numpy.core._internal",
                     "array_function_errmsg_formatter",
                     &errmsg_formatter);
    if (errmsg_formatter != NULL) {
        PyObject *errmsg = PyObject_CallFunctionObjArgs(
            errmsg_formatter, public_api, types, NULL);
        if (errmsg != NULL) {
            PyErr_SetObject(PyExc_TypeError, errmsg);
            Py_DECREF(errmsg);
        }
    }

cleanup:
    for (j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(array_function_methods[j]);
    }
    Py_XDECREF(types);
    Py_DECREF(relevant_args);
    return result;
}


/*
 * Python wrapper for get_implementing_args_and_methods, for testing purposes.
 */
NPY_NO_EXPORT PyObject *
array__get_implementing_args(
    PyObject *NPY_UNUSED(dummy), PyObject *positional_args)
{
    PyObject *relevant_args;
    int j;
    int num_implementing_args = 0;
    PyObject *implementing_args[NPY_MAXARGS];
    PyObject *array_function_methods[NPY_MAXARGS];
    PyObject *result = NULL;

    if (!PyArg_ParseTuple(positional_args, "O:array__get_implementing_args",
                          &relevant_args)) {
        return NULL;
    }

    relevant_args = PySequence_Fast(
        relevant_args,
        "dispatcher for __array_function__ did not return an iterable");
    if (relevant_args == NULL) {
        return NULL;
    }

    num_implementing_args = get_implementing_args_and_methods(
        relevant_args, implementing_args, array_function_methods);
    if (num_implementing_args == -1) {
        goto cleanup;
    }

    /* create a Python object for implementing_args */
    result = PyList_New(num_implementing_args);
    if (result == NULL) {
        goto cleanup;
    }
    for (j = 0; j < num_implementing_args; j++) {
        PyObject *argument = implementing_args[j];
        Py_INCREF(argument);
        PyList_SET_ITEM(result, j, argument);
    }

cleanup:
    for (j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(array_function_methods[j]);
    }
    Py_DECREF(relevant_args);
    return result;
}
