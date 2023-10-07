#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>
#include "structmember.h"

#include "npy_pycompat.h"
#include "get_attr_string.h"
#include "npy_import.h"
#include "multiarraymodule.h"

#include "arrayfunction_override.h"

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

    PyObject *array_function = PyArray_LookupSpecial(obj, npy_ma_str_array_function);
    if (array_function == NULL && PyErr_Occurred()) {
        PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
    }

    return array_function;
}


/*
 * Like list.insert(), but for C arrays of PyObject*. Skips error checking.
 */
static void
pyobject_array_insert(PyObject **array, int length, int index, PyObject *item)
{
    for (int j = length; j > index; j--) {
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

    PyObject **items = PySequence_Fast_ITEMS(relevant_args);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(relevant_args);

    for (Py_ssize_t i = 0; i < length; i++) {
        int new_class = 1;
        PyObject *argument = items[i];

        /* Have we seen this type before? */
        for (int j = 0; j < num_implementing_args; j++) {
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
                for (int j = 0; j < num_implementing_args; j++) {
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
    for (int j = 0; j < num_implementing_args; j++) {
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
    PyObject **items = PySequence_Fast_ITEMS(types);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(types);

    for (Py_ssize_t j = 0; j < length; j++) {
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

    PyObject *implementation = PyObject_GetAttr(func, npy_ma_str_implementation);
    if (implementation == NULL) {
        return NULL;
    }
    PyObject *result = PyObject_Call(implementation, args, kwargs);
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
 * Helper to convert from vectorcall convention, since the protocol requires
 * args and kwargs to be passed as tuple and dict explicitly.
 * We always pass a dict, so always returns it.
 */
static int
get_args_and_kwargs(
        PyObject *const *fast_args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject **out_args, PyObject **out_kwargs)
{
    len_args = PyVectorcall_NARGS(len_args);
    PyObject *args = PyTuple_New(len_args);
    PyObject *kwargs = NULL;

    if (args == NULL) {
        return -1;
    }
    for (Py_ssize_t i = 0; i < len_args; i++) {
        Py_INCREF(fast_args[i]);
        PyTuple_SET_ITEM(args, i, fast_args[i]);
    }
    kwargs = PyDict_New();
    if (kwargs == NULL) {
        Py_DECREF(args);
        return -1;
    }
    if (kwnames != NULL) {
        Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t i = 0; i < nkwargs; i++) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            PyObject *value = fast_args[i+len_args];
            if (PyDict_SetItem(kwargs, key, value) < 0) {
                Py_DECREF(args);
                Py_DECREF(kwargs);
                return -1;
            }
        }
    }
    *out_args = args;
    *out_kwargs = kwargs;
    return 0;
}


static void
set_no_matching_types_error(PyObject *public_api, PyObject *types)
{
    static PyObject *errmsg_formatter = NULL;
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
}

/*
 * Implements the __array_function__ protocol for C array creation functions
 * only. Added as an extension to NEP-18 in an effort to bring NEP-35 to
 * life with minimal dispatch overhead.
 *
 * The caller must ensure that `like != Py_None` or `like == NULL`.
 */
NPY_NO_EXPORT PyObject *
array_implement_c_array_function_creation(
    const char *function_name, PyObject *like,
    PyObject *args, PyObject *kwargs,
    PyObject *const *fast_args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *dispatch_types = NULL;
    PyObject *numpy_module = NULL;
    PyObject *public_api = NULL;
    PyObject *result = NULL;

    /* If `like` doesn't implement `__array_function__`, raise a `TypeError` */
    PyObject *method = get_array_function(like);
    if (method == NULL) {
        return PyErr_Format(PyExc_TypeError,
                "The `like` argument must be an array-like that "
                "implements the `__array_function__` protocol.");
    }
    if (is_default_array_function(method)) {
        /*
         * Return a borrowed reference of Py_NotImplemented to defer back to
         * the original function.
         */
        Py_DECREF(method);
        return Py_NotImplemented;
    }

    /* We needs args and kwargs for __array_function__ (when not using it). */
    if (fast_args != NULL) {
        assert(args == NULL);
        assert(kwargs == NULL);
        if (get_args_and_kwargs(
                fast_args, len_args, kwnames, &args, &kwargs) < 0) {
            goto finish;
        }
    }
    else {
        Py_INCREF(args);
        Py_INCREF(kwargs);
    }

    dispatch_types = PyTuple_Pack(1, Py_TYPE(like));
    if (dispatch_types == NULL) {
        goto finish;
    }

    /* The like argument must be present in the keyword arguments, remove it */
    if (PyDict_DelItem(kwargs, npy_ma_str_like) < 0) {
        goto finish;
    }

    /* Fetch the actual symbol (the long way right now) */
    numpy_module = PyImport_Import(npy_ma_str_numpy);
    if (numpy_module == NULL) {
        goto finish;
    }

    public_api = PyObject_GetAttrString(numpy_module, function_name);
    Py_DECREF(numpy_module);
    if (public_api == NULL) {
        goto finish;
    }
    if (!PyCallable_Check(public_api)) {
        PyErr_Format(PyExc_RuntimeError,
                "numpy.%s is not callable.", function_name);
        goto finish;
    }

    result = call_array_function(like, method,
            public_api, dispatch_types, args, kwargs);

    if (result == Py_NotImplemented) {
        /* This shouldn't really happen as there is only one type, but... */
        Py_DECREF(result);
        result = NULL;
        set_no_matching_types_error(public_api, dispatch_types);
    }

  finish:
    Py_DECREF(method);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    Py_XDECREF(dispatch_types);
    Py_XDECREF(public_api);
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

    int num_implementing_args = get_implementing_args_and_methods(
        relevant_args, implementing_args, array_function_methods);
    if (num_implementing_args == -1) {
        goto cleanup;
    }

    /* create a Python object for implementing_args */
    result = PyList_New(num_implementing_args);
    if (result == NULL) {
        goto cleanup;
    }
    for (int j = 0; j < num_implementing_args; j++) {
        PyObject *argument = implementing_args[j];
        Py_INCREF(argument);
        PyList_SET_ITEM(result, j, argument);
    }

cleanup:
    for (int j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(array_function_methods[j]);
    }
    Py_DECREF(relevant_args);
    return result;
}


typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    PyObject *dict;
    PyObject *relevant_arg_func;
    PyObject *default_impl;
    /* The following fields are used to clean up TypeError messages only: */
    PyObject *dispatcher_name;
    PyObject *public_name;
} PyArray_ArrayFunctionDispatcherObject;


static void
dispatcher_dealloc(PyArray_ArrayFunctionDispatcherObject *self)
{
    Py_CLEAR(self->relevant_arg_func);
    Py_CLEAR(self->default_impl);
    Py_CLEAR(self->dict);
    Py_CLEAR(self->dispatcher_name);
    Py_CLEAR(self->public_name);
    PyObject_FREE(self);
}


static void
fix_name_if_typeerror(PyArray_ArrayFunctionDispatcherObject *self)
{
    if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
        return;
    }

    PyObject *exc, *val, *tb, *message;
    PyErr_Fetch(&exc, &val, &tb);

    if (!PyUnicode_CheckExact(val)) {
        /*
         * We expect the error to be unnormalized, but maybe it isn't always
         * the case, so normalize and fetch args[0] if it isn't a string.
         */
        PyErr_NormalizeException(&exc, &val, &tb);

        PyObject *args = PyObject_GetAttrString(val, "args");
        if (args == NULL || !PyTuple_CheckExact(args)
                || PyTuple_GET_SIZE(args) != 1) {
            Py_XDECREF(args);
            goto restore_error;
        }
        message = PyTuple_GET_ITEM(args, 0);
        Py_INCREF(message);
        Py_DECREF(args);
        if (!PyUnicode_CheckExact(message)) {
            Py_DECREF(message);
            goto restore_error;
        }
    }
    else {
        Py_INCREF(val);
        message = val;
    }

    Py_ssize_t cmp = PyUnicode_Tailmatch(
            message, self->dispatcher_name, 0, -1, -1);
    if (cmp <= 0) {
        Py_DECREF(message);
        goto restore_error;
    }
    Py_SETREF(message, PyUnicode_Replace(
            message, self->dispatcher_name, self->public_name, 1));
    if (message == NULL) {
        goto restore_error;
    }
    PyErr_SetObject(PyExc_TypeError, message);
    Py_DECREF(exc);
    Py_XDECREF(val);
    Py_XDECREF(tb);
    Py_DECREF(message);
    return;

  restore_error:
    /* replacement not successful, so restore original error */
    PyErr_Restore(exc, val, tb);
}


static PyObject *
dispatcher_vectorcall(PyArray_ArrayFunctionDispatcherObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *result = NULL;
    PyObject *types = NULL;
    PyObject *relevant_args = NULL;

    PyObject *public_api;

    /* __array_function__ passes args, kwargs.  These may be filled: */
    PyObject *packed_args = NULL;
    PyObject *packed_kwargs = NULL;

    PyObject *implementing_args[NPY_MAXARGS];
    PyObject *array_function_methods[NPY_MAXARGS];

    int num_implementing_args;

    if (self->relevant_arg_func != NULL) {
        public_api = (PyObject *)self;

        /* Typical path, need to call the relevant_arg_func and unpack them */
        relevant_args = PyObject_Vectorcall(
                self->relevant_arg_func, args, len_args, kwnames);
        if (relevant_args == NULL) {
            fix_name_if_typeerror(self);
            return NULL;
        }
        Py_SETREF(relevant_args, PySequence_Fast(relevant_args,
                "dispatcher for __array_function__ did not return an iterable"));
        if (relevant_args == NULL) {
            return NULL;
        }

        num_implementing_args = get_implementing_args_and_methods(
                relevant_args, implementing_args, array_function_methods);
        if (num_implementing_args < 0) {
            Py_DECREF(relevant_args);
            return NULL;
        }
    }
    else {
        /* For like= dispatching from Python, the public_symbol is the impl */
        public_api = self->default_impl;

        /*
         * We are dealing with `like=` from Python.  For simplicity, the
         * Python code passes it on as the first argument.
         */
        if (PyVectorcall_NARGS(len_args) == 0) {
            PyErr_Format(PyExc_TypeError,
                    "`like` argument dispatching, but first argument is not "
                    "positional in call to %S.", self->default_impl);
            return NULL;
        }

        array_function_methods[0] = get_array_function(args[0]);
        if (array_function_methods[0] == NULL) {
            return PyErr_Format(PyExc_TypeError,
                    "The `like` argument must be an array-like that "
                    "implements the `__array_function__` protocol.");
        }
        num_implementing_args = 1;
        implementing_args[0] = args[0];
        Py_INCREF(implementing_args[0]);

        /* do not pass the like argument */
        len_args = PyVectorcall_NARGS(len_args) - 1;
        len_args |= PY_VECTORCALL_ARGUMENTS_OFFSET;
        args++;
    }

    /*
     * Handle the typical case of no overrides. This is merely an optimization
     * if some arguments are ndarray objects, but is also necessary if no
     * arguments implement __array_function__ at all (e.g., if they are all
     * built-in types).
     */
    int any_overrides = 0;
    for (int j = 0; j < num_implementing_args; j++) {
        if (!is_default_array_function(array_function_methods[j])) {
            any_overrides = 1;
            break;
        }
    }
    if (!any_overrides) {
        /* Directly call the actual implementation. */
        result = PyObject_Vectorcall(self->default_impl, args, len_args, kwnames);
        goto cleanup;
    }

    /* Find args and kwargs as tuple and dict, as we pass them out: */
    if (get_args_and_kwargs(
            args, len_args, kwnames, &packed_args, &packed_kwargs) < 0) {
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
    for (int j = 0; j < num_implementing_args; j++) {
        PyObject *arg_type = (PyObject *)Py_TYPE(implementing_args[j]);
        Py_INCREF(arg_type);
        PyTuple_SET_ITEM(types, j, arg_type);
    }

    /* Call __array_function__ methods */
    for (int j = 0; j < num_implementing_args; j++) {
        PyObject *argument = implementing_args[j];
        PyObject *method = array_function_methods[j];

        result = call_array_function(
                argument, method, public_api, types,
                packed_args, packed_kwargs);

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

    set_no_matching_types_error(public_api, types);

cleanup:
    for (int j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(array_function_methods[j]);
    }
    Py_XDECREF(packed_args);
    Py_XDECREF(packed_kwargs);
    Py_XDECREF(types);
    Py_XDECREF(relevant_args);
    return result;
}


static PyObject *
dispatcher_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwargs)
{
    PyArray_ArrayFunctionDispatcherObject *self;

    self = PyObject_New(
            PyArray_ArrayFunctionDispatcherObject,
            &PyArrayFunctionDispatcher_Type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }

    char *kwlist[] = {"", "", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO:_ArrayFunctionDispatcher", kwlist,
            &self->relevant_arg_func, &self->default_impl)) {
        Py_DECREF(self);
        return NULL;
    }

    self->vectorcall = (vectorcallfunc)dispatcher_vectorcall;
    Py_INCREF(self->default_impl);
    self->dict = NULL;
    self->dispatcher_name = NULL;
    self->public_name = NULL;

    if (self->relevant_arg_func == Py_None) {
        /* NULL in the relevant arg function means we use `like=` */
        Py_CLEAR(self->relevant_arg_func);
    }
    else {
        /* Fetch names to clean up TypeErrors (show actual name) */
        Py_INCREF(self->relevant_arg_func);
        self->dispatcher_name = PyObject_GetAttrString(
            self->relevant_arg_func, "__qualname__");
        if (self->dispatcher_name == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        self->public_name = PyObject_GetAttrString(
            self->default_impl, "__qualname__");
        if (self->public_name == NULL) {
            Py_DECREF(self);
            return NULL;
        }
    }

    /* Need to be like a Python function that has arbitrary attributes */
    self->dict = PyDict_New();
    if (self->dict == NULL) {
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *)self;
}


static PyObject *
dispatcher_str(PyArray_ArrayFunctionDispatcherObject *self)
{
    return PyObject_Str(self->default_impl);
}


static PyObject *
dispatcher_repr(PyObject *self)
{
    PyObject *name = PyObject_GetAttrString(self, "__name__");
    if (name == NULL) {
        return NULL;
    }
    /* Print like a normal function */
    return PyUnicode_FromFormat("<function %S at %p>", name, self);
}


static PyObject *
func_dispatcher___get__(PyObject *self, PyObject *obj, PyObject *cls)
{
    if (obj == NULL) {
        /* Act like a static method, no need to bind */
        Py_INCREF(self);
        return self;
    }
    return PyMethod_New(self, obj);
}


static PyObject *
dispatcher_get_implementation(
        PyArray_ArrayFunctionDispatcherObject *self, void *NPY_UNUSED(closure))
{
    Py_INCREF(self->default_impl);
    return self->default_impl;
}


static PyObject *
dispatcher_reduce(PyObject *self, PyObject *NPY_UNUSED(args))
{
    return PyObject_GetAttrString(self, "__qualname__");
}


static struct PyMethodDef func_dispatcher_methods[] = {
    {"__reduce__",
        (PyCFunction)dispatcher_reduce, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};


static struct PyGetSetDef func_dispatcher_getset[] = {
    {"__dict__", &PyObject_GenericGetDict, 0, NULL, 0},
    {"_implementation", (getter)&dispatcher_get_implementation, 0, NULL, 0},
    {0, 0, 0, 0, 0}
};


NPY_NO_EXPORT PyTypeObject PyArrayFunctionDispatcher_Type = {
     PyVarObject_HEAD_INIT(NULL, 0)
     .tp_name = "numpy._ArrayFunctionDispatcher",
     .tp_basicsize = sizeof(PyArray_ArrayFunctionDispatcherObject),
     /* We have a dict, so in theory could traverse, but in practice... */
     .tp_dictoffset = offsetof(PyArray_ArrayFunctionDispatcherObject, dict),
     .tp_dealloc = (destructor)dispatcher_dealloc,
     .tp_new = (newfunc)dispatcher_new,
     .tp_str = (reprfunc)dispatcher_str,
     .tp_repr = (reprfunc)dispatcher_repr,
     .tp_flags = (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL
                  | Py_TPFLAGS_METHOD_DESCRIPTOR),
     .tp_methods = func_dispatcher_methods,
     .tp_getset = func_dispatcher_getset,
     .tp_descr_get = func_dispatcher___get__,
     .tp_call = &PyVectorcall_Call,
     .tp_vectorcall_offset = offsetof(PyArray_ArrayFunctionDispatcherObject, vectorcall),
};
