#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>
#include "structmember.h"

#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"
#include "get_attr_string.h"
#include "npy_import.h"
#include "npy_static_data.h"
#include "module_state.h"
#include "multiarraymodule.h"

#include "arrayfunction_override.h"

/*
 * Get an object's __array_function__ method in the fastest way possible.
 * Never raises an exception. Returns NULL if the method doesn't exist.
 */
static PyObject *
get_array_function(PyObject *obj)
{
    multiarray_umath_state *state = _npy_module_state;
    /* Fast return for ndarray */
    if (PyArray_CheckExact(obj)) {
        Py_INCREF(state->static_pydata.ndarray_array_function);
        return state->static_pydata.ndarray_array_function;
    }

    PyObject *array_function;
    if (PyArray_LookupSpecial(
            obj, state->interned_str.array_function, &array_function) < 0) {
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
 * `items` is a C array of `length` borrowed references.
 * Returns the number of arguments, or -1 on failure.  Returns 0 without
 * collecting when no argument can carry an override (the caller then
 * dispatches to the default implementation).
 */
static int
get_implementing_args_and_methods(
        PyObject *const *items, Py_ssize_t length,
        PyObject **implementing_args, PyObject **methods)
{
    int num_implementing_args = 0;
    /*
     * Skip the leading run of args that cannot carry an override; the
     * first exact ndarray is remembered and only collected when a later
     * override candidate ends the prefix (keeping `types` unchanged).
     */
    int in_safe_prefix = 1;
    PyObject *deferred_ndarray = NULL;

    for (Py_ssize_t i = 0; i < length; i++) {
        int new_class = 1;
        PyObject *argument = items[i];

        if (in_safe_prefix) {
            if (PyArray_CheckExact(argument)) {
                if (deferred_ndarray == NULL) {
                    deferred_ndarray = argument;
                }
                continue;
            }
            if (_is_basic_python_type(Py_TYPE(argument))) {
                continue;
            }
            in_safe_prefix = 0;
            if (deferred_ndarray != NULL) {
                Py_INCREF(deferred_ndarray);
                Py_INCREF(_npy_module_state->static_pydata.ndarray_array_function);
                implementing_args[0] = deferred_ndarray;
                methods[0] = _npy_module_state->static_pydata.ndarray_array_function;
                num_implementing_args = 1;
            }
        }

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
    return obj == _npy_module_state->static_pydata.ndarray_array_function;
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
    /*
     * Python functions are wrapped, and we should now call their
     * implementation, so that we do not dispatch a second time
     * on possible subclasses.
     * C functions that can be overridden with "like" are not wrapped and
     * thus do not have an _implementation attribute, but since the like
     * keyword has been removed, we can safely call those directly.
     */
    PyObject *implementation;
    if (PyObject_GetOptionalAttr(
            func, _npy_module_state->interned_str.implementation, &implementation) < 0) {
        return NULL;
    }
    else if (implementation == NULL) {
        return PyObject_Call(func, args, kwargs);
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
    multiarray_umath_state *state = _npy_module_state;
    /* No acceptable override found, raise TypeError. */
    if (npy_cache_import_runtime(
            "numpy._core._internal",
            "array_function_errmsg_formatter",
            &state->runtime_imports.array_function_errmsg_formatter) == 0) {
        PyObject *errmsg = PyObject_CallFunctionObjArgs(
                state->runtime_imports.array_function_errmsg_formatter,
                public_api, types, NULL);
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
    multiarray_umath_state *state = _npy_module_state;
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
    if (PyDict_DelItem(kwargs, state->interned_str.like) < 0) {
        goto finish;
    }

    /* Fetch the actual symbol (the long way right now) */
    numpy_module = PyImport_Import(state->interned_str.numpy);
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

    relevant_args = PySequence_Fast( // noqa: borrowed-ref OK
        relevant_args,
        "dispatcher for __array_function__ did not return an iterable");
    if (relevant_args == NULL) {
        return NULL;
    }

    int num_implementing_args = get_implementing_args_and_methods(
            PySequence_Fast_ITEMS(relevant_args),
            PySequence_Fast_GET_SIZE(relevant_args),
            implementing_args, array_function_methods);
    if (num_implementing_args == -1) {
        goto cleanup;
    }
    if (num_implementing_args == 0) {
        /* Keep the documented NEP-18 result: the first exact ndarray
         * stands in for ndarray's default __array_function__. */
        PyObject **items = PySequence_Fast_ITEMS(relevant_args);
        Py_ssize_t length = PySequence_Fast_GET_SIZE(relevant_args);
        for (Py_ssize_t i = 0; i < length; i++) {
            if (PyArray_CheckExact(items[i])) {
                implementing_args[0] = Py_NewRef(items[i]);
                array_function_methods[0] = Py_NewRef(
                        _npy_module_state->static_pydata.ndarray_array_function);
                num_implementing_args = 1;
                break;
            }
        }
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


/* Maximum number of target argument slots (ufunc.reduce has 7). */
#define NPY_FORWARD_MAX_SLOTS 8

/*
 * Forward fast-path state (see _resolve_forward_spec): `call` is invoked
 * directly for exact-ndarray calls with `n_slots` arguments; `slots[i]`
 * is the target-slot of the i-th public parameter (-1: no slot, declines
 * when passed), `defaults` fills omitted slots.
 */
typedef struct {
    PyObject *call;
    PyObject *defaults;
    PyObject *kwnames;
    int n_slots;
    /* Positional argument count of the target call (n_slots minus the
     * trailing keyword slots). */
    int n_pos;
    int out_slot;
    int where_slot;
    /* Bit s set: the parameter for slot s defaults to np._NoValue, so
     * an explicit _NoValue means "not passed". */
    unsigned int novalue_slots;
    /* Bit s set: slot s has no default; missing it declines. */
    unsigned int required_slots;
    int slots[];
} npy_forward_info;

typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    PyObject *dict;
    PyObject *relevant_arg_func;
    PyObject *default_impl;
    /* Tuple-spec parameter table (set instead of relevant_arg_func).
     * Parameter i may be matched positionally iff i < n_pos_max and by
     * keyword iff param_names[i] is not None (None: positional-only). */
    Py_ssize_t n_params;
    int n_pos_max;
    int n_required;
    int has_varkw;
    PyObject *param_names;
    /* The relevant args are parameter indices into param_names. */
    Py_ssize_t n_relevant_args;
    uint8_t relevant_idx[NPY_MAXARGS];
    /* NULL unless this dispatcher has a forward fast path. */
    npy_forward_info *forward;
    /* The following fields are used to clean up TypeError messages only: */
    PyObject *dispatcher_name;
    PyObject *public_name;
} PyArray_ArrayFunctionDispatcherObject;

/*
 * Index of the str `needle` in `items[0..n)`, or -1 when absent.  Both
 * sides are normally interned, so the pointer pass usually decides.
 */
static inline Py_ssize_t
find_string(PyObject *needle, PyObject *const *items, Py_ssize_t n)
{
    assert(PyUnicode_Check(needle));
    for (Py_ssize_t i = 0; i < n; i++) {
        if (items[i] == needle) {
            return i;
        }
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *s = items[i];
        if (PyUnicode_Check(s) && PyUnicode_Compare(needle, s) == 0) {
            return i;
        }
    }
    return -1;
}


/* Index of keyword `kw` in the full parameter table, or -1 when unknown. */
static inline Py_ssize_t
find_param_index(const PyArray_ArrayFunctionDispatcherObject *self,
                 PyObject *kw)
{
    return find_string(
            kw, PySequence_Fast_ITEMS(self->param_names), self->n_params);
}


/*
 * Scatter an exact-ndarray call into the target's argument slots and call
 * the target (ufunc.reduce, an ndarray method, ...) directly, bypassing
 * the Python wrapper.
 * Returns 1 if handled (*result set), 0 to fall back, and -1 on error.
 */
static int
try_forward(PyArray_ArrayFunctionDispatcherObject *self,
        PyObject *const *args, Py_ssize_t nargsf, PyObject *kwnames,
        PyObject **result)
{
    const npy_forward_info *fwd = self->forward;
    PyObject *slots[NPY_FORWARD_MAX_SLOTS] = {NULL};
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    if (nargs > self->n_pos_max) {
        return 0;
    }
    for (Py_ssize_t p = 0; p < nargs; p++) {
        int slot = fwd->slots[p];
        if (slot < 0) {
            /* parameter with no target slot was passed */
            return 0;
        }
        slots[slot] = args[p];
    }
    Py_ssize_t nkwargs = (kwnames != NULL) ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t k = 0; k < nkwargs; k++) {
        PyObject *kw = PyTuple_GET_ITEM(kwnames, k);
        Py_ssize_t i = find_param_index(self, kw);
        /* Positional params form a prefix of param_names, so i < nargs is
         * a duplicate of a positional arg; i < 0 an unknown keyword. */
        if (i < nargs) {
            return 0;
        }
        int slot = fwd->slots[i];
        if (slot < 0) {
            return 0;
        }
        slots[slot] = args[nargs + k];
    }
    /* `a` must be an exact ndarray */
    if (slots[0] == NULL || !PyArray_CheckExact(slots[0])) {
        return 0;
    }
    for (int s = 1; s < fwd->n_slots; s++) {
        /* np._NoValue means "not passed", but only for parameters whose
         * own default is _NoValue. */
        if (slots[s] == NULL
                || (slots[s] == _npy_module_state->static_pydata._NoValue
                    && (fwd->novalue_slots & (1u << s)))) {
            if (fwd->required_slots & (1u << s)) {
                /* missing required argument */
                return 0;
            }
            slots[s] = PyTuple_GET_ITEM(fwd->defaults, s);
        }
    }

    if (fwd->out_slot >= 0) {
        PyObject *out = slots[fwd->out_slot];
        if (out != Py_None && !PyArray_CheckExact(out)) {
            return 0;
        }
    }
    if (fwd->where_slot >= 0) {
        PyObject *where = slots[fwd->where_slot];
        if (where != Py_None && !PyBool_Check(where)
                && !PyArray_CheckExact(where)) {
            return 0;
        }
    }
    /* Every dispatch-relevant arg must be override-free, not just
     * a/out/where: e.g. searchsorted also dispatches on `v` and `sorter`. */
    for (Py_ssize_t i = 0; i < self->n_relevant_args; i++) {
        int slot = fwd->slots[self->relevant_idx[i]];
        if (slot >= 0 && !PyArray_CheckExact(slots[slot])
                && !_is_basic_python_type(Py_TYPE(slots[slot]))) {
            return 0;
        }
    }

    *result = PyObject_Vectorcall(
            fwd->call, slots, fwd->n_pos, fwd->kwnames);
    return *result != NULL ? 1 : -1;
}


static void
dispatcher_dealloc(PyArray_ArrayFunctionDispatcherObject *self)
{
    Py_CLEAR(self->relevant_arg_func);
    Py_CLEAR(self->default_impl);
    Py_CLEAR(self->dict);
    Py_CLEAR(self->dispatcher_name);
    Py_CLEAR(self->public_name);
    Py_CLEAR(self->param_names);
    if (self->forward != NULL) {
        Py_XDECREF(self->forward->call);
        Py_XDECREF(self->forward->defaults);
        Py_XDECREF(self->forward->kwnames);
        PyMem_Free(self->forward);
    }
    PyObject_FREE(self);
}


static void
fix_name_if_typeerror(PyArray_ArrayFunctionDispatcherObject *self)
{
    if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
        return;
    }
    /* Nothing to rewrite for tuple-spec dispatchers. */
    if (self->dispatcher_name == NULL) {
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


/*
 * For a tuple-spec dispatcher, extract the value of the i-th relevant arg
 * from positional/keyword arguments.  Returns Py_None if the arg is missing
 * (so downstream checks can short-circuit on None).
 */
static inline PyObject *
lookup_relevant_arg(
        const PyArray_ArrayFunctionDispatcherObject *self, Py_ssize_t i,
        PyObject *const *args, Py_ssize_t nargs,
        PyObject *kwnames, Py_ssize_t nkwargs)
{
    assert(i >= 0 && i < self->n_relevant_args);
    /* idx is the positional slot iff idx < n_pos_max. */
    int idx = self->relevant_idx[i];
    if (idx < self->n_pos_max && idx < nargs) {
        return args[idx];
    }
    PyObject *name = PyTuple_GET_ITEM(self->param_names, idx);
    /* None: positional-only, never matched by keyword. */
    if (name != Py_None && nkwargs > 0) {
        Py_ssize_t k = find_string(
                name, PySequence_Fast_ITEMS(kwnames), nkwargs);
        if (k >= 0) {
            return args[nargs + k];
        }
    }
    return Py_None;
}


/*
 * Validate the call shape before forwarding to an __array_function__
 * override (the no-override path is validated by default_impl itself).
 * Returns 0 if valid, -1 with a TypeError set otherwise.
 */
static int
validate_call_signature(
        const PyArray_ArrayFunctionDispatcherObject *self,
        Py_ssize_t nargs, PyObject *kwnames, Py_ssize_t nkwargs)
{
    if (nargs > self->n_pos_max) {
        PyErr_Format(PyExc_TypeError,
                "%U() takes at most %d positional arguments but %zd were "
                "given", self->public_name, self->n_pos_max, nargs);
        return -1;
    }
    Py_ssize_t missing = self->n_required - nargs;  /* <= 0 when satisfied */
    for (Py_ssize_t k = 0; k < nkwargs; k++) {
        PyObject *kw = PyTuple_GET_ITEM(kwnames, k);
        Py_ssize_t i = find_param_index(self, kw);
        if (i < 0) {
            if (self->has_varkw) {
                continue;
            }
            /* %S: `kw` need not be str for misbehaving vectorcall callers */
            PyErr_Format(PyExc_TypeError,
                    "%U() got an unexpected keyword argument '%S'",
                    self->public_name, kw);
            return -1;
        }
        if (i < nargs) {
            PyErr_Format(PyExc_TypeError,
                    "%U() got multiple values for argument '%U'",
                    self->public_name, kw);
            return -1;
        }
        if (i < self->n_required) {
            missing--;
        }
    }
    if (missing > 0) {
        PyErr_Format(PyExc_TypeError,
                "%U() missing %zd required positional argument%s",
                self->public_name, missing, missing == 1 ? "" : "s");
        return -1;
    }
    return 0;
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

    if (self->forward != NULL) {
        int forward_status = try_forward(
                self, args, len_args, kwnames, &result);
        if (forward_status < 0) {
            return NULL;
        }
        if (forward_status == 1) {
            return result;
        }
    }

    if (self->param_names != NULL) {
        /* Tuple-spec: extract relevant args from the call directly.  When
         * all are safe, get_implementing_args_and_methods returns 0 and the
         * shared no-overrides path below calls default_impl. */
        public_api = (PyObject *)self;
        PyObject *items[NPY_MAXARGS];
        Py_ssize_t nargs = PyVectorcall_NARGS(len_args);
        Py_ssize_t nkwargs = (kwnames != NULL) ? PyTuple_GET_SIZE(kwnames) : 0;

        for (Py_ssize_t i = 0; i < self->n_relevant_args; i++) {
            items[i] = lookup_relevant_arg(
                    self, i, args, nargs, kwnames, nkwargs);
        }

        num_implementing_args = get_implementing_args_and_methods(
                items, self->n_relevant_args,
                implementing_args, array_function_methods);
        if (num_implementing_args < 0) {
            return NULL;
        }
        /* An override takes the call as-is, so check the signature first
         * (the no-override path is checked by default_impl itself). */
        if (num_implementing_args > 0
                && validate_call_signature(self, nargs, kwnames, nkwargs) < 0) {
            goto cleanup;
        }
    }
    else if (self->relevant_arg_func != NULL) {
        public_api = (PyObject *)self;

        /* Typical path, need to call the relevant_arg_func and unpack them */
        relevant_args = PyObject_Vectorcall(
                self->relevant_arg_func, args, len_args, kwnames);
        if (relevant_args == NULL) {
            fix_name_if_typeerror(self);
            return NULL;
        }
        Py_SETREF(relevant_args, PySequence_Fast(relevant_args, // noqa: borrowed-ref OK
                "dispatcher for __array_function__ did not return an iterable"));
        if (relevant_args == NULL) {
            return NULL;
        }

        num_implementing_args = get_implementing_args_and_methods(
                PySequence_Fast_ITEMS(relevant_args),
                PySequence_Fast_GET_SIZE(relevant_args),
                implementing_args, array_function_methods);
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


/*
 * Fill self's tuple-spec fields from ``(spec, sig_info, forward_spec)``
 * as built by array_function_dispatch.  Returns -1 with an exception set
 * on error; partial fields are released by dispatcher_dealloc.
 */
static int
init_relevant_arg_spec(
        PyArray_ArrayFunctionDispatcherObject *self, PyObject *full_spec)
{
    PyObject *spec, *sig_info, *forward_spec;
    if (!PyArg_ParseTuple(full_spec, "O!O!O:_ArrayFunctionDispatcher",
            &PyTuple_Type, &spec, &PyTuple_Type, &sig_info,
            &forward_spec)) {
        return -1;
    }

    PyObject *sig_names;
    int has_varkw;
    if (!PyArg_ParseTuple(sig_info, "O!iip:_ArrayFunctionDispatcher",
            &PyTuple_Type, &sig_names,
            &self->n_pos_max, &self->n_required, &has_varkw)) {
        return -1;
    }
    self->has_varkw = has_varkw;
    Py_ssize_t n_params = PyTuple_GET_SIZE(sig_names);
    if (n_params > UINT8_MAX) {
        /* relevant_idx stores uint8_t parameter indices */
        PyErr_Format(PyExc_ValueError,
                "too many parameters (%zd) for a tuple-spec dispatcher",
                n_params);
        return -1;
    }
    if (self->n_required < 0 || self->n_required > self->n_pos_max
            || self->n_pos_max > n_params) {
        PyErr_SetString(PyExc_ValueError,
                "inconsistent signature info (need "
                "0 <= n_required <= n_pos_max <= number of parameters)");
        return -1;
    }
    PyObject *names = PyTuple_New(n_params);
    if (names == NULL) {
        return -1;
    }
    /* Stored immediately: dispatcher_dealloc releases a partial tuple. */
    self->param_names = names;
    self->n_params = n_params;
    for (Py_ssize_t i = 0; i < n_params; i++) {
        PyObject *name = PyTuple_GET_ITEM(sig_names, i);
        if (name == Py_None) {
            /* positional-only: never matched by keyword */
            PyTuple_SET_ITEM(names, i, Py_NewRef(Py_None));
            continue;
        }
        if (!PyUnicode_Check(name)) {
            PyErr_Format(PyExc_TypeError,
                    "signature param name must be str or None, got %.200s",
                    Py_TYPE(name)->tp_name);
            return -1;
        }
        /* Own the reference before interning: InternInPlace may
         * replace (and release) the object it is given. */
        Py_INCREF(name);
        PyUnicode_InternInPlace(&name);
        PyTuple_SET_ITEM(names, i, name);
    }

    Py_ssize_t n = PyTuple_GET_SIZE(spec);
    if (n == 0) {
        /* User-facing validation lives in _resolve_relevant_arg_spec;
         * this only guards direct private-constructor calls. */
        PyErr_SetString(PyExc_ValueError, "empty relevant-argument spec");
        return -1;
    }
    if (n > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                "too many relevant args (%zd > %d)", n, NPY_MAXARGS);
        return -1;
    }
    self->n_relevant_args = n;
    for (Py_ssize_t i = 0; i < n; i++) {
        long idx = PyLong_AsLong(PyTuple_GET_ITEM(spec, i));
        if (idx < 0 || idx >= n_params) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError,
                        "relevant arg index out of range");
            }
            return -1;
        }
        self->relevant_idx[i] = (uint8_t)idx;
    }

    if (forward_spec != Py_None) {
        PyObject *fwd_call, *slots_tup, *defaults_tup, *fwd_kwnames;
        int n_slots, out_slot, where_slot;
        unsigned int novalue_slots, required_slots;
        if (!PyArg_ParseTuple(forward_spec, "OO!O!OiiiII:forward",
                &fwd_call, &PyTuple_Type, &slots_tup,
                &PyTuple_Type, &defaults_tup, &fwd_kwnames,
                &n_slots, &out_slot, &where_slot,
                &novalue_slots, &required_slots)) {
            return -1;
        }
        if (!PyCallable_Check(fwd_call)) {
            PyErr_SetString(PyExc_TypeError,
                    "forward target must be callable");
            return -1;
        }
        Py_ssize_t n_kw = 0;
        if (fwd_kwnames != Py_None) {
            if (!PyTuple_CheckExact(fwd_kwnames)) {
                PyErr_SetString(PyExc_TypeError,
                        "forward kwnames must be a tuple or None");
                return -1;
            }
            n_kw = PyTuple_GET_SIZE(fwd_kwnames);
        }
        if (n_slots <= 0 || n_slots > NPY_FORWARD_MAX_SLOTS
                || n_kw >= n_slots
                || PyTuple_GET_SIZE(slots_tup) != n_params
                || PyTuple_GET_SIZE(defaults_tup) != n_slots
                || out_slot < -1 || out_slot >= n_slots
                || where_slot < -1 || where_slot >= n_slots) {
            PyErr_SetString(PyExc_ValueError,
                    "forward spec does not match the signature table");
            return -1;
        }
        npy_forward_info *fwd = PyMem_Malloc(
                sizeof(npy_forward_info) + n_params * sizeof(int));
        if (fwd == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        for (Py_ssize_t i = 0; i < n_params; i++) {
            long slot = PyLong_AsLong(PyTuple_GET_ITEM(slots_tup, i));
            if (slot < -1 || slot >= n_slots) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_ValueError,
                            "forward slot out of range");
                }
                PyMem_Free(fwd);
                return -1;
            }
            fwd->slots[i] = (int)slot;
        }
        fwd->n_slots = n_slots;
        fwd->n_pos = n_slots - (int)n_kw;
        fwd->out_slot = out_slot;
        fwd->where_slot = where_slot;
        fwd->novalue_slots = novalue_slots;
        fwd->required_slots = required_slots;
        fwd->call = Py_NewRef(fwd_call);
        fwd->defaults = Py_NewRef(defaults_tup);
        fwd->kwnames = fwd_kwnames == Py_None ? NULL : Py_NewRef(fwd_kwnames);
        /* Set last: its presence enables try_forward. */
        self->forward = fwd;
    }
    return 0;
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

    /* Init all fields before any fallible call so dispatcher_dealloc is safe. */
    self->vectorcall = (vectorcallfunc)dispatcher_vectorcall;
    self->dict = NULL;
    self->dispatcher_name = NULL;
    self->public_name = NULL;
    self->relevant_arg_func = NULL;
    self->default_impl = NULL;
    self->n_relevant_args = 0;
    self->param_names = NULL;
    self->n_params = 0;
    self->n_pos_max = 0;
    self->n_required = 0;
    self->has_varkw = 0;
    self->forward = NULL;

    PyObject *relevant_arg_spec;
    PyObject *default_impl;
    char *kwlist[] = {"", "", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO:_ArrayFunctionDispatcher", kwlist,
            &relevant_arg_spec, &default_impl)) {
        goto fail;
    }
    Py_INCREF(default_impl);
    self->default_impl = default_impl;

    if (relevant_arg_spec == Py_None) {
        /* NULL relevant_arg_func means we use `like=` */
    }
    else if (PyTuple_CheckExact(relevant_arg_spec)) {
        /* dispatcher_name stays NULL (nothing to rewrite in errors);
         * public_name is used by validate_call_signature. */
        self->public_name = PyObject_GetAttrString(
            self->default_impl, "__qualname__");
        if (self->public_name == NULL) {
            goto fail;
        }
        if (init_relevant_arg_spec(self, relevant_arg_spec) < 0) {
            goto fail;
        }
    }
    else {
        self->relevant_arg_func = Py_NewRef(relevant_arg_spec);
        /* Fetch names to clean up TypeErrors (show actual name) */
        self->dispatcher_name = PyObject_GetAttrString(
            self->relevant_arg_func, "__qualname__");
        if (self->dispatcher_name == NULL) {
            goto fail;
        }
        self->public_name = PyObject_GetAttrString(
            self->default_impl, "__qualname__");
        if (self->public_name == NULL) {
            goto fail;
        }
    }

    /* Need to be like a Python function that has arbitrary attributes */
    self->dict = PyDict_New();
    if (self->dict == NULL) {
        goto fail;
    }
    return (PyObject *)self;

fail:
    Py_DECREF(self);
    return NULL;
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
