/*
 * NumPy Debug-only C-Level Function Call Tracing - Python Extension Module
 *
 * This module provides Python bindings for the C-level tracing functionality.
 * It is only compiled when NUMPY_DEBUG_CTRACE is defined.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <Python.h>
#include "npy_config.h"

#ifdef NUMPY_DEBUG_CTRACE

#include "npy_ctrace.h"
#include <stdint.h>

/* Python callback storage */
static PyObject *py_callback = NULL;

/*
 * Wrapper callback that invokes the Python callback.
 */
__attribute__((no_instrument_function))
static void
python_callback_wrapper(void *func, void *caller, uint32_t depth, int is_entry)
{
    if (!py_callback || py_callback == Py_None) {
        return;
    }

    /* Acquire GIL - we might be called from a non-Python thread */
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *args = Py_BuildValue(
        "(KKiO)",
        (unsigned long long)(uintptr_t)func,
        (unsigned long long)(uintptr_t)caller,
        (int)depth,
        is_entry ? Py_True : Py_False
    );

    if (args) {
        PyObject *result = PyObject_CallObject(py_callback, args);
        Py_DECREF(args);
        if (result) {
            Py_DECREF(result);
        }
        else {
            /* Clear any exception - we can't propagate from here */
            PyErr_Clear();
        }
    }

    PyGILState_Release(gstate);
}

/*
 * Python API: enable()
 */
static PyObject *
ctrace_enable(PyObject *self, PyObject *args)
{
    npy_ctrace_enable();
    Py_RETURN_NONE;
}

/*
 * Python API: disable()
 */
static PyObject *
ctrace_disable(PyObject *self, PyObject *args)
{
    npy_ctrace_disable();
    Py_RETURN_NONE;
}

/*
 * Python API: is_enabled()
 */
static PyObject *
ctrace_is_enabled(PyObject *self, PyObject *args)
{
    if (npy_ctrace_is_enabled()) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/*
 * Python API: get_depth()
 */
static PyObject *
ctrace_get_depth(PyObject *self, PyObject *args)
{
    return PyLong_FromUnsignedLong(npy_ctrace_get_depth());
}

/*
 * Python API: snapshot(max_size)
 */
static PyObject *
ctrace_snapshot(PyObject *self, PyObject *args)
{
    Py_ssize_t max_size = 100;

    if (!PyArg_ParseTuple(args, "|n", &max_size)) {
        return NULL;
    }

    if (max_size <= 0) {
        return PyList_New(0);
    }

    /* Allocate buffer */
    void **buffer = (void **)PyMem_Malloc(max_size * sizeof(void *));
    if (!buffer) {
        return PyErr_NoMemory();
    }

    /* Get snapshot */
    size_t count = npy_ctrace_snapshot(buffer, (size_t)max_size);

    /* Build Python list */
    PyObject *result = PyList_New(count);
    if (!result) {
        PyMem_Free(buffer);
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        PyObject *addr = PyLong_FromUnsignedLongLong(
            (unsigned long long)(uintptr_t)buffer[i]);
        if (!addr) {
            Py_DECREF(result);
            PyMem_Free(buffer);
            return NULL;
        }
        PyList_SET_ITEM(result, i, addr);
    }

    PyMem_Free(buffer);
    return result;
}

/*
 * Python API: resolve_symbol(addr)
 */
static PyObject *
ctrace_resolve_symbol(PyObject *self, PyObject *args)
{
    unsigned long long addr;

    if (!PyArg_ParseTuple(args, "K", &addr)) {
        return NULL;
    }

    char buf[512];
    const char *symbol = npy_ctrace_resolve_symbol(
        (void *)(uintptr_t)addr, buf, sizeof(buf));

    if (symbol) {
        return PyUnicode_FromString(symbol);
    }

    Py_RETURN_NONE;
}

/*
 * Python API: dump_stack()
 */
static PyObject *
ctrace_dump_stack(PyObject *self, PyObject *args)
{
    npy_ctrace_dump_stack();
    Py_RETURN_NONE;
}

/*
 * Python API: set_callback(callback)
 */
static PyObject *
ctrace_set_callback(PyObject *self, PyObject *args)
{
    PyObject *callback;

    if (!PyArg_ParseTuple(args, "O", &callback)) {
        return NULL;
    }

    /* Clear old callback */
    Py_XDECREF(py_callback);
    py_callback = NULL;

    if (callback == Py_None) {
        /* Reset to default C callback */
        npy_ctrace_set_callback(NULL);
    }
    else if (PyCallable_Check(callback)) {
        /* Set Python callback */
        Py_INCREF(callback);
        py_callback = callback;
        npy_ctrace_set_callback(python_callback_wrapper);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "callback must be callable or None");
        return NULL;
    }

    Py_RETURN_NONE;
}

/* Method table */
static PyMethodDef ctrace_methods[] = {
    {"enable", ctrace_enable, METH_NOARGS,
     "Enable C-level tracing for the current thread."},
    {"disable", ctrace_disable, METH_NOARGS,
     "Disable C-level tracing for the current thread."},
    {"is_enabled", ctrace_is_enabled, METH_NOARGS,
     "Check if tracing is enabled for the current thread."},
    {"get_depth", ctrace_get_depth, METH_NOARGS,
     "Get the current call stack depth."},
    {"snapshot", ctrace_snapshot, METH_VARARGS,
     "Snapshot the current C call stack."},
    {"resolve_symbol", ctrace_resolve_symbol, METH_VARARGS,
     "Resolve a function address to a symbol name."},
    {"dump_stack", ctrace_dump_stack, METH_NOARGS,
     "Dump the current C call stack to stderr."},
    {"set_callback", ctrace_set_callback, METH_VARARGS,
     "Set a custom callback for trace events."},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef ctrace_module = {
    PyModuleDef_HEAD_INIT,
    "_ctrace_impl",
    "NumPy C-level function call tracing (debug builds only)",
    -1,
    ctrace_methods,
    NULL, NULL, NULL, NULL
};

/* Module initialization */
PyMODINIT_FUNC
PyInit__ctrace_impl(void)
{
    /* Initialize the tracing subsystem */
    npy_ctrace_init();

    return PyModule_Create(&ctrace_module);
}

#else /* !NUMPY_DEBUG_CTRACE */

/*
 * Stub module when tracing is disabled.
 * This allows the Python code to check for availability.
 */

static PyMethodDef ctrace_methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ctrace_module = {
    PyModuleDef_HEAD_INIT,
    "_ctrace_impl",
    "NumPy C-level function call tracing (not available - build without NUMPY_DEBUG_CTRACE)",
    -1,
    ctrace_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__ctrace_impl(void)
{
    return PyModule_Create(&ctrace_module);
}

#endif /* NUMPY_DEBUG_CTRACE */
