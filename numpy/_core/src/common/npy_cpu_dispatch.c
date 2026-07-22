#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "npy_cpu_dispatch.h"
#include "numpy/ndarraytypes.h"
#include "npy_static_data.h"
#include "module_state.h"

NPY_VISIBILITY_HIDDEN int
npy_cpu_dispatch_tracer_init(PyObject *mod)
{
    multiarray_umath_state *state = get_module_state(mod);
    if (state->static_pydata.cpu_dispatch_registry != NULL) {
        PyErr_Format(PyExc_RuntimeError, "CPU dispatcher tracer already initialized");
        return -1;
    }
    PyObject *mod_dict = PyModule_GetDict(mod);
    if (mod_dict == NULL) {
        return -1;
    }
    PyObject *reg_dict = PyDict_New();
    if (reg_dict == NULL) {
        return -1;
    }
    int err = PyDict_SetItemString(mod_dict, "__cpu_targets_info__", reg_dict);
    Py_DECREF(reg_dict);
    if (err != 0) {
        return -1;
    }
    /*
     * state->static_pydata.cpu_dispatch_registry is a separate owned
     * reference from the one held by mod_dict above, since it is
     * visited/cleared by the module's GC hooks.
     */
    Py_INCREF(reg_dict);
    state->static_pydata.cpu_dispatch_registry = reg_dict;
    return 0;
}

NPY_VISIBILITY_HIDDEN void
npy_cpu_dispatch_trace(const char *fname, const char *signature,
                       const char **dispatch_info)
{
    npy_static_pydata_struct *static_pydata = &npy_get_module_state()->static_pydata;
    PyObject *func_dict = PyDict_GetItemString(static_pydata->cpu_dispatch_registry, fname); // noqa: borrowed-ref OK
    if (func_dict == NULL) {
        func_dict = PyDict_New();
        if (func_dict == NULL) {
            return;
        }
        int err = PyDict_SetItemString(static_pydata->cpu_dispatch_registry, fname, func_dict);
        Py_DECREF(func_dict);
        if (err != 0) {
            return;
        }
    }
    // target info for each signature
    PyObject *sig_dict = PyDict_New();
    if (sig_dict == NULL) {
        return;
    }
    int err = PyDict_SetItemString(func_dict, signature, sig_dict);
    Py_DECREF(sig_dict);
    if (err != 0) {
        return;
    }
    // current dispatched target
    PyObject *current_target = PyUnicode_FromString(dispatch_info[0]);
    if (current_target == NULL) {
        return;
    }
    err = PyDict_SetItemString(sig_dict, "current", current_target);
    Py_DECREF(current_target);
    if (err != 0) {
        return;
    }
    // available targets
    PyObject *available = PyUnicode_FromString(dispatch_info[1]);
    if (available == NULL) {
        return;
    }
    err = PyDict_SetItemString(sig_dict, "available", available);
    Py_DECREF(available);
    if (err != 0) {
        return;
    }
}
