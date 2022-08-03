#include "module.hpp"

#include "common.hpp"

#include <tuple>
#include <array>

static PyObject *
floatstatus(PyObject* NPY_UNUSED(self), PyObject *NPY_UNUSED(args))
{
    auto status = np::FloatStatus();
    PyObject *status_dict = PyDict_New();
    if (status_dict == nullptr) {
        return nullptr;
    }
    std::array<std::tuple<const char*, bool>, 5> values {{
        {"DivideByZero", status.IsDivideByZero()},
        {"Inexact", status.IsInexact()},
        {"Invalid", status.IsInvalid()},
        {"OverFlow", status.IsOverFlow()},
        {"UnderFlow", status.IsUnderFlow()}
    }};
    for (int i = 0; i < 5; ++i) {
        auto v = values[i];
        PyObject *obj = PyBool_FromLong(static_cast<long>(std::get<1>(v)));
        if (obj == nullptr) {
            Py_DECREF(status_dict);
            return nullptr;
        }
        if (PyDict_SetItemString(status_dict, std::get<0>(v), obj) < 0) {
            Py_DECREF(obj);
            Py_DECREF(status_dict);
            return nullptr;
        }
    }
    return status_dict;
}


PyMODINIT_FUNC PyInit__intrinsics(void)
{
    static PyMethodDef intrinsics_methods[] = {
        {"floatstatus", floatstatus, METH_NOARGS, NULL},
        {NULL, NULL, 0, NULL}
    };

    static struct PyModuleDef defs = {
        PyModuleDef_HEAD_INIT,
        "numpy.core._simd",
        "Simd module for testing purposes",
        -1,
        intrinsics_methods
    };
    if (npy_cpu_init() < 0) {
        return NULL;
    }
    PyObject *m = PyModule_Create(&defs);
    if (m == NULL) {
        return NULL;
    }
    PyObject *targets = PyDict_New();
    if (targets == NULL) {
        goto err;
    }
    if (PyModule_AddObject(m, "targets", targets) < 0) {
        Py_DECREF(targets);
        goto err;
    }
#ifndef NPY_DISABLE_OPTIMIZATION
    #include "intrinsics.dispatch.h"
#endif
    // add keys for non-supported optimizations with None value
    #define ATTACH_MODULE(TESTED_FEATURES, TARGET_NAME, MAKE_MSVC_HAPPY)       \
        {                                                                      \
            PyObject *simd_mod;                                                \
            if (!TESTED_FEATURES) {                                            \
                Py_INCREF(Py_None);                                            \
                simd_mod = Py_None;                                            \
            } else {                                                           \
                simd_mod = NPY_CAT(np::SimdExtention_, TARGET_NAME)();         \
                if (simd_mod == NULL) {                                        \
                    goto err;                                                  \
                }                                                              \
            }                                                                  \
            const char *target_name = NPY_TOSTRING(TARGET_NAME);               \
            if (PyDict_SetItemString(targets, target_name, simd_mod) < 0) {    \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
            Py_INCREF(simd_mod);                                               \
            if (PyModule_AddObject(m, target_name, simd_mod) < 0) {            \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
        }

    #define ATTACH_BASELINE_MODULE(MAKE_MSVC_HAPPY)                            \
        {                                                                      \
            PyObject *simd_mod = np::SimdExtention();                          \
            if (simd_mod == NULL) {                                            \
                goto err;                                                      \
            }                                                                  \
            if (PyDict_SetItemString(targets, "baseline", simd_mod) < 0) {     \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
            Py_INCREF(simd_mod);                                               \
            if (PyModule_AddObject(m, "baseline", simd_mod) < 0) {             \
                Py_DECREF(simd_mod);                                           \
                goto err;                                                      \
            }                                                                  \
        }

    NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, ATTACH_MODULE, MAKE_MSVC_HAPPY)
    NPY__CPU_DISPATCH_BASELINE_CALL(ATTACH_BASELINE_MODULE, MAKE_MSVC_HAPPY)
    return m;
err:
    Py_DECREF(m);
    return NULL;
}

