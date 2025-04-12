#include "_simd.h"

#include "numpy/npy_math.h"

static PyObject *
get_floatstatus(PyObject* NPY_UNUSED(self), PyObject *NPY_UNUSED(args))
{
    return PyLong_FromLong(npy_get_floatstatus());
}

static PyObject *
clear_floatstatus(PyObject* NPY_UNUSED(self), PyObject *NPY_UNUSED(args))
{
    npy_clear_floatstatus();
    Py_RETURN_NONE;
}

static PyMethodDef _simd_methods[] = {
    {"get_floatstatus", get_floatstatus, METH_NOARGS, NULL},
    {"clear_floatstatus", clear_floatstatus, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit__simd(void)
{
    static struct PyModuleDef defs = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "numpy._core._simd",
        .m_size = -1,
        .m_methods = _simd_methods
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
    // add keys for non-supported optimizations with None value
    #define ATTACH_MODULE(TESTED_FEATURES, TARGET_NAME, MAKE_MSVC_HAPPY)       \
        {                                                                      \
            PyObject *simd_mod;                                                \
            if (!TESTED_FEATURES) {                                            \
                Py_INCREF(Py_None);                                            \
                simd_mod = Py_None;                                            \
            } else {                                                           \
                simd_mod = NPY_CAT(simd_create_module_, TARGET_NAME)();        \
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
            PyObject *simd_mod = simd_create_module();                         \
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
    NPY_MTARGETS_CONF_DISPATCH(NPY_CPU_HAVE, ATTACH_MODULE, MAKE_MSVC_HAPPY)
    NPY_MTARGETS_CONF_BASELINE(ATTACH_BASELINE_MODULE, MAKE_MSVC_HAPPY)

#if Py_GIL_DISABLED
    // signal this module supports running with the GIL disabled
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    return m;
err:
    Py_DECREF(m);
    return NULL;
}
