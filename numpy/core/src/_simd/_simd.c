#include "_simd.h"

PyMODINIT_FUNC PyInit__simd(void)
{
    static struct PyModuleDef defs = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "numpy.core._simd",
        .m_size = -1
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

    NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, ATTACH_MODULE, MAKE_MSVC_HAPPY)
    NPY__CPU_DISPATCH_BASELINE_CALL(ATTACH_BASELINE_MODULE, MAKE_MSVC_HAPPY)
    return m;
err:
    Py_DECREF(m);
    return NULL;
}
