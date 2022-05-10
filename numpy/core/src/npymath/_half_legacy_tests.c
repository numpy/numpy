#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define NPY_USE_LEGACY_HALF
#include "numpy/halffloat.h"


static PyMethodDef TestMethods[] = {
        {NULL, NULL, 0, NULL}
};


static void
check_legacy_symbols(void)
{
  void *funcs[] = {(void *)&npy_half_to_float,
                   (void *)&npy_half_to_double,
                   (void *)&npy_float_to_half,
                   (void *)&npy_double_to_half,
                   (void *)&npy_half_eq,
                   (void *)&npy_half_ne,
                   (void *)&npy_half_le,
                   (void *)&npy_half_lt,
                   (void *)&npy_half_ge,
                   (void *)&npy_half_gt,
                   (void *)&npy_half_eq_nonan,
                   (void *)&npy_half_lt_nonan,
                   (void *)&npy_half_le_nonan,
                   (void *)&npy_half_iszero,
                   (void *)&npy_half_isnan,
                   (void *)&npy_half_isinf,
                   (void *)&npy_half_isfinite,
                   (void *)&npy_half_signbit,
                   (void *)&npy_half_copysign,
                   (void *)&npy_half_spacing,
                   (void *)&npy_half_nextafter,
                   (void *)&npy_half_divmod,
                   NULL};
  // Flagged as a volatile pointer to make sure the pointer is not dismissed,
  // preventing the compiler to *not* link with the above symbols.
  void **volatile checker = funcs;
  while (*checker)
    ++checker;
}


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_half_legacy_tests",
    NULL,
    -1,
    TestMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__half_legacy_tests(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) {
        goto fail;
    }

    check_legacy_symbols();

    return m;

fail:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load _half_legacy_tests module.");
    }
    if (m) {
        Py_DECREF(m);
        m = NULL;
    }
    return m;
}
