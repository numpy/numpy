/**
 * Testing the utilities of the CPU dispatcher
 *
 * @targets $werror baseline
 * SSE2 SSE41 AVX2
 * VSX VSX2 VSX3
 * NEON ASIMD ASIMDHP
 * VX VXE
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "npy_cpu_dispatch.h"

#ifndef NPY_DISABLE_OPTIMIZATION
    #include "_umath_tests.dispatch.h"
#endif

NPY_CPU_DISPATCH_DECLARE(const char *_umath_tests_dispatch_func, (void))
NPY_CPU_DISPATCH_DECLARE(extern const char *_umath_tests_dispatch_var)
NPY_CPU_DISPATCH_DECLARE(void _umath_tests_dispatch_attach, (PyObject *list))

const char *NPY_CPU_DISPATCH_CURFX(_umath_tests_dispatch_var) = NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(var));
const char *NPY_CPU_DISPATCH_CURFX(_umath_tests_dispatch_func)(void)
{
    static const char *current = NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(func));
    return current;
}

void NPY_CPU_DISPATCH_CURFX(_umath_tests_dispatch_attach)(PyObject *list)
{
    PyObject *item = PyUnicode_FromString(NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(func)));
    if (item) {
        PyList_Append(list, item);
        Py_DECREF(item);
    }
}
