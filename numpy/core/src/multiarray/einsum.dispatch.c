/**
 * Testing the utilites of the CPU dispatcher
 *
 * @targets $werror baseline
 * SSE2 SSE41 AVX2
 * VSX VSX2 VSX3
 * NEON ASIMD ASIMDHP
 */
#include <Python.h>
#include "npy_cpu_dispatch.h"

#ifndef NPY_DISABLE_OPTIMIZATION
    #include "einsum.dispatch.h"
#endif

NPY_CPU_DISPATCH_DECLARE(const char *einsum_dispatch_func, (void))
NPY_CPU_DISPATCH_DECLARE(extern const char *einsum_dispatch_var)
NPY_CPU_DISPATCH_DECLARE(void einsum_dispatch_attach, (PyObject *list))

const char *NPY_CPU_DISPATCH_CURFX(einsum_dispatch_var) = NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(var));
const char *NPY_CPU_DISPATCH_CURFX(einsum_dispatch_func)(void)
{
    static const char *current = NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(func));
    return current;
}

void NPY_CPU_DISPATCH_CURFX(einsum_dispatch_attach)(PyObject *list)
{
    PyObject *item = PyUnicode_FromString(NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(func)));
    if (item) {
        PyList_Append(list, item);
    }
}
