#define Py_LIMITED_API 0x03060000

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

static int
_limited_api1_exec(void)
{
    import_array1(-1);
    import_umath1(-1);
    return 0;
}

static struct PyModuleDef_Slot limited_api1_slots[] = {
    {Py_mod_exec, _limited_api1_exec},
#if PY_VERSION_HEX >= 0x030c00f0  // Python 3.12+
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d00f0  // Python 3.13+
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "limited_api1",
    .m_size = 0,
    .m_slots = limited_api1_slots,
};

PyMODINIT_FUNC PyInit_limited_api1(void)
{
    return PyModuleDef_Init(&moduledef);
}
