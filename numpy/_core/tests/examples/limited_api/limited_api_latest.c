#if Py_LIMITED_API != PY_VERSION_HEX & 0xffff0000
    # error "Py_LIMITED_API not defined to Python major+minor version"
#endif

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

static int
_limited_api_latest_exec(void)
{
    import_array1(-1);
    import_umath1(-1);
    return 0;
}

static struct PyModuleDef_Slot limited_api_latest_slots[] = {
    {Py_mod_exec, _limited_api_latest_exec},
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
    .m_name = "limited_api_latest",
    .m_size = 0,
    .m_slots = limited_api_latest_slots,
};

PyMODINIT_FUNC PyInit_limited_api_latest(void)
{
    return PyModuleDef_Init(&moduledef);
}
