/*
 * Test that third-party extensions that use the Numpy C API can be built with
 * the limited Python C API (see https://docs.python.org/3/c-api/stable.html).
 */

#define Py_LIMITED_API 0x03060000

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

static PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "example_limited_api"
};

PyMODINIT_FUNC PyInit_example_limited_api(void)
{
    import_array();
    import_umath();
    return PyModule_Create(&moduledef);
}
