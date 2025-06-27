/* Any file that includes Python.h must include it before any other files */
/* https://docs.python.org/3/extending/extending.html#a-simple-example */
/* npy_common.h includes Python.h so it also counts in this list */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#if Py_LIMITED_API != PY_VERSION_HEX & 0xffff0000
    # error "Py_LIMITED_API not defined to Python major+minor version"
#endif

static PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "limited_api_latest"
};

PyMODINIT_FUNC PyInit_limited_api_latest(void)
{
    import_array();
    import_umath();
    return PyModule_Create(&moduledef);
}
