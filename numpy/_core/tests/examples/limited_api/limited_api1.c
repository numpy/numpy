/* Any file that includes Python.h must include it before any other files */
/* https://docs.python.org/3/extending/extending.html#a-simple-example */
/* npy_common.h includes Python.h so it also counts in this list */
#define Py_LIMITED_API 0x03060000

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

static PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "limited_api1"
};

PyMODINIT_FUNC PyInit_limited_api1(void)
{
    import_array();
    import_umath();
    return PyModule_Create(&moduledef);
}
