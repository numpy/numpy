#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include "numpy/npy_3kcompat.h"
#include <math.h>
#include <structmember.h>


static PyMethodDef TestMethods[] = {
        {NULL, NULL, 0, NULL}
};


static void
inplace_add(char **args, npy_intp const *dimensions, npy_intp const *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp in2_step = steps[1];

    for (i = 0; i < n; i++) {
        (*(npy_intp *)in1) = *(npy_intp*)in1 + *(npy_intp*)in2;
        in1 += in1_step;
        in2 += in2_step;
    }
}


/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&inplace_add};

/* These are the input and return dtypes of logit.*/
static const char types[2] = {NPY_INTP, NPY_INTP};

static void *const data[1] = {NULL};

static int
_operand_flag_tests_exec(PyObject *m)
{
    PyObject *ufunc;

    import_array1(-1);
    import_umath1(-1);

    ufunc = PyUFunc_FromFuncAndData(funcs, data, types, 1, 2, 0,
                                    PyUFunc_None, "inplace_add",
                                    "inplace_add_docstring", 0);

    /*
     * Set flags to turn off buffering for first input operand,
     * so that result can be written back to input operand.
     */
    ((PyUFuncObject*)ufunc)->op_flags[0] = NPY_ITER_READWRITE;
    ((PyUFuncObject*)ufunc)->iter_flags = NPY_ITER_REDUCE_OK;
    PyModule_AddObject(m, "inplace_add", (PyObject*)ufunc);

    return 0;
}

static struct PyModuleDef_Slot _operand_flag_tests_slots[] = {
    {Py_mod_exec, _operand_flag_tests_exec},
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
    .m_name = "_operand_flag_tests",
    .m_size = 0,
    .m_methods = TestMethods,
    .m_slots = _operand_flag_tests_slots,
};

PyMODINIT_FUNC PyInit__operand_flag_tests(void) {
    return PyModuleDef_Init(&moduledef);
}
