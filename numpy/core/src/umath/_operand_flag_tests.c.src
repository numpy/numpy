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
        (*(long *)in1) = *(long*)in1 + *(long*)in2;
        in1 += in1_step;
        in2 += in2_step;
    }
}


/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&inplace_add};

/* These are the input and return dtypes of logit.*/
static char types[2] = {NPY_LONG, NPY_LONG};

static void *data[1] = {NULL};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_operand_flag_tests",
    NULL,
    -1,
    TestMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__operand_flag_tests(void)
{
    PyObject *m = NULL;
    PyObject *ufunc;

    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        goto fail;
    }

    import_array();
    import_umath();

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

    return m;

fail:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load _operand_flag_tests module.");
    }
    if (m) {
        Py_DECREF(m);
        m = NULL;
    }
    return m;
}
