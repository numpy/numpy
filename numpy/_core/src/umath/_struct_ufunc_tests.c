#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#include <math.h>


/*
 * struct_ufunc_test.c
 * This is the C code for creating your own
 * NumPy ufunc for a structured array dtype.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

static void add_uint64_triplet(char **args,
                               npy_intp const *dimensions,
                               npy_intp const* steps,
                               void* data)
{
    npy_intp i;
    npy_intp is1=steps[0];
    npy_intp is2=steps[1];
    npy_intp os=steps[2];
    npy_intp n=dimensions[0];
    npy_uint64 *x, *y, *z;

    char *i1=args[0];
    char *i2=args[1];
    char *op=args[2];

    for (i = 0; i < n; i++) {

        x = (npy_uint64*)i1;
        y = (npy_uint64*)i2;
        z = (npy_uint64*)op;

        z[0] = x[0] + y[0];
        z[1] = x[1] + y[1];
        z[2] = x[2] + y[2];

        i1 += is1;
        i2 += is2;
        op += os;
    }
}

static PyObject*
register_fail(PyObject* NPY_UNUSED(self), PyObject* NPY_UNUSED(args))
{
    PyObject *add_triplet;
    PyObject *dtype_dict;
    PyArray_Descr *dtype;
    PyArray_Descr *dtypes[3];
    int retval;

    add_triplet = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 2, 1,
                                    PyUFunc_None, "add_triplet",
                                    "add_triplet_docstring", 0);

    dtype_dict = Py_BuildValue("[(s, s), (s, s), (s, s)]",
                               "f0", "u8", "f1", "u8", "f2", "u8");
    PyArray_DescrConverter(dtype_dict, &dtype);
    Py_DECREF(dtype_dict);

    dtypes[0] = dtype;
    dtypes[1] = dtype;
    dtypes[2] = dtype;

    retval = PyUFunc_RegisterLoopForDescr((PyUFuncObject *)add_triplet,
                                dtype,
                                &add_uint64_triplet,
                                dtypes,
                                NULL);

    if (retval < 0) {
        Py_DECREF(add_triplet);
        Py_DECREF(dtype);
        return NULL;
    }
    retval = PyUFunc_RegisterLoopForDescr((PyUFuncObject *)add_triplet,
                                dtype,
                                &add_uint64_triplet,
                                dtypes,
                                NULL);
    Py_DECREF(add_triplet);
    Py_DECREF(dtype);
    if (retval < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef StructUfuncTestMethods[] = {
    {"register_fail",
        register_fail,
        METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_struct_ufunc_tests",
    NULL,
    -1,
    StructUfuncTestMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__struct_ufunc_tests(void)
{
    PyObject *m, *add_triplet, *d;
    PyObject *dtype_dict;
    PyArray_Descr *dtype;
    PyArray_Descr *dtypes[3];

    import_array();
    import_umath();

    m = PyModule_Create(&moduledef);

    if (m == NULL) {
        return NULL;
    }

    add_triplet = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 2, 1,
                                          PyUFunc_None, "add_triplet",
                                          NULL, 0);

    dtype_dict = Py_BuildValue("[(s, s), (s, s), (s, s)]",
                               "f0", "u8", "f1", "u8", "f2", "u8");
    PyArray_DescrConverter(dtype_dict, &dtype);
    Py_DECREF(dtype_dict);

    dtypes[0] = dtype;
    dtypes[1] = dtype;
    dtypes[2] = dtype;

    PyUFunc_RegisterLoopForDescr((PyUFuncObject *)add_triplet,
                                dtype,
                                &add_uint64_triplet,
                                dtypes,
                                NULL);

    Py_DECREF(dtype);
    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "add_triplet", add_triplet);
    Py_DECREF(add_triplet);

#ifdef Py_GIL_DISABLED
    // signal this module supports running with the GIL disabled
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    return m;
}
