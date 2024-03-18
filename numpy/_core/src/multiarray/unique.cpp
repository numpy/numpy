#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>

#include <unordered_map>
#include <vector>
#include <random>
#include <iostream>

#include "numpy/ndarraytypes.h"

#include "numpy/arrayobject.h"
#include "numpy/npy_common.h"

#include "numpy/npy_2_compat.h"

template<typename T>
npy_intp unique(PyArrayObject *self)
{
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp* strideptr,* innersizeptr;
    std::unordered_map<T, char> hashmap;

    iter = NpyIter_New(self, NPY_ITER_READONLY|
                             NPY_ITER_EXTERNAL_LOOP|
                             NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        return -1;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }
    /* The location of the data pointer which the iterator may update */
    dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    std::cout << "printing values: " << std::endl;
    do {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        while (count--) {
            std::cout << (T)* data << std::endl;
            hashmap[(T)* data] = 0;
            data += stride;
        }

        /* Increment the iterator to the next inner loop */
    } while(iternext(iter));

    std::vector<T> res;
    std::cout << "unique values :" << std::endl;
    res.reserve(hashmap.size());
    for (auto it = hashmap.begin(); it != hashmap.end(); it++) {
        res.emplace_back(it->first);
        std::cout << it->first << std::endl;
    }

    NpyIter_Deallocate(iter);
    return 0;
}

static PyObject *
PyArray_Unique(PyObject *dummy, PyObject *args)
{
    PyArrayObject *self = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &self))
        return NULL;

    npy_intp itemsize;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(self) == 0) {
        return Py_BuildValue("i", 0);
    }

    itemsize = PyArray_ITEMSIZE(self);
    std::cout << "Item size: " << itemsize << std::endl;

    if (sizeof(char) == itemsize) {
        unique<char>(self);
    } else if (sizeof(int) == itemsize) {
        unique<int>(self);
    }
    return Py_BuildValue("i", 0);
}

static PyMethodDef UniqueMethods[] = {
    {"unique_hash",  PyArray_Unique, METH_VARARGS,
     "Collect unique values via a hash map."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef uniquemodule = {
    PyModuleDef_HEAD_INIT,
    "unique", /* name of module */
    NULL, /* module docs */
    -1,  /* size of per-interpreter state of the module,
            or -1 if the module keeps state in global variables. */
    UniqueMethods
};

PyMODINIT_FUNC
PyInit_unique(void)
{
    // import_array();
    // PyArray_ImportNumPyAPI();
    return PyModule_Create(&uniquemodule);
}
