#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>

#include <unordered_map>
#include <iostream>

#include "numpy/ndarraytypes.h"

#include "numpy/arrayobject.h"
#include "numpy/npy_common.h"

#include "numpy/npy_2_compat.h"

template<typename T>
PyObject* unique(PyArrayObject *self)
{
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    size_t i;
    npy_intp* strideptr,* innersizeptr;
    std::unordered_map<T, char> hashmap;

    iter = NpyIter_New(self, NPY_ITER_READONLY|
                             NPY_ITER_EXTERNAL_LOOP|
                             NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        return Py_None;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return Py_None;
    }
    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    do {
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        while (count--) {
            hashmap[*((T *) data)] = 0;
            data += stride;
        }
    } while(iternext(iter));
    NpyIter_Deallocate(iter);

    T* res = new T[hashmap.size()];
    for (auto it = hashmap.begin(), i = 0; it != hashmap.end(); it++, i++) {
        res[i] = it->first;
    }

    // does this need to have the same lifetime as the array?
    npy_intp dims[1] = {(npy_intp)hashmap.size()};
    PyArray_Descr *descr = PyArray_DESCR(self);
    Py_INCREF(descr);
    return PyArray_NewFromDescr(
        &PyArray_Type,
        descr,
        1, // ndim
        dims, // shape
        NULL, // strides
        res, // data
        NPY_ARRAY_OUT_ARRAY, // flags
        NULL // obj
    );
}

static PyObject *
PyArray_Unique(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyArrayObject *self = NULL;
    PyObject *res = NULL;
    if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &self))
        return NULL;

    npy_intp itemsize;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(self) == 0) {
        Py_XDECREF(self);
        return PyArray_NewLikeArray(
            self,
            NPY_ANYORDER,
            NULL, // descr (use prototype's descr)
            1 // subok
        );
    }

    itemsize = PyArray_ITEMSIZE(self);

    /* for the purpose of finding unique values on dtypes that we support, we
    don't really care what dtype it is, and we can look at the data as if they
    were all uint values */
    if (sizeof(npy_uint8) == itemsize) {
        res = unique<npy_uint8>(self);
    } else if (sizeof(npy_uint16) == itemsize) {
        res = unique<npy_uint16>(self);
    } else if (sizeof(npy_uint32) == itemsize) {
        res = unique<npy_uint32>(self);
    } else if (sizeof(npy_uint64) == itemsize) {
        res = unique<npy_uint64>(self);
    // these don't seem to be available?
    // } else if (sizeof(npy_uint96) == itemsize) {
    //     unique<npy_uint96>(self);
    // } else if (sizeof(npy_uint128) == itemsize) {
    //     unique<npy_uint128>(self);
    } else {
        Py_XDECREF(self);
        return Py_None;
    }
    Py_XDECREF(self);
    return res;
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
    PyArray_ImportNumPyAPI();
    return PyModule_Create(&uniquemodule);
}
