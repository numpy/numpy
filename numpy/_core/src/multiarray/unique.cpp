#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>

#include <unordered_map>
#include <functional>

#include "numpy/arrayobject.h"
#include "numpy/npy_common.h"


template<typename T>
PyObject* unique(PyArrayObject *self)
{
    /* This function takes a numpy array and returns a numpy array containing
    the unique values.

    It assumes the numpy array includes data that can be viewed as unsigned integers
    of a certain size (sizeof(T)).

    It doesn't need to know the actual type, since it needs to find unique values
    among binary representations of the input data. This means it won't apply to
    custom or complicated dtypes or string values.
    */
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

    // first we put the data in a hash map
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

    // then we iterate through the map's keys to get the unique values
    T* res = new T[hashmap.size()];
    auto it = hashmap.begin();
    size_t i = 0;
    for (; it != hashmap.end(); it++, i++) {
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


// this map contains the functions used for each item size.
typedef std::function<PyObject *(PyArrayObject *)> function_type;
std::unordered_map<size_t, function_type> unique_funcs = {
    {sizeof(npy_uint8), unique<npy_uint8>},
    {sizeof(npy_uint16), unique<npy_uint16>},
    {sizeof(npy_uint32), unique<npy_uint32>},
    {sizeof(npy_uint64), unique<npy_uint64>},
};


static PyObject *
PyArray_Unique(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    /* This is called from Python space, and expects a single numpy array as input.

    It then returns a numpy array containing the unique values of the input array.

    If the input array is not supported, it returns None.
    */
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
    PyArray_Descr *descr = PyArray_DESCR(self);
    char kind = descr->kind;
    // we only support booleans, integers, unsigned integers, and floats
    // we also only support data sizes present in our unique_funcs map
    if (
        (kind != 'b' && kind != 'i' && kind != 'u' && kind !='f')
        || (unique_funcs.find(itemsize) == unique_funcs.end())
    ){
        Py_XDECREF(self);
        return Py_None;
    }


    /* for the purpose of finding unique values on dtypes that we support, we
    don't really care what dtype it is, and we can look at the data as if they
    were all uint values */
    res = unique_funcs[itemsize](self);
    Py_XDECREF(self);
    return res;
}


// The following is to expose the unique function to Python
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
