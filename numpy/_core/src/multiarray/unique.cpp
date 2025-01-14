#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>

#include <unordered_set>
#include <functional>

#include "numpy/arrayobject.h"
#include "numpy/npy_common.h"


// This is to use RAII pattern to handle cpp exceptions while avoiding memory leaks.
// Adapted from https://stackoverflow.com/a/25510879/2536294
template <typename F>
struct FinalAction {
    FinalAction(F f) : clean_{f} {}
    ~FinalAction() { clean_(); }
  private:
    F clean_;
};

template <typename F>
FinalAction<F> finally(F f) {
    return FinalAction<F>(f);
}

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
    std::unordered_set<T> hashset;

    iter = NpyIter_New(self, NPY_ITER_READONLY |
                             NPY_ITER_EXTERNAL_LOOP |
                             NPY_ITER_REFS_OK |
                             NPY_ITER_ZEROSIZE_OK |
                             NPY_ITER_GROWINNER,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    // Making sure the iterator is deallocated when the function returns, with
    // or w/o an exception
    auto iter_dealloc = finally([&]() { NpyIter_Deallocate(iter); });
    if (iter == NULL) {
        return Py_None;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
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
            hashset.insert(*((T *) data));
            data += stride;
        }
    } while(iternext(iter));

    // then we iterate through the map's keys to get the unique values
    T* res = new T[hashset.size()];
    auto it = hashset.begin();
    size_t i = 0;
    for (; it != hashset.end(); it++, i++) {
        res[i] = *it;
    }

    // does this need to have the same lifetime as the array?
    npy_intp dims[1] = {(npy_intp)hashset.size()};
    PyArray_Descr *descr = PyArray_DESCR(self);
    Py_INCREF(descr);
    return PyArray_NewFromDescr(
        &PyArray_Type,
        descr,
        1, // ndim
        dims, // shape
        NULL, // strides
        res, // data
        // This flag is needed to be able to call .sort on it.
        NPY_ARRAY_WRITEABLE, // flags
        NULL // obj
    );
}


// this map contains the functions used for each item size.
typedef std::function<PyObject *(PyArrayObject *)> function_type;
std::unordered_map<int, function_type> unique_funcs = {
    {NPY_BYTE, unique<npy_byte>},
    {NPY_UBYTE, unique<npy_ubyte>},
    {NPY_SHORT, unique<npy_short>},
    {NPY_USHORT, unique<npy_ushort>},
    {NPY_INT, unique<npy_int>},
    {NPY_UINT, unique<npy_uint>},
    {NPY_LONG, unique<npy_long>},
    {NPY_ULONG, unique<npy_ulong>},
    {NPY_LONGLONG, unique<npy_longlong>},
    {NPY_ULONGLONG, unique<npy_ulonglong>},
    {NPY_INT8, unique<npy_int8>},
    {NPY_INT16, unique<npy_int16>},
    {NPY_INT32, unique<npy_int32>},
    {NPY_INT64, unique<npy_int64>},
    {NPY_UINT8, unique<npy_uint8>},
    {NPY_UINT16, unique<npy_uint16>},
    {NPY_UINT32, unique<npy_uint32>},
    {NPY_UINT64, unique<npy_uint64>},
};


extern "C" NPY_NO_EXPORT PyObject *
array_unique(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    /* This is called from Python space, and expects a single numpy array as input.

    It then returns a numpy array containing the unique values of the input array.

    If the input array is not supported, it returns None.
    */
    // this is to allow grabbing the GIL before raising a python exception.
    NPY_ALLOW_C_API_DEF;
    PyArrayObject *self = NULL;
    PyObject *res = NULL;
    if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &self))
        return NULL;
    
    // Making sure the DECREF is called when the function returns, with
    // or w/o an exception
    auto self_decref = finally([&]() { Py_XDECREF(self); });


    try {
        /* Handle zero-sized arrays specially */
        if (PyArray_SIZE(self) == 0) {
            return PyArray_NewLikeArray(
                self,
                NPY_ANYORDER,
                NULL, // descr (use prototype's descr)
                1 // subok
            );
        }

        auto type = PyArray_TYPE(self);
        // we only support data types present in our unique_funcs map
        if (unique_funcs.find(type) == unique_funcs.end()) {
            // Grab GIL before raising an exception
            NPY_ALLOW_C_API;
            PyErr_SetString(PyExc_NotImplementedError, "Unsupported dtype");
            NPY_DISABLE_C_API;
            return NULL;
        }

        res = unique_funcs[type](self);
        return res;
    }
    catch (const std::bad_alloc &e) {
        // Grab GIL before raising an exception
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
        return NULL;
    }
    catch (const std::exception &e) {
        // Grab GIL before raising an exception
        NPY_ALLOW_C_API;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        NPY_DISABLE_C_API;
        return NULL;
    }
}
